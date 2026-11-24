"""Trial-continuous GRU with trial-consistent channel dropout for Task 3 OFF velocity decoding.

Comparison: keep the same augmented LFP features, velocity targets, trial splits,
time-only baseline and scored 50-ms bins as the 20-bin windowed GRU, but carry
the GRU hidden state through each trial instead of resetting it for every bin.
The model resets between trials. The first 19 bins are a causal warm-up; the
following 40 bins have exactly the same feature-history validity mask as the
windowed decoder. During training only, channel dropout randomly masks all
features of an electrode for an entire trial (default probability 0.10). Only valid target positions contribute to training or scoring.

Validation OFF days choose the best epoch. Held-out training blocks are scored
only after model selection. No test-day or stimulation-ON data are accessed.

Run from the project root, with Task3/task3_hf_velocity_v1.py caches prepared:
  python task3/task3_gru_trial_v2.py --mode augmented --target velocity --channel-dropout 0.10
  python task3/task3_gru_trial_v2.py --mode augmented --target velocity --channel-dropout 0.00

Requires task3_gru_v2_vxfocus.py, task3_hf_velocity_v1.py, the Task 3 modules,
PyTorch, NumPy, pandas, and the original Box project data.
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import task3_gru_v2_vxfocus as G
from task3_paths import GRU_RESULTS

D = G.D
C = G.C
F = G.F
H = G.H


class TrialVelocityGRU(nn.Module):
    """One output per scored bin, with state shared across bins of one trial."""

    def __init__(self, input_size, output_size=2, projection=48, hidden=64,
                 dropout=0.15, warmup=19, channel_dropout=0.0,
                 features_per_channel=1):
        super().__init__()
        if input_size % features_per_channel:
            raise ValueError('Input size must be divisible by features per channel')
        if not 0 <= channel_dropout < 1:
            raise ValueError('Channel dropout must be in [0, 1)')
        self.projection = nn.Linear(input_size, projection)
        self.gru = nn.GRU(projection, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden, output_size)
        self.warmup = warmup
        self.channel_dropout = channel_dropout
        self.features_per_channel = features_per_channel
        self.n_channels = input_size // features_per_channel
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def drop_channels(self, x):
        """Mask entire electrodes consistently across a trial; evaluation is unmodified."""
        if not self.training or self.channel_dropout == 0:
            return x
        n_trials, n_steps, _ = x.shape
        keep = (torch.rand(n_trials, 1, self.n_channels, 1, device=x.device)
                >= self.channel_dropout).to(x.dtype)
        return ((x.reshape(n_trials, n_steps, self.n_channels,
                           self.features_per_channel) * keep)
                / (1 - self.channel_dropout)).reshape(n_trials, n_steps, -1)

    def forward(self, x):
        x = torch.tanh(self.projection(self.drop_channels(x)))
        outputs, _ = self.gru(x)
        return self.readout(self.dropout(outputs[:, self.warmup:, :]))


def make_trials(feat, rows, kin, session, profile, n_lags):
    """Extract all trial steps and exactly the original 19-lag scored-bin mask."""
    n_score = int(round((C.PRE + C.POST) / C.L.BIN_S))
    warmup_s = n_lags * C.L.BIN_S
    # Original windowed decoder defines scored bins and their validity.
    _, Y, valid, score_bins = F.trial_arrays(
        feat, rows, kin=kin, align=C.ALIGN, pre=C.PRE, post=C.POST,
        n_lags=n_lags, kin_cols=C.KIN_COLS)
    # Include the n_lags immediately preceding the first scored bin, then
    # process each scored bin once. No recurrent state crosses trial boundaries.
    X_warm, _, _, all_bins = F.trial_arrays(
        feat, rows, kin=None, align=C.ALIGN, pre=C.PRE + warmup_s,
        post=C.POST, n_lags=0)
    if X_warm.shape[1] != n_lags + n_score:
        raise ValueError('Unexpected number of warm-up or score bins')
    if not np.array_equal(all_bins[:, n_lags:], score_bins):
        raise ValueError('Warm-up bins do not align with the original scored bins')
    if Y.shape[:2] != valid.shape or valid.shape[1] != n_score:
        raise ValueError('Unexpected scored velocity shape')
    if len(rows) == 0 or not np.any(valid):
        raise ValueError(f'{session}: no scored trials or valid bins')

    positions = np.broadcast_to(np.arange(n_score), valid.shape)
    baseline = D.profile_values(profile, session, positions)
    X = X_warm[:, :, :X_warm.shape[-1]].astype(np.float32, copy=False)
    # Missing input bins are represented by zeros in normalized feature units;
    # their corresponding target bins are never scored unless all historical
    # inputs and the target are valid (the original windowed mask).
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    base = np.asarray(baseline, dtype=np.float32)
    ids = rows.trial.to_numpy(dtype=int)
    return dict(session=session, X=X, Y=Y, baseline=base,
                valid=valid.astype(bool), trial_ids=ids,
                bin_ids=score_bins, position=positions)


def load_level(session, part, feature_set, n_lags, profile):
    norm_rows = C.trial_rows(session, 'pool') if session in C.splits()['calibration'] else None
    feat = F.load_features(session, C.PATHS, variant='export_none', normalize=True, trials=norm_rows)
    keep = [feat['feature_names'].index(name) for name in C.FEATURE_SETS[feature_set]]
    feat = dict(feat, X=np.clip(feat['X'][:, :, keep], -C.CLIP_Z, C.CLIP_Z),
                feature_names=C.FEATURE_SETS[feature_set])
    return make_trials(feat, C.trial_rows(session, part), C.kinematics(session),
                       session, profile, n_lags)


def session_score(batch, residual_prediction):
    mask = batch['valid']
    y = batch['Y'][mask].astype(np.float64)
    baseline = batch['baseline'][mask].astype(np.float64)
    predicted = baseline + residual_prediction[mask].astype(np.float64)
    r2 = C.r2(y, predicted)
    r2_time = C.r2(y, baseline)
    row = dict(n_trials=len(batch['trial_ids']), n_bins=int(mask.sum()),
               r2=float(r2.mean()), r2_time=float(r2_time.mean()),
               r2_gain=float((r2 - r2_time).mean()),
               mae=float(np.abs(y - predicted).mean()),
               mae_time=float(np.abs(y - baseline).mean()))
    for index, axis in enumerate(C.KIN_COLS):
        short = D.SHORT[axis]
        row[f'r2_{short}'] = float(r2[index])
        row[f'gain_{short}'] = float(r2[index] - r2_time[index])
    return row


def predict(model, batch, scale, device, batch_size):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(batch['X']), batch_size):
            x = torch.from_numpy(batch['X'][start:start + batch_size]).to(device)
            pieces.append(model(x).cpu().numpy())
    return np.concatenate(pieces, axis=0) * scale[None, None, :]


def validation_gains(model, batches, scale, device, batch_size):
    scores = [session_score(batch, predict(model, batch, scale, device, batch_size))
              for batch in batches]
    gains = {'joint': float(np.mean([r['r2_gain'] for r in scores])),
             'vx': float(np.mean([r['gain_vx'] for r in scores]))}
    if 'gain_vy' in scores[0]:
        gains['vy'] = float(np.mean([r['gain_vy'] for r in scores]))
    return gains


def masked_weighted_mse(prediction, target, mask, vx_loss_weight):
    """Mean squared error per valid bin, weighting output axes equally by default."""
    err = (prediction - target).square()
    if err.shape[2] == 2:
        weights = prediction.new_tensor([vx_loss_weight, 1.0])
        err = (err * weights[None, None, :]).sum(dim=2) / weights.sum()
    else:
        err = err.mean(dim=2)
    return (err * mask).sum() / mask.sum().clamp_min(1)


def train_model(train_batches, val_batches, args, device):
    all_x = np.concatenate([batch['X'] for batch in train_batches], axis=0)
    all_mask = np.concatenate([batch['valid'] for batch in train_batches], axis=0)
    all_y = np.concatenate([batch['Y'] - batch['baseline'] for batch in train_batches], axis=0)
    scale = np.std(all_y[all_mask].astype(np.float64), axis=0).astype(np.float32)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError('Invalid training residual scale')
    all_y = np.nan_to_num(all_y / scale[None, None, :], nan=0.0).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(all_x), torch.from_numpy(all_y),
                                      torch.from_numpy(all_mask.astype(np.float32))),
                        batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = TrialVelocityGRU(all_x.shape[2], all_y.shape[2], args.projection,
                             args.hidden, args.dropout, args.lags,
                             channel_dropout=args.channel_dropout,
                             features_per_channel=len(C.FEATURE_SETS[args.feature_set])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_gain, best_epoch = 0.0, 0
    best_state = copy.deepcopy(model.state_dict())
    history = [dict(epoch=0, train_loss=np.nan, val_gain_joint=0.0, val_gain_vx=0.0,
                    val_gain_vy=0.0)]
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, valid_count = 0.0, 0.0
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_weighted_mse(model(x), y, mask, args.vx_loss_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            valid = mask.sum().item()
            loss_sum += loss.item() * valid
            valid_count += valid
        gains = validation_gains(model, val_batches, scale, device, args.batch_size)
        selected_gain = gains[args.select_axis]
        history.append(dict(epoch=epoch, train_loss=loss_sum / valid_count,
                            val_gain_joint=gains['joint'], val_gain_vx=gains['vx'],
                            val_gain_vy=gains.get('vy', np.nan)))
        print(f'  epoch {epoch:3d}: train residual MSE (scaled) {loss_sum / valid_count:.5f}, '
              f'validation vx gain {gains["vx"]:+.5f}, '
              f'joint gain {gains["joint"]:+.5f}'
              + (' *' if selected_gain > best_gain + args.min_delta else ''))
        if selected_gain > best_gain + args.min_delta:
            best_gain, best_epoch = selected_gain, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f'  early stopping: no improvement for {args.patience} epochs')
                break
    model.load_state_dict(best_state)
    return model, scale, pd.DataFrame(history), best_epoch, best_gain, int(all_mask.sum()), len(all_mask)


def evaluate(model, scale, groups, args, device):
    rows = []
    predictions = []
    for level, batches in groups.items():
        for number, batch in enumerate(batches, 1):
            residual = predict(model, batch, scale, device, args.batch_size)
            row = dict(level=level, session=batch['session'], **session_score(batch, residual))
            rows.append(row)
            print(f'  {level}: {number}/{len(batches)} {batch["session"]} gain {row["r2_gain"]:+.4f}')
            keep = batch['valid']
            table = dict(level=level, session=batch['session'],
                         trial=np.broadcast_to(batch['trial_ids'][:, None], keep.shape)[keep],
                         bin_id=batch['bin_ids'][keep],
                         window_position=batch['position'][keep])
            full = batch['baseline'] + residual
            for axis_index, axis in enumerate(C.KIN_COLS):
                name = D.SHORT[axis]
                table[f'true_{name}'] = batch['Y'][:, :, axis_index][keep]
                table[f'baseline_{name}'] = batch['baseline'][:, :, axis_index][keep]
                table[f'pred_{name}'] = full[:, :, axis_index][keep]
            predictions.append(pd.DataFrame(table))
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def run(args):
    started = time.time()
    if args.target != 'velocity':
        raise SystemExit('This first trial-continuous comparison uses joint vx/vy (--target velocity)')
    if args.lags < 1 or args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        raise SystemExit('History, batch size, epochs and patience must be positive')
    if args.vx_loss_weight <= 0 or not np.isfinite(args.vx_loss_weight):
        raise SystemExit('--vx-loss-weight must be finite and positive')
    if not 0 <= args.dropout < 1 or args.min_delta < 0:
        raise SystemExit('Invalid dropout or min-delta')
    if not 0 <= args.channel_dropout < 1:
        raise SystemExit('--channel-dropout must be in [0, 1)')
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS not available; choose --device cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device == 'cpu':
        torch.set_num_threads(min(8, torch.get_num_threads()))

    feature_set = G.configure(args.mode, args.target, args.lags, args.feature_group)
    args.feature_set = feature_set
    train_sessions = [s for s in C.sessions('train', 'no_stim') if C.has_cm_velocity(s)]
    val_sessions = C.sessions('val', 'no_stim')
    names = [*train_sessions, *val_sessions]
    required = [os.path.join(C.PATHS.root, 'features', 'export_none', f'{s}.h5') for s in names]
    required += [os.path.join(C.PATHS.trial_dir, f'{s}_trials.csv') for s in names]
    required += [os.path.join(C.PATHS.kin_dir, f'{s}_bins50ms.csv') for s in names]
    if args.mode == 'augmented':
        required += [H.hf_path(s) for s in names]
    absent = [p for p in required if not os.path.isfile(p)]
    if absent:
        raise SystemExit('Missing data files (first examples):\n  ' + '\n  '.join(absent[:8]))

    print(f'Trial-continuous GRU {args.mode} {args.feature_group}: '
          f'{len(train_sessions)} train OFF and {len(val_sessions)} validation OFF; device={device}')
    print('Continuous hidden state within each trial; reset between trials; '
          f'{args.lags} warm-up bins, {int(round((C.PRE + C.POST) / C.L.BIN_S))} scored bins.')
    print('Validation selects epoch; held-out training blocks scored afterward. No test or ON data.')
    print(f'Epoch selection: {args.select_axis} full-prediction neural R2 gain')
    print(f'Channel dropout: {args.channel_dropout:.0%} per electrode and trial during training only')
    print('Features:', ', '.join(C.FEATURE_SETS[feature_set]))
    profile = D.fit_time_profile(train_sessions, 'export_none', feature_set, args.lags)
    train_batches = [load_level(s, 'train', feature_set, args.lags, profile) for s in train_sessions]
    val_batches = [load_level(s, 'query', feature_set, args.lags, profile) for s in val_sessions]
    model, scale, history, epoch, gain, n_bins, n_trials = train_model(train_batches, val_batches,
                                                                       args, device)
    print(f'Training: {n_trials:,} trials, {n_bins:,} valid bins, '
          f'{args.lags + int(round((C.PRE + C.POST) / C.L.BIN_S))} steps x '
          f'{train_batches[0]["X"].shape[2]} features')
    print(f'Chosen epoch {epoch}, validation {args.select_axis} gain {gain:+.5f}')
    held_batches = [load_level(s, 'val', feature_set, args.lags, profile) for s in train_sessions]
    rows, preds = evaluate(model, scale, {'train': train_batches,
                                         'held_out': held_batches,
                                         'val_days': val_batches}, args, device)
    summary = rows.groupby('level', sort=False).agg(
        n_sessions=('session', 'size'), bins_per_session=('n_bins', 'mean'),
        r2=('r2', 'mean'), r2_time=('r2_time', 'mean'), r2_gain=('r2_gain', 'mean'),
        mae=('mae', 'mean'), mae_time=('mae_time', 'mean'))
    print('\nGRU session-mean velocity R2:')
    print(summary.to_string(float_format=lambda x: f'{x:.5f}'))
    print('\nPer-axis session means:')
    print(rows.groupby('level', sort=False)[['r2_vx', 'gain_vx', 'r2_vy', 'gain_vy']]
          .mean().to_string(float_format=lambda x: f'{x:.5f}'))

    tag = f'{args.vx_loss_weight:g}'.replace('.', 'p')
    cd_tag = f'{args.channel_dropout:g}'.replace('.', 'p')
    stem = (f'task3_gru_trial_v2_{args.mode}_{args.feature_group.replace("-", "_")}_'
            f'{args.target}_lags{args.lags}_h{args.hidden}_seed{args.seed}_'
            f'sel{args.select_axis}_vxw{tag}_cd{cd_tag}')
    os.makedirs((GRU_RESULTS / 'training'), exist_ok=True)
    checkpoint_dir = os.path.join((GRU_RESULTS / 'training'), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    rows.to_csv(os.path.join((GRU_RESULTS / 'training'), stem + '_sessions.csv'), index=False)
    preds.to_csv(os.path.join((GRU_RESULTS / 'training'), stem + '_predictions.csv'), index=False)
    history.to_csv(os.path.join((GRU_RESULTS / 'training'), stem + '_training.csv'), index=False)
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    model_config=dict(input_size=train_batches[0]['X'].shape[2],
                                      output_size=len(C.KIN_COLS), projection=args.projection,
                                      hidden=args.hidden, dropout=args.dropout, warmup=args.lags,
                                      channel_dropout=args.channel_dropout,
                                      features_per_channel=len(C.FEATURE_SETS[feature_set])),
                    temporal_mode='trial_continuous', reference='export_none',
                    target_columns=list(C.KIN_COLS), feature_names=C.FEATURE_SETS[feature_set],
                    n_lags=args.lags, target_scale=scale.tolist(),
                    profile={k: v.tolist() for k, v in profile.items()},
                    best_epoch=epoch, val_gain=gain,
                    train_sessions=train_sessions, val_sessions=val_sessions,
                    run_config=vars(args)),
               os.path.join(checkpoint_dir, stem + '.pt'))
    with open(os.path.join((GRU_RESULTS / 'training'), stem + '_summary.json'), 'w') as file:
        json.dump(dict(best_epoch=epoch, val_gain=gain,
                       summary=summary.reset_index().to_dict('records'), args=vars(args)),
                  file, indent=2)
    print('Saved checkpoint, predictions, session scores and learning curve under results/task3/.')
    print(f'Done in {time.time() - started:.0f} s.')


def self_test():
    C.KIN_COLS = D.TARGETS['velocity']
    D.TIME_PROFILE = 'pooled'
    n_lags = 3
    n_score = int(round((C.PRE + C.POST) / C.L.BIN_S))
    bin_id = np.arange(200)
    feat = dict(bin_id=bin_id, X=np.arange(200, dtype=np.float32)[:, None, None],
                feature_names=['synthetic'], channel_ids=np.array([1]))
    rows = pd.DataFrame([dict(trial=1, t_go=5.0), dict(trial=2, t_go=7.0)])
    kin = pd.DataFrame({'Index_vx_cm': np.sin(bin_id / 5),
                        'Index_vy_cm': np.cos(bin_id / 6)}, index=bin_id)
    profile = {'all': np.zeros((n_score, 2), dtype=np.float32)}
    batch = make_trials(feat, rows, kin, 'synthetic', profile, n_lags)
    assert batch['X'].shape == (2, n_lags + n_score, 1)
    assert batch['Y'].shape == (2, n_score, 2)
    _, expected_y, expected_valid, expected_bins = F.trial_arrays(
        feat, rows, kin=kin, align=C.ALIGN, pre=C.PRE, post=C.POST, n_lags=n_lags,
        kin_cols=C.KIN_COLS)
    assert np.array_equal(batch['bin_ids'], expected_bins)
    assert np.array_equal(batch['valid'], expected_valid)
    assert np.allclose(batch['Y'][expected_valid], expected_y[expected_valid])
    model = TrialVelocityGRU(1, 2, projection=4, hidden=6, dropout=0, warmup=n_lags)
    zeros = predict(model, batch, np.array([1.0, 1.0], np.float32), 'cpu', 1)
    assert zeros.shape == batch['Y'].shape and np.allclose(zeros, 0)
    metrics = session_score(batch, zeros)
    assert metrics['r2_gain'] == 0.0
    torch.nn.init.normal_(model.readout.weight)
    model.eval()
    with torch.no_grad():
        batched = model(torch.from_numpy(batch['X']))
        separate = torch.cat([model(torch.from_numpy(batch['X'][j:j + 1])) for j in range(2)])
        assert torch.allclose(batched, separate, atol=1e-6)
        changed = batch['X'].copy()
        changed[:, -1, :] += 1000
        before = model(torch.from_numpy(batch['X']))[:, :-1]
        after = model(torch.from_numpy(changed))[:, :-1]
        assert torch.allclose(before, after, atol=1e-6)
    mask = torch.tensor([[1., 0.], [1., 1.]])
    pred = torch.tensor([[[1., 2.], [9., 9.]], [[3., 4.], [5., 6.]]])
    assert abs(masked_weighted_mse(pred, torch.zeros_like(pred), mask, 1).item()
               - ((1+4)/2 + (9+16)/2 + (25+36)/2)/3) < 1e-6
    torch.manual_seed(1)
    channel_model = TrialVelocityGRU(12, 2, projection=4, hidden=6, dropout=0,
                                     warmup=3, channel_dropout=0.5,
                                     features_per_channel=3)
    ones = torch.ones(64, 7, 12)
    channel_model.train()
    dropped = channel_model.drop_channels(ones).reshape(64, 7, 4, 3)
    assert torch.all((dropped == 0) | (dropped == 2))
    assert torch.equal(dropped[:, :1].expand_as(dropped), dropped)
    assert torch.equal(dropped[:, :, :, :1].expand_as(dropped), dropped)
    assert (dropped == 0).any() and (dropped == 2).any()
    out = channel_model(ones)
    loss = out.square().mean()
    loss.backward()
    assert channel_model.projection.weight.grad is not None
    channel_model.eval()
    assert torch.equal(channel_model.drop_channels(ones), ones)
    print('Self-test passed: exact scored-bin alignment, baseline, masked loss, '
          'causality, per-trial state reset, and channel-wise dropout.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('augmented', 'base'), default='augmented')
    parser.add_argument('--target', choices=('velocity',), default='velocity')
    parser.add_argument('--feature-group', choices=('all', 'high', 'without-beta'), default='all')
    parser.add_argument('--select-axis', choices=('joint', 'vx'), default='joint')
    parser.add_argument('--vx-loss-weight', type=float, default=1.0)
    parser.add_argument('--lags', type=int, default=19)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--projection', type=int, default=48)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--channel-dropout', type=float, default=0.10,
                        help='Training-only probability of masking an entire electrode per trial')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Number of complete trials in one optimizer batch')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--min-delta', type=float, default=0.00005)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)
