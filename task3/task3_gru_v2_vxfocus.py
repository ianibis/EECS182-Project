"""Causal windowed GRU baseline with optional vx-focused model selection.

Comparison: GRU versus residual ridge using exactly the same 19-lag LFP
windows and sequence-specific time-only baseline. The default augmented mode
uses the existing 200-400 Hz cache and nine LFP features per channel; base
uses the original eight features. Training uses OFF training-day training
blocks; early stopping uses only OFF validation-day query blocks; held-out
training blocks are evaluated only after model selection. No test sessions
or stimulation-ON recordings are read.

The GRU is one conventional temporal decoder baseline for later domain-shift
and adaptation studies, NOT the in-context transformer or LoRA implementation.
Its input windows, target definition and output metrics can be reused.

Run from the project root:
  python task3/task3_gru_v2_vxfocus.py --mode augmented --target velocity --feature-group all --select-axis vx
  python task3/task3_gru_v2_vxfocus.py --mode augmented --target velocity --feature-group all --select-axis vx --vx-loss-weight 2

Requires torch, numpy, pandas, h5py and the local Task3 modules. For augmented
mode, first run task3_hf_velocity_v1.py --prepare-only if HF caches are missing.
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

import task3_hf_velocity_v1 as H
from task3_paths import GRU_RESULTS

D = H.D
C = D.C
F = C.F


class VelocityGRU(nn.Module):
    """One prediction per causal feature window; hidden state reset per window."""

    def __init__(self, input_size, output_size=1, projection=48, hidden=64, dropout=0.15):
        super().__init__()
        self.projection = nn.Linear(input_size, projection)
        self.gru = nn.GRU(projection, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden, output_size)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x):
        x = torch.tanh(self.projection(x))
        _, h = self.gru(x)
        return self.readout(self.dropout(h[-1]))


def chronologic_windows(x_flat, n_lags):
    """Convert Task 3's [lag0, lag1, ..., lagN] to [oldest, ..., now]."""
    n_steps = n_lags + 1
    if x_flat.shape[1] % n_steps:
        raise ValueError('Ridge feature count is not divisible by the history length')
    n_features = x_flat.shape[1] // n_steps
    return np.ascontiguousarray(x_flat.reshape(-1, n_steps, n_features)[:, ::-1, :], dtype=np.float32)


def configure(mode, target, n_lags, feature_group):
    if target not in ('vx', 'velocity'):
        raise ValueError('Target must be vx or velocity')
    if mode != 'augmented' and feature_group != 'all':
        raise ValueError('--feature-group high/without-beta requires --mode augmented')
    C.KIN_COLS = D.TARGETS[target]
    C.REFERENCES = ('export_none',)
    C.POST = 1.5
    D.TIME_PROFILE = 'sequence'
    D.NORM = 'own'
    D.SMOOTH_HZ = None
    D.TRIAL_END_PAD = None
    if mode == 'augmented':
        F.load_features = H.load_augmented
        feature_groups = {
            'all': [*F.FEATURE_NAMES, H.NEW_NAME],
            'high': ['pow_70_110', 'pow_130_200', H.NEW_NAME],
            'without-beta': [name for name in [*F.FEATURE_NAMES, H.NEW_NAME]
                             if name not in ('pow_13_30', 'pow_30_50')],
        }
        key = f'task3_gru_augmented_{feature_group.replace("-", "_")}'
        C.FEATURE_SETS[key] = feature_groups[feature_group]
    else:
        F.load_features = H.RAW_LOAD
        key = 'lmp+bands'
    return key


def load_level(session, part, feature_set, n_lags, profile):
    X, Y, position, _ = D.arrays_with_position(session, 'export_none', part, feature_set, n_lags)
    baseline = D.profile_values(profile, session, position)
    if X.shape[0] == 0:
        raise ValueError(f'{session}, {part}: no valid bins')
    return dict(session=session, part=part, X=chronologic_windows(X, n_lags),
                Y=np.asarray(Y, dtype=np.float32),
                baseline=np.asarray(baseline, dtype=np.float32),
                position=position.astype(np.int16))


def predict(model, batch, scale, device, batch_size):
    model.eval()
    pieces = []
    with torch.no_grad():
        for i in range(0, len(batch['X']), batch_size):
            x = torch.from_numpy(batch['X'][i:i + batch_size]).to(device)
            pieces.append(model(x).cpu().numpy())
    return np.concatenate(pieces, axis=0) * scale


def session_score(batch, residual_prediction):
    y, base = batch['Y'].astype(np.float64), batch['baseline'].astype(np.float64)
    pred = base + residual_prediction.astype(np.float64)
    r2 = C.r2(y, pred)
    r2_time = C.r2(y, base)
    row = dict(n_bins=len(y), r2=float(r2.mean()), r2_time=float(r2_time.mean()),
               r2_gain=float((r2 - r2_time).mean()),
               mae=float(np.abs(y - pred).mean()), mae_time=float(np.abs(y - base).mean()))
    for col, axis in enumerate(C.KIN_COLS):
        short = D.SHORT[axis]
        row[f'r2_{short}'] = float(r2[col])
        row[f'gain_{short}'] = float(r2[col] - r2_time[col])
    return row


def validation_gains(model, batches, scale, device, batch_size):
    scores = [session_score(batch, predict(model, batch, scale, device, batch_size))
              for batch in batches]
    gains = {'joint': float(np.mean([score['r2_gain'] for score in scores])),
             'vx': float(np.mean([score['gain_vx'] for score in scores]))}
    if 'gain_vy' in scores[0]:
        gains['vy'] = float(np.mean([score['gain_vy'] for score in scores]))
    return gains


def weighted_mse(predicted, target, vx_loss_weight):
    """Maintain loss scale while changing vx emphasis in joint output."""
    error = (predicted - target).square()
    if error.shape[1] == 1:
        return error.mean()
    weights = predicted.new_tensor([vx_loss_weight, 1.0])
    return (error * weights).sum(dim=1).mean() / weights.sum()


def train_model(train_batches, val_batches, args, device):
    x_train = np.concatenate([b['X'] for b in train_batches], axis=0)
    y_train = np.concatenate([b['Y'] - b['baseline'] for b in train_batches], axis=0)
    scale = np.std(y_train, axis=0, dtype=np.float64).astype(np.float32)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('The training residual target has zero or invalid variance')
    y_train /= scale
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = VelocityGRU(x_train.shape[2], y_train.shape[1], args.projection,
                        args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # The initial readout is zero, i.e. exactly the time-only baseline. A learned
    # model is accepted only if its validation neural gain exceeds this reference.
    best_gain = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = [dict(epoch=0, train_loss=np.nan, val_r2_gain=0.0, val_gain_vx=0.0, val_gain_vy=0.0)]
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            residual = model(x)
            loss = weighted_mse(residual, y, args.vx_loss_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += loss.item() * len(x)
            count += len(x)
        gains = validation_gains(model, val_batches, scale, device, args.batch_size)
        val_gain = gains[args.select_axis]
        train_loss = loss_sum / count
        history.append(dict(epoch=epoch, train_loss=train_loss,
                            val_r2_gain=gains['joint'], val_gain_vx=gains['vx'],
                            val_gain_vy=gains.get('vy', np.nan)))
        print(f'  epoch {epoch:3d}: train residual MSE (scaled) {train_loss:.5f}, '
              f'validation vx gain {gains["vx"]:+.5f}, '
              f'joint gain {gains["joint"]:+.5f}'
              + (' *' if val_gain > best_gain + args.min_delta else ''))
        if val_gain > best_gain + args.min_delta:
            best_gain, best_epoch = val_gain, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f'  early stopping: no improvement for {args.patience} epochs')
                break
    model.load_state_dict(best_state)
    return model, scale, pd.DataFrame(history), best_epoch, best_gain


def evaluate(model, scale, groups, device, args):
    session_rows = []
    predictions = []
    for level, batches in groups.items():
        for idx, batch in enumerate(batches, 1):
            residual = predict(model, batch, scale, device, args.batch_size)
            row = dict(level=level, session=batch['session'],
                       **session_score(batch, residual))
            session_rows.append(row)
            print(f'  {level}: {idx}/{len(batches)} {batch["session"]} gain {row["r2_gain"]:+.4f}')
            pred = batch['baseline'] + residual
            columns = {f'true_{D.SHORT[axis]}': batch['Y'][:, j]
                       for j, axis in enumerate(C.KIN_COLS)}
            columns.update({f'baseline_{D.SHORT[axis]}': batch['baseline'][:, j]
                            for j, axis in enumerate(C.KIN_COLS)})
            columns.update({f'pred_{D.SHORT[axis]}': pred[:, j]
                            for j, axis in enumerate(C.KIN_COLS)})
            predictions.append(pd.DataFrame(dict(level=level, session=batch['session'],
                                                 row_in_subset=np.arange(len(pred)),
                                                 window_position=batch['position'], **columns)))
    return pd.DataFrame(session_rows), pd.concat(predictions, ignore_index=True)


def run(args):
    t0 = time.time()
    if args.lags < 0 or args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise SystemExit('Invalid history, epoch, patience or batch-size parameter')
    if args.vx_loss_weight <= 0 or not np.isfinite(args.vx_loss_weight):
        raise SystemExit('--vx-loss-weight must be finite and positive')
    if args.min_delta < 0 or not 0 <= args.dropout < 1:
        raise SystemExit('--min-delta must be nonnegative and --dropout must be between 0 and 1')
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS unavailable in this PyTorch environment; use --device cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device == 'cpu':
        torch.set_num_threads(min(torch.get_num_threads(), 8))

    key = configure(args.mode, args.target, args.lags, args.feature_group)
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
        raise SystemExit('Missing required files, first examples:\n  ' + '\n  '.join(absent[:8]) +
                         ('\nPrepare high-frequency caches using task3_hf_velocity_v1.py --prepare-only'
                          if args.mode == 'augmented' else ''))

    print(f'GRU {args.mode} {args.feature_group} {args.target}: {len(train_sessions)} train OFF, {len(val_sessions)} validation OFF; '
          f'{args.lags + 1} x 50-ms causal bins, device={device}')
    print('Validation chooses the training epoch; held-out train blocks are evaluated afterwards.')
    print('No stimulation-ON sessions or test days are accessed.')
    print(f'Model selection: validation {args.select_axis} R2 gain; vx loss weight {args.vx_loss_weight:g}')
    print('Selected features:', ', '.join(C.FEATURE_SETS[key]))
    profile = D.fit_time_profile(train_sessions, 'export_none', key, args.lags)
    train_batches = [load_level(s, 'train', key, args.lags, profile) for s in train_sessions]
    val_batches = [load_level(s, 'query', key, args.lags, profile) for s in val_sessions]
    n_train = sum(len(b['Y']) for b in train_batches)
    print(f'Training: {n_train:,} valid bins, '
          f'{train_batches[0]["X"].shape[1]} time steps x {train_batches[0]["X"].shape[2]} features')
    model, scale, history, best_epoch, best_gain = train_model(train_batches, val_batches, args, device)
    print(f'Chosen epoch {best_epoch}, validation {args.select_axis} gain {best_gain:+.5f}')
    held_batches = [load_level(s, 'val', key, args.lags, profile) for s in train_sessions]
    rows, preds = evaluate(model, scale, {'train': train_batches,
                                         'held_out': held_batches, 'val_days': val_batches}, device, args)
    summary = rows.groupby('level', sort=False).agg(
        n_sessions=('session', 'size'), bins_per_session=('n_bins', 'mean'),
        r2=('r2', 'mean'), r2_time=('r2_time', 'mean'), r2_gain=('r2_gain', 'mean'),
        mae=('mae', 'mean'), mae_time=('mae_time', 'mean'))
    print('\nGRU session-mean velocity R2:')
    print(summary.to_string(float_format=lambda x: f'{x:.5f}'))
    axes = ['vx', 'vy'] if args.target == 'velocity' else ['vx']
    axis_columns = [f'{metric}_{axis}' for axis in axes for metric in ('r2', 'gain')]
    print('\nPer-axis session means:')
    print(rows.groupby('level', sort=False)[axis_columns].mean().to_string(float_format=lambda x: f'{x:.5f}'))
    print('\nAugmented ridge reference (vx, 19 lags): held-out gain +0.00733, '
          'validation gain +0.00707.' if args.mode == 'augmented' and args.target == 'vx' and args.lags == 19
          else '\nSelect the corresponding ridge reference for this target/mode.')

    weight_tag = f'{args.vx_loss_weight:g}'.replace('.', 'p')
    stem = (f'task3_gru_v2_vxfocus_{args.mode}_{args.feature_group.replace("-", "_")}_'
            f'{args.target}_lags{args.lags}_h{args.hidden}_seed{args.seed}_'
            f'sel{args.select_axis}_vxw{weight_tag}')
    os.makedirs((GRU_RESULTS / 'training'), exist_ok=True)
    checkpoint_dir = os.path.join((GRU_RESULTS / 'training'), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    rows.to_csv(os.path.join((GRU_RESULTS / 'training'), stem + '_sessions.csv'), index=False)
    preds.to_csv(os.path.join((GRU_RESULTS / 'training'), stem + '_predictions.csv'), index=False)
    history.to_csv(os.path.join((GRU_RESULTS / 'training'), stem + '_training.csv'), index=False)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(dict(state_dict=state,
                    model_config=dict(input_size=train_batches[0]['X'].shape[2],
                                      output_size=len(C.KIN_COLS), projection=args.projection,
                                      hidden=args.hidden, dropout=args.dropout),
                    target_columns=list(C.KIN_COLS), feature_names=C.FEATURE_SETS[key],
                    reference='export_none', n_lags=args.lags,
                    profile={k: v.tolist() for k, v in profile.items()},
                    target_scale=scale.tolist(), best_epoch=best_epoch,
                    val_gain=best_gain, train_sessions=train_sessions,
                    val_sessions=val_sessions,
                    run_config=dict(vars(args))),
               os.path.join(checkpoint_dir, stem + '.pt'))
    with open(os.path.join((GRU_RESULTS / 'training'), stem + '_summary.json'), 'w') as f:
        json.dump(dict(best_epoch=best_epoch, val_gain=best_gain,
                       summary=summary.reset_index().to_dict(orient='records'),
                       args=vars(args)), f, indent=2)
    print(f'Saved checkpoint, predictions, session scores and learning curve under results/task3/.')
    print(f'Done in {time.time() - t0:.0f} s.')


def self_test():
    lagged = np.array([[5, 4, 3, 2, 1, 0]], dtype=np.float32)
    x = chronologic_windows(lagged, 2)
    assert x.shape == (1, 3, 2) and x[0].tolist() == [[1., 0.], [3., 2.], [5., 4.]]
    model = VelocityGRU(2, output_size=1, projection=4, hidden=6, dropout=0)
    batch = dict(X=np.ones((5, 3, 2), np.float32), Y=np.array([[0.], [1.], [2.], [3.], [4.]]),
                 baseline=np.zeros((5, 1)), position=np.arange(5), session='synthetic')
    pred = predict(model, batch, np.array([4.], np.float32), 'cpu', 2)
    assert pred.shape == (5, 1) and np.allclose(pred, 0)
    C.KIN_COLS = ('Index_vx_cm',)
    metrics = session_score(batch, pred)
    assert abs(metrics['r2'] - metrics['r2_time']) < 1e-10
    assert abs(metrics['r2_gain']) < 1e-10
    fake_pred = torch.tensor([[1., 2.], [3., 4.]])
    fake_target = torch.zeros_like(fake_pred)
    assert abs(weighted_mse(fake_pred, fake_target, 1.0).item() - 7.5) < 1e-6
    assert abs(weighted_mse(fake_pred, fake_target, 2.0).item() - 40.0 / 6.0) < 1e-6
    print('Self-test passed: lag chronology, causal inputs, baseline, and weighted loss.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('augmented', 'base'), default='augmented')
    parser.add_argument('--select-axis', choices=('joint', 'vx'), default='joint',
                        help='Choose best epoch by mean joint or horizontal velocity validation R2 gain')
    parser.add_argument('--vx-loss-weight', type=float, default=1.0,
                        help='Relative vx loss weight during joint training; vy weight stays 1')
    parser.add_argument('--feature-group', choices=('all', 'high', 'without-beta'), default='all',
                        help='High uses 70-110, 130-200, 200-400 Hz; without-beta omits 13-30 and 30-50 Hz')
    parser.add_argument('--target', choices=('vx', 'velocity'), default='vx')
    parser.add_argument('--lags', type=int, default=19)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--projection', type=int, default=48)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=256)
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
