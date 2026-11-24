"""Trial-continuous GRU with auxiliary speed and horizontal-direction targets.

Comparison: joint vx/vy GRU with 10% channel dropout versus the same GRU
trained additionally to predict movement speed and horizontal direction.
Auxiliary labels are computed only from the existing training velocity targets.
The velocity heads, validation selection metric, time-only baseline, scored
bins, and OFF train/validation sessions match task3_gru_trial_v2.py.
No ON recordings or test-day sessions are read.

From project root:
  python task3/task3_gru_multitask_v1.py --mode augmented --target velocity
  python task3/task3_gru_multitask_v1.py --mode augmented --target velocity --aux-speed-weight 0 --aux-dir-weight 0

Requires task3_gru_trial_v2.py, its dependencies, and prepared HF caches.
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
from torch.nn import functional as TF
from torch.utils.data import DataLoader, TensorDataset

import task3_gru_trial_v2 as T
from task3_paths import GRU_RESULTS

C = T.C
D = T.D
G = T.G
H = T.H


class MultitaskTrialGRU(T.TrialVelocityGRU):
    """Velocity regression plus auxiliary scalar speed and vx direction heads."""

    def __init__(self, input_size, output_size=2, projection=48, hidden=64,
                 dropout=0.15, warmup=19, channel_dropout=0.10,
                 features_per_channel=9):
        super().__init__(input_size, output_size, projection, hidden, dropout,
                         warmup, channel_dropout, features_per_channel)
        # Preserve the base GRU's initialization and RNG stream when comparing
        # an auxiliary-weight-zero model against task3_gru_trial_v2.py.
        state = torch.get_rng_state()
        try:
            self.speed_head = nn.Linear(hidden, 1)
            self.direction_head = nn.Linear(hidden, 1)
        finally:
            torch.set_rng_state(state)

    def forward(self, x):
        x = torch.tanh(self.projection(self.drop_channels(x)))
        states, _ = self.gru(x)
        states = self.dropout(states[:, self.warmup:, :])
        velocity = self.readout(states)
        speed = self.speed_head(states).squeeze(-1)
        direction = self.direction_head(states).squeeze(-1)
        return velocity, speed, direction


class VelocityOnly(nn.Module):
    """Use the existing scoring/prediction functions without changing them."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


def auxiliary_targets(velocity, valid, speed_mean, speed_scale, direction_threshold):
    """Auxiliary targets from true velocity, never from model predictions."""
    speed = np.sqrt(np.sum(np.square(velocity.astype(np.float64)), axis=2))
    standardized_speed = (speed - speed_mean) / speed_scale
    signed_right = (velocity[:, :, 0] > 0).astype(np.float32)
    direction_valid = valid & (np.abs(velocity[:, :, 0]) >= direction_threshold)
    return (np.nan_to_num(standardized_speed, nan=0.0).astype(np.float32),
            signed_right, direction_valid.astype(np.float32))


def multitask_loss(outputs, residual, valid, speed, direction, direction_valid,
                   args):
    vel, pred_speed, pred_dir = outputs
    loss = T.masked_weighted_mse(vel, residual, valid, args.vx_loss_weight)
    if args.aux_speed_weight > 0:
        speed_err = (pred_speed - speed).square()
        loss = loss + args.aux_speed_weight * (
            (speed_err * valid).sum() / valid.sum().clamp_min(1))
    if args.aux_dir_weight > 0:
        dir_err = TF.binary_cross_entropy_with_logits(pred_dir, direction,
                                                      reduction='none')
        loss = loss + args.aux_dir_weight * (
            (dir_err * direction_valid).sum() / direction_valid.sum().clamp_min(1))
    return loss


def train_model(train_batches, val_batches, args, device):
    all_x = np.concatenate([b['X'] for b in train_batches], axis=0)
    all_valid = np.concatenate([b['valid'] for b in train_batches], axis=0)
    all_velocity = np.concatenate([b['Y'] for b in train_batches], axis=0)
    all_base = np.concatenate([b['baseline'] for b in train_batches], axis=0)
    residual = all_velocity - all_base
    scale = np.std(residual[all_valid].astype(np.float64), axis=0).astype(np.float32)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError('Invalid residual-velocity scale')
    target = np.nan_to_num(residual / scale[None, None, :], nan=0.0).astype(np.float32)

    speeds = np.linalg.norm(all_velocity[all_valid].astype(np.float64), axis=1)
    speed_mean = float(speeds.mean())
    speed_scale = float(speeds.std())
    if not np.isfinite(speed_scale) or speed_scale <= 0:
        raise ValueError('Invalid speed scale')
    aux_speed, aux_dir, dir_valid = auxiliary_targets(
        all_velocity, all_valid, speed_mean, speed_scale, args.direction_threshold)
    loader = DataLoader(TensorDataset(
        torch.from_numpy(all_x), torch.from_numpy(target),
        torch.from_numpy(all_valid.astype(np.float32)),
        torch.from_numpy(aux_speed), torch.from_numpy(aux_dir),
        torch.from_numpy(dir_valid)),
        batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = MultitaskTrialGRU(
        all_x.shape[2], all_velocity.shape[2], args.projection, args.hidden,
        args.dropout, args.lags, args.channel_dropout,
        len(C.FEATURE_SETS[args.feature_set])).to(device)
    vel_only = VelocityOnly(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    best_gain, best_epoch, stale = 0.0, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    history = [dict(epoch=0, train_loss=np.nan, val_gain_joint=0.,
                    val_gain_vx=0., val_gain_vy=0.)]
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, valid_count = 0., 0.
        for x, y, mask, speed, direction, dir_mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            speed, direction = speed.to(device), direction.to(device)
            dir_mask = dir_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = multitask_loss(model(x), y, mask, speed, direction,
                                  dir_mask, args)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            n = mask.sum().item()
            loss_sum += loss.item() * n
            valid_count += n
        gains = T.validation_gains(vel_only, val_batches, scale, device,
                                    args.batch_size)
        current = gains[args.select_axis]
        history.append(dict(epoch=epoch, train_loss=loss_sum / valid_count,
                            val_gain_joint=gains['joint'], val_gain_vx=gains['vx'],
                            val_gain_vy=gains.get('vy', np.nan)))
        print(f'  epoch {epoch:3d}: train multitask loss {loss_sum/valid_count:.5f}, '
              f'validation vx gain {gains["vx"]:+.5f}, '
              f'joint gain {gains["joint"]:+.5f}'
              + (' *' if current > best_gain + args.min_delta else ''))
        if current > best_gain + args.min_delta:
            best_gain, best_epoch = current, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f'  early stopping: no improvement for {args.patience} epochs')
                break
    model.load_state_dict(best_state)
    return (model, scale, pd.DataFrame(history), best_epoch, best_gain,
            int(all_valid.sum()), len(all_valid), speed_mean, speed_scale,
            int(dir_valid.sum()))


def run(args):
    started = time.time()
    if args.target != 'velocity':
        raise SystemExit('This multitask diagnostic requires --target velocity')
    if args.lags < 1 or args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        raise SystemExit('History, batch size, epochs and patience must be positive')
    if not 0 <= args.dropout < 1 or not 0 <= args.channel_dropout < 1:
        raise SystemExit('Dropout probabilities must lie in [0, 1)')
    if args.vx_loss_weight <= 0 or args.aux_speed_weight < 0 or args.aux_dir_weight < 0:
        raise SystemExit('Loss weights must be nonnegative; velocity weight positive')
    if args.direction_threshold <= 0 or args.min_delta < 0:
        raise SystemExit('Invalid direction threshold or min-delta')
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS not available; use --device cpu')
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
    print(f'Trial-continuous multitask GRU: {len(train_sessions)} train OFF, '
          f'{len(val_sessions)} val OFF; device={device}')
    print('Velocity, speed and horizontal-direction supervision; '
          'joint velocity alone determines validation selection and R2.')
    print('No test days or stimulation-ON sessions are accessed.')
    print(f'Feature group: {args.feature_group}; channel dropout: {args.channel_dropout:.0%}; '
          f'aux speed weight: {args.aux_speed_weight:g}; '
          f'aux direction weight: {args.aux_dir_weight:g}; '
          f'direction threshold: {args.direction_threshold:g} cm/s')
    print('Features:', ', '.join(C.FEATURE_SETS[feature_set]))
    profile = D.fit_time_profile(train_sessions, 'export_none', feature_set, args.lags)
    train_batches = [T.load_level(s, 'train', feature_set, args.lags, profile)
                     for s in train_sessions]
    val_batches = [T.load_level(s, 'query', feature_set, args.lags, profile)
                   for s in val_sessions]
    (model, scale, history, epoch, gain, n_bins, n_trials,
     speed_mean, speed_scale, n_dir) = train_model(train_batches, val_batches,
                                                 args, device)
    print(f'Training: {n_trials:,} trials, {n_bins:,} velocity bins, '
          f'{n_dir:,} active-direction bins')
    print(f'Chosen epoch {epoch}, validation {args.select_axis} gain {gain:+.5f}')
    held_batches = [T.load_level(s, 'val', feature_set, args.lags, profile)
                    for s in train_sessions]
    rows, preds = T.evaluate(VelocityOnly(model), scale,
                            {'train': train_batches,
                             'held_out': held_batches,
                             'val_days': val_batches}, args, device)
    summary = rows.groupby('level', sort=False).agg(
        n_sessions=('session', 'size'), bins_per_session=('n_bins', 'mean'),
        r2=('r2', 'mean'), r2_time=('r2_time', 'mean'),
        r2_gain=('r2_gain', 'mean'), mae=('mae', 'mean'),
        mae_time=('mae_time', 'mean'))
    print('\nGRU session-mean velocity R2:')
    print(summary.to_string(float_format=lambda x: f'{x:.5f}'))
    print('\nPer-axis session means:')
    print(rows.groupby('level', sort=False)[['r2_vx', 'gain_vx', 'r2_vy', 'gain_vy']]
          .mean().to_string(float_format=lambda x: f'{x:.5f}'))
    stem = (f'task3_gru_multitask_v1_{args.mode}_{args.feature_group.replace("-", "_")}_'
            f'lags{args.lags}_h{args.hidden}_seed{args.seed}_'
            f'sel{args.select_axis}_cd{args.channel_dropout:g}_'
            f'sw{args.aux_speed_weight:g}_dw{args.aux_dir_weight:g}')
    os.makedirs((GRU_RESULTS / 'training'), exist_ok=True)
    checkpoint_dir = os.path.join((GRU_RESULTS / 'training'), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    rows.to_csv(os.path.join((GRU_RESULTS / 'training'), stem + '_sessions.csv'), index=False)
    preds.to_csv(os.path.join((GRU_RESULTS / 'training'), stem + '_predictions.csv'), index=False)
    history.to_csv(os.path.join((GRU_RESULTS / 'training'), stem + '_training.csv'), index=False)
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    model_config=dict(input_size=train_batches[0]['X'].shape[2],
                                      output_size=len(C.KIN_COLS),
                                      projection=args.projection, hidden=args.hidden,
                                      dropout=args.dropout, warmup=args.lags,
                                      channel_dropout=args.channel_dropout,
                                      features_per_channel=len(C.FEATURE_SETS[feature_set])),
                    temporal_mode='trial_continuous', reference='export_none',
                    target_columns=list(C.KIN_COLS), feature_names=C.FEATURE_SETS[feature_set],
                    n_lags=args.lags, target_scale=scale.tolist(),
                    speed_mean=speed_mean, speed_scale=speed_scale,
                    direction_threshold=args.direction_threshold,
                    profile={k: v.tolist() for k, v in profile.items()},
                    best_epoch=epoch, val_gain=gain,
                    train_sessions=train_sessions, val_sessions=val_sessions,
                    run_config=vars(args)),
               os.path.join(checkpoint_dir, stem + '.pt'))
    with open(os.path.join((GRU_RESULTS / 'training'), stem + '_summary.json'), 'w') as f:
        json.dump(dict(best_epoch=epoch, val_gain=gain,
                       summary=summary.reset_index().to_dict('records'),
                       args=vars(args)), f, indent=2)
    print('Saved checkpoint, predictions, session scores and learning curve under results/task3/.')
    print(f'Done in {time.time() - started:.0f} s.')


def self_test():
    C.KIN_COLS = D.TARGETS['velocity']
    model = MultitaskTrialGRU(18, output_size=2, projection=8, hidden=12,
                              dropout=0, warmup=3, channel_dropout=0.1,
                              features_per_channel=3)
    model.eval()
    x = torch.randn(2, 8, 18)
    velocity, speed, direction = model(x)
    assert velocity.shape == (2, 5, 2)
    assert speed.shape == (2, 5) and direction.shape == (2, 5)
    assert torch.equal(velocity, torch.zeros_like(velocity))
    a = np.array([[[3., 4.], [-8., 6.], [0., 0.]]], np.float32)
    valid = np.ones((1, 3), dtype=bool)
    speed_target, sign_target, direction_valid = auxiliary_targets(a, valid, 0., 1., 5.)
    assert np.allclose(speed_target, [[5., 10., 0.]])
    assert np.allclose(sign_target, [[1., 0., 0.]])
    assert np.array_equal(direction_valid, [[0., 1., 0.]])
    args = argparse.Namespace(vx_loss_weight=1., aux_speed_weight=0.2,
                              aux_dir_weight=0.2)
    model.train()
    loss = multitask_loss(model(x), torch.zeros(2, 5, 2),
                          torch.ones(2, 5), torch.zeros(2, 5),
                          torch.zeros(2, 5), torch.ones(2, 5), args)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.projection.weight.grad is not None
    assert model.speed_head.weight.grad is not None
    assert model.direction_head.weight.grad is not None
    # With zero auxiliary weights, the original velocity output and loss match.
    torch.manual_seed(17)
    base = T.TrialVelocityGRU(18, 2, 8, 12, 0., 3, 0., 3)
    torch.manual_seed(17)
    multi = MultitaskTrialGRU(18, 2, 8, 12, 0., 3, 0., 3)
    assert all(torch.equal(a, b) for (name, a), b in
               zip(base.state_dict().items(), [multi.state_dict()[name]
                                               for name in base.state_dict()]))
    base.eval()
    multi.eval()
    assert torch.equal(base(x), multi(x)[0])
    print('Self-test passed: shapes, auxiliary labels and masking, gradients, '
          'zero-residual initialization, and base-velocity equivalence.')


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
    parser.add_argument('--channel-dropout', type=float, default=0.10)
    parser.add_argument('--aux-speed-weight', type=float, default=0.2)
    parser.add_argument('--aux-dir-weight', type=float, default=0.2)
    parser.add_argument('--direction-threshold', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=8)
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
