"""Train causal window MLPs on OFF, recorded ON, and 10x ON data."""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'task3'))

from task3_gru_evaluate import CHECKPOINT, ROOT, gru
from task3_paths import MLP_RESULTS, feature_root
from task3_gru_multitask_v1 import auxiliary_targets, multitask_loss

OUTPUT = MLP_RESULTS
CONFIG = dict(input_size=315, projection=48, hidden=64, lags=19, dropout=0.15,
              channel_dropout=0.10, features_per_channel=9)


class WindowMLP(nn.Module):
    """Predict each bin from the current and preceding 19 feature bins."""

    def __init__(self, input_size, projection, hidden, lags, dropout,
                 channel_dropout, features_per_channel):
        super().__init__()
        self.lags = lags
        self.features_per_channel = features_per_channel
        self.channel_dropout = channel_dropout
        self.projection = nn.Linear(input_size, projection)
        self.hidden = nn.Linear((lags + 1) * projection, hidden)
        self.dropout = nn.Dropout(dropout)
        self.velocity = nn.Linear(hidden, 2)
        self.speed = nn.Linear(hidden, 1)
        self.direction = nn.Linear(hidden, 1)

    def hidden_features(self, x):
        if self.training:
            n_trials, n_bins, width = x.shape
            channels = width // self.features_per_channel
            keep = x.new_empty((n_trials, 1, channels, 1)).bernoulli_(1 - self.channel_dropout)
            x = (x.reshape(n_trials, n_bins, channels, self.features_per_channel)
                 * keep / (1 - self.channel_dropout)).reshape(n_trials, n_bins, width)
        projected = torch.tanh(self.projection(x))
        windows = projected.unfold(1, self.lags + 1, 1).transpose(-1, -2).flatten(2)
        return self.dropout(torch.tanh(self.hidden(windows)))

    def forward(self, x):
        hidden = self.hidden_features(x)
        return self.velocity(hidden), self.speed(hidden).squeeze(-1), self.direction(hidden).squeeze(-1)


def read_batches(sessions, source, split, manifest, splits, paths, profile):
    batches = {'train': {}, 'held_out': {}} if split == 'train' else {'val_days': {}}
    normalization = {**splits, 'calibration': dict(splits['calibration'])}
    for session in sessions:
        if split == 'train':
            # Original GRU training uses the whole included-trial timing window.
            trial_table = pd.read_csv(Path(paths.trial_dir) / f'{session}_trials.csv')
            normalization['calibration'][session] = {
                'pool': trial_table.loc[trial_table.include == 1, 'trial'].tolist()
            }
        base = feature_root(ROOT) / 'features' / gru.feature_variant(source) / f'{session}.h5'
        if not base.is_file():
            raise FileNotFoundError(base)
        feat = gru.session_features(ROOT, session, source, paths, normalization, manifest)
        parts = {'train': 'train', 'held_out': 'val'} if split == 'train' else {'val_days': 'query'}
        for level, part in parts.items():
            batches[level][session] = gru.trial_batch(
                feat, session, part, paths, splits, manifest, profile, CONFIG['lags'])
        print(f'Loaded {split} {source} {session}', flush=True)
    return batches


def time_profile(batches, manifest):
    profiles = {}
    for sequence in sorted({str(manifest.loc[s, 'sequence']) for s in batches}):
        records = [b for s, b in batches.items() if str(manifest.loc[s, 'sequence']) == sequence]
        velocity = np.concatenate([b['Y'] for b in records])
        valid = np.concatenate([b['valid'] for b in records])
        counts = valid.sum(axis=0)
        overall_mean = velocity[valid].mean(axis=0, dtype=np.float64)
        means = np.broadcast_to(overall_mean, velocity.shape[1:]).copy()
        np.divide(np.where(valid[..., None], velocity, 0).sum(axis=0, dtype=np.float64),
                  counts[:, None], out=means, where=counts[:, None] > 0)
        profiles[sequence] = means.astype(np.float32)
    return profiles


def sessions_for_split(manifest, splits, split, off):
    return [
        session for session in splits['sessions'][split]
        if (manifest.loc[session, 'condition'] == 'no_stim') == off
        and bool(manifest.loc[session, 'touch_available'])
    ]


def verify_splits(manifest, splits):
    for split in ('train', 'val', 'test'):
        sessions = splits['sessions'][split]
        days = set(splits['days'][split])
        if len(set(sessions)) != len(sessions) or set(manifest.loc[sessions, 'date']) != days:
            raise ValueError(f'{split}: session assignments differ from the split days')
        for other in ('train', 'val', 'test'):
            if other != split and days.intersection(splits['days'][other]):
                raise ValueError(f'{split} and {other} overlap by day')


def set_baseline(batches, profile, manifest):
    for level in batches.values():
        for session, batch in level.items():
            sequence = str(manifest.loc[session, 'sequence'])
            if sequence not in profile:
                raise ValueError(f'{session}: no training trajectory for sequence {sequence}')
            batch['baseline'] = profile[sequence][batch['position']]


def score_batches(model, batches, scale, device):
    rows = []
    for session, batch in batches.items():
        pred = gru.predict(model, batch, device)
        rows.append(dict(**gru.model_scores(session, pred, batch, scale),
                         n_trials=len(batch['trial_ids'])))
    return pd.DataFrame(rows)


def train(name, batches, profile, args, manifest):
    torch.manual_seed(args.seed)
    fields = ('X', 'Y', 'baseline', 'valid')
    x, velocity, baseline, mask = (
        np.concatenate([b[key] for b in batches['train'].values()]) for key in fields
    )
    residual = velocity - baseline
    scale = residual[mask].astype(np.float64).std(axis=0).astype(np.float32)
    speed = np.linalg.norm(velocity[mask].astype(np.float64), axis=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0) or speed.std() <= 0:
        raise ValueError(f'{name}: invalid training target scales')
    target = np.nan_to_num(residual / scale)
    aux = auxiliary_targets(velocity, mask, speed.mean(), speed.std(), args.direction_threshold)
    loader = DataLoader(TensorDataset(*(torch.from_numpy(a) for a in
                        (x, target, mask.astype(np.float32), *aux))),
                        batch_size=8, shuffle=True)
    model = WindowMLP(**CONFIG).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    best_gain, best_epoch, stale = -np.inf, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_bins = 0.0, 0
        for items in loader:
            inputs, y, valid, aux_speed, direction, dir_valid = (a.to(args.device) for a in items)
            optimizer.zero_grad(set_to_none=True)
            loss = multitask_loss(model(inputs), y, valid, aux_speed, direction, dir_valid, args)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(valid.sum().item())
            n_bins += count
            total_loss += loss.item() * count
        scores = score_batches(model, batches['val_days'], scale, args.device)
        gain = scores.neural_gain.mean()
        if not np.isfinite(gain):
            raise ValueError(f'{name}: nonfinite validation gain')
        history.append(dict(epoch=epoch, train_loss=total_loss / n_bins,
                            val_r2=scores.r2.mean(), val_neural_gain=gain))
        print(f'{name} epoch {epoch:02d} loss={total_loss / n_bins:.5f} '
              f'val R²={scores.r2.mean():.5f} gain={gain:+.5f}', flush=True)
        if gain > best_gain + 0.00005:
            best_epoch, best_gain, stale = epoch, gain, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale == 8:
                break
    model.load_state_dict(best_state)
    data = pd.concat([score_batches(model, part, scale, args.device).assign(level=level)
                      for level, part in batches.items()], ignore_index=True)
    data['condition'] = data.session.map(manifest.condition)
    data.insert(0, 'run', name)
    metrics = ['r2_vx', 'r2_vy', 'r2', 'r2_time', 'neural_gain']
    grouped = data.groupby(['run', 'level'], sort=False)
    summary = grouped[metrics].mean().join(grouped.agg(
        n_sessions=('session', 'size'), n_trials=('n_trials', 'sum'), n_bins=('n_bins', 'sum')
    )).reset_index()
    output = (getattr(args, 'output', None) or OUTPUT) / name
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / 'sessions.csv', index=False)
    summary.to_csv(output / 'aggregate.csv', index=False)
    pd.DataFrame(history).to_csv(output / 'training.csv', index=False)
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in best_state.items()},
                    run=name, model_config=CONFIG, best_epoch=best_epoch, seed=args.seed,
                    train_sessions=list(batches['train']), val_sessions=list(batches['val_days']),
                    profile=profile, target_scale=scale,
                    feature_names=[*gru.features.FEATURE_NAMES, 'pow_200_400'],
                    target_columns=['Index_vx_cm', 'Index_vy_cm']),
               output / 'checkpoint.pt')
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', choices=('off', 'on', 'all'), default='all')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gru-checkpoint', nargs='?', type=Path, const=CHECKPOINT,
                        help='Reuse the saved GRU OFF trajectory and verify its splits. '
                             'Supply a path or omit the path to use the project checkpoint.')
    parser.add_argument('--output', type=Path,
                        help='Write training results to this directory instead of the default.')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    args.vx_loss_weight = 1.0
    args.aux_speed_weight = args.aux_dir_weight = 0.2
    args.direction_threshold = 5.0
    torch.set_num_threads(4)

    original_high_path = gru.high_band_path

    def high_band_path(root, session, source):
        existing = original_high_path(root, session, source)
        if source == 'recorded' and not existing.is_file():
            archived = ROOT / 'Task4' / 'results' / 'mlp' / 'recorded_high_band' / existing.name
            return archived if archived.is_file() else OUTPUT / 'recorded_high_band' / existing.name
        return existing

    gru.high_band_path = high_band_path
    manifest, splits = gru.read_project(ROOT)
    paths = gru.paths(ROOT)
    verify_splits(manifest, splits)
    # Batch loading needs a profile to allocate its baseline arrays. The
    # placeholder is replaced before any model fitting or scoring.
    score_bins = round(2.0 / gru.loader.BIN_S)
    empty_profile = {
        str(sequence): np.zeros((score_bins, 2), dtype=np.float32)
        for sequence in manifest.sequence.unique()
    }

    if args.gru_checkpoint is not None and args.run == 'on':
        parser.error('--gru-checkpoint applies only to an OFF training run')
    if args.run in ('off', 'all'):
        off_train = sessions_for_split(manifest, splits, 'train', off=True)
        off_val = sessions_for_split(manifest, splits, 'val', off=True)
        batches = {}
        for split, sessions in [('train', off_train), ('val', off_val)]:
            batches.update(read_batches(sessions, 'recorded', split, manifest, splits,
                                        paths, empty_profile))
        if args.gru_checkpoint is None:
            off_profile = time_profile(batches['train'], manifest)
        else:
            saved = torch.load(args.gru_checkpoint, map_location='cpu', weights_only=False)
            if (set(off_train) != set(saved['train_sessions'])
                    or set(off_val) != set(saved['val_sessions'])):
                raise ValueError('OFF split differs from the supplied GRU checkpoint')
            off_profile = {seq: np.asarray(value, dtype=np.float32)
                           for seq, value in saved['profile'].items()}
        set_baseline(batches, off_profile, manifest)
        train('off', batches, off_profile, args, manifest)
        del batches
    if args.run in ('on', 'all'):
        on_train = sessions_for_split(manifest, splits, 'train', off=False)
        on_val = sessions_for_split(manifest, splits, 'val', off=False)
        meta = pd.read_csv(ROOT / 'semisynthetic/parrm_amp10/metadata.csv').set_index('session')
        for split, ids in [('train', on_train), ('val', on_val)]:
            for s in ids:
                row = meta.loc[s]
                if row.split != split or row.artifact_scale != 10 or row.condition != manifest.loc[s, 'condition']:
                    raise ValueError(f'{s}: incorrect 10x metadata')
        both = {}
        for source in ('recorded', 'amp10'):
            both[source] = {}
            for split, ids in [('train', on_train), ('val', on_val)]:
                both[source].update(read_batches(ids, source, split, manifest, splits, paths, empty_profile))
        for level, part in both['recorded'].items():
            for session, first in part.items():
                second = both['amp10'][level][session]
                for key in ('trial_ids', 'bin_ids', 'valid', 'Y'):
                    if not np.array_equal(first[key], second[key], equal_nan=True):
                        raise ValueError(f'{level}/{session}: paired ON {key} differ')
        profile = time_profile(both['recorded']['train'], manifest)
        for source, batch_levels in both.items():
            set_baseline(batch_levels, profile, manifest)
            train({'recorded': 'on_1x', 'amp10': 'on_10x'}[source],
                  batch_levels, profile, args, manifest)
    output = args.output or OUTPUT
    tables = [p for p in (output / name / 'aggregate.csv' for name in ('off', 'on_1x', 'on_10x'))
              if p.is_file()]
    pd.concat([pd.read_csv(p) for p in tables], ignore_index=True).to_csv(
        output / 'training_comparison.csv', index=False)


if __name__ == '__main__':
    main()
