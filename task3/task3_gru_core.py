"""Shared data loading and calibration for the five-seed GRU decoder."""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn

import EECS182_NHP_data_loader as loader
import EECS182_NHP_features as features


BAND = (200, 400)
WINDOW = 0.256
SEEDS = (1, 2, 3, 4, 5)


def paths(root):
    return loader.Paths(
        root=str(root),
        export_dir=str(root / 'export_outputs' / 'h5_export_outputs'),
        trial_dir=str(root / 'export_outputs' / 'trials'),
        parrm_dir=str(root / 'parrm_outputs' / 'parrm_h5'),
        kin_dir=str(root / 'kinematics_v3' / 'bins50ms'),
    )


def read_project(root):
    manifest = pd.read_csv(root / 'EECS182_data_manifest.csv').set_index('session', drop=False)
    with (root / 'splits.json').open() as file:
        splits = json.load(file)
    return manifest, splits


def trials(session, part, project_paths, splits):
    table = pd.read_csv(Path(project_paths.trial_dir) / f'{session}_trials.csv')
    table = table[table.include == 1]
    if part in ('pool', 'query'):
        numbers = splits['calibration'][session][part]
    elif part in ('train', 'val'):
        held = set(splits['train_val_trials'][session])
        numbers = held if part == 'val' else set(table.trial) - held
    else:
        raise ValueError(f'Unknown trial part: {part}')
    return table[table.trial.isin(numbers)]


def feature_variant(source):
    if source == 'recorded':
        return 'export_none'
    if source == 'parrm':
        return 'parrm_none'
    if source == 'amp10':
        return 'task3_parrm_amp_s0010p000'
    raise ValueError(f'Unknown source: {source}')


def high_band_path(root, session, source):
    if source == 'amp10':
        folder = 'task3_parrm_amp_hf_s0010p000'
    elif source == 'recorded':
        # These already exist for the OFF data and the 1x artifact sweep.
        legacy = root / 'features' / 'hf200_400_off' / f'{session}.h5'
        if legacy.is_file():
            return legacy
        scale_one = root / 'features' / 'task3_parrm_amp_hf_s0001p000' / f'{session}.h5'
        if scale_one.is_file():
            return scale_one
        folder = 'hf200_400_export'
    else:
        folder = 'hf200_400_parrm'
    return root / 'features' / folder / f'{session}.h5'


def prepare_features(root, session, source, project_paths, manifest):
    variant = feature_variant(source)
    base = root / 'features' / variant / f'{session}.h5'
    high = high_band_path(root, session, source)

    if source == 'amp10':
        if not base.is_file() or not high.is_file():
            raise FileNotFoundError(
                f'{session}: missing 10x feature cache. Run task3_make_semisynthetic_h5_v1.py first.')
        return

    if not base.is_file():
        base.parent.mkdir(parents=True, exist_ok=True)
        recording = features.compute_session(session, project_paths,
                                             source='parrm' if source == 'parrm' else 'export',
                                             reference='none', manifest=manifest)
        features.save_session(recording, str(base))
        print(f'    made {base.relative_to(root)}', flush=True)

    if not high.is_file():
        recording = loader.load_session(session, project_paths,
                                        source='parrm' if source == 'parrm' else 'export',
                                        reference='none', manifest=manifest)
        first = int(np.ceil(recording.t[0] / loader.BIN_S - 1e-9))
        last = int(np.floor(recording.t[-1] / loader.BIN_S + 1e-9))
        bin_ids = np.arange(first, last)
        ends = np.searchsorted(recording.t,
                               (bin_ids + 1) * loader.BIN_S - 0.5 / recording.fs) - 1
        power = features._band_power(recording.lfp, ends, recording.fs, WINDOW, *BAND)
        high.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(high, 'w') as file:
            file.create_dataset('bin_id', data=bin_ids)
            file.create_dataset('channel_ids', data=recording.channels)
            file.create_dataset('logpow', data=power, compression='gzip')
            file.attrs['band_hz'] = BAND
        print(f'    made {high.relative_to(root)}', flush=True)


def session_features(root, session, source, project_paths, splits, manifest):
    prepare_features(root, session, source, project_paths, manifest)
    norm_rows = trials(session, 'pool', project_paths, splits)
    feat = features.load_features(session, project_paths, variant=feature_variant(source),
                                  normalize=True, trials=norm_rows)
    high = high_band_path(root, session, source)
    with h5py.File(high, 'r') as file:
        ids = file['bin_id'][:]
        channels = file['channel_ids'][:]
        power = file['logpow'][:].T.astype(np.float32)
    if not np.array_equal(ids, feat['bin_id']):
        raise ValueError(f'{session}: high-frequency bins do not match LFP features')
    if not np.array_equal(channels, feat['channel_ids']):
        raise ValueError(f'{session}: high-frequency channel order differs')

    start, stop = feat['norm']['window']
    inside = (ids * loader.BIN_S >= start) & ((ids + 1) * loader.BIN_S <= stop)
    median = np.nanmedian(power[inside], axis=0)
    spread = 1.4826 * np.nanmedian(abs(power[inside] - median), axis=0)
    spread[~(spread > 0)] = 1.0
    extra = (power - median) / spread
    x = np.concatenate([feat['X'], extra[:, :, None]], axis=2)
    feat['X'] = np.clip(x, -10.0, 10.0)
    feat['feature_names'] = [*features.FEATURE_NAMES, 'pow_200_400']
    return feat


def trial_batch(feat, session, part, project_paths, splits, manifest, profile, n_lags):
    rows = trials(session, part, project_paths, splits)
    kin = loader.load_kinematics(session, project_paths)
    _, y, valid, score_bins = features.trial_arrays(
        feat, rows, kin=kin, align='t_go', pre=0.5, post=1.5,
        n_lags=n_lags, kin_cols=('Index_vx_cm', 'Index_vy_cm'))
    x, _, _, all_bins = features.trial_arrays(
        feat, rows, kin=None, align='t_go', pre=0.5 + n_lags * loader.BIN_S,
        post=1.5, n_lags=0)
    if not np.array_equal(all_bins[:, n_lags:], score_bins):
        raise ValueError(f'{session}: trial history does not line up with targets')
    seq = str(manifest.loc[session, 'sequence'])
    if seq not in profile:
        raise ValueError(f'No pretrained time baseline for sequence {seq}')
    positions = np.broadcast_to(np.arange(y.shape[1]), valid.shape)
    baseline = profile[seq][positions].astype(np.float32)
    return {
        'X': np.nan_to_num(x, nan=0., posinf=0., neginf=0.).astype(np.float32),
        'Y': y.astype(np.float32),
        'valid': valid,
        'baseline': baseline,
        'position': positions,
        'trial_ids': rows.trial.to_numpy(),
        'bin_ids': score_bins,
    }


class GRUDecoder(nn.Module):
    """Same layer names and sizes as the saved multitask GRU checkpoints."""

    def __init__(self, input_size, output_size=2, projection=48, hidden=64,
                 dropout=0.15, warmup=19, channel_dropout=0.10,
                 features_per_channel=9):
        super().__init__()
        self.projection = nn.Linear(input_size, projection)
        self.gru = nn.GRU(projection, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden, output_size)
        self.speed_head = nn.Linear(hidden, 1)
        self.direction_head = nn.Linear(hidden, 1)
        self.warmup = warmup
        self.channel_dropout = channel_dropout
        self.features_per_channel = features_per_channel

    def forward(self, x):
        if self.training and self.channel_dropout > 0:
            n, t, nf = x.shape
            channels = nf // self.features_per_channel
            keep = (torch.rand(n, 1, channels, 1, device=x.device) >= self.channel_dropout)
            x = x.reshape(n, t, channels, self.features_per_channel)
            x = (x * keep.to(x.dtype) / (1 - self.channel_dropout)).reshape(n, t, nf)
        x = torch.tanh(self.projection(x))
        states, _ = self.gru(x)
        states = self.dropout(states[:, self.warmup:, :])
        return self.readout(states), self.speed_head(states), self.direction_head(states)


def device_name(choice):
    if choice != 'auto':
        return choice
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def load_models(root, checkpoint_path, device):
    # Only load checkpoint files created by this project. torch.load uses pickle.
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if saved.get('temporal_mode') != 'trial_continuous':
        raise ValueError('Not the trial-continuous GRU checkpoint')
    if saved.get('target_columns') != ['Index_vx_cm', 'Index_vy_cm']:
        raise ValueError('Checkpoint targets do not match fingertip velocity')
    if saved['model_config']['features_per_channel'] != 9:
        raise ValueError('Expected nine LFP features per channel')
    if saved['run_config']['seed'] != 1:
        raise ValueError('Expected seed-1 checkpoint')
    checkpoints = root / 'results' / 'task3' / 'checkpoints'
    states = {1: saved['state_dict']}
    epoch = int(saved['best_epoch'])
    for seed in SEEDS[1:]:
        path = checkpoints / f'task3_gru_ensemble_v1_seed{seed}_epoch{epoch}.pt'
        if not path.is_file():
            raise FileNotFoundError(f'Missing ensemble checkpoint: {path}')
        another = torch.load(path, map_location='cpu', weights_only=False)
        if another['seed'] != seed or another['epochs'] != epoch or \
                another['source_checkpoint'] != checkpoint_path.name:
            raise ValueError(f'Checkpoint metadata differs: {path}')
        states[seed] = another['state_dict']
    cfg = saved['model_config']
    expected = cfg['input_size']
    if expected != 35 * 9:
        raise ValueError(f'Expected 315 inputs, got {expected}')
    return saved, states


def calibration_settings(root):
    results = root / 'results' / 'task3'
    full = pd.read_csv(results / 'task3_gru_full_finetune_v1_pool_cv.csv')
    selected = full.sort_values('pool_cv_mse').iloc[0]
    lr, epochs = float(selected.lr), int(selected.epochs)
    table = pd.read_csv(results / 'task3_gru_temporal_adapt_v1_pool_cv.csv')
    options = {}
    for method in ('time', 'full'):
        choices = []
        for axis in ('vx', 'vy'):
            part = table[(table.method == method) & (table.axis == axis)]
            row = part.loc[part.cv_mse.idxmin()]
            choices.append((int(row.k), float(row.lam)))
        options[method] = choices
    return lr, epochs, options


def fit_session(config, state, pool, scale, device, lr, epochs, seed):
    model = GRUDecoder(**config).to(device)
    model.load_state_dict(state, strict=True)
    for param in model.parameters():
        param.requires_grad_(False)
    for layer in (model.projection, model.gru, model.readout):
        for param in layer.parameters():
            param.requires_grad_(True)
    model.eval()  # same as the validated fit: no dropout during fine-tuning
    parameters = [p for p in model.parameters() if p.requires_grad]
    original = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    x = torch.as_tensor(pool['X'], dtype=torch.float32, device=device)
    y = np.nan_to_num((pool['Y'] - pool['baseline']) / scale,
                      nan=0., posinf=0., neginf=0.)
    target = torch.as_tensor(y, dtype=torch.float32, device=device)
    valid = torch.as_tensor(pool['valid'], dtype=torch.bool, device=device)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for indices in np.array_split(order, max(1, int(np.ceil(len(x) / 8)))):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[indices])[0]
            errors = (prediction - target[indices]).square().mean(dim=2)
            loss = errors[valid[indices]].mean()
            loss += sum((p - before).square().mean()
                        for p, before in zip(parameters, original))
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
    return model


def predict(model, batch, device):
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(batch['X']), 8):
            x = torch.as_tensor(batch['X'][start:start + 8],
                                dtype=torch.float32, device=device)
            output.append(model(x)[0].cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def average(models, batch, device):
    return np.mean([predict(model, batch, device) for model in models], axis=0)


def cosine_basis(positions, count, n_steps):
    j = np.arange(count)
    x = np.cos(np.pi * (positions[..., None] + .5) * j / n_steps)
    x[..., 1:] *= np.sqrt(2.)
    return x


def correct_time(pool, pool_pred, target, target_pred, scale, choices):
    fixed = target_pred.copy()
    for axis, (count, lam) in enumerate(choices):
        if count == 0:
            continue
        mask = pool['valid']
        length = pool['position'].shape[1]
        matrix = cosine_basis(pool['position'][mask], count, length)
        residual = ((pool['Y'] - pool['baseline']) / scale - pool_pred)[mask, axis]
        coef = np.linalg.solve(matrix.T @ matrix / len(residual) + lam * np.eye(count),
                               matrix.T @ residual / len(residual))
        fixed[..., axis] += cosine_basis(target['position'], count, length) @ coef
    return fixed


def model_scores(session, prediction, batch, scale):
    mask = batch['valid']
    truth = batch['Y'][mask].astype(np.float64)
    baseline = batch['baseline'][mask].astype(np.float64)
    estimate = (batch['baseline'] + prediction * scale)[mask]
    errors = ((truth - estimate) ** 2).sum(axis=0)
    total = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - errors / total
    time_r2 = 1.0 - ((truth - baseline) ** 2).sum(axis=0) / total
    return dict(session=session, n_bins=int(mask.sum()), r2=float(r2.mean()),
                r2_vx=float(r2[0]), r2_vy=float(r2[1]),
                r2_time=float(time_r2.mean()), neural_gain=float((r2-time_r2).mean()))
