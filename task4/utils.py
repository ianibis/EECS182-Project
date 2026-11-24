"""Shared helpers for the Task 4 notebooks.

Data, splits and loaders, metrics, the training loop, decoder models, the
calibration-draw sampler, and the reporting helpers used by task 4.1-4.3.

Sections
  1  Imports and paths
  2  Data, splits and loaders
  3  Metrics
  4  Training utilities
  5  Evaluation utilities
  6  Decoders
  7  Calibration draws (task 4.1)
  8  Pipelines and reporting
"""

# ---------------------------------------------------------------------------
# 1. Imports and paths
# ---------------------------------------------------------------------------

import copy
import glob
import json
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

TASK4_DIR = os.path.dirname(os.path.abspath(__file__))   # .../task4
PROJECT_ROOT = os.path.dirname(TASK4_DIR)                # repository root
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')      # earlier baseline artifacts
TASK4_RESULTS = os.path.join(TASK4_DIR, 'results')
CHECKPOINT_DIR = os.path.join(TASK4_DIR, 'checkpoints')
TRAINING_LOG_DIR = os.path.join(TASK4_DIR, 'training_logs')
DRAWS_PATH = os.path.join(TASK4_RESULTS, 'draws.json')
BIN_S = 0.05  # 50 ms bins

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import EECS182_NHP_data_loader as lab_loader
import EECS182_NHP_features as lab_features

LOADER_PATHS = lab_loader.Paths(
    root=DATA_DIR,
    export_dir=os.path.join(DATA_DIR, 'export_outputs', 'h5_export_outputs'),
    trial_dir=os.path.join(DATA_DIR, 'export_outputs', 'trials'),
    kin_dir=os.path.join(DATA_DIR, 'kinematics_v3', 'bins50ms'),
)

# Semi-synthetic 10x condition: recorded LFP + 9 * PARRM artifact, stimulation-ON only.
# Built by semisynthetic/EECS182_make_semisynthetic_h5.py; see task4/README.md.
SEMISYNTHETIC_SCALE = 10.0
SEMISYNTHETIC_EXPORT_DIR = os.path.join(DATA_DIR, 'semi_synthetic', 'parrm_amp10',
                                        'h5_export_outputs')
AMP10_VARIANT = 'amp10'  # feature_variant label for rows scored on the 10x recording

# PARRM-cleaned counterpart of a stimulation-ON recording: the artifact subtracted.
PARRM_DIR = os.path.join(DATA_DIR, 'parrm_outputs', 'parrm_h5')

MANIFEST_PATH = os.path.join(PROJECT_ROOT, 'EECS182_data_manifest.csv')
SPLITS_PATH = os.path.join(PROJECT_ROOT, 'splits.json')
MANIFEST_TABLE = lab_loader.load_manifest(LOADER_PATHS, MANIFEST_PATH)
SESSION_TABLE = MANIFEST_TABLE.set_index('session')  # metadata, sessions stored row-wise
SPLITS = lab_loader.load_splits(LOADER_PATHS, SPLITS_PATH)  # fixed train/val/test and calibration splits

# ---------------------------------------------------------------------------
# 2. Data, splits and loaders
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path) as file:
        return json.load(file)

def load_kinematics(session):
    return lab_loader.load_kinematics(session, LOADER_PATHS)

def load_session_metadata(session):
    return lab_loader.load_session(session, LOADER_PATHS, with_lfp=False, manifest=MANIFEST_TABLE)

def trials_in_split_part(session, part):
    return lab_loader.split_trials(SPLITS, load_session_metadata(session), part)

TARGET_VELOCITY_COLS = ('Index_vx_cm', 'Index_vy_cm')  # target velocities
NORMALIZATION_PAD_S = 1.0  # extra seconds for normalization
MAD_TO_STD = 1.4826  # MAD to standard deviation

def load_unnormalized_features(session, variant):
    return lab_features.load_features(session, LOADER_PATHS, variant=variant,
                                      normalize=False)

def _select_trial_rows(session_metadata, trials):
    if trials is None:
        return session_metadata.included
    trial_numbers = {int(trial) for trial in trials}
    selected = session_metadata.trials[session_metadata.trials.trial.isin(trial_numbers)]
    return selected.sort_values('t_go')

def bins_within_trial_span(trial_rows, bin_ids):
    start = float(trial_rows.t_go.min() - NORMALIZATION_PAD_S)
    stop = float(trial_rows.t_reward.max() + NORMALIZATION_PAD_S)
    return (bin_ids * BIN_S >= start) & ((bin_ids + 1) * BIN_S <= stop)

def normalize_features(session, variant, normalization_trials):
    raw = load_unnormalized_features(session, variant)
    trial_rows = _select_trial_rows(load_session_metadata(session), normalization_trials)
    reference = raw['X'][bins_within_trial_span(trial_rows, raw['bin_id'])]
    median = np.nanmedian(reference, axis=0)
    scale = MAD_TO_STD * np.nanmedian(np.abs(reference - median), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return raw | {'X': ((raw['X'] - median) / scale).astype(np.float32)}

def load_trial_dataset(session, trials, variant, normalization_trials, lag_count):
    features = normalize_features(session, variant, normalization_trials)
    trial_rows = _select_trial_rows(load_session_metadata(session), trials)
    inputs, targets, valid_bins, bin_ids = lab_features.trial_arrays(
        features, trial_rows, kin=load_kinematics(session), align='t_go', pre=0.5, post=1.5,
        n_lags=lag_count, kin_cols=TARGET_VELOCITY_COLS)
    return dict(features=inputs, targets=targets, valid_bins=valid_bins, bin_ids=bin_ids,
                trials=trial_rows.trial.to_numpy())

def load_split_dataset(session, part, variant, lag_count):
    trials = trials_in_split_part(session, part)
    normalization_trials = (trials if part in ('train', 'val')
                            else SPLITS['calibration'][session]['pool'])
    return load_trial_dataset(session, trials, variant, normalization_trials, lag_count)

def load_same_day_off_dataset(stimulated_session, variant, lag_count):
    off_session = SPLITS['off_reference'][stimulated_session]
    trials = load_session_metadata(off_session).included.trial.astype(int).tolist()
    return load_trial_dataset(off_session, trials, variant, trials, lag_count)

def windows_from_valid_bins(dataset):
    return dataset['features'][dataset['valid_bins']], dataset['targets'][dataset['valid_bins']]

def semisynthetic_paths(export_dir=SEMISYNTHETIC_EXPORT_DIR):
    """Loader paths that read LFP from the 10x semi-synthetic export folder.

    Trials and kinematics are unchanged by the amplification, so only export_dir moves.
    """
    return lab_loader.Paths(
        root=DATA_DIR,
        export_dir=export_dir,
        trial_dir=os.path.join(DATA_DIR, 'export_outputs', 'trials'),
        kin_dir=os.path.join(DATA_DIR, 'kinematics_v3', 'bins50ms'),
    )

def parrm_paths(parrm_dir=PARRM_DIR):
    """Loader paths that can read the PARRM-cleaned LFP via source='parrm'."""
    return lab_loader.Paths(
        root=DATA_DIR,
        export_dir=os.path.join(DATA_DIR, 'export_outputs', 'h5_export_outputs'),
        trial_dir=os.path.join(DATA_DIR, 'export_outputs', 'trials'),
        parrm_dir=parrm_dir,
        kin_dir=os.path.join(DATA_DIR, 'kinematics_v3', 'bins50ms'),
    )

def load_lfp_dataset(session, trials, paths=None, source='export'):
    session_data = lab_loader.load_session(session, paths or LOADER_PATHS, with_lfp=True,
                                           manifest=MANIFEST_TABLE, source=source)
    lab_loader.normalize(session_data)
    session_data.trials = session_data.trials[session_data.trials.include == 1]
    examples = lab_loader.trial_examples(
        session_data, load_kinematics(session), align='t_go', pre=0.5, post=1.5,
        trials=trials, kin_cols=TARGET_VELOCITY_COLS)
    lfp = [example['lfp'].reshape(example['lfp'].shape[0], -1) for example in examples]
    return dict(features=np.stack(lfp).astype(np.float32),
                targets=np.stack([example['kin'] for example in examples]).astype(np.float32),
                valid_bins=np.stack([example['kin_valid'] for example in examples]).astype(bool))

def load_lfp_split(session, part, paths=None, source='export'):
    return load_lfp_dataset(session, trials_in_split_part(session, part), paths, source)

STIMULATED_ON = 1    # manifest 'stim' flag for recordings with stimulation on
STIMULATED_OFF = 0   # same-day no-stimulation reference recordings

def sessions_in_split(part, stimulated):
    return [s for s in SPLITS['sessions'][part]
            if int(SESSION_TABLE.loc[s, 'stim']) == stimulated]

# Stimulation-off sessions used to fit the decoders.
OFF_TRAINING_SESSIONS = sessions_in_split('train', STIMULATED_OFF)
CALIBRATION_SIZES = (2, 5, 10, 20)  # calibration trial counts

def _as_ints(trials):
    return [int(trial) for trial in trials]

# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------

NEGLIGIBLE_R2_GAP = 1e-3  # ignore R² gaps smaller than this

def r2_per_target(y, y_hat, valid=None):
    y = np.asarray(y, np.float64)
    y_hat = np.asarray(y_hat, np.float64)
    if valid is not None:
        valid = np.asarray(valid, bool)
        y, y_hat = y[valid], y_hat[valid]
    if len(y) == 0:
        return np.full(y.shape[1], np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        scores = r2_score(y, y_hat, multioutput='raw_values', force_finite=False)
    return np.where(np.isfinite(scores), scores, np.nan)

def mean_r2(y, y_hat, valid=None):
    return float(np.nanmean(r2_per_target(y, y_hat, valid)))

def gap_recovery_fraction(r2_model, r2_base, r2_off):
    r2_model = np.asarray(r2_model, np.float64)
    r2_base = np.asarray(r2_base, np.float64)
    gap = np.asarray(r2_off, np.float64) - r2_base
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(np.abs(gap) > NEGLIGIBLE_R2_GAP, (r2_model - r2_base) / gap, np.nan)

# ---------------------------------------------------------------------------
# 4. Training utilities
# ---------------------------------------------------------------------------

ADAM_LR = 1e-3  # Adam learning rate for biases and normalisation parameters
MUON_LR = 0.02  # Muon learning rate for the weight matrices
MUON_MOMENTUM = 0.95
MUON_NS_STEPS = 5
LR_FLOOR = 0.01  # cosine decay ends at this fraction of the initial rate
TRIAL_BATCH_SIZE = 16
WINDOW_BATCH_SIZE = 256

def save_checkpoint(model, name):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'{name}.pt'))

def save_ridge(decoder, name):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    np.savez(os.path.join(CHECKPOINT_DIR, f'{name}.npz'),
             coefficients=decoder.coefficients, intercept=decoder.intercept, alpha=decoder.alpha)

def select_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def masked_mse(y_hat, y, valid):
    if not valid.any():
        return y_hat.sum() * 0.0
    return F.mse_loss(y_hat[valid], y[valid])

def to_tensors(arrays):
    return [torch.as_tensor(np.asarray(array),
                            dtype=torch.bool if np.asarray(array).dtype == bool else torch.float32)
            for array in arrays]

def predict(model, X):
    device = select_device()
    model = model.to(device).eval()
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32).to(device)
    with torch.inference_mode():
        return model(X).cpu().numpy()

def orthogonal_polar_factor(gradient, steps=MUON_NS_STEPS, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = gradient.float()
    X = X / (X.norm() + eps)
    if X.shape[0] > X.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if gradient.shape[0] > gradient.shape[1] else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=MUON_LR, momentum=MUON_MOMENTUM, nesterov=True,
                 ns_steps=MUON_NS_STEPS):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = (parameter.grad.reshape(parameter.shape[0], -1)
                            if parameter.ndim > 2 else parameter.grad)
                buffer = self.state[parameter].setdefault(
                    'momentum_buffer', torch.zeros_like(gradient))
                buffer.mul_(group['momentum']).add_(gradient)
                update = (gradient.add(buffer, alpha=group['momentum'])
                          if group['nesterov'] else buffer)
                orthogonal = orthogonal_polar_factor(update, group['ns_steps'])
                # sqrt(max(1, rows / cols)) matches the RMS of an Adam step at the same lr.
                step_size = group['lr'] * max(1.0, orthogonal.shape[0] / orthogonal.shape[1]) ** 0.5
                parameter.add_(orthogonal.reshape(parameter.shape).to(parameter.dtype),
                               alpha=-step_size)

def build_optimizers(model):
    optimizers = [Muon([p for p in model.parameters() if p.ndim >= 2])]
    vectors = [p for p in model.parameters() if p.ndim < 2]
    if vectors:
        optimizers.append(torch.optim.Adam(vectors, lr=ADAM_LR))
    return optimizers

def cosine_lr_multiplier(epoch, epochs):
    return LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * epoch / epochs))

def set_learning_rates(optimizers, base_rates, factor):
    for optimizer, optimizer_rates in zip(optimizers, base_rates):
        for group, base_rate in zip(optimizer.param_groups, optimizer_rates):
            group['lr'] = base_rate * factor

def train_epoch(model, tensors, loss_fn, optimizers, order, batch_size, device):
    model.train()
    loss_sum = 0.0
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        X, *y = [tensor[batch].to(device) for tensor in tensors]
        loss = loss_fn(model(X), *y)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        loss_sum += float(loss.detach()) * len(batch)
    return loss_sum / len(order)

def _train_model(model, train, val, loss_fn, seed, epochs, batch_size):
    # Set device, seeds, tensors, and the optimizers.
    device = select_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    tensors = to_tensors(train)
    X_val, *y_val = [tensor.numpy() for tensor in to_tensors(val)]
    optimizers = build_optimizers(model)
    base_rates = [[group['lr'] for group in optimizer.param_groups] for optimizer in optimizers]

    training_log = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(tensors[0]))  # shuffle each epoch to avoid trial-order bias
        train_loss = train_epoch(model, tensors, loss_fn, optimizers, order, batch_size, device)
        val_r2 = mean_r2(y_val[0], predict(model, X_val), *y_val[1:])
        training_log.append(dict(epoch=epoch, train_loss=train_loss, val_r2=val_r2))
        set_learning_rates(optimizers, base_rates, cosine_lr_multiplier(epoch, epochs))

    model.eval()
    best = max(training_log, key=lambda row: row['val_r2'])
    return model, dict(training_log=training_log,
                       best_val_r2=best['val_r2'], best_epoch=best['epoch'])

def train_trial_model(model, train, val, seed, epochs=150):
    keys = ('features', 'targets', 'valid_bins')
    return _train_model(model, [train[key] for key in keys], [val[key] for key in keys],
                        masked_mse, seed, epochs, TRIAL_BATCH_SIZE)

def train_window_model(model, train, val, seed, epochs=200):
    return _train_model(model, train, val, F.mse_loss, seed, epochs, WINDOW_BATCH_SIZE)

def predict_trial_bins(model, data):
    return predict(model, data['features'])

# ---------------------------------------------------------------------------
# 5. Evaluation utilities
# ---------------------------------------------------------------------------

RAW_VARIANT = 'export_none'
PARRM_VARIANT = 'parrm_none'
FEATURE_VARIANTS = ((RAW_VARIANT, 'raw'), (PARRM_VARIANT, 'parrm'))

def concat_session_windows(sessions, part, variant, lag_count):
    pairs = [windows_from_valid_bins(load_split_dataset(s, part, variant, lag_count)) for s in sessions]
    X, y = zip(*pairs)
    return np.concatenate(X), np.concatenate(y)

def load_query_and_off_windows(session, variant, lag_count):
    query = load_split_dataset(session, 'query', variant, lag_count)
    off = load_same_day_off_dataset(session, variant, lag_count)
    return windows_from_valid_bins(query), windows_from_valid_bins(off)

def load_calibration_windows(session, trials, normalization_trials, variant, lag_count):
    return windows_from_valid_bins(load_trial_dataset(session, trials, variant, normalization_trials, lag_count))

def iter_calibration_windows(draws, session, variant, lag_count):
    # v2 draws have no 'pool' key: each draw normalizes on its own trials.
    entry = draws['sessions'][session]
    for k in draws['ks']:
        for index, subset in enumerate(entry['subsets'][str(k)]):
            trials = subset['draw']
            yield k, index, load_calibration_windows(session, trials, trials, variant, lag_count)

def metrics_row(session, method, feature_label, y, y_hat, y_off, y_hat_off, k=0, draw=0, seed=0):
    r2 = r2_per_target(y, y_hat)
    r2_off = r2_per_target(y_off, y_hat_off)
    return dict(method=method, feature_variant=feature_label, session=session,
                am_freq=float(SESSION_TABLE.loc[session, 'am_freq_hz']), k=k, draw=draw, seed=seed,
                r2_vx=float(r2[0]), r2_vy=float(r2[1]),
                r2_ave=float(np.nanmean(r2)),
                r2_off_vx=float(r2_off[0]), r2_off_vy=float(r2_off[1]),
                r2_off_ave=float(np.nanmean(r2_off)),
                n_bins=len(y))

def score_window_decoder(decoder, session, method, feature_label, query, off, k=0, draw=0):
    X, y = query
    X_off, y_off = off
    return metrics_row(session, method, feature_label, y, decoder.predict(X),
                     y_off, decoder.predict(X_off), k, draw)

def save_training_log(summary, method, seed, training_log_dir):
    if not training_log_dir:
        return
    os.makedirs(training_log_dir, exist_ok=True)
    with open(os.path.join(training_log_dir, f'{method}_seed{seed}.json'), 'w') as file:
        json.dump(dict(method=method, seed=seed, **summary), file)

def concat_off_training_sessions(load_part, part, excluded_session=None):
    datasets = [load_part(s, part) for s in OFF_TRAINING_SESSIONS if s != excluded_session]
    return {key: np.concatenate([d[key] for d in datasets])
            for key in ('features', 'targets', 'valid_bins')}

def score_trial_model(model, session, method, feature_label, seed, query, off):
    y_hat = predict_trial_bins(model, query)[query['valid_bins']]
    y_hat_off = predict_trial_bins(model, off)[off['valid_bins']]
    return metrics_row(session, method, feature_label,
                     query['targets'][query['valid_bins']], y_hat,
                     off['targets'][off['valid_bins']], y_hat_off, seed=seed)

# ---------------------------------------------------------------------------
# 6. Decoders
# ---------------------------------------------------------------------------

class RidgeRegressor:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.float64)

        N_features = X.shape[1]

        x_mean = X.mean(axis=0)
        y_mean = y.mean(axis=0)
        X_centred = X - x_mean
        y_centred = y - y_mean

        # Ridge normal equations: (X^T X + alpha I) W = X^T Y.
        system = X_centred.T @ X_centred + self.alpha * np.eye(N_features)
        self.coefficients = np.linalg.solve(system, X_centred.T @ y_centred)
        self.intercept = y_mean - x_mean @ self.coefficients
        return self

    def predict(self, X):
        return np.asarray(X, np.float64) @ self.coefficients + self.intercept

RIDGE_PENALTIES = (100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000)  # ridge penalties

def pick_ridge_penalty(X_train, y_train, variant, lag_count):
    validation = [windows_from_valid_bins(load_split_dataset(s, 'query', variant, lag_count))
                  for s in sessions_in_split('val', STIMULATED_ON)]

    def score(alpha):
        decoder = RidgeRegressor(alpha).fit(X_train, y_train)
        return np.nanmean([mean_r2(y, decoder.predict(X)) for X, y in validation])

    return max(RIDGE_PENALTIES, key=score)

def train_ridge(variant, lag_count):
    X_train, y_train = concat_session_windows(OFF_TRAINING_SESSIONS, 'train', variant, lag_count)
    alpha = pick_ridge_penalty(X_train, y_train, variant, lag_count)
    return RidgeRegressor(alpha).fit(X_train, y_train), alpha

def evaluate_ridge(draws, decoder, alpha, variant, feature_label, lag_count, sessions):
    rows = []
    for session in sessions:
        query, off = load_query_and_off_windows(session, variant, lag_count)
        for k, draw, cal in iter_calibration_windows(draws, session, variant, lag_count):
            refit = RidgeRegressor(alpha).fit(*cal)
            rows.append(score_window_decoder(refit, session, f'ridge_{feature_label}_refit',
                                      feature_label, query, off, k, draw))
        rows.append(score_window_decoder(decoder, session, f'ridge_{feature_label}_off',
                                  feature_label, query, off))
    return rows

def run_ridge(draws, variant, feature_label, lag_count, sessions):
    decoder, alpha = train_ridge(variant, lag_count)
    save_ridge(decoder, f'ridge_{feature_label}')
    return evaluate_ridge(draws, decoder, alpha, variant, feature_label, lag_count, sessions)

def ridge_smoke_test():
    SANITY_SESSION = OFF_TRAINING_SESSIONS[0]
    SANITY_VARIANT = 'export_none'

    sanity_train = load_split_dataset(SANITY_SESSION, 'train', SANITY_VARIANT, 0)
    sanity_val = load_split_dataset(SANITY_SESSION, 'val', SANITY_VARIANT, 0)
    X_train, y_train = windows_from_valid_bins(sanity_train)
    X_val, y_val = windows_from_valid_bins(sanity_val)

    print('session       ', SANITY_SESSION)
    print('X_train       ', X_train.shape, ' y_train', y_train.shape)
    print('X_val         ', X_val.shape, ' y_val  ', y_val.shape)
    print('targets       ', TARGET_VELOCITY_COLS)

    sanity_model = RidgeRegressor(1e3).fit(X_train, y_train)
    sanity_pred = sanity_model.predict(X_val)
    print('R2 (vx, vy)   ', r2_per_target(y_val, sanity_pred))
    print('R2 mean       ', mean_r2(y_val, sanity_pred))
    print('one bin       ')
    print('  x[0][:8]    ', X_val[0, :8])
    print('  y[0]        ', y_val[0])
    print('  y_hat[0]     ', sanity_pred[0])

KNN_NEIGHBOR_GRID = (5, 10, 25, 50, 100, 200)  # KNN neighbours parameter sweep

class KNNRegressor:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.model = KNeighborsRegressor(
            n_neighbors=min(self.n_neighbors, len(X)), weights='distance'
        ).fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

def pick_best_neighbor_count(X_bank, y_bank, X_val, y_val):
    def validation_r2(neighbor_count):
        model = KNNRegressor(neighbor_count).fit(X_bank, y_bank)
        return mean_r2(y_val, model.predict(X_val))

    return max(KNN_NEIGHBOR_GRID, key=validation_r2)

CALIBRATION_TUNE_TRIALS = 10  # calibration trials used to tune KNN

def pick_knn_neighbor_counts(variant, lag_count):
    val_sessions = sessions_in_split('val', STIMULATED_ON)
    X_bank, y_bank = concat_session_windows(OFF_TRAINING_SESSIONS, 'train', variant, lag_count)
    X_val, y_val = concat_session_windows(val_sessions, 'query', variant, lag_count)

    calibration = SPLITS['calibration']
    cal = [load_calibration_windows(s, calibration[s]['subsets'][str(CALIBRATION_TUNE_TRIALS)][0],
                                    calibration[s]['pool'], variant, lag_count)
           for s in val_sessions]
    X_cal, y_cal = [np.concatenate(part) for part in zip(*cal)]

    return (pick_best_neighbor_count(X_bank, y_bank, X_val, y_val),
            pick_best_neighbor_count(X_cal, y_cal, X_val, y_val))

def train_knn(variant, lag_count):
    bank_neighbors, calibration_neighbors = pick_knn_neighbor_counts(variant, lag_count)
    bank_decoder = None
    if variant == RAW_VARIANT:
        X_bank, y_bank = concat_session_windows(OFF_TRAINING_SESSIONS, 'train', variant, lag_count)
        bank_decoder = KNNRegressor(bank_neighbors).fit(X_bank, y_bank)
    return bank_decoder, calibration_neighbors

def evaluate_knn(draws, variant, feature_label, lag_count, sessions, bank_decoder, calibration_neighbors):
    rows = []
    for session in sessions:
        query, off = load_query_and_off_windows(session, variant, lag_count)
        if bank_decoder is not None:
            rows.append(score_window_decoder(bank_decoder, session, 'knn_off', feature_label, query, off))
        for k, draw, cal in iter_calibration_windows(draws, session, variant, lag_count):
            decoder = KNNRegressor(calibration_neighbors).fit(*cal)
            rows.append(score_window_decoder(decoder, session, 'knn_incontext', feature_label,
                                      query, off, k, draw))
    return rows

MLP_DROPOUT = 0.3
EEGNET_DROPOUT = 0.25

class MLPRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32), nn.GELU(), nn.Dropout(MLP_DROPOUT),
            nn.Linear(32, 16), nn.GELU(), nn.Dropout(MLP_DROPOUT),
            nn.Linear(16, 2),
        )

    def forward(self, windows):
        return self.layers(windows)

def train_mlp(train, val, seed):
    return train_window_model(MLPRegressor(train[0].shape[1]), train, val, seed)

def evaluate_mlp(model, seed, test, feature_label):
    rows = []
    for session, (query, off) in test.items():
        X_query, y_query = query
        X_off, y_off = off
        rows.append(metrics_row(session, 'mlp', feature_label,
                              y_query, predict(model, X_query),
                              y_off, predict(model, X_off), seed=seed))
    return rows

def run_mlp(variant, feature_label, lag_count, sessions, seeds, training_log_dir):
    train = concat_session_windows(OFF_TRAINING_SESSIONS, 'train', variant, lag_count)
    val = concat_session_windows(OFF_TRAINING_SESSIONS, 'val', variant, lag_count)
    test = {s: load_query_and_off_windows(s, variant, lag_count) for s in sessions}
    rows = []
    for seed in seeds:
        print('mlp seed', seed)
        model, summary = train_mlp(train, val, seed)
        save_checkpoint(model, f'mlp_seed{seed}')
        save_training_log(summary, 'mlp', seed, training_log_dir)
        rows += evaluate_mlp(model, seed, test, feature_label)
    return rows

GRU_HIDDEN_SIZE = 128

class GRURegressor(nn.Module):
    """Per-bin velocity from a trial's sequence of cached features.

    `hidden`, `layers` and `dropout` default to the values the saved seed-0 checkpoint
    was trained with, so GRURegressor(input_size) still loads it strictly.
    """

    def __init__(self, input_size, hidden=GRU_HIDDEN_SIZE, layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, 2)

    def forward(self, trials):
        hidden_states, _ = self.gru(trials)
        return self.output(self.dropout(hidden_states))

def load_gru_datasets(variant, sessions):
    def load_part(s, part):
        return load_split_dataset(s, part, variant, 0)

    train = concat_off_training_sessions(load_part, 'train')
    val = concat_off_training_sessions(load_part, 'val')
    test = {s: (load_part(s, 'query'), load_same_day_off_dataset(s, variant, 0))
            for s in sessions}
    return train, val, test

def train_gru(train, val, seed):
    return train_trial_model(GRURegressor(train['features'].shape[-1]), train, val, seed)

def evaluate_gru(model, seed, test, feature_label):
    return [score_trial_model(model, s, 'gru', feature_label, seed, *data)
            for s, data in test.items()]

def run_gru(variant, feature_label, sessions, seeds, training_log_dir):
    train, val, test = load_gru_datasets(variant, sessions)
    rows = []
    for seed in seeds:
        print('gru seed', seed)
        model, summary = train_gru(train, val, seed)
        save_checkpoint(model, f'gru_seed{seed}')
        save_training_log(summary, 'gru', seed, training_log_dir)
        rows += evaluate_gru(model, seed, test, feature_label)
    return rows

TEMPORAL_KERNEL_SAMPLES = 500
SEPARABLE_KERNEL_SAMPLES = 62
TEMPORAL_FILTERS = 8
SPATIAL_FILTERS = 16
SEPARABLE_FILTERS = 16
FIRST_POOL_SAMPLES = 2
SECOND_POOL_SAMPLES = 25

def _left_only_padding(kernel_samples):
    return nn.ZeroPad2d((kernel_samples - 1, 0, 0, 0))

class EEGNetRegressor(nn.Module):
    def __init__(self, channel_count):
        super().__init__()
        self.layers = nn.Sequential(
            _left_only_padding(TEMPORAL_KERNEL_SAMPLES),
            nn.Conv2d(1, TEMPORAL_FILTERS, (1, TEMPORAL_KERNEL_SAMPLES), bias=False),
            nn.BatchNorm2d(TEMPORAL_FILTERS),
            nn.Conv2d(TEMPORAL_FILTERS, SPATIAL_FILTERS, (channel_count, 1),
                      groups=TEMPORAL_FILTERS, bias=False),
            nn.BatchNorm2d(SPATIAL_FILTERS),
            nn.ELU(),
            nn.AvgPool2d((1, FIRST_POOL_SAMPLES)),
            nn.Dropout(EEGNET_DROPOUT),
            _left_only_padding(SEPARABLE_KERNEL_SAMPLES),
            nn.Conv2d(SPATIAL_FILTERS, SPATIAL_FILTERS, (1, SEPARABLE_KERNEL_SAMPLES),
                      groups=SPATIAL_FILTERS, bias=False),
            nn.Conv2d(SPATIAL_FILTERS, SEPARABLE_FILTERS, (1, 1), bias=False),
            nn.BatchNorm2d(SEPARABLE_FILTERS),
            nn.ELU(),
            nn.AvgPool2d((1, SECOND_POOL_SAMPLES)),
            nn.Dropout(EEGNET_DROPOUT),
            nn.Conv2d(SEPARABLE_FILTERS, 2, (1, 1)),
        )

    def forward(self, lfp):
        return self.layers(lfp.unsqueeze(1)).squeeze(2).transpose(1, 2)


class CausalEEGNet(nn.Module):
    """braindecode EEGNet with left-only padding: one (vx, vy) per 50 ms bin.

    Single definition for task 4.2 and the Kaggle runner. Every padded convolution is
    rebuilt as an explicit left-only pad plus an unpadded convolution, so a bin can only
    see samples up to its own end.
    """

    def __init__(self, channel_count=35, samples=2000, outputs=2, temporal_filters=8,
                 depth_multiplier=2, kernel_samples=500, separable_samples=62,
                 pool_first=2, pool_second=25, dropout=0.5):
        super().__init__()
        from braindecode.models import EEGNet  # only task 4.2 needs braindecode
        self.bin_count = samples // (pool_first * pool_second)
        if self.bin_count * pool_first * pool_second != samples:
            raise ValueError('pool_first * pool_second must divide samples into whole bins')

        backbone = EEGNet(n_chans=channel_count, n_outputs=outputs, n_times=samples,
                          final_conv_length=1, F1=temporal_filters, D=depth_multiplier,
                          kernel_length=kernel_samples,
                          depthwise_kernel_length=separable_samples,
                          pool1_kernel_size=pool_first, pool2_kernel_size=pool_second,
                          drop_prob=dropout)

        layers = []
        for module in backbone.children():
            if isinstance(module, nn.Conv2d) and module.padding[1] > 0:
                layers.append(nn.ConstantPad2d((module.kernel_size[1] - 1, 0, 0, 0), 0.0))
                module.padding = (module.padding[0], 0)
            layers.append(module)
        self.layers = nn.Sequential(*layers)

    def forward(self, lfp):
        return self.layers(lfp).transpose(1, 2)


def load_eegnet_checkpoint(name, checkpoint_dir=None):
    """Load a saved CausalEEGNet state dict, e.g. 'task4_2_eegnet_seed0'."""
    path = os.path.join(checkpoint_dir or CHECKPOINT_DIR, f'{name}.pt')
    model = CausalEEGNet()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.eval()

# ---------------------------------------------------------------------------
# 7. Calibration draws (task 4.1)
# ---------------------------------------------------------------------------

DRAW_SEED = 20261106
DRAWS_PER_SIZE = 10
# 'fixed' calibrates from the pool that splits.json declares (blocks 1-2) and scores the
# declared query (blocks 3-5), so k is the only thing that varies. 'remaining' scores
# whatever the draw leaves behind and is kept only to reproduce the earlier numbers.
QUERY_DEFINITION = 'fixed'

def included_on_trials(session):
    return sorted(int(trial) for trial in load_session_metadata(session).included.trial)

def calibration_pool_and_query(session):
    """The declared calibration pool and query for one session, from splits.json."""
    entry = SPLITS['calibration'][session]
    return sorted(int(t) for t in entry['pool']), sorted(int(t) for t in entry['query'])

def distinct_subsets(pool, k, n_draws, rng):
    """Up to n_draws distinct k-subsets of pool, redrawn until no subset repeats.

    C(len(pool), k) caps how many exist, so k equal to the pool size gives one.
    """
    target = min(n_draws, math.comb(len(pool), k))
    seen, subsets, attempts = set(), [], 0
    while len(subsets) < target and attempts < 1000 * target:
        subset = frozenset(rng.sample(pool, k))
        attempts += 1
        if subset not in seen:
            seen.add(subset)
            subsets.append(sorted(subset))
    return subsets

def sample_draws_for_session(session, ks=CALIBRATION_SIZES, n_draws=DRAWS_PER_SIZE,
                             seed=DRAW_SEED, query_definition=QUERY_DEFINITION):
    """The k-trial calibration draws for one stimulation-on session."""
    trials = included_on_trials(session)
    rng = random.Random(f'{seed}:{session}')  # str seeds hash with sha512, so runs repeat

    if query_definition == 'fixed':
        pool, query = calibration_pool_and_query(session)
        if not set(query) <= set(trials):
            raise ValueError(f'{session}: the declared query leaves the included ON trials')
        draws_by_k = {str(k): [dict(draw=subset, query=query)
                               for subset in distinct_subsets(pool, k, n_draws, rng)]
                      for k in ks}
        return dict(pool=pool, query=query, subsets=draws_by_k)

    if query_definition != 'remaining':
        raise ValueError(f'unknown query_definition {query_definition!r}')

    orders = []  # one shuffled trial order per draw index
    for _ in range(n_draws):
        order = list(trials)
        rng.shuffle(order)
        orders.append(order)

    # A draw is the first k entries of an order; its query is everything left over.
    draws_by_k = {str(k): [dict(draw=order[:k], query=sorted(set(trials) - set(order[:k])))
                           for order in orders]
                  for k in ks}
    return dict(pool=None, query=None, subsets=draws_by_k)

def build_draws(sessions=None, ks=CALIBRATION_SIZES, n_draws=DRAWS_PER_SIZE, seed=DRAW_SEED,
                query_definition=QUERY_DEFINITION):
    if sessions is None:
        sessions = sessions_in_split('test', STIMULATED_ON)  # default: all stimulated test sessions

    by_session = {}
    for session in sessions:
        sampled = sample_draws_for_session(session, ks, n_draws, seed, query_definition)
        by_session[session] = dict(
            condition=str(SESSION_TABLE.loc[session, 'condition']),
            am_freq_hz=float(SESSION_TABLE.loc[session, 'am_freq_hz']),
            off_reference=SPLITS['off_reference'].get(session),  # same-day no-stimulation session
            on_trials=included_on_trials(session),  # every ON trial of the session
            pool=sampled['pool'],                   # calibration pool, or None when 'remaining'
            query=sampled['query'],                 # fixed query trials, or None when 'remaining'
            subsets=sampled['subsets'],             # str(k) -> entries of {'draw': [...], 'query': [...]}
        )

    payload = dict(version=3, created_by='task4/utils.py', unit='trial',
                   query_definition=query_definition, ks=list(ks), n_draws=n_draws,
                   seed=seed, sessions=by_session)
    payload['checks'] = verify_draws(payload)  # refuse to hand back an invalid payload
    return payload

def verify_draws(payload):
    failures, notes = [], []
    fixed = payload.get('query_definition') == 'fixed'
    for session, entry in payload['sessions'].items():
        all_on = set(entry['on_trials'])
        pool = set(entry['pool']) if entry.get('pool') else all_on
        query = set(entry['query']) if entry.get('query') else None
        if len(all_on) != len(entry['on_trials']):
            failures.append(f'{session}: on_trials holds duplicates')
        if query is not None:
            if not query <= all_on:
                failures.append(f'{session}: the query leaves the ON trials')
            if pool & query:
                failures.append(f'{session}: the pool and the query overlap')

        for k in payload['ks']:
            draws = entry['subsets'][str(k)]
            if len(draws) > payload['n_draws']:
                failures.append(f'{session}: k={k} has {len(draws)} draws, more than {payload["n_draws"]}')
            if k > len(pool):
                failures.append(f'{session}: k={k} exceeds the {len(pool)}-trial pool')

            for index, draw in enumerate(draws):
                calibration, scored = set(draw['draw']), set(draw['query'])
                label = f'{session}: k={k} draw {index}'
                if len(calibration) != k:
                    failures.append(f'{label} is not {k} distinct trials')
                if not calibration <= pool:
                    failures.append(f'{label} leaves the calibration pool')
                if calibration & scored:
                    failures.append(f'{label} reuses a query trial')
                if fixed and scored != query:
                    failures.append(f'{label} does not use the declared query')
                if not fixed and calibration | scored != all_on:
                    failures.append(f'{label} misses ON trials')

            distinct = len({frozenset(d['draw']) for d in draws})
            if distinct < payload['n_draws']:
                notes.append(f'{session}: k={k} has {distinct} distinct draws among '
                             f'{payload["n_draws"]}; only {math.comb(len(pool), k)} exist')

    return dict(passed=not failures, failures=failures, notes=notes)

def save_draws(payload, out=DRAWS_PATH):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as file:
        json.dump(payload, file, indent=1)

def iter_draws(draws, session):
    for k in draws['ks']:
        for index, draw in enumerate(draws['sessions'][session]['subsets'][str(k)]):
            yield k, index, draw

def drawn_trial_windows(session, draw, variant, lag_count):
    return windows_from_valid_bins(load_trial_dataset(session, draw['draw'], variant,
                                            draw['draw'], lag_count))

def query_trial_windows(session, draw, variant, lag_count):
    return windows_from_valid_bins(load_trial_dataset(session, draw['query'], variant,
                                            draw['draw'], lag_count))

# ---------------------------------------------------------------------------
# 8. Pipelines and reporting
# ---------------------------------------------------------------------------

def run_all_baselines(draws_path=DRAWS_PATH,
             out=os.path.join(RESULTS_DIR, 'results.parquet'),
             methods=('ridge',), n_lags=9, sessions=None, seed_count=5, epochs=150,
             training_log_dir=None):
    draws = load_json(draws_path)
    sessions = sessions or sessions_in_split('test', STIMULATED_ON)
    seeds = range(seed_count)
    rows = []

    if 'ridge' in methods:
        print('ridge')
        for variant, feature_label in FEATURE_VARIANTS:
            rows += run_ridge(draws, variant, feature_label, n_lags, sessions)
    if 'mlp' in methods:
        rows += run_mlp(RAW_VARIANT, 'raw', n_lags, sessions, seeds, training_log_dir)
    if 'gru' in methods:
        rows += run_gru(RAW_VARIANT, 'raw', sessions, seeds, training_log_dir)
    if 'knn' in methods:
        print('knn')
        bank, calibration_neighbors = train_knn(RAW_VARIANT, n_lags)
        for variant, feature_label in FEATURE_VARIANTS:
            rows += evaluate_knn(draws, variant, feature_label, n_lags, sessions,
                                 bank if variant == RAW_VARIANT else None, calibration_neighbors)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)

LAG_COUNT = 9
FIXED_RIDGE_ALPHA = 1e5  # fixed penalty for prediction plots
FIXED_KNN_NEIGHBOURS = 100  # fixed KNN k for prediction plots
PRE_GO_SECONDS = 0.5  # seconds before go cue
DECODER_COLORS = ['#7f7f7f', '#d62728', '#2ca02c']
DEFAULT_PREDICTION_SESSIONS = 'Butters-250616-102124,Butters-250520-101021'

def measured_finger_positions(session, bin_ids):
    kinematics = load_kinematics(session)
    flat_bin_ids = bin_ids.reshape(-1)
    x_positions = kinematics['Index_x_cm'].reindex(flat_bin_ids).to_numpy(np.float64)
    y_positions = kinematics['Index_y_cm'].reindex(flat_bin_ids).to_numpy(np.float64)
    return np.stack([x_positions, y_positions], axis=-1).reshape(*bin_ids.shape, 2)

def predict_with_reference_decoders(session, draws_path):
    draws = load_json(draws_path)
    entry = draws['sessions'][session]
    largest_calibration_size = max(draws['ks'])

    train = concat_session_windows(OFF_TRAINING_SESSIONS, 'train', RAW_VARIANT, LAG_COUNT)
    query = load_split_dataset(session, 'query', RAW_VARIANT, LAG_COUNT)
    query_features, _ = windows_from_valid_bins(query)
    calibration_trials = entry['subsets'][str(largest_calibration_size)][0]['draw']
    calibration_windows = windows_from_valid_bins(load_trial_dataset(
                                            session,
                                            calibration_trials,
                                            RAW_VARIANT,
                                            calibration_trials,  # normalize on the drawn trials
                                            LAG_COUNT,
                                        )
                                        )

    decoders = {
        'ridge, no adaptation': RidgeRegressor(FIXED_RIDGE_ALPHA).fit(*train),
        f'ridge, refit on k={largest_calibration_size}':
            RidgeRegressor(FIXED_RIDGE_ALPHA).fit(*calibration_windows),
        f'KNN, in-context k={largest_calibration_size}':
            KNNRegressor(FIXED_KNN_NEIGHBOURS).fit(*calibration_windows),
    }

    measured = query['targets'].astype(np.float64)
    predictions = {}
    for name, decoder in decoders.items():
        predicted_trials = np.full(measured.shape, np.nan)
        predicted_trials[query['valid_bins']] = decoder.predict(query_features)
        predictions[name] = predicted_trials
    return dict(session=session, measured=measured, predictions=predictions,
                valid_bins=query['valid_bins'], trials=query['trials'],
                positions=measured_finger_positions(session, query['bin_ids']))

def pick_most_complete_trials(result, count):
    valid_bin_counts = result['valid_bins'].sum(axis=1)
    mean_speed = np.nanmean(velocity_magnitude(result['measured']), axis=1)
    return np.lexsort((-np.nan_to_num(mean_speed), -valid_bin_counts))[:count]

def reference_decoder_r2(result, name):
    valid_bins = result['valid_bins']
    return mean_r2(result['measured'][valid_bins], result['predictions'][name][valid_bins])

def velocity_magnitude(velocity):
    return np.linalg.norm(velocity, axis=-1)

def bin_times_from_go_cue(result):
    return np.arange(result['measured'].shape[1]) * BIN_S - PRE_GO_SECONDS

def plot_measured_and_decoded(axis, times, measured_trace, decoded_traces):
    axis.plot(times, measured_trace, color='k', linewidth=2.4, label='measured')
    for color, (name, trace) in zip(DECODER_COLORS, decoded_traces.items()):
        axis.plot(times, trace, color=color, linewidth=1.3, label=name)
    axis.set_xlim(times[0], times[-1])
    axis.grid(alpha=0.25)

def plot_session_summary(result, out_path):
    names = list(result['predictions'])
    times = bin_times_from_go_cue(result)
    trial_index = pick_most_complete_trials(result, 1)[0]
    trial_number = result['trials'][trial_index]
    measured_trial = result['measured'][trial_index]
    decoded_trial = {name: result['predictions'][name][trial_index] for name in names}

    figure = plt.figure(figsize=(13, 7.4))
    grid = figure.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    for column, component in enumerate(('vx', 'vy')):
        axis = figure.add_subplot(grid[0, column])
        plot_measured_and_decoded(axis, times, measured_trial[:, column],
                                  {name: trace[:, column] for name, trace in decoded_trial.items()})
        axis.axhline(0, color='k', linewidth=0.5, alpha=0.4)
        axis.set_xlabel('time from go cue (s)')
        axis.set_ylabel(f'Index {component} (cm/s)')
        axis.set_title(f'Trial {trial_number} — velocity {component}', fontsize=9)
        if column == 0:
            axis.legend(fontsize=7)

    axis = figure.add_subplot(grid[0, 2])
    plot_measured_and_decoded(axis, times, velocity_magnitude(measured_trial),
                              {name: velocity_magnitude(trace) for name, trace in decoded_trial.items()})
    axis.set_xlabel('time from go cue (s)')
    axis.set_ylabel('speed (cm/s)')
    axis.set_title(f'Trial {trial_number} — speed', fontsize=9)
    axis.legend(fontsize=7)

    axis = figure.add_subplot(grid[1, 0])
    measured_bins = result['measured'][result['valid_bins']]
    for color, name in zip(DECODER_COLORS, names):
        decoded_bins = result['predictions'][name][result['valid_bins']]
        axis.scatter(measured_bins[:, 0], decoded_bins[:, 0], s=4, alpha=0.25, color=color,
                     label=f'{name} (R²={reference_decoder_r2(result, name):+.3f})')
    limit = np.nanpercentile(np.abs(measured_bins[:, 0]), 99)
    axis.plot([-limit, limit], [-limit, limit], color='k', linewidth=0.8)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xlabel('measured vx (cm/s)')
    axis.set_ylabel('decoded vx (cm/s)')
    axis.set_title('All query bins', fontsize=9)
    axis.legend(fontsize=6, loc='upper left')
    axis.grid(alpha=0.25)

    axis = figure.add_subplot(grid[1, 1])
    plot_measured_and_decoded(axis, times, np.nanmean(velocity_magnitude(result['measured']), axis=0),
                              {name: np.nanmean(velocity_magnitude(result['predictions'][name]), axis=0)
                               for name in names})
    axis.set_xlabel('time from go cue (s)')
    axis.set_ylabel('mean speed (cm/s)')
    axis.set_title(f'Mean speed profile over {len(result["trials"])} query trials', fontsize=9)
    axis.legend(fontsize=7)

    axis = figure.add_subplot(grid[1, 2])
    r2_values = [reference_decoder_r2(result, name) for name in names]
    axis.barh(range(len(names)), r2_values, color=DECODER_COLORS, height=0.6)
    for row, (name, value) in enumerate(zip(names, r2_values)):
        label = name.replace('ridge, ', '').replace('KNN, in-context', 'KNN')
        axis.text(value + 0.002, row, f'{label}  {value:+.3f}', va='center', ha='left', fontsize=7)
    axis.set_yticks([])
    axis.set_xlim(0, max(r2_values) * 1.9)
    axis.axvline(0, color='k', linewidth=0.8)
    axis.set_xlabel('R² (2-target mean, query bins)')
    axis.set_title('Decoder accuracy on this session', fontsize=9)
    axis.grid(alpha=0.25, axis='x')

    figure.suptitle(f'{result["session"]} — decoded Index-finger action', fontsize=11)
    figure.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(figure)

def plot_trial_speeds(result, out_path):
    times = bin_times_from_go_cue(result)
    figure, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for axis, trial_index in zip(axes.ravel(), pick_most_complete_trials(result, 4)):
        plot_measured_and_decoded(axis, times, velocity_magnitude(result['measured'][trial_index]),
                                  {name: velocity_magnitude(trace[trial_index])
                                   for name, trace in result['predictions'].items()})
        axis.set_title(f'Trial {result["trials"][trial_index]}', fontsize=9)
        axis.set_ylabel('speed (cm/s)')
    for axis in axes[-1]:
        axis.set_xlabel('time from go cue (s)')
    axes[0, 0].legend(fontsize=7)
    figure.suptitle(f'{result["session"]} — speed envelope, measured vs decoded', fontsize=11)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(figure)

def write_prediction_outputs(draws_path=DRAWS_PATH,
                    sessions=DEFAULT_PREDICTION_SESSIONS,
                    out_dir=os.path.join(RESULTS_DIR, 'predictions')):
    os.makedirs(out_dir, exist_ok=True)
    for session in sessions.split(','):
        result = predict_with_reference_decoders(session, draws_path)
        plot_session_summary(result, os.path.join(out_dir, f'{session}_actions.png'))
        plot_trial_speeds(result, os.path.join(out_dir, f'{session}_trials_speed.png'))
        np.savez_compressed(os.path.join(out_dir, f'{session}_predictions.npz'),
                            measured=result['measured'], mask=result['valid_bins'],
                            trials=result['trials'], positions=result['positions'],
                            **{f'pred__{name}': trace for name, trace in result['predictions'].items()})

METHOD_LABELS = {
    'ridge_raw_off': 'Ridge, raw, no adapt',
    'ridge_parrm_off': 'Ridge, PARRM, no adapt',
    'ridge_raw_refit': 'Ridge, raw, refit',
    'ridge_parrm_refit': 'Ridge, PARRM, refit',
    'knn_off': 'KNN, stim-off bank',
    'knn_incontext': 'KNN, k calibration',
    'mlp': 'MLP',
    'eegnet': 'EEGNet',
    'gru': 'GRU',
}
SHORT_METHOD_LABELS = {
    'ridge_raw_off': 'Ridge raw\nno adapt',
    'ridge_parrm_off': 'Ridge PARRM\nno adapt',
    'ridge_raw_refit': 'Ridge raw\nrefit k=20',
    'ridge_parrm_refit': 'Ridge PARRM\nrefit k=20',
    'knn_off': 'KNN\nstim-off bank',
    'knn_incontext': 'KNN\nk calib.',
    'mlp': 'MLP',
    'eegnet': 'EEGNet',
    'gru': 'GRU',
}
METHOD_COLORS = {
    'ridge_raw_off': '#4c72b0', 'ridge_parrm_off': '#8ab4d8',
    'ridge_raw_refit': '#c44e52', 'ridge_parrm_refit': '#e08a7f',
    'knn_off': '#8172b2', 'knn_incontext': '#55a868',
    'mlp': '#937860', 'eegnet': '#ccb974', 'gru': '#6b6b6b',
}
NON_ADAPTIVE_METHODS = ['ridge_raw_off', 'ridge_parrm_off', 'knn_off', 'mlp', 'eegnet', 'gru']
ADAPTIVE_METHODS = ['ridge_raw_refit', 'ridge_parrm_refit', 'knn_incontext']
METHOD_ORDER = NON_ADAPTIVE_METHODS + ['knn_incontext', 'ridge_raw_refit', 'ridge_parrm_refit']
RESULT_KEYS = ['method', 'feature_variant', 'session', 'k', 'draw', 'seed']
RIDGE_FACTORIAL_CELLS = [('raw', 'no refit', 'ridge_raw_off', 0),
                         ('raw', 'refit k=20', 'ridge_raw_refit', 20),
                         ('PARRM', 'no refit', 'ridge_parrm_off', 0),
                         ('PARRM', 'refit k=20', 'ridge_parrm_refit', 20)]
MAIN_TABLE_COLUMNS = {'method_label': 'method',
                      'r2_mean_3Hz': 'R² 3 Hz', 'r2_mean_20Hz': 'R² 20 Hz',
                      'r2_off_mean_3Hz': 'off 3 Hz', 'r2_off_mean_20Hz': 'off 20 Hz'}
LARGEST_CALIBRATION_SIZE = 20  # largest calibration set

def load_result_frames(pattern):
    frames = [pd.read_parquet(path) for path in sorted(glob.glob(pattern))]
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=RESULT_KEYS)

def average_scores_per_session(results):
    return (results.groupby(['method', 'am_freq', 'k', 'session'], dropna=False)
            .agg(r2=('r2_ave', 'mean'), r2_off=('r2_off_ave', 'mean'))
            .reset_index())

def bootstrap_sample_means(values, resample_count):
    return np.random.default_rng(0).choice(values, size=(resample_count, len(values)),
                                           replace=True).mean(axis=1)

def bootstrap_confidence_interval(values, resample_count=2000):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    resampled = bootstrap_sample_means(values, resample_count)
    return float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))

def two_level_bootstrap_confidence_interval(groups, resample_count=2000, seed=0):
    """95% CI for a mean over sessions, resampling sessions and the draws inside each.

    `groups` holds one array of per-draw scores per session, so the interval covers both
    which sessions were tested and which calibration sets were drawn inside them.
    """
    arrays = [np.asarray(group, float) for group in groups]
    arrays = [array[np.isfinite(array)] for array in arrays]
    arrays = [array for array in arrays if array.size]
    if len(arrays) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    resampled = np.empty(resample_count)
    for index in range(resample_count):
        sessions = rng.integers(0, len(arrays), len(arrays))
        resampled[index] = np.mean([arrays[session][rng.integers(0, arrays[session].size,
                                                                 arrays[session].size)].mean()
                                    for session in sessions])
    return float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))

def select_method_scores(session_scores, method, frequency, k):
    return session_scores[(session_scores.method == method) &
                          (session_scores.am_freq == frequency) &
                          (session_scores.k == k)]

def session_r2_values(session_scores, method, frequency, k):
    return select_method_scores(session_scores, method, frequency, k).r2.to_numpy()

def am_frequencies(session_scores):
    return sorted(session_scores.am_freq.unique())

def save_figure(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(figure)

def plot_r2_vs_calibration_trials(session_scores, path):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for axis, frequency in zip(axes, am_frequencies(session_scores)):
        for method in NON_ADAPTIVE_METHODS:
            mean_r2 = np.nanmean(session_r2_values(session_scores, method, frequency, 0))
            axis.axhline(mean_r2, linestyle='--', linewidth=1.4, color=METHOD_COLORS[method],
                         label=f'{METHOD_LABELS[method]}  {mean_r2:+.3f}')
        for method in ADAPTIVE_METHODS:
            calibration_sizes = sorted(session_scores[(session_scores.method == method) &
                                                      (session_scores.am_freq == frequency)].k.unique())
            values_by_size = [session_r2_values(session_scores, method, frequency, k)
                              for k in calibration_sizes]
            lows, highs = zip(*[bootstrap_confidence_interval(values) for values in values_by_size])
            axis.plot(calibration_sizes, [np.nanmean(values) for values in values_by_size],
                      marker='o', color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            axis.fill_between(calibration_sizes, lows, highs, color=METHOD_COLORS[method], alpha=0.16)
        axis.set_title(f'{int(frequency)} Hz AM')
        axis.set_xlabel('calibration trials k')
        axis.set_xticks([0, 2, 5, 10, 20])
        axis.grid(alpha=0.25)
        axis.set_xlim(0, 21.6)
        axis.set_ylim(-0.05, 0.09)
    axes[0].set_ylabel('R² (Index velocity, 2-target mean)')
    axes[0].legend(fontsize=6.2, loc='lower right', framealpha=0.9)
    figure.suptitle('Decoding R² vs number of labelled stimulation trials — dashed = no adaptation; '
                    'bands = 95% session bootstrap',
                    fontsize=8.6)
    save_figure(figure, path)

def plot_method_bar(axis, position, width, values, **style):
    mean_r2 = np.nanmean(values) if len(values) else np.nan
    low, high = bootstrap_confidence_interval(values)
    axis.bar(position, mean_r2, width, edgecolor='k', linewidth=0.5,
             yerr=[[mean_r2 - low], [high - mean_r2]], capsize=2, error_kw=dict(lw=0.7), **style)

def plot_method_comparison(session_scores, path):
    paired_bar_width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    for axis, frequency in zip(axes, am_frequencies(session_scores)):
        for position, method in enumerate(METHOD_ORDER):
            color = METHOD_COLORS[method]
            if method in NON_ADAPTIVE_METHODS:
                plot_method_bar(axis, position, 0.72, session_r2_values(session_scores, method, frequency, 0),
                                color=color, hatch='///')
                continue
            for offset, k in zip([-paired_bar_width, paired_bar_width], [0, LARGEST_CALIBRATION_SIZE]):
                plot_method_bar(axis, position + offset, paired_bar_width,
                                session_r2_values(session_scores, method, frequency, k), color=color,
                                alpha=1.0 if k == LARGEST_CALIBRATION_SIZE else 0.45)
        axis.set_xticks(np.arange(len(METHOD_ORDER)))
        axis.set_xticklabels([SHORT_METHOD_LABELS[method] for method in METHOD_ORDER], fontsize=6.4)
        axis.axhline(0, color='k', linewidth=0.8)
        axis.set_title(f'{int(frequency)} Hz AM')
        axis.grid(alpha=0.25, axis='y')
    axes[0].set_ylabel('R² (2-target mean)')
    axes[0].legend(handles=[Patch(facecolor='w', edgecolor='k', hatch='///', label='no adaptation'),
                            Patch(facecolor='grey', edgecolor='k', label='k = 20'),
                            Patch(facecolor='grey', edgecolor='k', alpha=0.45, label='k = 0')],
                   fontsize=7, loc='lower left')
    figure.suptitle('Baseline comparison on held-out stimulation sessions '
                    '(error bars: 95% session bootstrap)', fontsize=9.5)
    save_figure(figure, path)

def plot_ridge_factorial(session_scores, path):
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for axis, frequency in zip(axes, am_frequencies(session_scores)):
        mean_r2 = np.array([[np.nanmean(session_r2_values(session_scores, method, frequency, k))
                             for _, _, method, k in RIDGE_FACTORIAL_CELLS[start:start + 2]]
                            for start in (0, 2)])
        image = axis.imshow(mean_r2, cmap='RdYlGn', vmin=-0.03, vmax=0.06, aspect='auto')
        for (row, column), value in np.ndenumerate(mean_r2):
            axis.text(column, row, f'{value:+.3f}', ha='center', va='center', fontsize=11,
                      fontweight='bold', color='k')
        axis.set_xticks([0, 1])
        axis.set_xticklabels(['decoder from\nother days', 'refit on\nk=20 ON'])
        axis.set_yticks([0, 1])
        axis.set_yticklabels(['raw', 'PARRM\ncleaned'])
        axis.set_title(f'{int(frequency)} Hz AM')
        axis.set_xlabel('adaptation')
        axis.set_ylabel('input cleaning')
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03).set_label('R²', fontsize=8)
    figure.suptitle('Ridge factorial: does cleaning help, or do labels?', fontsize=10)
    save_figure(figure, path)

def plot_session_heatmap(session_scores, path):
    method_rows = []
    for method in METHOD_ORDER:
        k = 0 if method in NON_ADAPTIVE_METHODS else LARGEST_CALIBRATION_SIZE
        selected = session_scores[(session_scores.method == method) & (session_scores.k == k)]
        method_rows.append(selected.set_index('session').r2)
    session_order = (session_scores[session_scores.method == 'ridge_raw_off']
                     .sort_values('r2').session.tolist())
    matrix = pd.concat(method_rows, axis=1).T[session_order]

    figure, axis = plt.subplots(figsize=(12, 3.6))
    image = axis.imshow(matrix.to_numpy(), cmap='RdBu_r', aspect='auto',
                        norm=TwoSlopeNorm(vmin=-0.16, vcenter=0.0, vmax=0.16))
    axis.set_yticks(range(len(METHOD_ORDER)))
    axis.set_yticklabels([METHOD_LABELS[method] for method in METHOD_ORDER], fontsize=7.5)
    axis.set_xticks(range(len(session_order)))
    axis.set_xticklabels([session.replace('Butters-', '') for session in session_order],
                         rotation=90, fontsize=7)
    axis.set_title('Per-session R² (k=20 when the method adapts); sessions sorted by '
                   'no-adaptation ridge', fontsize=9)
    figure.colorbar(image, ax=axis, fraction=0.022, pad=0.012).set_label('R²', fontsize=8)
    save_figure(figure, path)

def plot_training_curves(training_log_dir, path):
    training_log_files = sorted(glob.glob(os.path.join(training_log_dir, '*.json')))
    figure, axis = plt.subplots(figsize=(7.6, 4.2))
    for training_log_file in training_log_files:
        record = load_json(training_log_file)
        color = METHOD_COLORS[record['method']]
        axis.plot([epoch['epoch'] for epoch in record['training_log']],
                  [epoch['val_r2'] for epoch in record['training_log']],
                  linewidth=1.1, color=color, alpha=0.85)
        axis.scatter([record['best_epoch']], [record['best_val_r2']], color=color, s=18, zorder=5,
                     edgecolor='k', linewidth=0.4)
    axis.legend(handles=[plt.Line2D([], [], color=METHOD_COLORS[method], label=method)
                         for method in ('mlp', 'gru', 'eegnet')
                         if any(method in os.path.basename(name) for name in training_log_files)],
                fontsize=8)
    axis.set_xlabel('epoch')
    axis.set_ylabel('validation R² (held-out block of training sessions)')
    axis.set_title('Training curves; dots mark the early-stopping checkpoint. '
                   'High validation R² does not transfer across days.', fontsize=8.5)
    axis.grid(alpha=0.25)
    save_figure(figure, path)

def plot_noise_floor(results, results_dir, path):
    stimulated_sessions_by_off_session = {}
    for stimulated_session, off_session in SPLITS['off_reference'].items():
        stimulated_sessions_by_off_session.setdefault(off_session, []).append(stimulated_session)
    cross_day_r2 = results[results.method == 'ridge_raw_off'].groupby('session').r2_off_ave.mean()
    diagnostic = pd.read_csv(os.path.join(results_dir, 'diagnostic_within_day_off.csv'))
    diagnostic['cross_day'] = diagnostic.session.map(
        lambda off_session: cross_day_r2.reindex(
            stimulated_sessions_by_off_session.get(off_session, [])).mean())

    figure, axis = plt.subplots(figsize=(7.4, 3.8))
    positions = np.arange(len(diagnostic))
    axis.bar(positions - 0.2, diagnostic.within_day_r2, 0.4, color='#4c72b0',
             label='same-day fit, held-out trials')
    axis.bar(positions + 0.2, diagnostic.cross_day, 0.4, color='#c44e52',
             label='decoder from other days')
    axis.set_xticks(positions)
    axis.set_xticklabels([session.replace('Butters-', '') for session in diagnostic.session],
                         rotation=20, fontsize=7.5)
    axis.axhline(0, color='k', linewidth=0.8)
    axis.set_ylabel('R² (no stimulation)')
    axis.set_title('The noise floor: even without stimulation, accuracy is ~0.07', fontsize=9.5)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25, axis='y')
    save_figure(figure, path)

def build_gap_recovery_table(results, resample_count=500):
    reference = (results[results.method == 'ridge_raw_off'][['session', 'r2_ave', 'r2_off_ave']]
                 .drop_duplicates('session').set_index('session'))
    rows = []
    for (method, frequency, k), group in results.groupby(['method', 'am_freq', 'k']):
        fractions = np.array([
            float(gap_recovery_fraction(session_rows.r2_ave.mean(),
                           reference.loc[session].r2_ave,
                           reference.loc[session].r2_off_ave))
            for session, session_rows in group.groupby('session')])
        resampled = bootstrap_sample_means(fractions, resample_count)
        low, high = np.nanpercentile(resampled, [2.5, 97.5])
        rows.append(dict(method=method, am_freq=frequency, k=int(k), n_sessions=len(fractions),
                         rho=float(np.nanmean(fractions)), rho_lo=float(low), rho_hi=float(high)))
    return pd.DataFrame(rows)

def plot_recovery(table, path):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    adaptive = table[table.method.isin(ADAPTIVE_METHODS)]
    for axis, (frequency, frequency_rows) in zip(axes, adaptive.groupby('am_freq')):
        for method, series in frequency_rows.groupby('method'):
            series = series.sort_values('k')
            axis.errorbar(series.k, series.rho,
                          yerr=[series.rho - series.rho_lo, series.rho_hi - series.rho],
                          marker='o', capsize=3, color=METHOD_COLORS[method],
                          label=METHOD_LABELS[method])
        axis.axhline(0, color='k', linewidth=0.8)
        axis.axhline(1, color='k', linewidth=0.8, linestyle=':')
        axis.set_title(f'{int(frequency)} Hz AM')
        axis.set_xlabel('calibration trials k')
        axis.set_ylabel('fraction of loss recovered (rho)')
        axis.grid(alpha=0.3)
        axis.set_xticks([2, 5, 10, 20])
    axes[0].legend(fontsize=7)
    save_figure(figure, path)

def write_main_tables(session_scores, out_dir):
    rows = []
    for method in METHOD_ORDER:
        calibration_sizes = ([0] if method in NON_ADAPTIVE_METHODS else
                             sorted(session_scores[session_scores.method == method].k.unique()))
        for k in calibration_sizes:
            for frequency in am_frequencies(session_scores):
                selected = select_method_scores(session_scores, method, frequency, k)
                values = selected.r2.to_numpy()
                low, high = bootstrap_confidence_interval(values)
                rows.append(dict(method=method, method_label=METHOD_LABELS[method],
                                 am_freq_hz=frequency, k=int(k), n_sessions=len(values),
                                 r2_mean=np.nanmean(values), r2_ci_lo=low, r2_ci_hi=high,
                                 r2_off_mean=np.nanmean(selected.r2_off),
                                 r2_sem=np.nanstd(values, ddof=1) / np.sqrt(len(values))))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, 'table_main.csv'), index=False)

    wide = table.pivot_table(index=['method_label', 'k'], columns='am_freq_hz',
                             values=['r2_mean', 'r2_off_mean'])
    wide.columns = [f'{metric}_{int(frequency)}Hz' for metric, frequency in wide.columns]
    wide = wide.reset_index()
    label_rank = {label: rank for rank, label in enumerate(METHOD_LABELS.values())}
    wide = (wide.assign(label_rank=wide.method_label.map(label_rank))
            .sort_values(['label_rank', 'k'])
            .drop(columns='label_rank')
            .rename(columns=MAIN_TABLE_COLUMNS)
            .round(4))
    with open(os.path.join(out_dir, 'table_main.md'), 'w') as file:
        file.write('# Main results\n\n')
        file.write('R² on the fixed 30-trial query set; mean over sessions, draws and seeds. '
                   '`off` is the same decoder on that day\'s no-stimulation session.\n\n')
        file.write(wide.to_markdown(index=False))
        file.write('\n')
    with open(os.path.join(out_dir, 'table_main.tex'), 'w') as file:
        file.write(wide.to_latex(index=False, float_format='%.4f', escape=True))

def write_ridge_factorial_table(session_scores, out_dir):
    rows = []
    for cleaning, adaptation, method, k in RIDGE_FACTORIAL_CELLS:
        row = dict(cleaning=cleaning, adaptation=adaptation, method=method, k=k)
        for frequency in am_frequencies(session_scores):
            row[f'r2_{int(frequency)}Hz'] = np.nanmean(session_r2_values(session_scores, method, frequency, k))
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'table_ridge_factorial.csv'), index=False)

def run_figures(pattern=os.path.join(RESULTS_DIR, 'results*.parquet'),
                out_dir=RESULTS_DIR,
                training_log_dir=os.path.join(RESULTS_DIR, 'training_logs')):
    def output_path(filename):
        return os.path.join(out_dir, filename)

    results = load_result_frames(pattern)
    session_scores = average_scores_per_session(results)
    recovery_table = build_gap_recovery_table(results)

    plot_r2_vs_calibration_trials(session_scores, output_path('fig1_r2_vs_k.png'))
    plot_method_comparison(session_scores, output_path('fig2_method_comparison.png'))
    plot_ridge_factorial(session_scores, output_path('fig3_ridge_factorial.png'))
    plot_session_heatmap(session_scores, output_path('fig4_session_heatmap.png'))
    plot_training_curves(training_log_dir, output_path('fig5_training_curves.png'))
    plot_noise_floor(results, out_dir, output_path('fig6_noise_floor.png'))
    plot_recovery(recovery_table, output_path('recovery_vs_k.png'))

    recovery_table.to_csv(output_path('table_recovery.csv'), index=False)
    write_main_tables(session_scores, out_dir)
    write_ridge_factorial_table(session_scores, out_dir)
    session_scores.pivot_table(index='session', columns=['method', 'k'], values='r2').to_csv(
        output_path('table_per_session.csv'))

# ---------------------------------------------------------------------------
# 9. EEGNet baseline
# ---------------------------------------------------------------------------

SAMPLE_RATE_HZ = 1000
MAX_TRAINING_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-4
STIMULATION_TONE_FREQUENCIES = (3.0, 20.0)
STIMULATION_TONE_STRENGTH = (0.5, 3.0)
EEGNET_METHOD = 'eegnet'
EEGNET_TONE_METHOD = 'eegnet_stimulation_tones'
EEGNET_ENSEMBLE_METHOD = 'eegnet_ensemble'
ENSEMBLE_SEED = -1  # seed column value for a score averaged over every seed
EEGNET_DEFAULT_SETTINGS = dict(temporal_filters=8, depth_multiplier=2, kernel_samples=500,
                               separable_samples=62, dropout=0.5)


def off_training_datasets():
    return concat_off_training_sessions(load_lfp_split, 'train')


def off_validation_datasets():
    datasets = [load_lfp_dataset(session, None)
                for session in sessions_in_split('val', STIMULATED_OFF)]
    return {key: np.concatenate([dataset[key] for dataset in datasets])
            for key in ('features', 'targets', 'valid_bins')}


def stimulated_test_datasets():
    return {session: (load_lfp_split(session, 'query'),
                      load_lfp_dataset(SPLITS['off_reference'][session], None))
            for session in sessions_in_split('test', STIMULATED_ON)}


def amp10_test_datasets(sessions=None):
    """Query trials read from the 10x semi-synthetic recording, same-day OFF unchanged.

    Only the ON recording is amplified, so the OFF control and the targets are the same
    arrays stimulated_test_datasets() uses; the pair differs only in the input signal.
    """
    if not os.path.isdir(SEMISYNTHETIC_EXPORT_DIR):
        raise FileNotFoundError(
            f'No semi-synthetic exports at {SEMISYNTHETIC_EXPORT_DIR}; run\n'
            '  python semisynthetic/EECS182_make_semisynthetic_h5.py '
            '--root data --split val-test --h5-only')
    sessions = sessions or sessions_in_split('test', STIMULATED_ON)
    paths = semisynthetic_paths()
    return {session: (load_lfp_split(session, 'query', paths),
                      load_lfp_dataset(SPLITS['off_reference'][session], None))
            for session in sessions}


def parrm_test_datasets(sessions=None, parrm_dir=None):
    """Query trials read from the PARRM-cleaned ON recording, same-day OFF unchanged.

    The cleaned file only exists for stimulation-ON sessions, so the OFF control stays on
    the recorded export and is identical to stimulated_test_datasets()'s.
    """
    sessions = sessions or sessions_in_split('test', STIMULATED_ON)
    paths = parrm_paths(parrm_dir) if parrm_dir else parrm_paths()
    return {session: (load_lfp_split(session, 'query', paths, source='parrm'),
                      load_lfp_dataset(SPLITS['off_reference'][session], None))
            for session in sessions}


# The four input arms of the 10x analysis: recorded, amplified, and the two ways of
# cleaning the amplified recording. See task4/make_parrm_amp10.py for why
# 'parrm_reestimated' is the existing PARRM output rather than a new file.
PARRM_AMP10_REUSE_DIR = os.path.join(DATA_DIR, 'semi_synthetic', 'parrm_amp10_reuse', 'parrm_h5')
# feature_variant labels written into the result tables, one per arm. Distinct from
# PARRM_VARIANT ('parrm_none'), which names the cached-feature directory.
PARRM_LFP_VARIANT = 'parrm_lfp'
PARRM_REUSE_VARIANT = 'parrm_amp10_reuse'

ARM_VARIANTS = {
    'recorded': 'raw',
    'amp10': AMP10_VARIANT,
    'parrm_recorded': PARRM_LFP_VARIANT,
    'parrm_amp10_reuse': PARRM_REUSE_VARIANT,
}


def amp10_arms(sessions=None, arms=None):
    """Score-ready datasets for each input arm, keyed by arm name.

    'parrm_recorded' is PARRM re-estimated on the amplified signal: the estimate scales with
    the artifact, so subtracting it returns the recorded cleaned LFP. 'parrm_amp10_reuse'
    subtracts one copy of the 1x template from the 10x signal, leaving eight copies.
    """
    arms = arms or list(ARM_VARIANTS)
    datasets = {}
    if 'recorded' in arms:
        datasets['recorded'] = stimulated_test_datasets()
    if 'amp10' in arms:
        datasets['amp10'] = amp10_test_datasets(sessions)
    if 'parrm_recorded' in arms:
        datasets['parrm_recorded'] = parrm_test_datasets(sessions)
    if 'parrm_amp10_reuse' in arms:
        if not os.path.isdir(PARRM_AMP10_REUSE_DIR):
            raise FileNotFoundError(
                f'No reused-template files at {PARRM_AMP10_REUSE_DIR}; run\n'
                '  python task4/make_parrm_amp10.py --split val-test')
        datasets['parrm_amp10_reuse'] = parrm_test_datasets(sessions,
                                                            parrm_dir=PARRM_AMP10_REUSE_DIR)
    return datasets


def train_with_early_stopping(model, train, validation, seed,
                              max_epochs=MAX_TRAINING_EPOCHS,
                              patience=EARLY_STOPPING_PATIENCE):
    device = select_device()
    # seeds shuffling and dropout; the caller seeds before building the model
    torch.manual_seed(seed)
    shuffling = np.random.default_rng(seed)
    model = model.to(device)

    keys = ('features', 'targets', 'valid_bins')
    training_tensors = to_tensors([train[key] for key in keys])
    validation_features, validation_targets, validation_valid = [
        tensor.numpy() for tensor in to_tensors([validation[key] for key in keys])]

    optimizers = build_optimizers(model)
    initial_rates = [[group['lr'] for group in optimizer.param_groups] for optimizer in optimizers]

    best_validation_r2 = -np.inf
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    history = []
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        order = shuffling.permutation(len(training_tensors[0]))
        training_loss = train_epoch(model, training_tensors, masked_mse, optimizers, order,
                                    TRIAL_BATCH_SIZE, device)
        validation_r2 = mean_r2(validation_targets, predict(model, validation_features),
                                validation_valid)
        history.append(dict(epoch=epoch, train_loss=training_loss, val_r2=validation_r2))

        if validation_r2 > best_validation_r2 + EARLY_STOPPING_MIN_DELTA:
            best_validation_r2, best_epoch = validation_r2, epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
        set_learning_rates(optimizers, initial_rates, cosine_lr_multiplier(epoch, max_epochs))

    model.load_state_dict(best_weights)
    model.eval()
    return model, dict(training_log=history, best_val_r2=best_validation_r2,
                       best_epoch=best_epoch, stopped_epoch=len(history))


def eegnet_sweep_settings():
    settings = EEGNET_DEFAULT_SETTINGS
    return [
        settings,
        settings | dict(kernel_samples=250),
        settings | dict(kernel_samples=1000),
        settings | dict(dropout=0.25),
        settings | dict(dropout=0.75),
        settings | dict(depth_multiplier=1),
        settings | dict(kernel_samples=250, dropout=0.25),
        settings | dict(kernel_samples=1000, dropout=0.25),
    ]


def sweep_eegnet_settings(settings, train, validation, build_model, seed=0,
                          max_epochs=MAX_TRAINING_EPOCHS):
    rows = []
    for setting in settings:
        torch.manual_seed(seed)  # seed the init as well, not just the shuffling
        model, history = train_with_early_stopping(build_model(**setting), train, validation,
                                                   seed, max_epochs)
        rows.append(dict(**setting, seed=seed, best_val_r2=history['best_val_r2'],
                         best_epoch=history['best_epoch'], stopped_epoch=history['stopped_epoch']))
        print(setting, 'validation R2', round(history['best_val_r2'], 4))
    return (pd.DataFrame(rows)
            .sort_values('best_val_r2', ascending=False)
            .reset_index(drop=True))


def best_eegnet_settings(sweep):
    best = sweep.iloc[0]
    return dict(temporal_filters=int(best.temporal_filters),
                depth_multiplier=int(best.depth_multiplier),
                kernel_samples=int(best.kernel_samples),
                separable_samples=int(best.separable_samples),
                dropout=float(best.dropout))


def add_stimulation_tones(dataset, seed, strength_range=STIMULATION_TONE_STRENGTH,
                          frequencies=STIMULATION_TONE_FREQUENCIES):
    generator = np.random.default_rng(seed)
    features = dataset['features'].copy()
    trial_count, channel_count, sample_count = features.shape
    time = np.arange(sample_count) / SAMPLE_RATE_HZ

    for trial in range(trial_count):
        frequency = generator.choice(frequencies)
        strength = generator.uniform(*strength_range)
        phase = generator.uniform(0, 2 * np.pi)
        channel_gain = generator.uniform(0.5, 1.5, size=(channel_count, 1))
        features[trial] += strength * channel_gain * np.sin(2 * np.pi * frequency * time + phase)

    return dataset | dict(features=features)


def score_ensemble(models, session, method, feature_label, query, off, seed=ENSEMBLE_SEED):
    query_prediction = np.mean([predict(model, query['features']) for model in models], axis=0)
    off_prediction = np.mean([predict(model, off['features']) for model in models], axis=0)
    return metrics_row(session, method, feature_label,
                       query['targets'][query['valid_bins']], query_prediction[query['valid_bins']],
                       off['targets'][off['valid_bins']], off_prediction[off['valid_bins']],
                       seed=seed)


def scores_per_day(scores, column):
    scored = scores.assign(day=scores.session.str.split('-').str[1])
    per_session = scored.groupby(['method', 'day', 'session'], as_index=False)[column].mean()
    return per_session.groupby(['method', 'day'], as_index=False)[column].mean()


def day_level_summary(scores, column):
    per_day = scores_per_day(scores, column)
    rows = []
    for method, method_days in per_day.groupby('method'):
        values = method_days[column].to_numpy()
        low, high = bootstrap_confidence_interval(values)
        rows.append(dict(method=method, column=column, days=len(values),
                         mean=float(np.mean(values)), low=low, high=high))
    return pd.DataFrame(rows)


def write_baseline_parquet(scores, path):
    columns = ['method', 'feature_variant', 'session', 'am_freq', 'k', 'draw', 'seed',
               'r2_vx', 'r2_vy', 'r2_ave', 'r2_off_vx', 'r2_off_vy', 'r2_off_ave', 'n_bins']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scores[columns].to_parquet(path, index=False)
