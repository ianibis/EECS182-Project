"""Calibrate the frozen multitask GRU velocity readout using 20 OFF trials/session.

Comparison: original frozen GRU vs calibration-pool bias correction vs regularized
GRU readout correction on the same 30 query trials of each validation OFF day.
The model, features, original time-only profile and all neural normalization
remain frozen. Readout regularization is selected exclusively by cross-fitting
the two calibration blocks; query trials are never used for selection.

Run from the EECS282_finalProject root:
    python task3/task3_gru_readout_recal_v1.py

Requires the existing task3_gru_multitask_v1.py and its dependencies and the
original saved GRU checkpoint. Does not access stimulation-ON or test days.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import task3_gru_multitask_v1 as M
from task3_paths import GRU_RESULTS, gru_artifact

T, G, C, D, H = M.T, M.G, M.C, M.D, M.H
LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_CHECKPOINT = (
    'task3_gru_multitask_v1_augmented_all_lags19_h64_seed1_'
    'seljoint_cd0.1_sw0.2_dw0.2.pt'
)


def readout_features(model, batch, device='cpu', batch_size=8):
    model.eval()
    all_states = []
    all_residual = []
    with torch.no_grad():
        for start in range(0, len(batch['X']), batch_size):
            x = torch.as_tensor(batch['X'][start:start + batch_size], device=device)
            hidden, _ = model.gru(torch.tanh(model.projection(x)))
            hidden = hidden[:, model.warmup:, :]
            pred = model.readout(hidden)
            all_states.append(hidden.cpu().numpy())
            all_residual.append(pred.cpu().numpy())
    states = np.concatenate(all_states, axis=0).astype(np.float64)
    original = np.concatenate(all_residual, axis=0).astype(np.float64)
    return states, original


def fit_correction(states, old_predictions, targets, valid, lam, scale):
    """Fit ΔW/Δb relative to the pretrained readout; intercept unpenalized.

    Objective: mean ||Y_normalized - Y_frozen - H ΔW - Δb||²
               + lambda * ||ΔW||².
    """
    x = states[valid]
    residual = ((targets / scale) - old_predictions)[valid]
    if len(x) < x.shape[1] + 1:
        raise ValueError(f'Only {len(x)} valid calibration bins for {x.shape[1]} hidden features')
    mx = x.mean(axis=0)
    my = residual.mean(axis=0)
    xc = x - mx
    yc = residual - my
    gram = xc.T @ xc
    weight = np.linalg.solve(gram + len(x) * lam * np.eye(gram.shape[0]), xc.T @ yc)
    intercept = my - mx @ weight
    return weight, intercept


def predict_adapt(states, original, coef):
    weights, intercept = coef
    return original + states @ weights + intercept


def package(session, part, feature_set, n_lags, profile, model, device, batch_size):
    batch = T.load_level(session, part, feature_set, n_lags, profile)
    states, original = readout_features(model, batch, device, batch_size)
    if states.shape[:2] != batch['valid'].shape:
        raise ValueError('GRU readout and target bin shapes differ')
    return dict(batch=batch, states=states, original=original,
                targets=np.asarray(batch['Y'] - batch['baseline'], dtype=np.float64))


def crossfit_mse(pool, lam, scale):
    """Two-fold CV on the calibration trials, split chronologically 10/10."""
    batch = pool['batch']
    n = len(batch['trial_ids'])
    if n != 20:
        raise ValueError(f'Expected 20 calibration trials, got {n}')
    loss, count = 0.0, 0
    for first in (slice(0, 10), slice(10, 20)):
        second = slice(10, 20) if first.start == 0 else slice(0, 10)
        fit = fit_correction(pool['states'][first], pool['original'][first],
                             pool['targets'][first], batch['valid'][first], lam, scale)
        prediction = predict_adapt(pool['states'][second], pool['original'][second], fit)
        mask = batch['valid'][second]
        errors = ((pool['targets'][second] / scale - prediction)[mask]) ** 2
        loss += errors.sum()
        count += errors.size
    return loss / count


def score(query, normalized_prediction, scale, label, session):
    batch = query['batch']
    pred_residual = normalized_prediction * scale
    row = T.session_score(batch, pred_residual)
    return dict(session=session, method=label, **row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=gru_artifact('checkpoints', DEFAULT_CHECKPOINT))
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if not args.checkpoint.is_file():
        raise SystemExit(f'Missing checkpoint: {args.checkpoint}')
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS is not available')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['model_config']
    run_config = checkpoint['run_config']
    if checkpoint.get('temporal_mode') != 'trial_continuous':
        raise SystemExit('Checkpoint must be a trial-continuous GRU')
    if checkpoint.get('target_columns') != list(D.TARGETS['velocity']):
        raise SystemExit('Checkpoint target columns do not match vx/vy')
    mode = run_config.get('mode', 'augmented')
    group = run_config.get('feature_group', 'all')
    n_lags = int(checkpoint['n_lags'])
    feature_set = G.configure(mode, 'velocity', n_lags, group)
    if list(C.FEATURE_SETS[feature_set]) != list(checkpoint['feature_names']):
        raise SystemExit('Checkpoint features do not match configured feature set')
    profile = {k: np.asarray(v, np.float32) for k, v in checkpoint['profile'].items()}
    scale = np.asarray(checkpoint['target_scale'], np.float64)
    if scale.shape != (2,) or np.any(scale <= 0):
        raise SystemExit('Invalid target scale in checkpoint')
    model = M.MultitaskTrialGRU(**config).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    sessions = C.sessions('val', 'no_stim')
    if set(sessions) != set(checkpoint['val_sessions']):
        raise SystemExit('Current validation sessions differ from checkpoint')

    print(f'Checkpoint: {args.checkpoint.name}, selected epoch {checkpoint["best_epoch"]}')
    print(f'Unchanged pretrained GRU, {len(sessions)} OFF validation days, {device}')
    print('20 pool trials: select lambda by two-fold block CV; fit on all 20.')
    print('30 query trials: evaluate only; no query labels used in adaptation or selection.')
    pools, queries = {}, {}
    for session in sessions:
        pools[session] = package(session, 'pool', feature_set, n_lags, profile, model, device, args.batch_size)
        queries[session] = package(session, 'query', feature_set, n_lags, profile, model, device, args.batch_size)
        print(f'  {session}: pool {int(pools[session]["batch"]["valid"].sum())} valid bins, '
              f'query {int(queries[session]["batch"]["valid"].sum())} valid bins')

    choices = []
    for lam in LAMBDAS:
        losses = [crossfit_mse(pools[s], lam, scale) for s in sessions]
        choices.append(dict(lam=lam, cv_mse=float(np.mean(losses)),
                            **{f'cv_mse_{s}': v for s, v in zip(sessions, losses)}))
    tuning = pd.DataFrame(choices)
    chosen = float(tuning.loc[tuning.cv_mse.idxmin(), 'lam'])
    print('\nPool-only cross-validation (normalized velocity MSE):')
    print(tuning[['lam', 'cv_mse']].to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    print(f'Selected lambda: {chosen:g}')

    rows = []
    for session in sessions:
        pool, query = pools[session], queries[session]
        rows.append(score(query, query['original'], scale, 'frozen', session))
        for label, lam in (('bias_only', 1e12), ('readout_recal', chosen)):
            if label == 'bias_only':
                valid = pool['batch']['valid']
                delta_bias = (pool['targets'] / scale - pool['original'])[valid].mean(axis=0)
                adapted = query['original'] + delta_bias
            else:
                fit = fit_correction(pool['states'], pool['original'], pool['targets'],
                                     pool['batch']['valid'], lam, scale)
                adapted = predict_adapt(query['states'], query['original'], fit)
            rows.append(score(query, adapted, scale, label, session))
    scores = pd.DataFrame(rows)
    print('\nQuery-trial scores (each session, unchanged bins):')
    print(scores[['session', 'method', 'n_bins', 'r2', 'r2_vx', 'r2_vy', 'r2_gain']]
          .to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nSession-mean query R² (each day weighted equally):')
    summary = scores.groupby('method', sort=False)[['r2', 'r2_vx', 'r2_vy', 'r2_time',
                                                     'r2_gain', 'gain_vx', 'gain_vy']].mean()
    print(summary.to_string(float_format=lambda x: f'{x:.4f}'))
    out = (GRU_RESULTS / 'calibration')
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / 'task3_gru_readout_recal_v1_sessions.csv', index=False)
    tuning.to_csv(out / 'task3_gru_readout_recal_v1_pool_cv.csv', index=False)
    print(f'\nSaved CSVs under {out}')


def self_test():
    rng = np.random.default_rng(3)
    hidden = rng.normal(size=(20, 40, 64))
    old = rng.normal(size=(20, 40, 2))
    scale = np.array([25., 30.])
    delta = rng.normal(scale=.03, size=(64, 2))
    shift = np.array([.6, -.4])
    targets = (old + hidden @ delta + shift) * scale
    valid = np.ones((20, 40), bool)
    coeff = fit_correction(hidden, old, targets, valid, 0.001, scale)
    pred = predict_adapt(hidden, old, coeff)
    before = np.mean((targets / scale - old) ** 2)
    after = np.mean((targets / scale - pred) ** 2)
    assert after < .005 * before, (before, after)
    assert pred.shape == old.shape
    test_pool = dict(states=hidden, original=old, targets=targets,
                     batch={'valid': valid, 'trial_ids': np.arange(20)})
    assert crossfit_mse(test_pool, 0.001, scale) < .005 * before
    print('Self-test passed: regularized correction, intercept, and 20-trial cross-fit')


if __name__ == '__main__':
    if '--self-test' in __import__('sys').argv:
        self_test()
    else:
        main()
