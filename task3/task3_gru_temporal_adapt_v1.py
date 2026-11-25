"""Test session-specific time-profile correction of frozen and full-tuned GRU predictions.

Comparison: original frozen GRU and calibrated full GRU versus those same models
with a smooth, time-within-trial residual correction learned from 20 pool trials.
A pool-only, two-block cross-validation selects DCT complexity and shrinkage
separately for vx and vy. Thirty query trials remain untouched until scoring.
The adapted time-only baseline is reported to isolate neural gain.

Run from EECS282_finalProject root:
    python task3/task3_gru_temporal_adapt_v1.py

Requires task3_gru_full_finetune_v1.py, the pretrained multitask checkpoint,
and the previous full-finetune pool-CV CSV. Does not change any input files.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import task3_gru_full_finetune_v1 as F
from task3_paths import GRU_RESULTS, gru_artifact

P, R, M, T, G, C, D = F.P, F.R, F.M, F.T, F.G, F.C, F.D
KS = (0, 1, 3, 6, 10)
LAMBDAS = (0.01, 0.1, 1.0, 10.0)


def basis(positions, k, n_positions):
    if k == 0:
        return np.zeros((*positions.shape, 0), dtype=np.float64)
    freq = np.arange(k, dtype=np.float64)
    values = np.cos(np.pi * (positions[..., None] + 0.5) * freq / n_positions)
    values[..., 1:] *= np.sqrt(2.0)
    return values


def fit_correction(batch, prediction, scale, axis, k, lam):
    if k == 0:
        return np.empty(0, dtype=np.float64)
    valid = batch['valid']
    n_positions = batch['position'].shape[1]
    x = basis(batch['position'][valid], k, n_positions)
    residual = ((batch['Y'] - batch['baseline']) / scale - prediction)[valid, axis]
    n = len(residual)
    if n < 20:
        raise ValueError('Insufficient valid calibration samples')
    return np.linalg.solve(x.T @ x / n + lam * np.eye(k), x.T @ residual / n)


def correction(batch, prediction, scale, choices, coefficients=None):
    delta = np.zeros_like(prediction, dtype=np.float64)
    for axis in range(2):
        k, lam = choices[axis]
        if k == 0:
            continue
        w = (fit_correction(batch, prediction, scale, axis, k, lam)
             if coefficients is None else coefficients[axis])
        delta[..., axis] = basis(batch['position'], k, batch['position'].shape[1]) @ w
    return delta


def apply_correction(train, train_pred, evaluate, eval_pred, scale, choices):
    delta = np.zeros_like(eval_pred, dtype=np.float64)
    for axis, (k, lam) in enumerate(choices):
        if k:
            coef = fit_correction(train, train_pred, scale, axis, k, lam)
            delta[..., axis] = basis(evaluate['position'], k, evaluate['position'].shape[1]) @ coef
    return eval_pred + delta


def subset(batch, indices):
    return {name: batch[name][indices] for name in ('X', 'Y', 'baseline', 'valid', 'position')}


def fold_predictions(pools, config, state, scale, device, lr, epochs, anchor, seed):
    folds = {kind: {} for kind in ('time', 'frozen', 'full')}
    for session, pool in pools.items():
        batch = pool['batch']
        if len(batch['trial_ids']) != 20:
            raise ValueError(f'{session}: expected 20 pool trials')
        for fold in range(2):
            train_idx = slice(0, 10) if fold == 0 else slice(10, 20)
            held_idx = slice(10, 20) if fold == 0 else slice(0, 10)
            train = subset(batch, train_idx)
            held = subset(batch, held_idx)
            zero_train = np.zeros_like(train['Y'], dtype=np.float64)
            zero_held = np.zeros_like(held['Y'], dtype=np.float64)
            folds['time'].setdefault(session, []).append((train, held, zero_train, zero_held))
            folds['frozen'].setdefault(session, []).append((train, held,
                pool['original'][train_idx], pool['original'][held_idx]))
            fitted = F.fit_full(config, state, train, scale, device, lr, epochs,
                                anchor, seed + fold)
            full_train = P.predict_partial(fitted, train, device)
            full_held = P.predict_partial(fitted, held, device)
            folds['full'].setdefault(session, []).append((train, held, full_train, full_held))
        print(f'  Prepared calibration folds: {session}', flush=True)
    return folds


def select_choices(folds, scale, sessions):
    options = [(0, 0.0)] + [(k, lam) for k in KS if k for lam in LAMBDAS]
    rows = []
    best = {}
    for method in ('time', 'frozen', 'full'):
        selected = []
        for axis in range(2):
            for k, lam in options:
                errors = []
                for session in sessions:
                    fold_errors = []
                    for train, held, p_train, p_held in folds[method][session]:
                        corr = apply_correction(train, p_train, held, p_held,
                                                scale, [(k, lam)] * 2)
                        true = (held['Y'] - held['baseline']) / scale
                        masked = held['valid']
                        fold_errors.append(float(np.mean((true - corr)[masked, axis] ** 2)))
                    errors.append(float(np.mean(fold_errors)))
                rows.append(dict(method=method, axis=('vx', 'vy')[axis],
                                 k=k, lam=lam, cv_mse=np.mean(errors)))
            axis_rows = [r for r in rows if r['method'] == method and
                         r['axis'] == ('vx', 'vy')[axis]]
            choice = min(axis_rows, key=lambda r: r['cv_mse'])
            selected.append((int(choice['k']), float(choice['lam'])))
        best[method] = selected
        print(f'  {method}: vx (K={selected[0][0]}, lambda={selected[0][1]:g}), '
              f'vy (K={selected[1][0]}, lambda={selected[1][1]:g})', flush=True)
    return best, pd.DataFrame(rows)


def score(session, method, batch, prediction, scale, adapted_time):
    valid = batch['valid']
    truth = batch['Y'][valid].astype(np.float64)
    pred = (batch['baseline'] + prediction * scale)[valid].astype(np.float64)
    original_time = batch['baseline'][valid].astype(np.float64)
    new_time = (batch['baseline'] + adapted_time * scale)[valid].astype(np.float64)
    r = C.r2(truth, pred)
    r_time = C.r2(truth, original_time)
    r_new_time = C.r2(truth, new_time)
    return dict(session=session, method=method, n_bins=int(valid.sum()),
                r2=float(r.mean()), r2_vx=float(r[0]), r2_vy=float(r[1]),
                r2_time=float(r_time.mean()), r2_time_adapted=float(r_new_time.mean()),
                gain_over_adapted_time=float((r-r_new_time).mean()),
                gain_vx=float(r[0]-r_new_time[0]),
                gain_vy=float(r[1]-r_new_time[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=gru_artifact('checkpoints', R.DEFAULT_CHECKPOINT))
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
    parser.add_argument('--anchor', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    if args.anchor < 0:
        parser.error('--anchor must be nonnegative')
    cv_path = gru_artifact('calibration', 'task3_gru_full_finetune_v1_pool_cv.csv')
    if not args.checkpoint.is_file() or not cv_path.is_file():
        raise SystemExit('Requires pretrained GRU checkpoint and full-finetune pool CV CSV: '
                         f'{args.checkpoint} / {cv_path}')
    cv = pd.read_csv(cv_path).sort_values('pool_cv_mse')
    lr, epochs = float(cv.iloc[0]['lr']), int(cv.iloc[0]['epochs'])
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('temporal_mode') != 'trial_continuous' or ck.get('target_columns') != list(D.TARGETS['velocity']):
        raise SystemExit('Checkpoint is not the expected trial-continuous velocity GRU')
    config, state = ck['model_config'], ck['state_dict']
    n_lags = int(ck['n_lags'])
    run_config = ck['run_config']
    feature_set = G.configure(run_config.get('mode', 'augmented'), 'velocity', n_lags,
                              run_config.get('feature_group', 'all'))
    if list(C.FEATURE_SETS[feature_set]) != list(ck['feature_names']):
        raise SystemExit('Checkpoint feature list mismatch')
    scale = np.asarray(ck['target_scale'], dtype=np.float64)
    profile = {k: np.asarray(v, np.float32) for k, v in ck['profile'].items()}
    sessions = C.sessions('val', 'no_stim')
    if set(sessions) != set(ck['val_sessions']):
        raise SystemExit('Validation sessions do not match checkpoint')
    frozen = F.full_model(config, state, device)
    pools, queries = {}, {}
    for s in sessions:
        pools[s] = R.package(s, 'pool', feature_set, n_lags, profile, frozen, device, 8)
        queries[s] = R.package(s, 'query', feature_set, n_lags, profile, frozen, device, 8)
    print(f'Full fine-tune lr={lr:g}, epochs={epochs}, anchor={args.anchor:g} (previous pool CV)')
    print('Adapt 20 calibration trials, score 30 untouched query trials per session')
    folds = fold_predictions(pools, config, state, scale, device,
                             lr, epochs, args.anchor, args.seed)
    choices, tuning = select_choices(folds, scale, sessions)
    rows = []
    for s in sessions:
        pool, query = pools[s], queries[s]
        pb, qb = pool['batch'], query['batch']
        time_pool = np.zeros_like(pool['original'])
        time_query = np.zeros_like(query['original'])
        adapt_time = apply_correction(pb, time_pool, qb, time_query, scale, choices['time'])
        full = F.fit_full(config, state, pb, scale, device, lr, epochs, args.anchor, args.seed)
        full_pool = P.predict_partial(full, pb, device)
        full_query = P.predict_partial(full, qb, device)
        freeze_time = apply_correction(pb, pool['original'], qb,
                                       query['original'], scale, choices['frozen'])
        full_time = apply_correction(pb, full_pool, qb, full_query, scale, choices['full'])
        methods = dict(frozen=query['original'], full_finetune=full_query,
                       time_only_adapted=adapt_time, frozen_time_adapted=freeze_time,
                       full_time_adapted=full_time)
        for method, prediction in methods.items():
            rows.append(score(s, method, qb, prediction, scale, adapt_time))
    result = pd.DataFrame(rows)
    metrics = ['r2','r2_vx','r2_vy','r2_time','r2_time_adapted','gain_over_adapted_time',
               'gain_vx','gain_vy']
    print('\nSession-mean query R² (each day weighted equally):')
    print(result.groupby('method', sort=False)[metrics].mean()
          .to_string(float_format=lambda x: f'{x:.4f}'))
    print('\nPer-session query R²:')
    print(result[['session','method','n_bins','r2','r2_vx','r2_vy','gain_over_adapted_time']]
          .to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    output = (GRU_RESULTS / 'calibration')
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output/'task3_gru_temporal_adapt_v1_sessions.csv', index=False)
    tuning.to_csv(output/'task3_gru_temporal_adapt_v1_pool_cv.csv', index=False)
    print(f'\nSaved CSVs under {output}')


def self_test():
    rng = np.random.default_rng(21)
    n, t = 20, 40
    position = np.tile(np.arange(t), (n, 1))
    scale = np.array([20., 30.])
    baseline = np.zeros((n,t,2))
    phase = 0.5 * np.cos(np.pi*(np.arange(t)+.5)*3/t)
    y = rng.normal(0,.02,(n,t,2)) * scale + phase[None,:,None] * scale
    batch = dict(Y=y, baseline=baseline, valid=np.ones((n,t),bool), position=position)
    prediction = np.zeros_like(y)
    choices = [(4, .01), (4, .01)]
    p = apply_correction({**batch,'X':np.zeros((n,1,1))},prediction,
                         batch,prediction,scale,choices)
    assert np.mean((y/scale-p)**2) < np.mean((y/scale)**2)*.02
    vals = basis(position,4,t)
    assert np.isfinite(vals).all()
    assert np.allclose(correction(batch,prediction,scale,choices),p)
    print('Self-test passed: smooth time-dependent correction recovers a known profile')


if __name__ == '__main__':
    import sys
    if '--self-test' in sys.argv:
        self_test()
    else:
        main()
