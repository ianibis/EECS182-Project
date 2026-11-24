"""Compare frozen, readout-recalibrated, and projection+readout-tuned GRUs.

Uses the saved OFF multitask GRU and the unchanged OFF validation-day splits.
The GRU recurrent weights and auxiliary heads stay frozen; only the input
projection and velocity readout are tuned. Select LR and epoch using only the
20 calibration trials (two-fold by block), then fit on all 20 and score the
30 untouched query trials. Original checkpoint, features and labels unchanged.

From project root:
    python task3/task3_gru_partial_finetune_v1.py

Depends on task3_gru_readout_recal_v1.py and its existing Task 3 modules.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import task3_gru_readout_recal_v1 as R
from task3_paths import GRU_RESULTS, gru_artifact

M, T, G, C, D = R.M, R.T, R.G, R.C, R.D
DEFAULT_LRS = (0.0001, 0.0003, 0.001)
DEFAULT_EPOCHS = (5, 10, 20)


def new_model(config, state, device):
    model = M.MultitaskTrialGRU(**config).to(device)
    model.load_state_dict(state, strict=True)
    for param in model.parameters():
        param.requires_grad_(False)
    for layer in (model.projection, model.readout):
        for param in layer.parameters():
            param.requires_grad_(True)
    model.eval()
    return model


def fit_partial(config, state, batch, scale, device, lr, epochs, anchor, seed=1):
    model = new_model(config, state, device)
    params = [p for p in model.parameters() if p.requires_grad]
    initial = [p.detach().clone() for p in params]
    optimizer = torch.optim.Adam(params, lr=lr)
    x = torch.as_tensor(batch['X'], device=device, dtype=torch.float32)
    target = torch.as_tensor(
        np.nan_to_num((batch['Y'] - batch['baseline']) / scale, nan=0.0),
        device=device, dtype=torch.float32)
    mask = torch.as_tensor(batch['valid'], device=device, dtype=torch.bool)
    if int(mask.sum()) < 20:
        raise ValueError('Too few valid calibration bins')
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for ids in np.array_split(order, max(1, int(np.ceil(len(order) / 8)))):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[ids])[0]
            errors = (prediction - target[ids]).square().mean(dim=2)
            loss = errors[mask[ids]].mean()
            if anchor:
                loss = loss + anchor * sum(
                    (p - p0).square().mean() for p, p0 in zip(params, initial))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
    return model


def predict_partial(model, batch, device, batch_size=8):
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(batch['X']), batch_size):
            x = torch.as_tensor(batch['X'][start:start + batch_size],
                                dtype=torch.float32, device=device)
            output.append(model(x)[0].cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float64)


def calibration_cv(config, state, pool, scale, device, lr, epochs, anchor, seed):
    batch = pool['batch']
    n = len(batch['trial_ids'])
    if n != 20:
        raise ValueError(f'Expected 20 pool trials, got {n}')
    mse = 0.0
    count = 0
    for fold in range(2):
        train_idx = np.arange(0, 10) if fold == 0 else np.arange(10, 20)
        val_idx = np.arange(10, 20) if fold == 0 else np.arange(0, 10)
        training = {k: v[train_idx] for k, v in batch.items()
                    if k in ('X', 'Y', 'baseline', 'valid')}
        model = fit_partial(config, state, training, scale, device,
                            lr, epochs, anchor, seed + fold)
        validating = {k: v[val_idx] for k, v in batch.items()
                      if k in ('X', 'Y', 'baseline', 'valid')}
        predicted = predict_partial(model, validating, device)
        truth = (validating['Y'] - validating['baseline']) / scale
        valid = validating['valid']
        err = (truth - predicted)[valid]
        mse += float(np.square(err).sum())
        count += err.size
    return mse / count


def score_method(query, normalized, scale, name, session):
    row = T.session_score(query['batch'], normalized * scale)
    return dict(session=session, method=name, **row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=gru_artifact('checkpoints', R.DEFAULT_CHECKPOINT))
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
    parser.add_argument('--anchor', type=float, default=1.0,
                        help='Penalty on squared changes relative to pretrained parameters')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    if args.anchor < 0:
        parser.error('--anchor must be nonnegative')
    if not args.checkpoint.is_file():
        raise SystemExit(f'Missing checkpoint: {args.checkpoint}')
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS is unavailable; use --device cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('temporal_mode') != 'trial_continuous':
        raise SystemExit('Expected a trial-continuous checkpoint')
    if checkpoint.get('target_columns') != list(D.TARGETS['velocity']):
        raise SystemExit('Checkpoint target columns do not match vx/vy')
    config = checkpoint['model_config']
    state = checkpoint['state_dict']
    n_lags = int(checkpoint['n_lags'])
    run_config = checkpoint['run_config']
    feature_set = G.configure(run_config.get('mode', 'augmented'), 'velocity',
                              n_lags, run_config.get('feature_group', 'all'))
    if list(C.FEATURE_SETS[feature_set]) != list(checkpoint['feature_names']):
        raise SystemExit('Feature mismatch between checkpoint and local code')
    scale = np.asarray(checkpoint['target_scale'], np.float64)
    profile = {k: np.asarray(v, np.float32)
               for k, v in checkpoint['profile'].items()}
    sessions = C.sessions('val', 'no_stim')
    if set(sessions) != set(checkpoint['val_sessions']):
        raise SystemExit('Validation sessions do not match the checkpoint')
    frozen = new_model(config, state, device)
    print(f'Checkpoint: {args.checkpoint.name}; {len(sessions)} OFF validation days')
    print('Partial tuning: projection+readout only; recurrent GRU and auxiliary heads frozen')
    print('20 calibration trials: two-fold block CV; 30 query trials: score only')
    pools, queries = {}, {}
    for session in sessions:
        pools[session] = R.package(session, 'pool', feature_set, n_lags,
                                   profile, frozen, device, 8)
        queries[session] = R.package(session, 'query', feature_set, n_lags,
                                     profile, frozen, device, 8)
        print(f'  {session}: calibration bins {pools[session]["batch"]["valid"].sum()}, '
              f'query bins {queries[session]["batch"]["valid"].sum()}')
    readout_cv = pd.DataFrame([dict(lam=lam, mse=float(np.mean([
        R.crossfit_mse(pools[s], lam, scale) for s in sessions])))
        for lam in R.LAMBDAS])
    best_lam = float(readout_cv.loc[readout_cv.mse.idxmin(), 'lam'])
    print(f'Readout-only ridge lambda from calibration CV: {best_lam:g}')
    cv_rows = []
    for lr, epochs in itertools.product(DEFAULT_LRS, DEFAULT_EPOCHS):
        vals = [calibration_cv(config, state, pools[s], scale, device,
                               lr, epochs, args.anchor, args.seed)
                for s in sessions]
        row = dict(lr=lr, epochs=epochs, pool_cv_mse=float(np.mean(vals)),
                   **{f'cv_{s}': v for s, v in zip(sessions, vals)})
        cv_rows.append(row)
        print(f'  CV lr={lr:.4g}, epochs={epochs}: normalized MSE {row["pool_cv_mse"]:.5f}', flush=True)
    tuning = pd.DataFrame(cv_rows).sort_values('pool_cv_mse')
    best = tuning.iloc[0]
    lr, epochs = float(best.lr), int(best.epochs)
    print(f'Selected partial fine-tune: lr={lr:g}, epochs={epochs} (pool-only CV)')
    rows = []
    for session in sessions:
        pool, query = pools[session], queries[session]
        rows.append(score_method(query, query['original'], scale, 'frozen', session))
        readout = R.fit_correction(pool['states'], pool['original'], pool['targets'],
                                   pool['batch']['valid'], best_lam, scale)
        readout_pred = R.predict_adapt(query['states'], query['original'], readout)
        rows.append(score_method(query, readout_pred, scale, 'readout_recal', session))
        tuned = fit_partial(config, state, pool['batch'], scale, device,
                            lr, epochs, args.anchor, args.seed)
        prediction = predict_partial(tuned, query['batch'], device)
        rows.append(score_method(query, prediction, scale, 'partial_finetune', session))
    scores = pd.DataFrame(rows)
    print('\nSession-mean query R² (each day weighted equally):')
    keys = ['r2', 'r2_vx', 'r2_vy', 'r2_time', 'r2_gain', 'gain_vx', 'gain_vy']
    print(scores.groupby('method', sort=False)[keys].mean().to_string(float_format=lambda x: f'{x:.4f}'))
    print('\nPer-session query R²:')
    print(scores[['session', 'method', 'n_bins', 'r2', 'r2_vx', 'r2_vy', 'r2_gain']].to_string(index=False,
                                                                                   float_format=lambda x: f'{x:.4f}'))
    out = (GRU_RESULTS / 'calibration')
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / 'task3_gru_partial_finetune_v1_sessions.csv', index=False)
    tuning.to_csv(out / 'task3_gru_partial_finetune_v1_pool_cv.csv', index=False)
    print(f'\nSaved CSVs: {out}/task3_gru_partial_finetune_v1_*.csv')


def self_test():
    rng = np.random.default_rng(9)
    config = dict(input_size=4, output_size=2, projection=5, hidden=6,
                  dropout=0.0, warmup=2, channel_dropout=0.0,
                  features_per_channel=2)
    init_model = M.MultitaskTrialGRU(**config)
    state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
    x = rng.normal(size=(20, 12, 4)).astype(np.float32)
    raw = np.zeros((20, 10, 2), dtype=np.float32)
    valid = np.ones((20, 10), dtype=bool)
    scale = np.array([10.0, 15.0])
    target_model = new_model(config, state, 'cpu')
    with torch.no_grad():
        target_model.readout.bias.add_(0.7)
        raw[:] = target_model(torch.from_numpy(x))[0].numpy() * scale
    batch = dict(X=x, Y=raw, baseline=np.zeros_like(raw), valid=valid,
                 trial_ids=np.arange(20))
    original = new_model(config, state, 'cpu')
    before = predict_partial(original, batch, 'cpu')
    trained = fit_partial(config, state, batch, scale, 'cpu', .01, 40, .1)
    after = predict_partial(trained, batch, 'cpu')
    truth = raw / scale
    assert np.mean((truth - after) ** 2) < np.mean((truth - before) ** 2)
    assert all(torch.equal(trained.state_dict()[k], v)
               for k, v in state.items() if k.startswith(('gru.', 'speed_head.', 'direction_head.')))
    cv = calibration_cv(config, state, dict(batch=batch), scale, 'cpu', .01, 20, .1, 1)
    assert np.isfinite(cv)
    print('Self-test passed: partial adaptation, frozen GRU, calibration CV and prediction shapes')


if __name__ == '__main__':
    if '--self-test' in __import__('sys').argv:
        self_test()
    else:
        main()
