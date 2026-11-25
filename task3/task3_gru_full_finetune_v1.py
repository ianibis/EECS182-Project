"""Compare frozen, readout, partial, and full GRU adaptation on OFF validation days.

The pretrained multitask GRU is fixed initially. For each validation session,
fit on 20 pool trials and score only its 30 query trials. Full adaptation
updates input projection, recurrent GRU and velocity readout, keeping auxiliary
heads frozen. LR and epoch count use two-fold, block-wise pool-only CV,
aggregated across validation sessions. Original checkpoints/data unchanged.

From the project root:
    python task3/task3_gru_full_finetune_v1.py

Requires task3_gru_partial_finetune_v1.py and its existing dependencies.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import task3_gru_partial_finetune_v1 as P
from task3_paths import GRU_RESULTS, gru_artifact

R, M, T, G, C, D = P.R, P.M, P.T, P.G, P.C, P.D
LRS = (3e-5, 1e-4, 3e-4)
EPOCHS = (8, 20)


def full_model(config, state, device):
    model = M.MultitaskTrialGRU(**config).to(device)
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in (model.projection, model.gru, model.readout):
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    model.eval()
    return model


def fit_full(config, state, batch, scale, device, lr, epochs, anchor, seed=1):
    model = full_model(config, state, device)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    initial = [parameter.detach().clone() for parameter in params]
    optimizer = torch.optim.Adam(params, lr=lr)
    x = torch.as_tensor(batch['X'], dtype=torch.float32, device=device)
    targets = torch.as_tensor(
        np.nan_to_num((batch['Y'] - batch['baseline']) / scale,
                      nan=0.0, posinf=0.0, neginf=0.0),
        dtype=torch.float32, device=device)
    valid = torch.as_tensor(batch['valid'], dtype=torch.bool, device=device)
    if int(valid.sum()) < 20:
        raise ValueError('Too few valid calibration bins')
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for indices in np.array_split(order, max(1, int(np.ceil(len(order) / 8)))):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[indices])[0]
            error = (prediction - targets[indices]).square().mean(dim=2)
            loss = error[valid[indices]].mean()
            if anchor:
                loss = loss + anchor * sum(
                    (parameter - pretrained).square().mean()
                    for parameter, pretrained in zip(params, initial))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
    return model


def pool_cv(config, state, pool, scale, device, lr, epochs, anchor, seed):
    batch = pool['batch']
    if len(batch['trial_ids']) != 20:
        raise ValueError('Expected exactly 20 calibration trials')
    error_sum, count = 0.0, 0
    for fold in range(2):
        train = slice(0, 10) if fold == 0 else slice(10, 20)
        held = slice(10, 20) if fold == 0 else slice(0, 10)
        fields = ('X', 'Y', 'baseline', 'valid')
        train_batch = {key: batch[key][train] for key in fields}
        held_batch = {key: batch[key][held] for key in fields}
        model = fit_full(config, state, train_batch, scale, device, lr, epochs, anchor, seed + fold)
        prediction = P.predict_partial(model, held_batch, device)
        target = (held_batch['Y'] - held_batch['baseline']) / scale
        mask = held_batch['valid']
        error_sum += float(np.square((target - prediction)[mask]).sum())
        count += int(2 * mask.sum())
    return error_sum / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=gru_artifact('checkpoints', R.DEFAULT_CHECKPOINT))
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
    parser.add_argument('--anchor', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    if args.anchor < 0:
        parser.error('--anchor must be nonnegative')
    if not args.checkpoint.is_file():
        raise SystemExit(f'Missing checkpoint: {args.checkpoint}')
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS unavailable')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('temporal_mode') != 'trial_continuous':
        raise SystemExit('Expected trial-continuous GRU checkpoint')
    if checkpoint.get('target_columns') != list(D.TARGETS['velocity']):
        raise SystemExit('Checkpoint velocity targets mismatch')
    config = checkpoint['model_config']
    state = checkpoint['state_dict']
    n_lags = int(checkpoint['n_lags'])
    run_config = checkpoint['run_config']
    feature_set = G.configure(run_config.get('mode', 'augmented'), 'velocity', n_lags,
                              run_config.get('feature_group', 'all'))
    if list(C.FEATURE_SETS[feature_set]) != list(checkpoint['feature_names']):
        raise SystemExit('Checkpoint feature names differ from local feature pipeline')
    scale = np.asarray(checkpoint['target_scale'], dtype=np.float64)
    profile = {key: np.asarray(value, dtype=np.float32)
               for key, value in checkpoint['profile'].items()}
    sessions = C.sessions('val', 'no_stim')
    if set(sessions) != set(checkpoint['val_sessions']):
        raise SystemExit('Validation-day sessions differ from checkpoint')
    frozen = full_model(config, state, device)
    print(f'Checkpoint: {args.checkpoint.name}; device={device}')
    print('Full fine-tuning: projection + recurrent GRU + velocity readout; auxiliary heads frozen')
    print('Pool-only two-fold 10/10 trial CV; 30 query trials scored only after tuning')
    pools = {}
    queries = {}
    for session in sessions:
        pools[session] = R.package(session, 'pool', feature_set, n_lags, profile, frozen, device, 8)
        queries[session] = R.package(session, 'query', feature_set, n_lags, profile, frozen, device, 8)
        print(f'  {session}: pool={int(pools[session]["batch"]["valid"].sum())}, '
              f'query={int(queries[session]["batch"]["valid"].sum())} bins')
    ridge_cv = [(lam, float(np.mean([R.crossfit_mse(pools[s], lam, scale) for s in sessions])))
                for lam in R.LAMBDAS]
    best_lam = min(ridge_cv, key=lambda item: item[1])[0]
    print(f'Readout ridge lambda (pool CV): {best_lam:g}')
    trials = []
    for lr, epochs in itertools.product(LRS, EPOCHS):
        losses = [pool_cv(config, state, pools[session], scale, device,
                          lr, epochs, args.anchor, args.seed) for session in sessions]
        row = dict(lr=lr, epochs=epochs, pool_cv_mse=float(np.mean(losses)),
                   **{f'cv_{session}': loss for session, loss in zip(sessions, losses)})
        trials.append(row)
        print(f'  CV full lr={lr:g}, epochs={epochs}: MSE {row["pool_cv_mse"]:.5f}', flush=True)
    cv = pd.DataFrame(trials).sort_values('pool_cv_mse')
    best = cv.iloc[0]
    print(f'Selected full fine-tune: lr={best.lr:g}, epochs={int(best.epochs)}')
    rows = []
    for session in sessions:
        pool, query = pools[session], queries[session]
        rows.append(P.score_method(query, query['original'], scale, 'frozen', session))
        coefficients = R.fit_correction(pool['states'], pool['original'], pool['targets'],
                                         pool['batch']['valid'], best_lam, scale)
        readout = R.predict_adapt(query['states'], query['original'], coefficients)
        rows.append(P.score_method(query, readout, scale, 'readout_recal', session))
        partial = P.fit_partial(config, state, pool['batch'], scale, device,
                                3e-4, 20, args.anchor, args.seed)
        rows.append(P.score_method(query, P.predict_partial(partial, query['batch'], device),
                                    scale, 'partial_finetune', session))
        full = fit_full(config, state, pool['batch'], scale, device,
                         float(best.lr), int(best.epochs), args.anchor, args.seed)
        rows.append(P.score_method(query, P.predict_partial(full, query['batch'], device),
                                    scale, 'full_finetune', session))
    scores = pd.DataFrame(rows)
    cols = ['r2', 'r2_vx', 'r2_vy', 'r2_time', 'r2_gain', 'gain_vx', 'gain_vy']
    print('\nSession-mean query R² (each day weighted equally):')
    print(scores.groupby('method', sort=False)[cols].mean().to_string(float_format=lambda x: f'{x:.4f}'))
    print('\nPer-session query R²:')
    print(scores[['session', 'method', 'n_bins', 'r2', 'r2_vx', 'r2_vy', 'r2_gain']].to_string(
        index=False, float_format=lambda x: f'{x:.4f}'))
    out = (GRU_RESULTS / 'calibration')
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / 'task3_gru_full_finetune_v1_sessions.csv', index=False)
    cv.to_csv(out / 'task3_gru_full_finetune_v1_pool_cv.csv', index=False)
    print(f'\nSaved CSVs under {out}')


def self_test():
    rng = np.random.default_rng(11)
    config = dict(input_size=4, output_size=2, projection=5, hidden=6,
                  dropout=0.0, warmup=2, channel_dropout=0.0,
                  features_per_channel=2)
    model = M.MultitaskTrialGRU(**config)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    x = rng.normal(size=(20, 12, 4)).astype(np.float32)
    original = full_model(config, state, 'cpu')
    with torch.no_grad():
        desired = original(torch.from_numpy(x))[0].numpy().copy()
    targets = (desired + 0.5) * 10.0
    batch = dict(X=x, Y=targets.astype(np.float32),
                 baseline=np.zeros_like(targets, dtype=np.float32),
                 valid=np.ones((20, 10), bool), trial_ids=np.arange(20))
    before = np.square(desired - targets / 10).mean()
    tuned = fit_full(config, state, batch, np.array([10., 10.]), 'cpu',
                     lr=0.01, epochs=15, anchor=0.1)
    after = P.predict_partial(tuned, batch, 'cpu')
    assert np.square(after - targets / 10).mean() < before
    assert all(torch.equal(tuned.state_dict()[key], value)
               for key, value in state.items() if key.startswith(('speed_head.', 'direction_head.')))
    assert any(not torch.equal(tuned.state_dict()[key], value)
               for key, value in state.items() if key.startswith('gru.'))
    assert np.isfinite(pool_cv(config, state, {'batch': batch}, np.array([10., 10.]),
                               'cpu', 0.01, 3, 0.1, 1))
    print('Self-test passed: recurrent parameters updated, auxiliary heads frozen, pool CV works')


if __name__ == '__main__':
    if '--self-test' in __import__('sys').argv:
        self_test()
    else:
        main()
