"""Compare one GRU with five averaged GRUs on validation-day OFF trials."""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from task3_paths import GRU_RESULTS, gru_artifact

import task3_gru_temporal_adapt_v1 as temporal

fine_tune = temporal.F
prediction_tools = temporal.P
readout = temporal.R
multitask = temporal.M
trial_gru = temporal.T
feature_setup = temporal.G
project = temporal.C
targets = temporal.D
DEFAULT_SEEDS = (1, 2, 3, 4, 5)


def load_cv_choices(path):
    cv = pd.read_csv(path)
    choices = {}
    for method in ('time', 'full'):
        values = []
        for axis in ('vx', 'vy'):
            rows = cv[(cv.method == method) & (cv.axis == axis)]
            if rows.empty:
                raise ValueError(f'Missing {method}/{axis} from {path}')
            best = rows.loc[rows.cv_mse.idxmin()]
            values.append((int(best.k), float(best.lam)))
        choices[method] = values
    return choices


def pretrained_fixed_epoch(train_batches, checkpoint, seed, epochs, device):
    """Match the original multitask training, but fix epoch count for all seeds."""
    config = checkpoint['model_config']
    scale = np.asarray(checkpoint['target_scale'], dtype=np.float32)
    all_x = np.concatenate([b['X'] for b in train_batches]).astype(np.float32)
    all_valid = np.concatenate([b['valid'] for b in train_batches])
    all_y = np.concatenate([b['Y'] for b in train_batches]).astype(np.float32)
    all_base = np.concatenate([b['baseline'] for b in train_batches]).astype(np.float32)
    residual = (all_y - all_base) / scale[None, None, :]
    target = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    speed, direction, direction_valid = multitask.auxiliary_targets(
        all_y, all_valid, float(checkpoint['speed_mean']),
        float(checkpoint['speed_scale']), float(checkpoint['direction_threshold']))
    data = TensorDataset(torch.from_numpy(all_x), torch.from_numpy(target),
                         torch.from_numpy(all_valid.astype(np.float32)),
                         torch.from_numpy(speed), torch.from_numpy(direction),
                         torch.from_numpy(direction_valid))
    hparams = checkpoint['run_config']
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    model = multitask.MultitaskTrialGRU(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(hparams['lr']),
                                   weight_decay=float(hparams['weight_decay']))
    loader = DataLoader(data, batch_size=int(hparams['batch_size']), shuffle=True, num_workers=0)
    objective = argparse.Namespace(vx_loss_weight=float(hparams['vx_loss_weight']),
                                   aux_speed_weight=float(hparams['aux_speed_weight']),
                                   aux_dir_weight=float(hparams['aux_dir_weight']))
    for epoch in range(1, epochs + 1):
        model.train()
        weighted_loss, count = 0.0, 0.0
        for x, y, valid, spd, direction_target, direction_mask in loader:
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            spd = spd.to(device)
            direction_target = direction_target.to(device)
            direction_mask = direction_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = multitask.multitask_loss(outputs, y, valid, spd, direction_target,
                                    direction_mask, objective)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(hparams['grad_clip']))
            optimizer.step()
            n = float(valid.sum().item())
            weighted_loss += float(loss.item()) * n
            count += n
        print(f'    Seed {seed}, pretrained epoch {epoch}/{epochs}: loss={weighted_loss/count:.5f}',
              flush=True)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def states_for_seeds(checkpoint, train_batches, seeds, epochs, device, original_path):
    states = {1: checkpoint['state_dict']}
    new_checkpoint_dir = GRU_RESULTS / 'ensemble' / 'checkpoints'
    for seed in seeds:
        if seed == 1:
            continue
        filename = f'task3_gru_ensemble_v1_seed{seed}_epoch{epochs}.pt'
        path = new_checkpoint_dir / filename
        if not original_path.resolve().is_relative_to(GRU_RESULTS):
            path = gru_artifact('ensemble/checkpoints', filename)
        if path.is_file():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if (saved.get('seed') != seed or saved.get('epochs') != epochs or
                saved.get('source_checkpoint') != original_path.name):
                raise ValueError(f'Cached checkpoint metadata mismatch: {path}')
            states[seed] = saved['state_dict']
            print(f'  Reusing pretrained seed {seed}: {path.name}', flush=True)
            continue
        print(f'  Training independent seed {seed}', flush=True)
        state = pretrained_fixed_epoch(train_batches, checkpoint, seed, epochs, device)
        saved = dict(seed=seed, epochs=epochs, state_dict=state,
                     source_checkpoint=original_path.name)
        path = new_checkpoint_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix('.tmp')
        torch.save(saved, temp_path)
        os.replace(temp_path, path)
        states[seed] = state
    return states


def average_predictions(models, batch, device):
    predictions = [prediction_tools.predict_partial(model, batch, device) for model in models]
    return np.mean(np.stack(predictions, axis=0), axis=0)


def build_batches(sessions, feature_set, n_lags, profile):
    pools = {}
    queries = {}
    for session in sessions:
        pools[session] = trial_gru.load_level(session, 'pool', feature_set, n_lags, profile)
        queries[session] = trial_gru.load_level(session, 'query', feature_set, n_lags, profile)
        if len(pools[session]['trial_ids']) != 20 or len(queries[session]['trial_ids']) != 30:
            raise ValueError(f'{session}: expected 20 pool and 30 query trials')
        print(f'  {session}: pool {int(pools[session]["valid"].sum())}, '
              f'query {int(queries[session]["valid"].sum())} valid bins')
    return pools, queries


def run(args):
    if len(args.seeds) != 5 or len(set(args.seeds)) != 5 or 1 not in args.seeds:
        raise SystemExit('Use exactly five unique seeds, including seed 1')
    if args.anchor < 0:
        raise SystemExit('--anchor must be nonnegative')
    if not args.checkpoint.is_file():
        raise SystemExit(f'Missing seed-1 checkpoint: {args.checkpoint}')
    cv_full = gru_artifact('calibration', 'task3_gru_full_finetune_v1_pool_cv.csv')
    cv_time = gru_artifact('calibration', 'task3_gru_temporal_adapt_v1_pool_cv.csv')
    if not cv_full.is_file() or not cv_time.is_file():
        raise SystemExit(f'Missing pool CV CSV(s): {cv_full} / {cv_time}')
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('temporal_mode') != 'trial_continuous':
        raise SystemExit('Expected trial-continuous checkpoint')
    if ck.get('target_columns') != list(targets.TARGETS['velocity']):
        raise SystemExit('Expected vx/vy checkpoint')
    if int(ck['run_config'].get('seed', 1)) != 1:
        raise SystemExit('Source checkpoint must be seed 1')
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS is unavailable')
    n_lags = int(ck['n_lags'])
    feature_set = feature_setup.configure(ck['run_config']['mode'], 'velocity', n_lags,
                              ck['run_config']['feature_group'])
    if list(project.FEATURE_SETS[feature_set]) != list(ck['feature_names']):
        raise SystemExit('Feature names differ from checkpoint')
    profile = {k: np.asarray(v, dtype=np.float32) for k, v in ck['profile'].items()}
    scale = np.asarray(ck['target_scale'], dtype=np.float64)
    config = ck['model_config']
    if not all(key in ck for key in ('speed_mean', 'speed_scale', 'direction_threshold')):
        raise SystemExit('Checkpoint lacks multitask auxiliary-label statistics')
    pretrain_epochs = int(args.pretrain_epochs or ck['best_epoch'])
    full_cv = pd.read_csv(cv_full).sort_values('pool_cv_mse')
    lr, epochs = float(full_cv.iloc[0].lr), int(full_cv.iloc[0].epochs)
    choices = load_cv_choices(cv_time)
    sessions = project.sessions('val', 'no_stim')
    if set(sessions) != set(ck['val_sessions']):
        raise SystemExit('Validation sessions differ from saved checkpoint')
    print(f'Five seeds: {args.seeds}; device={device}')
    print(f'New seed pretraining fixed to epoch {pretrain_epochs}; seed 1 uses original checkpoint')
    print(f'Full-session adaptation: lr={lr:g}, epochs={epochs}, anchor={args.anchor:g}')
    print(f'Previous calibration-only DCT choices: {choices}')
    print('No test days or ON sessions. Query trials excluded from all training and calibration.')
    pools, queries = build_batches(sessions, feature_set, n_lags, profile)
    train_sessions = [s for s in project.sessions('train', 'no_stim') if project.has_cm_velocity(s)]
    train_batches = [trial_gru.load_level(s, 'train', feature_set, n_lags, profile)
                     for s in train_sessions] if any(seed != 1 for seed in args.seeds) else []
    states = states_for_seeds(ck, train_batches, args.seeds, pretrain_epochs, device,
                             args.checkpoint)
    rows = []
    for session in sessions:
        pool, query = pools[session], queries[session]
        adapted_models = []
        for seed in args.seeds:
            fitted = fine_tune.fit_full(config, states[seed], pool, scale, device,
                                lr, epochs, args.anchor, seed)
            adapted_models.append(fitted)
        normalized_pool = average_predictions(adapted_models, pool, device)
        normalized_query = average_predictions(adapted_models, query, device)
        seed1_pool = prediction_tools.predict_partial(adapted_models[args.seeds.index(1)], pool, device)
        seed1_query = prediction_tools.predict_partial(adapted_models[args.seeds.index(1)], query, device)
        zero_pool = np.zeros_like(seed1_pool)
        zero_query = np.zeros_like(seed1_query)
        adapted_time = temporal.apply_correction(pool, zero_pool, query, zero_query,
                                          scale, choices['time'])
        model_preds = {
            'single_full': seed1_query,
            'single_full_time': temporal.apply_correction(pool, seed1_pool, query,
                                                  seed1_query, scale, choices['full']),
            'ensemble_full': normalized_query,
            'ensemble_full_time': temporal.apply_correction(pool, normalized_pool, query,
                                                    normalized_query, scale, choices['full']),
        }
        first_score = len(rows)
        for method, prediction in model_preds.items():
            rows.append(temporal.score(session, method, query, prediction, scale, adapted_time))
        for seed, model in zip(args.seeds, adapted_models):
            pred_pool = prediction_tools.predict_partial(model, pool, device)
            pred_query = prediction_tools.predict_partial(model, query, device)
            corrected = temporal.apply_correction(pool, pred_pool, query, pred_query,
                                           scale, choices['full'])
            rows.append(temporal.score(session, f'seed{seed}_full_time', query,
                                corrected, scale, adapted_time))
        print(f'  Scored {session}: single={rows[first_score]["r2"]:.4f}', flush=True)
    scores = pd.DataFrame(rows)
    metrics = ['r2', 'r2_vx', 'r2_vy', 'r2_time_adapted',
               'gain_over_adapted_time', 'gain_vx', 'gain_vy']
    print('\nSession-mean query R² (each day equally weighted):')
    print(scores.groupby('method', sort=False)[metrics].mean().to_string(float_format=lambda v: f'{v:.4f}'))
    print('\nPer-session query R²:')
    print(scores[['session', 'method', 'n_bins', 'r2', 'r2_vx', 'r2_vy',
                  'gain_over_adapted_time']].to_string(index=False,
                    float_format=lambda v: f'{v:.4f}'))
    output = GRU_RESULTS / 'ensemble'
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / 'task3_gru_ensemble_v1_sessions.csv', index=False)
    print(f'\nSaved {output / "task3_gru_ensemble_v1_sessions.csv"}')


def self_test():
    rng = np.random.default_rng(7)
    config = dict(input_size=6, output_size=2, projection=5, hidden=6,
                  dropout=0.0, warmup=2, channel_dropout=0.1,
                  features_per_channel=3)
    model = multitask.MultitaskTrialGRU(**config)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    n, steps = 20, 12
    x = rng.standard_normal((n, steps, 6)).astype(np.float32)
    y = rng.standard_normal((n, steps - 2, 2)).astype(np.float32)
    batch = dict(X=x, Y=y, baseline=np.zeros_like(y),
                 valid=np.ones((n, steps - 2), bool),
                 trial_ids=np.arange(n), position=np.tile(np.arange(steps - 2), (n, 1)))
    ck = dict(model_config=config, target_scale=[1.0, 1.0],
              speed_mean=1.0, speed_scale=0.5, direction_threshold=0.2,
              run_config=dict(lr=0.001, weight_decay=0.001, batch_size=8,
                              vx_loss_weight=1.0, aux_speed_weight=0.2,
                              aux_dir_weight=0.2, grad_clip=1.0))
    trained = pretrained_fixed_epoch([batch], ck, 2, 2, 'cpu')
    assert any(not torch.equal(trained[k], original[k]) for k in original)
    fitted = fine_tune.fit_full(config, original, batch, np.ones(2), 'cpu', 0.001, 2, 1.0, 1)
    a = prediction_tools.predict_partial(fitted, batch, 'cpu')
    b = average_predictions([fitted, fitted], batch, 'cpu')
    assert np.allclose(a, b)
    choices = {'time': [(1, 1.0), (1, 1.0)], 'full': [(1, 1.0), (1, 1.0)]}
    corrected = temporal.apply_correction(batch, a, batch, b, np.ones(2), choices['full'])
    assert corrected.shape == a.shape and np.isfinite(corrected).all()
    assert np.isfinite(temporal.score('synthetic', 'model', batch, corrected, np.ones(2),
                               np.zeros_like(corrected))['r2'])
    print('Self-test passed: independent pretraining, fine-tuning, averaging, correction, scoring')


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        self_test()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--checkpoint', type=Path,
                            default=gru_artifact('checkpoints', readout.DEFAULT_CHECKPOINT))
        parser.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
        parser.add_argument('--seeds', type=lambda s: tuple(int(k) for k in s.split(',')),
                            default=DEFAULT_SEEDS)
        parser.add_argument('--pretrain-epochs', type=int, default=None)
        parser.add_argument('--anchor', type=float, default=1.0)
        arguments = parser.parse_args()
        run(arguments)
