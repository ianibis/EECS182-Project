"""Evaluate saved MLPs with the Task 4 GRU's OFF calibration protocol."""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'task3'))

from task3_gru_evaluate import ROOT, check_inputs, gru, parrm_high_path
from task3_paths import GRU_RESULTS, mlp_checkpoint
from task4_mlp_train import OUTPUT, WindowMLP

RUNS = ('off', 'on_1x', 'on_10x')
METRICS = ['r2_vx', 'r2_vy', 'r2', 'r2_time', 'neural_gain',
           'r2_time_adapted', 'gain_over_adapted_time']


def calibrate(frozen, pool, scale, lr, epochs, seed):
    model = copy.deepcopy(frozen).eval()
    for head in (model.speed, model.direction):
        head.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    original = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    device = next(model.parameters()).device
    x = torch.as_tensor(pool['X'], device=device)
    target = torch.as_tensor(np.nan_to_num((pool['Y'] - pool['baseline']) / scale,
                                          nan=0., posinf=0., neginf=0.),
                             dtype=torch.float32, device=device)
    valid = torch.as_tensor(pool['valid'], device=device)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for indices in np.array_split(order, int(np.ceil(len(x) / 8))):
            optimizer.zero_grad(set_to_none=True)
            predicted = model(x[indices])[0]
            mse = (predicted - target[indices]).square().mean(dim=2)[valid[indices]].mean()
            anchor = sum((p - before).square().mean() for p, before in zip(parameters, original))
            (mse + anchor).backward()
            nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
    return model


def ridge_readout(frozen, pool, scale, penalty):
    """Fit an anchored readout on OFF residuals with hidden features frozen."""
    model = copy.deepcopy(frozen)
    device = next(model.parameters()).device
    with torch.no_grad():
        hidden = model.hidden_features(torch.as_tensor(pool['X'], device=device))
        hidden = hidden.cpu().numpy()[pool['valid']].astype(np.float64)
        target = ((pool['Y'] - pool['baseline']) / scale)[pool['valid']]
        weight = model.velocity.weight.detach().cpu().numpy().astype(np.float64).T
        bias = model.velocity.bias.detach().cpu().numpy().astype(np.float64)
        residual = target - (hidden @ weight + bias)
        # Center the system so the intercept is fit without a penalty.
        center = hidden.mean(axis=0)
        centered = hidden - center
        n, width = centered.shape
        update = np.linalg.solve(centered.T @ centered / n + penalty * np.eye(width),
                                 centered.T @ (residual - residual.mean(axis=0)) / n)
        model.velocity.weight.copy_(torch.as_tensor((weight + update).T, device=device))
        model.velocity.bias.copy_(torch.as_tensor(bias + residual.mean(axis=0) - center @ update,
                                                  device=device))
    return model


def evaluation_features(session, source, paths, splits, manifest):
    high = parrm_high_path(ROOT, session, source)
    return gru.session_features(ROOT, session, source, paths, splits, manifest,
                                high=high, build_missing=False)


def evaluate(split, device, readout_penalty=None, *, runs=RUNS, checkpoint=None,
             calibration=None, compare_gru=True, output_root=OUTPUT):
    manifest, splits = gru.read_project(ROOT)
    paths = gru.paths(ROOT)
    on_sessions = [s for s in splits['sessions'][split]
                   if manifest.loc[s, 'condition'] in ('am3hz', 'am20hz')]
    off_sessions = list(dict.fromkeys(splits['off_reference'][s] for s in on_sessions))
    check_inputs(manifest, splits, split, on_sessions, off_sessions)
    lr, epochs, choices = calibration or gru.calibration_settings(ROOT)
    models = {}
    for run in runs:
        path = checkpoint if checkpoint is not None else mlp_checkpoint(run)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if (saved['run'] != run or saved['target_columns'] != ['Index_vx_cm', 'Index_vy_cm']
                or saved['feature_names'] != [*gru.features.FEATURE_NAMES, 'pow_200_400']
                or saved['model_config']['lags'] != 19):
            raise ValueError(f'{run}: incompatible training checkpoint')
        if set(saved['train_sessions']) & set(splits['sessions'][split]):
            raise ValueError(f'{run}: evaluation overlaps training')
        if split == 'test' and set(saved['val_sessions']) & set(splits['sessions'][split]):
            raise ValueError(f'{run}: test overlaps checkpoint selection')
        frozen = WindowMLP(**saved['model_config']).to(device).eval()
        frozen.load_state_dict(saved['state_dict'])
        models[run] = (frozen, saved['profile'], np.asarray(saved['target_scale'], dtype=np.float64),
                       saved['seed'])
    common_profile = models['off'][1]
    rows = []
    for off in off_sessions:
        # Read each feature source once and use its same query mask for all models.
        feat = evaluation_features(off, 'recorded', paths, splits, manifest)
        pool = gru.trial_batch(feat, off, 'pool', paths, splits, manifest, common_profile, 19)
        if len(pool['trial_ids']) != 20:
            raise ValueError(f'{off}: expected 20 eligible OFF calibration trials')
        adapted = {}
        for run, (frozen, profile, scale, seed) in models.items():
            baseline = profile[str(manifest.loc[off, 'sequence'])][pool['position']]
            fit_pool = {**pool, 'baseline': baseline}
            model = calibrate(frozen, fit_pool, scale, lr, epochs, seed)
            methods = {'full': model}
            if readout_penalty is not None:
                methods['frozen'] = frozen
                methods['ridge'] = ridge_readout(frozen, fit_pool, scale, readout_penalty)
            adapted[run] = (fit_pool, {key: (candidate, gru.predict(candidate, fit_pool, device))
                                       for key, candidate in methods.items()})
        print(f'{split}: calibrated {len(models)} MLP checkpoint(s) on {off}', flush=True)
        comparisons = [(off, 'off', 'recorded')]
        comparisons += [(s, dataset, source) for s in on_sessions
                        if splits['off_reference'][s] == off
                        for dataset, source in [('on_1x', 'recorded'), ('on_10x', 'amp10'),
                                                ('on_parrm', 'parrm')]]
        for session, dataset, source in comparisons:
            if session != off:
                feat = evaluation_features(session, source, paths, splits, manifest)
            query = gru.trial_batch(feat, session, 'query', paths, splits, manifest, common_profile, 19)
            if len(query['trial_ids']) != 30:
                raise ValueError(f'{session}: expected 30 eligible query trials')
            if dataset == 'on_1x':
                original = query
            elif dataset != 'off' and any(not np.array_equal(original[k], query[k])
                                          for k in ('trial_ids', 'bin_ids', 'valid')):
                raise ValueError(f'{session}: query alignment differs across variants')
            for run, (frozen, profile, scale, _) in models.items():
                fit_pool, methods = adapted[run]
                batch = {**query, 'baseline':
                         profile[str(manifest.loc[session, 'sequence'])][query['position']]}
                zeros = np.zeros_like(batch['Y'], dtype=np.float64)
                time = gru.correct_time(fit_pool, np.zeros_like(fit_pool['Y']), batch,
                                        zeros, scale, choices['time'])
                time_r2 = gru.model_scores(session, time, batch, scale)['r2']
                frozen_r2 = gru.model_scores(session, gru.predict(frozen, batch, device), batch, scale)['r2']
                for method, (model, pool_pred) in methods.items():
                    pred = gru.predict(model, batch, device)
                    full = gru.correct_time(fit_pool, pool_pred, batch, pred, scale, choices['full'])
                    uncorrected = gru.model_scores(session, pred, batch, scale)
                    corrected = gru.model_scores(session, full, batch, scale)
                    variants = [('on', corrected)]
                    if readout_penalty is not None:
                        variants.append(('off', uncorrected))
                    for correction, score in variants:
                        score.update(run=run, split=split, dataset=dataset,
                                     condition=manifest.loc[session, 'condition'], off_reference=off,
                                     r2_time_adapted=time_r2, gain_over_adapted_time=score['r2'] - time_r2,
                                     r2_frozen=frozen_r2, r2_before_correction=uncorrected['r2'])
                        if readout_penalty is not None:
                            score.update(method=method, time_correction=correction,
                                         ridge_penalty=readout_penalty if method == 'ridge' else np.nan)
                        rows.append(score)
            print(f'{split}: scored {session} {dataset}', flush=True)
    data = pd.DataFrame(rows)
    split_name = 'validation' if split == 'val' else 'test'
    method_keys = ['method', 'time_correction'] if readout_penalty is not None else []
    metrics = METRICS + ['r2_frozen', 'r2_before_correction']
    display_metrics = ['r2']
    if compare_gru:
        data = compare_with_gru(data, split_name, method_keys)
        metrics += [f'{m}_gru' for m in METRICS] + [f'delta_{m}' for m in METRICS]
        display_metrics += ['r2_gru', 'delta_r2']
    if not np.isfinite(data[metrics].to_numpy()).all():
        raise ValueError('Nonfinite score')
    grouped = data.groupby(['run', *method_keys, 'dataset', 'condition'], sort=False)
    summary = grouped[metrics].mean().join(
        grouped.agg(n_sessions=('session', 'size'), n_bins=('n_bins', 'sum'))
    ).reset_index()
    output = (output_root / 'calibration_ablation' if method_keys else output_root) / split_name
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / 'sessions.csv', index=False)
    summary.to_csv(output / 'aggregate.csv', index=False)
    print(summary[['run', *method_keys, 'dataset', 'condition', *display_metrics]].to_string(index=False))


def compare_with_gru(data, split_name, method_keys):
    path = GRU_RESULTS / split_name / 'sessions.csv'
    if not path.is_file():
        path = ROOT / 'Task4' / 'results' / split_name / 'sessions.csv'
    reference = pd.read_csv(path)
    keys = ['session', 'dataset', 'condition', 'off_reference']
    for variant, subset in data.groupby(['run', *method_keys]):
        check = subset[keys + ['n_bins']].merge(
            reference[keys + ['n_bins']], on=keys, how='outer',
            suffixes=('', '_gru'), indicator=True, validate='one_to_one')
        if not check['_merge'].eq('both').all() or not check.n_bins.eq(check.n_bins_gru).all():
            raise ValueError(f'{variant}: session, dataset or scored-bin counts differ from GRU')
    reference = reference[keys + METRICS].rename(columns={m: f'{m}_gru' for m in METRICS})
    comparison = data.merge(reference, on=keys, validate='many_to_one')
    for field in METRICS:
        comparison[f'delta_{field}'] = comparison[field] - comparison[f'{field}_gru']
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--ablation', action='store_true',
                        help='Score frozen, ridge readout and full calibration, with and without correction')
    parser.add_argument('--ridge-penalty', type=float, default=0.1)
    args = parser.parse_args()
    if not np.isfinite(args.ridge_penalty) or args.ridge_penalty <= 0:
        parser.error('--ridge-penalty must be finite and positive')
    torch.set_num_threads(4)
    evaluate(args.split, args.device, args.ridge_penalty if args.ablation else None)


if __name__ == '__main__':
    main()
