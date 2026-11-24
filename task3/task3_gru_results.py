"""Task 3 GRU: OFF/ON, PARRM, and 10x artifact comparisons."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from task3_paths import DATA_ROOT, GRU_RESULTS, gru_artifact

import task3_gru_core as gru
from task3_gru_evaluate import parrm_high_path


def evaluate(root, split, dataset, checkpoint, output, device):
    manifest, splits = gru.read_project(root)
    project_paths = gru.paths(root)
    saved, states = gru.load_models(root, checkpoint, device)
    lr, epochs, choices = gru.calibration_settings(root)
    scale = np.asarray(saved['target_scale'], dtype=np.float64)
    profile = {name: np.asarray(data, dtype=np.float32)
               for name, data in saved['profile'].items()}
    config = saved['model_config']
    history = int(saved['n_lags'])
    if list(saved['feature_names']) != [*gru.features.FEATURE_NAMES, 'pow_200_400']:
        raise ValueError('Saved GRU features differ from this pipeline')
    print(f'Using five pretrained GRUs; fine-tuning lr={lr}, epochs={epochs}', flush=True)
    print(f'Temporal settings: {choices}', flush=True)
    print(f'Evaluating {dataset} on {split} queries')

    on_sessions = [name for name in splits['sessions'][split]
                   if manifest.loc[name, 'condition'] in ('am3hz', 'am20hz')]
    if not on_sessions:
        raise ValueError(f'No stimulation sessions in {split}')
    sources = {
        'all': ('recorded', 'amp10', 'parrm'),
        'recorded': ('recorded', 'parrm'),
        'amp10': ('recorded', 'amp10'),
    }[dataset]
    off_models = {}
    off_rows = []
    rows = []
    for session in on_sessions:
        off = splits['off_reference'][session]
        if off not in off_models:
            print(f'  Calibrating on OFF {off}', flush=True)
            feat = gru.session_features(root, off, 'recorded', project_paths, splits, manifest)
            pool = gru.trial_batch(feat, off, 'pool', project_paths, splits,
                                   manifest, profile, history)
            query = gru.trial_batch(feat, off, 'query', project_paths, splits,
                                    manifest, profile, history)
            if len(pool['trial_ids']) != 20 or len(query['trial_ids']) != 30:
                raise ValueError(f'{off}: expected 20 calibration and 30 query trials')
            models = [gru.fit_session(config, states[seed], pool, scale, device,
                                      lr, epochs, seed) for seed in gru.SEEDS]
            pool_pred = gru.average(models, pool, device)
            query_pred = gru.average(models, query, device)
            pred = gru.correct_time(pool, pool_pred, query, query_pred,
                                    scale, choices['full'])
            score = gru.model_scores(off, pred, query, scale)
            score.update(condition='no_stim', method='ensemble_full_time')
            off_rows.append(score)
            off_models[off] = (pool, pool_pred, models, score)
            print(f'    OFF R² = {score["r2"]:.4f}', flush=True)

        pool, off_pred, models, off_score = off_models[off]
        recorded_mask = None
        for source in sources:
            print(f'  {session}: {source}', flush=True)
            high_path = parrm_high_path(root, session, source)
            feat = gru.session_features(root, session, source, project_paths, splits, manifest,
                                        high=high_path)
            query = gru.trial_batch(feat, session, 'query', project_paths, splits,
                                    manifest, profile, history)
            if len(query['trial_ids']) != 30:
                raise ValueError(f'{session}: expected 30 query trials')
            if recorded_mask is None:
                recorded_mask = query['valid'].copy()
            elif not np.array_equal(recorded_mask, query['valid']):
                raise ValueError(f'{session}: scored bins differ between data variants')
            prediction = gru.average(models, query, device)
            corrected = gru.correct_time(pool, off_pred, query, prediction,
                                         scale, choices['full'])
            score = gru.model_scores(session, corrected, query, scale)
            score.update(condition=manifest.loc[session, 'condition'],
                         date=manifest.loc[session, 'date'],
                         off_reference=off, dataset=source,
                         r2_off=off_score['r2'], r2_off_vx=off_score['r2_vx'],
                         r2_off_vy=off_score['r2_vy'],
                         gap_on_minus_off=score['r2'] - off_score['r2'])
            rows.append(score)
            print(f'    R² = {score["r2"]:.4f}', flush=True)

    data = pd.DataFrame(rows)
    off_data = pd.DataFrame(off_rows)
    summary = (data.groupby(['dataset', 'condition'], as_index=False)
               [['r2', 'r2_vx', 'r2_vy', 'r2_time', 'neural_gain',
                 'r2_off', 'gap_on_minus_off']].mean())
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / f'task3_gru_{split}_sessions.csv', index=False)
    off_data.to_csv(output / f'task3_gru_{split}_off.csv', index=False)
    summary.to_csv(output / f'task3_gru_{split}_summary.csv', index=False)
    print('\nMean R² by ON condition and dataset:')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    if {'recorded', 'parrm'} <= set(data['dataset']):
        on = data[data.dataset == 'recorded'].set_index('session')
        clean = data[data.dataset == 'parrm'].set_index('session')
        comparison = on[['condition', 'off_reference', 'r2_off', 'n_bins']].copy()
        comparison['r2_on_raw'] = on['r2']
        comparison['r2_on_parrm'] = clean['r2']
        comparison['parrm_minus_raw'] = clean['r2'] - on['r2']
        comparison['gap_raw'] = on['r2'] - on['r2_off']
        comparison['gap_parrm'] = clean['r2'] - on['r2_off']
        comparison.to_csv(output / f'task3_gru_{split}_parrm.csv')

    print(f'\nResults saved to {output}')


def self_test():
    import torch

    torch.manual_seed(13)
    cfg = dict(input_size=18, output_size=2, projection=8, hidden=10,
               dropout=.15, warmup=3, channel_dropout=.10, features_per_channel=3)
    model = gru.GRUDecoder(**cfg)
    model.eval()
    x = torch.randn(2, 12, 18)
    with torch.no_grad():
        result = model(x)[0]
        states, _ = model.gru(torch.tanh(model.projection(x)))
        expected = model.readout(states[:, 3:, :])
    assert result.shape == (2, 9, 2)
    assert torch.allclose(result, expected)

    rng = np.random.default_rng(14)
    n, bins = 20, 9
    y = rng.normal(size=(n, bins, 2)).astype(np.float32)
    pool = dict(X=rng.normal(size=(n, 12, 18)).astype(np.float32), Y=y,
                valid=np.ones((n, bins), bool), baseline=np.zeros_like(y),
                position=np.tile(np.arange(bins), (n, 1)), trial_ids=np.arange(n))
    tuned = gru.fit_session(cfg, model.state_dict(), pool, np.ones(2), 'cpu', .001, 2, 1)
    preds = gru.average([tuned, tuned], pool, 'cpu')
    assert np.allclose(preds, gru.predict(tuned, pool, 'cpu'))
    corrected = gru.correct_time(pool, preds, pool, preds, np.ones(2),
                                 [(3, .1), (3, .1)])
    assert np.isfinite(corrected).all()
    score = gru.model_scores('toy', corrected, pool, np.ones(2))
    assert np.isfinite(score['r2'])
    print('Synthetic model, calibration, correction and scoring tests passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DATA_ROOT,
                        help='Folder containing exports, features, splits.json, and checkpoints')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--dataset', choices=('recorded', 'amp10', 'all'), default='all')
    parser.add_argument('--output', type=Path, default=GRU_RESULTS / 'recording_modes')
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    root = args.root.expanduser().resolve()
    device = gru.device_name(args.device)
    ck = args.checkpoint
    if ck is None:
        ck = gru_artifact('checkpoints',
                          'task3_gru_multitask_v1_augmented_all_lags19_h64_seed1_seljoint_cd0.1_sw0.2_dw0.2.pt',
                          root)
    evaluate(root, args.split, args.dataset, ck, args.output, device)


if __name__ == '__main__':
    main()
