"""Evaluate Task 3's saved GRU ensemble with same-day OFF calibration."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from task3_paths import DATA_ROOT, GRU_RESULTS, REPO, gru_artifact, feature_root, stored_path

ROOT = DATA_ROOT
import task3_gru_core as gru

CHECKPOINT = gru_artifact('checkpoints',
    'task3_gru_multitask_v1_augmented_all_lags19_h64_seed1_seljoint_cd0.1_sw0.2_dw0.2.pt'
)
SCORES = ['r2_vx', 'r2_vy', 'r2', 'r2_time', 'neural_gain',
          'r2_time_adapted', 'gain_over_adapted_time']


ON_SOURCES = (('on_1x', 'recorded'), ('on_10x', 'amp10'), ('on_parrm', 'parrm'))


def parrm_high_path(root, session, source):
    path = gru.high_band_path(root, session, source)
    if source != 'parrm' or path.is_file():
        return path
    archived = root / 'Task4' / 'results' / 'feature_cache' / path.name
    if archived.is_file():
        return archived
    return GRU_RESULTS / 'feature_cache' / 'hf200_400_parrm' / path.name


def query_sources(off, on_sessions, off_reference):
    yield off, 'off', 'recorded'
    for session in on_sessions:
        if off_reference[session] == off:
            for dataset, source in ON_SOURCES:
                yield session, dataset, source


def check_inputs(manifest, splits, split, on_sessions, off_sessions):
    """Check the inputs before allowing Task 3 to load or cache features."""
    metadata = pd.read_csv(ROOT / 'semisynthetic/parrm_amp10/metadata.csv').set_index('session')
    for session in on_sessions + off_sessions:
        if session not in splits['sessions'][split]:
            raise ValueError(f'{session} is outside {split}')
        partition = splits['calibration'][session]
        pool, query = set(partition['pool']), set(partition['query'])
        if len(pool) != 20 or len(query) != 30 or pool & query:
            raise ValueError(f'{session}: expected disjoint 20/30 trial sets')
        sources = ('recorded', 'amp10', 'parrm') if session in on_sessions else ('recorded',)
        for source in sources:
            base = feature_root(ROOT) / 'features' / gru.feature_variant(source) / f'{session}.h5'
            high = parrm_high_path(ROOT, session, source)
            if not base.is_file():
                raise FileNotFoundError(base)
            if not high.is_file():
                if source != 'parrm':
                    raise FileNotFoundError(high)
                recording = Path(gru.paths(ROOT).parrm_dir) / f'{session}.h5'
                if not recording.is_file():
                    raise FileNotFoundError(recording)
        if session in on_sessions:
            off = splits['off_reference'][session]
            row = metadata.loc[session]
            if (manifest.loc[off, 'condition'] != 'no_stim'
                    or manifest.loc[off, 'date'] != manifest.loc[session, 'date']
                    or row.off_reference != off or row.artifact_scale != 10
                    or row.split != split or row.condition != manifest.loc[session, 'condition']):
                raise ValueError(f'{session}: mismatched same-day OFF or 10x metadata')
            for column in ('base_features', 'hf_features'):
                path = stored_path(ROOT, row[column])
                if not path.is_file():
                    raise FileNotFoundError(path)
            recording = ROOT / 'semisynthetic/parrm_amp10/h5_export_outputs' / f'{session}.h5'
            if not recording.is_file():
                raise FileNotFoundError(recording)


def evaluate(split, device):
    manifest, splits = gru.read_project(ROOT)
    paths = gru.paths(ROOT)
    on_sessions = [s for s in splits['sessions'][split]
                   if manifest.loc[s, 'condition'] in ('am3hz', 'am20hz')]
    off_sessions = list(dict.fromkeys(splits['off_reference'][s] for s in on_sessions))
    if set(off_sessions) != {s for s in splits['sessions'][split]
                             if manifest.loc[s, 'condition'] == 'no_stim'}:
        raise ValueError(f'{split}: matched OFF sessions differ from the saved split')
    check_inputs(manifest, splits, split, on_sessions, off_sessions)
    saved, states = gru.load_models(ROOT, CHECKPOINT, device)
    if set(splits['sessions']['test']) & (set(saved['train_sessions']) | set(saved['val_sessions'])):
        raise ValueError('Test sessions overlap checkpoint training or validation sessions')
    if set(saved['val_sessions']) != {s for s in splits['sessions']['val']
                                      if manifest.loc[s, 'condition'] == 'no_stim'}:
        raise ValueError('Checkpoint validation sessions differ from the saved split')
    if list(saved['feature_names']) != [*gru.features.FEATURE_NAMES, 'pow_200_400']:
        raise ValueError('Checkpoint feature definitions differ from Task 3')
    lr, epochs, choices = gru.calibration_settings(ROOT)
    scale = np.asarray(saved['target_scale'], dtype=np.float64)
    profile = {name: np.asarray(data, dtype=np.float32)
               for name, data in saved['profile'].items()}
    config = saved['model_config']
    history = int(saved['n_lags'])
    print(f'{split} on {device}: {len(off_sessions)} OFF, {len(on_sessions)} ON; '
          f'fine-tune lr={lr}, epochs={epochs}, correction={choices}', flush=True)

    rows = []
    for off in off_sessions:
        feat = gru.session_features(ROOT, off, 'recorded', paths, splits, manifest)
        pool = gru.trial_batch(feat, off, 'pool', paths, splits, manifest, profile, history)
        if len(pool['trial_ids']) != 20:
            raise ValueError(f'{off}: expected 20 eligible OFF calibration trials')
        models = [gru.fit_session(config, states[seed], pool, scale, device, lr, epochs, seed)
                  for seed in gru.SEEDS]
        pool_pred = gru.average(models, pool, device)
        for session, variant, source in query_sources(off, on_sessions, splits['off_reference']):
            if session != off:
                feature_path = parrm_high_path(ROOT, session, source)
                feat = gru.session_features(ROOT, session, source, paths, splits, manifest,
                                            high=feature_path)
            query = gru.trial_batch(feat, session, 'query', paths, splits, manifest, profile, history)
            if len(query['trial_ids']) != 30:
                raise ValueError(f'{session}: expected 30 eligible query trials')
            comparable = [query['trial_ids'], query['bin_ids'], query['valid']]
            if variant == 'on_1x':
                original = comparable
            elif variant != 'off' and any(not np.array_equal(a, b)
                                          for a, b in zip(original, comparable)):
                raise ValueError(f'{session}: query trial IDs, bins or masks differ across variants')
            pred = gru.average(models, query, device)
            pred = gru.correct_time(pool, pool_pred, query, pred, scale, choices['full'])
            time = gru.correct_time(pool, np.zeros_like(pool_pred), query, np.zeros_like(pred),
                                    scale, choices['time'])
            score = gru.model_scores(session, pred, query, scale)
            time_r2 = gru.model_scores(session, time, query, scale)['r2']
            score.update(split=split, dataset=variant, condition=manifest.loc[session, 'condition'],
                         date=manifest.loc[session, 'date'], off_reference=off,
                         r2_time_adapted=time_r2, gain_over_adapted_time=score['r2'] - time_r2)
            if variant == 'off':
                off_r2 = score['r2']
            score.update(r2_off=off_r2, gap_on_minus_off=score['r2'] - off_r2)
            rows.append(score)
            print(f'{session} {variant}: R²={score["r2"]:.6f}', flush=True)

    data = pd.DataFrame(rows)
    grouped = data.groupby(['split', 'dataset', 'condition'])
    aggregate = grouped[SCORES + ['r2_off', 'gap_on_minus_off']].mean().join(
        grouped.agg(n_sessions=('session', 'size'), n_bins=('n_bins', 'sum'))
    ).reset_index()
    output = GRU_RESULTS / ('validation' if split == 'val' else 'test')
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / 'sessions.csv', index=False)
    aggregate.to_csv(output / 'aggregate.csv', index=False)
    print(aggregate.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    evaluate(args.split, args.device)


if __name__ == '__main__':
    main()
