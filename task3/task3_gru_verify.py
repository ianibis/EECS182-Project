"""Compare Task 4 scores with the saved Task 3 session benchmarks."""
import argparse
from pathlib import Path

import pandas as pd

from task3_paths import GRU_RESULTS, gru_artifact
SCORE_TOLERANCE = 0.0001
SCORE_COLUMNS = ['r2_vx', 'r2_vy', 'r2', 'r2_time', 'r2_time_adapted']


def verify(split):
    output = GRU_RESULTS / split
    actual = pd.read_csv(output / 'sessions.csv')
    suffix = 'val' if split == 'validation' else 'test'
    off_file = ('task3_gru_ensemble_v1_sessions.csv' if suffix == 'val'
                else 'task3_gru_ensemble_test_v1_sessions.csv')
    off = pd.read_csv(gru_artifact('ensemble', off_file)).query("method == 'ensemble_full_time'")
    on = pd.read_csv(gru_artifact('reference', f'task3_gru_parrm_scale_sweep_v1_{suffix}_sessions.csv'))
    on = on[on.artifact_scale.isin([1, 10])].copy()
    on['dataset'] = on.artifact_scale.map({1: 'on_1x', 10: 'on_10x'})
    archived = pd.concat([off.assign(dataset='off'), on], ignore_index=True)
    selected = ['session', 'dataset', 'n_bins', *SCORE_COLUMNS]
    comparison = actual[actual.dataset != 'on_parrm'][selected].merge(
        archived[selected], on=['session', 'dataset'], how='outer',
        suffixes=('', '_task3'), validate='one_to_one', indicator=True)
    if (comparison['_merge'] != 'both').any():
        raise ValueError('Task 3 and Task 4 benchmark sessions differ')
    comparison = comparison.drop(columns='_merge')
    for column in ['n_bins', *SCORE_COLUMNS]:
        comparison[f'delta_{column}'] = comparison[column] - comparison[f'{column}_task3']
    comparison.to_csv(output / 'benchmark.csv', index=False)
    print(comparison.groupby('dataset')[['r2', 'r2_task3', 'delta_r2']].mean())
    largest_change = comparison.delta_r2.abs().groupby(comparison.dataset).max()
    print('Largest absolute session delta in R²:', largest_change.to_dict())
    if (comparison.delta_n_bins != 0).any():
        raise ValueError('Scored bin counts differ from Task 3')
    score_changes = comparison[[f'delta_{score}' for score in SCORE_COLUMNS]]
    if score_changes.abs().max().max() > SCORE_TOLERANCE:
        raise ValueError('Task 3 score discrepancy exceeds 0.0001; inspect benchmark.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('validation', 'test'), required=True)
    verify(parser.parse_args().split)
