"""One results table for every baseline decoder, on the shared task-4 protocol.

The paper reports test metrics by averaging per day, so every cell here is a day-level
mean: sessions are averaged inside their recording day, then days are averaged. That is
stricter than pooling sessions, because days hold unequal numbers of sessions.

The retrained GRU (hidden 256, the sweep winner) is scored here from its own CSV, since
the archived GRU in results_gru.parquet is the older 128-unit network.

    python task4/make_baseline_table.py            # print and write the LaTeX table
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compare_decoders as C  # noqa: E402
import utils as U  # noqa: E402

OUT = os.path.join(U.TASK4_RESULTS, 'baseline_table.tex')

# Row order and labels for the paper. `source` says where the per-session rows come from.
ROWS = [
    ('ridge_raw_off', 'Ridge, raw, no adaptation', 'fixed'),
    ('ridge_parrm_off', 'Ridge, PARRM-cleaned, no adaptation', 'fixed'),
    ('ridge_raw_refit', 'Ridge, raw, refit on $k=20$', 'adapted'),
    ('ridge_parrm_refit', 'Ridge, PARRM-cleaned, refit on $k=20$', 'adapted'),
    ('knn_off', 'kNN, stimulation-off bank', 'fixed'),
    ('knn_incontext', 'kNN, in-context $k=20$', 'adapted'),
    ('mlp', 'MLP', 'fixed'),
    ('gru', 'GRU, 128 units', 'fixed'),
    ('gru_h256', 'GRU, 256 units (sweep winner)', 'fixed'),
    ('eegnet', 'EEGNet, single seed', 'fixed'),
    ('eegnet_ensemble', 'EEGNet, 5-seed ensemble', 'fixed'),
    ('eegnet_stimulation_tones_ensemble', 'EEGNet $+$ tones, 5-seed ensemble', 'fixed'),
]


def add_retrained_gru(per_session):
    """Append the retrained GRU's ensemble rows, which live only in its own CSV."""
    path = os.path.join(U.TASK4_RESULTS, 'task4_gru_final.csv')
    if not os.path.exists(path):
        return per_session
    final = pd.read_csv(path)
    ensemble = final[final.seed == -1]
    if ensemble.empty:
        return per_session
    rows = ensemble[['session', 'am_freq', 'r2_ave', 'r2_off_ave']].copy()
    rows['method'] = 'gru_h256'
    rows['seed'] = -1
    return pd.concat([per_session, rows], ignore_index=True)


def day_level(frame, column, frequency=None):
    part = frame if frequency is None else frame[frame.am_freq == frequency]
    if part.empty:
        return np.nan
    part = part.assign(day=part.session.str.split('-').str[1])
    per_session = part.groupby(['day', 'session'])[column].mean()
    return float(per_session.groupby(level=0).mean().mean())


def build():
    per_session = add_retrained_gru(C.load_scores())
    frequencies = sorted(per_session.am_freq.unique())
    rows = []
    for method, label, adaptation in ROWS:
        part = per_session[per_session.method == method]
        if part.empty:
            continue
        cell = {'method': method, 'label': label, 'adaptation': adaptation,
                'clean': day_level(part, 'r2_off_ave'),
                'stim': day_level(part, 'r2_ave')}
        for frequency in frequencies:
            cell[f'stim_{frequency:g}'] = day_level(part, 'r2_ave', frequency)
        rows.append(cell)
    return pd.DataFrame(rows), frequencies


def latex(table, frequencies):
    metrics = ['clean', 'stim'] + [f'stim_{f:g}' for f in frequencies]
    best = {metric: table[metric].idxmax() for metric in metrics}

    def cell(row_index, metric):
        value = table.loc[row_index, metric]
        # Four decimals: at three, the 3 Hz column shows +0.045 for both the MLP and the
        # GRU, and the bold winner then looks arbitrary even though it is the true maximum.
        text = f'{value:+.4f}'
        return f'$\\mathbf{{{text}}}$' if best[metric] == row_index else f'${text}$'

    header = ' & '.join(['Decoder', 'Clean $R^2$', 'Stim.\\ $R^2$']
                        + [f'{f:g}~Hz' for f in frequencies])
    caption = (
        "Every baseline decoder on the shared test protocol. All cells are axis-averaged "
        "$R^2$ on the fixed 30-trial query set, averaged per recording day. ``Clean'' is the "
        "same-day stimulation-off recording; the last three columns are the stimulated "
        "recordings. Bold marks the best decoder in each column. Decoders marked refit or "
        "in-context spend $k=20$ stimulated calibration trials; the rest never see a "
        "stimulated label."
    )
    lines = [
        r'\begin{table}[h]\centering',
        r'\caption{' + caption + '}',
        r'\label{tab:baselines}',
        r'\begin{tabular}{l' + 'c' * len(metrics) + r'}',
        r'\hline \\',
        header + r' \\ \hline \\',
    ]
    for index in table.index:
        lines.append(f"{table.loc[index, 'label']} & "
                     + ' & '.join(cell(index, metric) for metric in metrics)
                     + r' \\')
    lines += [r'\hline', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def main():
    table, frequencies = build()
    text = latex(table, frequencies)
    with open(OUT, 'w') as handle:
        handle.write(text)
    show = table.copy()
    for column in show.columns:
        if show[column].dtype.kind == 'f':
            show[column] = show[column].map(lambda v: f'{v:+.4f}')
    print(show.to_string(index=False))
    print()
    print(text)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
