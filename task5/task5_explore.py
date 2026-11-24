"""Exploratory follow-up, written AFTER the test days were scored.

It asks where the k = 20 neural gain comes from, by stimulation frequency and by session.
Nothing here changes a pre-specified verdict. Stimulation frequency is also confounded with
recording day, because each day used one frequency, so these are leads for future work.

Run from the repository root:  python task5/task5_explore.py
"""

import os

import pandas as pd

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task5_analyze as analysis

K = 20


def session_gains(split):
    """In-context R², labels-only R² and their difference, per stimulation session at k = 20."""
    scores = pd.read_csv(os.path.join(cfg.RESULTS_DIR, f'scores_{split}.csv'), keep_default_na=False)
    model = analysis.session_scores(scores).rename('r2_in_context')
    labels = analysis.labels_only_scores(scores).rename('r2_labels_only')
    table = pd.concat([model, labels], axis=1).reset_index()
    table = table[(table.group != 'no_stim') & (table.k == K)].copy()
    table['neural_gain'] = table.r2_in_context - table.r2_labels_only
    table['day'] = [analysis.day_of(session) for session in table.session]
    table.insert(0, 'split', split)
    return table


def by_frequency(table):
    """Mean gain and 95% session-bootstrap interval, per split, input and stimulation frequency."""
    rows = []
    for (split, input_name, group), part in table.groupby(['split', 'input', 'group']):
        low, high = analysis.bootstrap_interval(part.neural_gain.to_numpy())
        rows.append(dict(split=split, input=input_name, stimulation=group, n_sessions=len(part),
                         n_days=part.day.nunique(), r2_in_context=part.r2_in_context.mean(),
                         r2_labels_only=part.r2_labels_only.mean(), neural_gain=part.neural_gain.mean(),
                         ci_low=low, ci_high=high, sessions_positive=int((part.neural_gain > 0).sum())))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    sessions = pd.concat([session_gains('val'), session_gains('test')])
    summary = by_frequency(sessions)
    sessions.to_csv(os.path.join(cfg.RESULTS_DIR, 'exploratory_session_gains.csv'), index=False)
    summary.to_csv(os.path.join(cfg.RESULTS_DIR, 'exploratory_by_frequency.csv'), index=False)
    print(summary.round(4).to_string())
