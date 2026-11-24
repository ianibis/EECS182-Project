"""Validation check: is the calibrated decoder better with stimulation-ON context than with OFF context?

This repeats check C2 from the main analysis for the 5 calibrated seeds. It trains nothing and never reads
the test days. Each query trial is decoded twice, and only the source of the context changes:

    on    context trials from the same stimulation session (signal and labels stay paired)
    off   context trials from the same-day stimulation-OFF session (signal and labels stay paired)

Queries come from the stimulation session in both cases. For the 10x input, the ON context and the query
carry the 10x artifact. The OFF context stays as recorded, because an OFF session has no artifact to scale.

Decision rule, the same as C2: at k = 20, ON minus OFF passes if its 95% interval over sessions is above 0.

Run from the repository root, on one free GPU:
    CUDA_VISIBLE_DEVICES=1 python task5/task5_calibrated_on_off.py
"""

import json
import os

import pandas as pd

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task3_common as C
import task5_analyze as analysis
import task5_data as data
from task5_calibrated import CHOSEN_PATH, INPUTS, load_calibrated, run_name
from task5_evaluate import context_positions, load_held_out, n_draws, predict_queries, session_r2
from task5_run_final import SEEDS

SPLIT = 'val'
K_VALUES = (2, 5, 10, 20)
DRAWS = 3               # the same first draws the other context controls use
MAX_CHECK_DIFFERENCE = 1e-3


def score_seed(seed, alpha, sessions, device):
    """R² of one calibrated model for every session, input, k, draw and context source."""
    model, scale, _ = load_calibrated(run_name(alpha, seed), device)
    rows = []
    for session in sessions:
        off_session = load_held_out(C.splits()['off_reference'][session], 'recorded', scale, device)
        for input_name in INPUTS:
            held = load_held_out(session, input_name, scale, device)
            for k in K_VALUES:
                for draw in range(n_draws(k, DRAWS)):
                    on_positions = context_positions(held, k, draw)
                    off_positions = context_positions(off_session, k, draw)
                    predictions = dict(
                        on=predict_queries(model, held, on_positions),
                        off=predict_queries(model, held, off_positions, 'off_context', off_session))
                    for context, scaled in predictions.items():
                        r2, _, _ = session_r2((scaled * scale).cpu().numpy(), held.query_velocity, held.query_valid)
                        rows.append(dict(seed=seed, session=session, input=input_name, k=k, draw=draw,
                                         context=context, r2=r2))
    return rows


def check_on_matches_saved_scores(table):
    """With ON context, this script must reproduce the saved calibrated validation scores."""
    saved = pd.read_csv(os.path.join(cfg.RESULTS_DIR, 'calibrated_comparison_val.csv'))
    saved = saved.set_index(['input', 'k']).r2_calibrated
    on = session_means(table[table.context == 'on']).groupby(['input', 'k']).mean()
    largest = (on - saved).abs().max()
    print(f'Check: largest difference from saved ON scores is {largest:.6f}', flush=True)
    if largest > MAX_CHECK_DIFFERENCE:
        raise RuntimeError('this script does not reproduce the saved calibrated scores')


def session_means(table):
    """Average over draws within each seed, then over seeds within each session."""
    per_seed = table.groupby(['session', 'input', 'k', 'seed']).r2.mean()
    return per_seed.groupby(['session', 'input', 'k']).mean()


def first_model_difference():
    """ON minus OFF for the first in-context model, on the same draws."""
    saved = pd.read_csv(os.path.join(cfg.RESULTS_DIR, f'scores_{SPLIT}.csv'), keep_default_na=False)
    saved = saved[(saved.model == 'in_context') & (saved.group != 'no_stim') & (saved.draw < DRAWS)]
    on = session_means(saved[saved.control == 'none'])
    off = session_means(saved[saved.control == 'off_context'])
    return (on - off).dropna()


def summarize(table):
    on = session_means(table[table.context == 'on'])
    off = session_means(table[table.context == 'off'])
    first = first_model_difference()
    rows = []
    for (input_name, k), difference in (on - off).groupby(level=['input', 'k']):
        difference = difference.droplevel(['input', 'k'])
        low, high = analysis.bootstrap_interval(difference.to_numpy())
        day_low, day_high = analysis.bootstrap_interval(difference.to_numpy(),
                                                        [analysis.day_of(s) for s in difference.index])
        first_part = first.xs((input_name, k), level=('input', 'k'))[difference.index]
        rows.append(dict(input=input_name, k=k, n_sessions=len(difference),
                         r2_on=on.xs((input_name, k), level=('input', 'k')).mean(),
                         r2_off=off.xs((input_name, k), level=('input', 'k')).mean(),
                         on_minus_off=difference.mean(), ci_low=low, ci_high=high,
                         day_ci_low=day_low, day_ci_high=day_high,
                         sessions_positive=int((difference > 0).sum()),
                         first_model_on_minus_off=first_part.mean(),
                         passes_c2=bool(k == 20 and low > 0)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    with open(CHOSEN_PATH) as file:
        chosen_alpha = json.load(file)['alpha_max']
    stim_sessions = data.usable_sessions(SPLIT, 'stim')
    all_rows = []
    for model_seed in SEEDS:
        all_rows.extend(score_seed(model_seed, chosen_alpha, stim_sessions, 'cuda'))
        print(f'seed {model_seed} done', flush=True)
    scores = pd.DataFrame(all_rows)
    check_on_matches_saved_scores(scores)
    summary = summarize(scores)
    scores.to_csv(os.path.join(cfg.RESULTS_DIR, 'calibrated_on_off_scores_val.csv'), index=False)
    summary.to_csv(os.path.join(cfg.RESULTS_DIR, 'calibrated_on_off_val.csv'), index=False)
    print(summary.round(4).T.to_string())
    print('DONE', flush=True)
