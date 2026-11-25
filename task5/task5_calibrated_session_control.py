"""Validation control: is the ON-context advantage about stimulation, or about being the same session?

ON context normally comes from the query's own session, while OFF context comes from a different session.
So "ON beats OFF" could mean either thing. This script decodes each stimulation session at k = 20 with:

    own        context from the query's own stimulation session
    other_on   context from the other stimulation session recorded that day
    off        context from that day's stimulation-OFF session

For the 10x input, own and other_on carry the 10x artifact. OFF stays as recorded. Nothing is trained and
the test days are not read.

How the result is read, fixed before running. A difference "holds" if it is at least 0.01 and its 95% interval
over sessions is above 0, for that input.

    own - other_on holds     part of the gain is specific to the query's own session
    other_on - off holds     ON context from a different session still beats OFF context

Limits: the validation set has 3 days and 6 stimulation sessions. Two of the 3 day pairs mix 3 Hz and 20 Hz, and
OFF was recorded first each day, so other_on - off can also reflect time between sessions.

Run from the repository root, on one free GPU:
    CUDA_VISIBLE_DEVICES=1 python task5/task5_calibrated_session_control.py
"""

import json
import os

import pandas as pd

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task3_common as C
import task5_analyze as analysis
import task5_data as data
from task5_calibrated import CHOSEN_PATH, INPUTS, load_calibrated, run_name
from task5_evaluate import context_positions, load_held_out, predict_queries, session_r2
from task5_run_final import SEEDS

SPLIT = 'val'
K = 20
DRAW = 0                # k = 20 uses the full pool, so it has one draw
MIN_DIFFERENCE = 0.01
COMPARISONS = (('own', 'other_on'), ('other_on', 'off'), ('own', 'off'))


def other_session_same_day(session, sessions):
    """The one other stimulation session recorded on the same day."""
    others = [s for s in sessions if s != session and analysis.day_of(s) == analysis.day_of(session)]
    if len(others) != 1:
        raise RuntimeError(f'expected one other stimulation session on the day of {session}, found {others}')
    return others[0]


def score_seed(seed, alpha, sessions, device):
    model, scale, _ = load_calibrated(run_name(alpha, seed), device)
    rows = []
    for session in sessions:
        off_session = load_held_out(C.splits()['off_reference'][session], 'recorded', scale, device)
        other_name = other_session_same_day(session, sessions)
        for input_name in INPUTS:
            held = load_held_out(session, input_name, scale, device)
            other = load_held_out(other_name, input_name, scale, device)
            sources = dict(own=None, other_on=other, off=off_session)
            for context, source in sources.items():
                if source is None:
                    scaled = predict_queries(model, held, context_positions(held, K, DRAW))
                else:   # 'off_context' tells predict_queries to take the context from `source`
                    scaled = predict_queries(model, held, context_positions(source, K, DRAW), 'off_context', source)
                r2, _, _ = session_r2((scaled * scale).cpu().numpy(), held.query_velocity, held.query_valid)
                rows.append(dict(seed=seed, session=session, input=input_name, context=context, r2=r2))
    return rows


def summarize(table):
    per_session = table.groupby(['session', 'input', 'context']).r2.mean().unstack('context')
    rows = []
    for input_name, part in per_session.groupby(level='input'):
        part = part.droplevel('input')
        for first, second in COMPARISONS:
            difference = part[first] - part[second]
            low, high = analysis.bootstrap_interval(difference.to_numpy())
            day_low, day_high = analysis.bootstrap_interval(difference.to_numpy(),
                                                            [analysis.day_of(s) for s in difference.index])
            rows.append(dict(input=input_name, comparison=f'{first} - {second}', n_sessions=len(difference),
                             r2_first=part[first].mean(), r2_second=part[second].mean(),
                             difference=difference.mean(), ci_low=low, ci_high=high,
                             day_ci_low=day_low, day_ci_high=day_high,
                             sessions_positive=int((difference > 0).sum()),
                             holds=bool(difference.mean() >= MIN_DIFFERENCE and low > 0)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    with open(CHOSEN_PATH) as file:
        chosen_alpha = json.load(file)['alpha_max']
    stim_sessions = list(data.usable_sessions(SPLIT, 'stim'))
    all_rows = []
    for model_seed in SEEDS:
        all_rows.extend(score_seed(model_seed, chosen_alpha, stim_sessions, 'cuda'))
        print(f'seed {model_seed} done', flush=True)
    scores = pd.DataFrame(all_rows)
    summary = summarize(scores)
    scores.to_csv(os.path.join(cfg.RESULTS_DIR, 'calibrated_session_control_scores_val.csv'), index=False)
    summary.to_csv(os.path.join(cfg.RESULTS_DIR, 'calibrated_session_control_val.csv'), index=False)
    print(summary.round(4).to_string())
    print(scores.groupby(['input', 'session', 'context']).r2.mean().unstack('context').round(3).to_string())
    print('DONE', flush=True)
