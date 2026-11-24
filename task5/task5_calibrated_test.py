"""Score the 5 calibrated decoder seeds on one split. On the test days this is an EXPLORATORY analysis.

Why exploratory: this decoder was designed after the first model's test results were seen. Its test score
cannot replace the pre-specified test of the first model, and nothing here may be tuned on the test days.

Everything below was fixed before the test days were read. Stimulation sessions only, both inputs:

    calibrated - labels-only     the neural gain, against the same reference as the main analysis
    calibrated - first model     whether the new decoder improves on the first one
    own - rotated                signal-label pairing broken: does the decoder read the context signal?
    own - off                    own ON context against same-day OFF context
    own - other_on               own ON context against context from other ON sessions that day
    other_on - off               other ON sessions' context against OFF context

The first two are reported at k = 2, 5, 10 and 20. The others are reported at k = 20. Each is reported for all
sessions and separately for 3 Hz and 20 Hz. A difference is "above zero" if its 95% session interval is above 0.
Frequency and day are confounded, so the frequency rows only describe the data.

Run from the repository root, on one free GPU. Run validation first: it must reproduce the saved scores.
    CUDA_VISIBLE_DEVICES=1 python task5/task5_calibrated_test.py --split val
    CUDA_VISIBLE_DEVICES=1 python task5/task5_calibrated_test.py --split test
The test run refuses to start if its result files already exist, so the test days are scored only once.
"""

import argparse
import json
import os

import pandas as pd

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task3_common as C
import task5_analyze as analysis
import task5_data as data
import task5_diagnose_context as diagnose
from task5_calibrated import CHOSEN_PATH, INPUTS, load_calibrated, run_name
from task5_evaluate import context_positions, load_held_out, n_draws, predict_queries, session_r2
from task5_run_final import SEEDS

K_VALUES = (2, 5, 10, 20)
CHECK_K = 20
CHECK_DRAW = 0          # k = 20 uses the full pool, so it has one draw
DRAWS = 3               # the same first draws the other context controls use
MAX_CHECK_DIFFERENCE = 1e-3
FREQUENCY_GROUPS = {'all': ('am3hz', 'am20hz'), '3 Hz': ('am3hz',), '20 Hz': ('am20hz',)}


def output_path(split, table_name):
    return os.path.join(cfg.RESULTS_DIR, f'calibrated_{split}_{table_name}.csv')


def other_stim_sessions(session, sessions):
    """The other stimulation sessions recorded on the same day. A day may have none."""
    return [s for s in sessions if s != session and analysis.day_of(s) == analysis.day_of(session)]


def score_seed(seed, alpha, sessions, device):
    """R² of one model for each session, input, k, draw and context source."""
    model, scale, _ = load_calibrated(run_name(alpha, seed), device)
    rows = []

    def add(session, input_name, k, draw, context, scaled, held):
        r2, _, _ = session_r2((scaled * scale).cpu().numpy(), held.query_velocity, held.query_valid)
        rows.append(dict(seed=seed, session=session, input=input_name, k=k, draw=draw, context=context, r2=r2))

    for session in sessions:
        off = load_held_out(C.splits()['off_reference'][session], 'recorded', scale, device)
        for input_name in INPUTS:
            held = load_held_out(session, input_name, scale, device)
            for k in K_VALUES:
                for draw in range(n_draws(k, DRAWS)):
                    add(session, input_name, k, draw, 'own', predict_queries(model, held, context_positions(held, k, draw)),
                        held)

            positions = context_positions(held, CHECK_K, CHECK_DRAW)
            rotated_signal = diagnose.context_signal('rotated', held, None, positions, CHECK_K, CHECK_DRAW)
            add(session, input_name, CHECK_K, CHECK_DRAW, 'rotated',
                diagnose.predict(model, held, rotated_signal, positions), held)
            # 'off_context' tells predict_queries to take signal and labels from the session passed after it.
            add(session, input_name, CHECK_K, CHECK_DRAW, 'off',
                predict_queries(model, held, context_positions(off, CHECK_K, CHECK_DRAW), 'off_context', off), held)
            for other_name in other_stim_sessions(session, sessions):
                other = load_held_out(other_name, input_name, scale, device)
                add(session, input_name, CHECK_K, CHECK_DRAW, 'other_on',
                    predict_queries(model, held, context_positions(other, CHECK_K, CHECK_DRAW), 'off_context', other),
                    held)
    return rows


def session_table(scores):
    """One row per session, input and k. Averages over draws and other sessions, then over seeds."""
    per_seed = scores.groupby(['session', 'input', 'k', 'context', 'seed']).r2.mean()
    return per_seed.groupby(['session', 'input', 'k', 'context']).mean().unstack('context')


def saved_references(split):
    """The first model and the labels-only reference, on the same context draws."""
    saved = pd.read_csv(os.path.join(cfg.RESULTS_DIR, f'scores_{split}.csv'), keep_default_na=False)
    saved = saved[saved.draw < DRAWS]
    first = analysis.session_scores(saved).droplevel('group')
    labels = analysis.labels_only_scores(saved).droplevel('group')
    return first, labels


def check_row(input_name, group, k, name, difference):
    difference = difference.dropna()
    low, high = analysis.bootstrap_interval(difference.to_numpy())
    day_low, day_high = analysis.bootstrap_interval(difference.to_numpy(),
                                                    [analysis.day_of(s) for s in difference.index])
    return dict(input=input_name, group=group, k=k, comparison=name, n_sessions=len(difference),
                n_days=len({analysis.day_of(s) for s in difference.index}), difference=difference.mean(),
                ci_low=low, ci_high=high, day_ci_low=day_low, day_ci_high=day_high,
                sessions_positive=int((difference > 0).sum()), above_zero=bool(low > 0))


def summarize(scores, split):
    table = session_table(scores)
    first, labels = saved_references(split)
    conditions = C.manifest().condition
    r2_rows, check_rows = [], []

    for input_name in INPUTS:
        for k in K_VALUES:
            part = table.xs((input_name, k), level=('input', 'k'))
            first_part = first.xs((input_name, k), level=('input', 'k'))[part.index]
            labels_part = labels.xs((input_name, k), level=('input', 'k'))[part.index]
            r2_rows.append(dict(input=input_name, k=k, n_sessions=len(part), calibrated=part.own.mean(),
                                first_model=first_part.mean(), labels_only=labels_part.mean()))

            for group, group_conditions in FREQUENCY_GROUPS.items():
                keep = conditions[part.index].isin(group_conditions).to_numpy()
                chosen = part[keep]
                check_rows.append(check_row(input_name, group, k, 'calibrated - labels-only',
                                            chosen.own - labels_part[keep]))
                check_rows.append(check_row(input_name, group, k, 'calibrated - first model',
                                            chosen.own - first_part[keep]))
                if k == CHECK_K:
                    for name, high, low in (('own - rotated', 'own', 'rotated'), ('own - off', 'own', 'off'),
                                            ('own - other_on', 'own', 'other_on'),
                                            ('other_on - off', 'other_on', 'off')):
                        check_rows.append(check_row(input_name, group, k, name, chosen[high] - chosen[low]))
    return pd.DataFrame(r2_rows), pd.DataFrame(check_rows)


def check_validation_reproduces_saved_scores(r2_table, checks):
    """The new scoring code must give the validation numbers already saved by the earlier scripts."""
    saved = pd.read_csv(os.path.join(cfg.RESULTS_DIR, 'calibrated_comparison_val.csv')).set_index(['input', 'k'])
    own = r2_table.set_index(['input', 'k']).calibrated
    differences = [(own - saved.r2_calibrated).abs().max()]

    at_k = checks[(checks.k == CHECK_K) & (checks.group == 'all')].set_index(['input', 'comparison']).difference
    on_off = pd.read_csv(os.path.join(cfg.RESULTS_DIR, 'calibrated_on_off_val.csv'))
    on_off = on_off[on_off.k == CHECK_K].set_index('input').on_minus_off
    control = pd.read_csv(os.path.join(cfg.RESULTS_DIR, 'calibrated_session_control_val.csv'))
    control = control.set_index(['input', 'comparison']).difference
    signal = pd.read_csv(os.path.join(cfg.RESULTS_DIR, 'calibrated_context_signal_val.csv'))
    signal = signal[(signal.k == CHECK_K) & (signal.condition == 'rotated')].set_index('input')['drop']
    for input_name in INPUTS:
        differences.append(abs(at_k[input_name, 'own - off'] - on_off[input_name]))
        differences.append(abs(at_k[input_name, 'own - rotated'] - signal[input_name]))
        for comparison in ('own - other_on', 'other_on - off'):
            differences.append(abs(at_k[input_name, comparison] - control[input_name, comparison]))

    largest = max(differences)
    print(f'Check: largest difference from saved validation scores is {largest:.6f}', flush=True)
    if largest > MAX_CHECK_DIFFERENCE:
        raise RuntimeError('this script does not reproduce the saved validation scores')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    split_name = parser.parse_args().split

    outputs = [output_path(split_name, name) for name in ('scores', 'r2', 'checks')]
    if split_name == 'test' and any(os.path.exists(path) for path in outputs):
        raise SystemExit('Test results already exist. The test days are scored only once.')

    with open(CHOSEN_PATH) as file:
        chosen_alpha = json.load(file)['alpha_max']
    stim_sessions = list(data.usable_sessions(split_name, 'stim'))
    all_rows = []
    for model_seed in SEEDS:
        all_rows.extend(score_seed(model_seed, chosen_alpha, stim_sessions, 'cuda'))
        print(f'seed {model_seed} done', flush=True)

    all_scores = pd.DataFrame(all_rows)
    r2_summary, check_summary = summarize(all_scores, split_name)
    if split_name == 'val':
        check_validation_reproduces_saved_scores(r2_summary, check_summary)

    all_scores.to_csv(outputs[0], index=False)
    r2_summary.to_csv(outputs[1], index=False)
    check_summary.to_csv(outputs[2], index=False)
    print(r2_summary.round(4).to_string())
    print(check_summary[check_summary.k == CHECK_K].round(4).to_string())
    print('DONE', flush=True)
