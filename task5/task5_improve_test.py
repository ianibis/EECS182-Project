"""Score the final decoder of Section 8.2 (the jitter decoder) on one split. On the test days this is EXPLORATORY.

Why exploratory: this decoder was designed on validation days after the test days had already been read for the
first model and for the calibrated decoder. Its test score cannot replace the pre-specified test of the first model,
and nothing here may be tuned on the test days.

Everything below was fixed before the test days were read. Final setting: improve_choice.json["step2"], seeds 1-5.
Stimulation sessions only, both inputs, the first 3 context draws (k = 20 has one draw), k = 2, 5, 10 and 20.
"One model" is the R² of each seed, averaged over seeds. "Average" is the R² of the mean of the 5 seeds' predictions.

    final - calibrated               one model each: does the Section 8.2 change help?
    final - first model              one model against the first model (whose numbers average R² over seeds)
    final - labels-only              the neural gain
    average - one model              what averaging the 5 seeds' predictions adds
    average - calibrated average     the same comparison, for 5-seed averages
    average - first model
    average - labels-only
    own - rotated, own - off, own - other_on, other_on - off      k = 20 only, one final model, as in Section 8.1

Each is reported for all sessions and separately for 3 Hz and 20 Hz (frequency and day are confounded, so the
frequency rows only describe the data). A difference is "above zero" if its 95% session interval is above 0.

Checks before anything is saved:
    validation   must reproduce improve_comparison_val.csv and the rotated drop in improve_context_signal_val.csv
    test         the calibrated decoder's one-model R² must reproduce calibrated_test_r2.csv

Run from the repository root, on one free GPU. Run validation first.
    CUDA_VISIBLE_DEVICES=0 python task5/task5_improve_test.py --split val
    CUDA_VISIBLE_DEVICES=0 python task5/task5_improve_test.py --split test
The test run refuses to start if its result files already exist, so the test days are scored only once.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task3_common as C
import task5_data as data
import task5_diagnose_context as diagnose
from task5_calibrated_test import (CHECK_DRAW, CHECK_K, DRAWS, FREQUENCY_GROUPS, K_VALUES, MAX_CHECK_DIFFERENCE,
                                   check_row, other_stim_sessions, saved_references)
from task5_evaluate import context_positions, load_held_out, n_draws, predict_queries, session_r2
from task5_improve import BASELINE, INPUTS, load_choice, load_run, run_name
from task5_run_final import SEEDS

AVERAGE = 0     # the `seed` of rows scored on the mean of the 5 seeds' predictions
R2_COLUMNS = ('final', 'final_average', 'calibrated', 'calibrated_average', 'first_model', 'labels_only')
COMPARISONS = (  # name, higher column, lower column
    ('final - calibrated', 'final', 'calibrated'),
    ('final - first model', 'final', 'first_model'),
    ('final - labels-only', 'final', 'labels_only'),
    ('average - one model', 'final_average', 'final'),
    ('average - calibrated average', 'final_average', 'calibrated_average'),
    ('average - first model', 'final_average', 'first_model'),
    ('average - labels-only', 'final_average', 'labels_only'),
)
CONTEXT_COMPARISONS = (
    ('own - rotated', 'final', 'rotated'),
    ('own - off', 'final', 'off'),
    ('own - other_on', 'final', 'other_on'),
    ('other_on - off', 'other_on', 'off'),
)


def output_path(split, table_name):
    return os.path.join(cfg.RESULTS_DIR, f'improve_{split}_{table_name}.csv')


def score_model(model_name, setting, sessions, device, with_controls):
    """Per-seed R² rows, each seed's own-context predictions (cm/s), and the query trials' recorded velocity."""
    rows, predictions, truth = [], {}, {}
    for seed in SEEDS:
        model, scale, _ = load_run(run_name(setting, seed), device)

        def add(session, input_name, k, draw, context, scaled, held):
            prediction = (scaled * scale).cpu().numpy()
            r2, _, _ = session_r2(prediction, held.query_velocity, held.query_valid)
            rows.append(dict(model=model_name, seed=seed, session=session, input=input_name, k=k, draw=draw,
                             context=context, r2=r2))
            return prediction

        for session in sessions:
            off = load_held_out(C.splits()['off_reference'][session], 'recorded', scale, device)
            for input_name in INPUTS:
                held = load_held_out(session, input_name, scale, device)
                truth[session] = (held.query_velocity, held.query_valid)
                for k in K_VALUES:
                    for draw in range(n_draws(k, DRAWS)):
                        scaled = predict_queries(model, held, context_positions(held, k, draw))
                        prediction = add(session, input_name, k, draw, 'own', scaled, held)
                        predictions.setdefault((session, input_name, k, draw), []).append(prediction)
                if not with_controls:
                    continue

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
                        predict_queries(model, held, context_positions(other, CHECK_K, CHECK_DRAW), 'off_context',
                                        other), held)
        print(f'{model_name} seed {seed} done', flush=True)
        del model
    return rows, predictions, truth


def average_rows(model_name, predictions, truth):
    """R² of the mean of the 5 seeds' predictions, for every own-context row."""
    rows = []
    for (session, input_name, k, draw), outputs in predictions.items():
        velocity, valid = truth[session]
        r2, _, _ = session_r2(np.mean(outputs, axis=0), velocity, valid)
        rows.append(dict(model=model_name, seed=AVERAGE, session=session, input=input_name, k=k, draw=draw,
                         context='own', r2=r2))
    return rows


def column(scores, model, context='own', average=False):
    """One value per session, input and k: the mean over draws (and other sessions), then over seeds."""
    chosen = scores[(scores.model == model) & (scores.context == context) & ((scores.seed == AVERAGE) == average)]
    per_seed = chosen.groupby(['session', 'input', 'k', 'seed']).r2.mean()
    return per_seed.groupby(['session', 'input', 'k']).mean()


def session_table(scores, split):
    first, labels = saved_references(split)
    table = pd.DataFrame(dict(
        final=column(scores, 'final'), final_average=column(scores, 'final', average=True),
        calibrated=column(scores, 'calibrated'), calibrated_average=column(scores, 'calibrated', average=True),
        rotated=column(scores, 'final', 'rotated'), off=column(scores, 'final', 'off'),
        other_on=column(scores, 'final', 'other_on')))
    table['first_model'] = first.reindex(table.index)
    table['labels_only'] = labels.reindex(table.index)
    return table


def summarize(scores, split):
    table = session_table(scores, split)
    conditions = C.manifest().condition
    r2_rows, check_rows = [], []
    for input_name in INPUTS:
        for k in K_VALUES:
            part = table.xs((input_name, k), level=('input', 'k'))
            r2_rows.append(dict(input=input_name, k=k, n_sessions=len(part), **part[list(R2_COLUMNS)].mean().to_dict()))
            comparisons = COMPARISONS + (CONTEXT_COMPARISONS if k == CHECK_K else ())
            for group, group_conditions in FREQUENCY_GROUPS.items():
                chosen = part[conditions[part.index].isin(group_conditions).to_numpy()]
                for name, high, low in comparisons:
                    check_rows.append(check_row(input_name, group, k, name, chosen[high] - chosen[low]))
    return pd.DataFrame(r2_rows), pd.DataFrame(check_rows)


def check_reproduces_saved_scores(r2_table, checks, split):
    """Validation must give the Section 8.2 numbers; on test, the calibrated decoder must give Section 8.1's."""
    new = r2_table.set_index(['input', 'k'])
    differences = []
    if split == 'val':
        saved = pd.read_csv(os.path.join(cfg.RESULTS_DIR, 'improve_comparison_val.csv')).set_index(['input', 'k'])
        for new_column, saved_column in (('final', 'final_one_model'), ('final_average', 'final_seed_average'),
                                         ('calibrated', 'calibrated_one_model'),
                                         ('calibrated_average', 'calibrated_seed_average')):
            differences.append((new[new_column] - saved[saved_column]).abs().max())
        signal = pd.read_csv(os.path.join(cfg.RESULTS_DIR, 'improve_context_signal_val.csv'))
        signal = signal[(signal.k == CHECK_K) & (signal.condition == 'rotated')].set_index('input')['drop']
        rotated = checks[(checks.k == CHECK_K) & (checks.group == 'all') & (checks.comparison == 'own - rotated')]
        rotated = rotated.set_index('input').difference
        differences.extend(abs(rotated[name] - signal[name]) for name in INPUTS)
    else:
        saved = pd.read_csv(os.path.join(cfg.RESULTS_DIR, 'calibrated_test_r2.csv')).set_index(['input', 'k'])
        differences.append((new.calibrated - saved.calibrated).abs().max())

    largest = max(differences)
    print(f'Check: largest difference from saved scores is {largest:.6f}', flush=True)
    if largest > MAX_CHECK_DIFFERENCE:
        raise RuntimeError('this script does not reproduce the saved scores')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    split_name = parser.parse_args().split

    outputs = [output_path(split_name, name) for name in ('scores', 'r2', 'checks')]
    if split_name == 'test' and any(os.path.exists(path) for path in outputs):
        raise SystemExit('Test results already exist. The test days are scored only once.')

    final_setting = load_choice('step2')
    print(f'Final setting: {final_setting}', flush=True)
    stim_sessions = list(data.usable_sessions(split_name, 'stim'))
    with torch.no_grad():
        final_rows, final_predictions, query_truth = score_model('final', final_setting, stim_sessions, 'cuda',
                                                                 with_controls=True)
        calibrated_rows, calibrated_predictions, _ = score_model('calibrated', BASELINE, stim_sessions, 'cuda',
                                                                 with_controls=False)
    all_scores = pd.DataFrame(final_rows + calibrated_rows + average_rows('final', final_predictions, query_truth)
                              + average_rows('calibrated', calibrated_predictions, query_truth))

    r2_summary, check_summary = summarize(all_scores, split_name)
    r2_summary.insert(2, 'final_setting', final_setting.name)
    check_reproduces_saved_scores(r2_summary, check_summary, split_name)

    all_scores.to_csv(outputs[0], index=False)
    r2_summary.to_csv(outputs[1], index=False)
    check_summary.to_csv(outputs[2], index=False)
    print(r2_summary.round(4).to_string())
    print(check_summary[check_summary.group == 'all'].round(4).to_string())
    print('DONE', flush=True)
