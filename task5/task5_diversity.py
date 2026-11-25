"""Task 5.7 (optional): how many distinct shifts does training need?

Trains one model per pool size, with the final learning-rate setting, and scores each on
held-out recordings, whose artifacts the model has never seen. No choice is made from
these runs, so they do not affect the main result.

Run from the repository root, one model at a time:
    CUDA_VISIBLE_DEVICES=1 python task5/task5_diversity.py --train
    CUDA_VISIBLE_DEVICES=1 python task5/task5_diversity.py --score val
    CUDA_VISIBLE_DEVICES=1 python task5/task5_diversity.py --score test
    python task5/task5_diversity.py --analyze val
    python task5/task5_diversity.py --analyze test
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task5_data as data
import task5_plot_style as style
from task5_analyze import INPUT_TITLES, bootstrap_interval, ridge_scores
from task5_baselines import LabelsOnlyBaseline
from task5_evaluate import evaluate
from task5_run_final import FINAL_ITERATIONS
from task5_score import load_model
from task5_train import train

POOL_SIZES = (4, 16, 64, 256, 1024, 4096)
SEED = 1
UNLIMITED = 'new every episode'     # the main seed-1 model, which never sees the same shift twice
RIDGE = 'ridge refit'               # least-squares reference fitted on the same k trials
FIGURE_K = (5, 20)


def run_name(n_shifts):
    return f'diversity_n{n_shifts}_seed{SEED}'


def train_all():
    with open(os.path.join(cfg.RESULTS_DIR, 'chosen_setting.json')) as file:
        chosen = json.load(file)
    for n_shifts in POOL_SIZES:
        name = run_name(n_shifts)
        if os.path.exists(os.path.join(cfg.RUNS_DIR, name, 'summary.json')):
            print(f'{name}: already finished', flush=True)
            continue
        train(name, cfg.TrainConfig(seed=SEED, iterations=FINAL_ITERATIONS, learning_rate=chosen['learning_rate'],
                                    warmup_iterations=chosen['warmup'], eval_every=500), n_shifts=n_shifts)
    print('ALL DIVERSITY RUNS DONE', flush=True)


def score_all(split):
    baseline = LabelsOnlyBaseline()
    sessions = data.usable_sessions(split, 'stim')
    tables = []
    for n_shifts in POOL_SIZES:
        model, scale, step = load_model(run_name(n_shifts), 'cuda')
        table = evaluate(model, sessions, ('recorded', 'amp10'), cfg.K_VALUES, scale, baseline)
        table.insert(0, 'n_shifts', n_shifts)
        table['checkpoint_step'] = step
        tables.append(table)
        print(f'n_shifts {n_shifts}: mean R2 at k=20 {table[table.k == 20].r2.mean():+.4f}', flush=True)
    pd.concat(tables).to_csv(os.path.join(cfg.RESULTS_DIR, f'diversity_{split}.csv'), index=False)


def summary_rows(per_session, label):
    """Mean R² and 95% bootstrap interval over sessions, per input and k."""
    rows = []
    for (input_name, k), part in per_session.groupby(['input', 'k']):
        low, high = bootstrap_interval(part.r2.to_numpy())
        rows.append(dict(training_shifts=label, input=input_name, k=k, n_sessions=len(part),
                         r2=part.r2.mean(), ci_low=low, ci_high=high))
    return rows


def summarize(split):
    """Collect the pool-size models, the unlimited-shift model and the ridge reference."""
    table = pd.read_csv(os.path.join(cfg.RESULTS_DIR, f'diversity_{split}.csv'), keep_default_na=False)
    rows = []
    for n_shifts in POOL_SIZES:
        part = table[table.n_shifts == n_shifts]
        rows += summary_rows(part.groupby(['session', 'input', 'k'], as_index=False).r2.mean(), str(n_shifts))

    scores = pd.read_csv(os.path.join(cfg.RESULTS_DIR, f'scores_{split}.csv'), keep_default_na=False)
    main = scores[(scores.model == 'in_context') & (scores.control == 'none') & (scores.seed == SEED)
                  & (scores.group != 'no_stim')]
    rows += summary_rows(main.groupby(['session', 'input', 'k'], as_index=False).r2.mean(), UNLIMITED)

    ridge = ridge_scores(split).reset_index()
    rows += summary_rows(ridge[ridge.group != 'no_stim'], RIDGE)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(cfg.RESULTS_DIR, f'diversity_summary_{split}.csv'), index=False)
    return summary


def diversity_figure(summary, split):
    """R² against the number of distinct training shifts, with the ridge refit as a reference line."""
    style.apply_style()
    pool_labels = [str(n) for n in POOL_SIZES]
    x = np.arange(len(pool_labels))
    unlimited_x = len(pool_labels) + 0.5
    figure, axes = style.plt.subplots(len(FIGURE_K), 2, figsize=(11, 6.2), sharex=True, sharey=True)

    for row, k in enumerate(FIGURE_K):
        for column, (input_name, title) in enumerate(INPUT_TITLES.items()):
            axis = axes[row, column]
            chosen = summary[(summary.input == input_name) & (summary.k == k)].set_index('training_shifts')
            curve = chosen.loc[pool_labels]
            axis.fill_between(x, curve.ci_low, curve.ci_high, color=style.BLUE, alpha=0.14, linewidth=0)
            axis.plot(x, curve.r2, color=style.BLUE, marker='o', markersize=5, markeredgecolor=style.SURFACE,
                      markeredgewidth=1.5, label='In-context, fixed set of shifts')

            best = chosen.loc[UNLIMITED]
            axis.errorbar([unlimited_x], [best.r2], yerr=[[best.r2 - best.ci_low], [best.ci_high - best.r2]],
                          fmt='D', color=style.BLUE, markersize=6, capsize=3, elinewidth=1,
                          markeredgecolor=style.SURFACE, label='In-context, new shift every episode')

            ridge = chosen.loc[RIDGE]
            axis.axhline(ridge.r2, color=style.AQUA, linestyle='--', linewidth=1.5,
                         label='Ridge refit on the same k trials')
            axis.axhline(0, color=style.AXIS, linewidth=0.8)
            axis.set_title(f'{title}, k = {k}')
            axis.set_xticks([*x, unlimited_x], [*pool_labels, 'New each\nepisode'])
        axes[row, 0].set_ylabel('Velocity R²')
    for axis in axes[-1]:
        axis.set_xlabel('Distinct simulated shifts seen in training')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04))
    figure.tight_layout()
    figure.savefig(os.path.join(cfg.RESULTS_DIR, f'fig_diversity_{split}.png'))
    style.plt.close(figure)


def analyze(split):
    summary = summarize(split)
    diversity_figure(summary, split)
    print(summary.pivot_table(index=['input', 'k'], columns='training_shifts', values='r2',
                              sort=False).round(4).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--score', choices=('val', 'test'))
    parser.add_argument('--analyze', choices=('val', 'test'))
    arguments = parser.parse_args()
    if arguments.train:
        train_all()
    if arguments.score:
        score_all(arguments.score)
    if arguments.analyze:
        analyze(arguments.analyze)
