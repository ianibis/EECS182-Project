"""Task 5 aggregation: turn the raw score tables into summary tables, checks and figures.

Scores are averaged over draws and seeds within each session. Intervals are 95% bootstrap
intervals over sessions. Because sessions from one day share electrode state, each paired
check also reports an interval that resamples whole days, which is stricter.

Run from the repository root:  python task5/task5_analyze.py --split val
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task3_common as C
import task5_plot_style as style

N_BOOTSTRAP = 10000
INPUT_TITLES = {'recorded': 'Recorded stimulation-ON trials', 'amp10': 'Semi-synthetic (10x artifact)'}
METHODS = (('in_context', 'In-context transformer', style.BLUE),
           ('blind', 'Blind transformer (no brain signal)', style.ORANGE),
           ('ridge', 'Ridge refit on k trials', style.AQUA),
           ('labels_only', 'Labels-only average', '#eda100'))


def day_of(session):
    return str(C.manifest().loc[session, 'date'])


def bootstrap_interval(values, groups=None, seed=0):
    """95% interval of the mean. With `groups`, whole groups (days) are resampled."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    if groups is None:
        picks = rng.integers(0, len(values), (N_BOOTSTRAP, len(values)))
        means = values[picks].mean(axis=1)
    else:
        groups = np.asarray(groups)
        names = np.unique(groups)
        members = [values[groups == name] for name in names]
        means = np.empty(N_BOOTSTRAP)
        for index in range(N_BOOTSTRAP):
            chosen = rng.integers(0, len(names), len(names))
            means[index] = np.concatenate([members[c] for c in chosen]).mean()
    return tuple(np.percentile(means, [2.5, 97.5]))


def session_scores(scores, model='in_context', control='none', max_draw=None):
    """R² per (session, input, k), averaged over draws and then over seeds."""
    chosen = scores[(scores.model == model) & (scores.control == control)]
    if max_draw is not None:
        chosen = chosen[chosen.draw < max_draw]
    per_seed = chosen.groupby(['session', 'group', 'input', 'k', 'seed']).r2.mean()
    return per_seed.groupby(['session', 'group', 'input', 'k']).mean()


def labels_only_scores(scores):
    chosen = scores[(scores.model == 'in_context') & (scores.control == 'none')]
    return chosen.groupby(['session', 'group', 'input', 'k']).r2_labels_only.mean().rename('r2')


def ridge_scores(split):
    path = os.path.join(cfg.RESULTS_DIR, f'ridge_reference_{split}.csv')
    if not os.path.exists(path):
        return None
    table = pd.read_csv(path)
    return table.groupby(['session', 'group', 'input', 'k']).r2.mean()


def stim_only(series, input_name):
    """Keep the stimulation-ON sessions of one input."""
    frame = series.reset_index()
    return frame[(frame.input == input_name) & (frame.group != 'no_stim')]


def stronger_labels_only(scores, split):
    """The harder no-brain reference at each input and k.

    The choice between the blind transformer and the labels-only average is made on the
    validation days and saved, so the test days cannot influence it.
    """
    sources = dict(blind=session_scores(scores, model='blind'), labels_only=labels_only_scores(scores))
    path = os.path.join(cfg.RESULTS_DIR, 'labels_only_choice.json')
    if split == 'val':
        choice = {}
        for input_name in INPUT_TITLES:
            for k in cfg.K_VALUES:
                means = {name: stim_only(series, input_name).query('k == @k').r2.mean()
                         for name, series in sources.items()}
                choice[f'{input_name}_{k}'] = max(means, key=means.get)
        with open(path, 'w') as file:
            json.dump(choice, file, indent=2)
    else:
        with open(path) as file:
            choice = json.load(file)

    parts = []
    for input_name in INPUT_TITLES:
        for k in cfg.K_VALUES:
            frame = sources[choice[f'{input_name}_{k}']].reset_index()
            parts.append(frame[(frame.input == input_name) & (frame.k == k)])
    return pd.concat(parts).set_index(['session', 'group', 'input', 'k']).r2


def summary_table(series_by_method):
    rows = []
    for method, series in series_by_method.items():
        if series is None:
            continue
        for input_name in INPUT_TITLES:
            frame = stim_only(series, input_name)
            for k, group in frame.groupby('k'):
                low, high = bootstrap_interval(group.r2.to_numpy())
                rows.append(dict(method=method, input=input_name, k=k, n_sessions=len(group),
                                 r2=group.r2.mean(), ci_low=low, ci_high=high))
    return pd.DataFrame(rows)


def paired(first, second, input_name, k_first, k_second=None):
    """Paired difference first(k_first) - second(k_second) over stimulation sessions."""
    k_second = k_first if k_second is None else k_second
    a = stim_only(first, input_name)
    b = stim_only(second, input_name)
    a = a[a.k == k_first].set_index('session').r2
    b = b[b.k == k_second].set_index('session').r2
    shared = a.index.intersection(b.index)
    return (a[shared] - b[shared]).rename('difference')


def check_row(name, input_name, difference, passes):
    values = difference.to_numpy()
    low, high = bootstrap_interval(values)
    day_low, day_high = bootstrap_interval(values, [day_of(s) for s in difference.index])
    return dict(check=name, input=input_name, mean_difference=values.mean(), ci_low=low, ci_high=high,
                day_ci_low=day_low, day_ci_high=day_high, sessions_positive=int((values > 0).sum()),
                n_sessions=len(values), passed=bool(passes(values.mean(), low, high)))


def hypothesis_table(scores, split):
    main = session_scores(scores)
    blind = session_scores(scores, model='blind')
    labels = labels_only_scores(scores)
    stronger = stronger_labels_only(scores, split)
    ridge = ridge_scores(split)
    above_zero = lambda mean, low, high: low > 0
    rows = []

    for input_name in INPUT_TITLES:
        rows.append(check_row('H1: k=20 beats k=0', input_name, paired(main, main, input_name, 20, 0), above_zero))
        for k in cfg.K_VALUES:
            rows.append(check_row(f'PRIMARY neural gain vs stronger labels-only reference, k={k}', input_name,
                                  paired(main, stronger, input_name, k), above_zero))
            rows.append(check_row(f'In-context minus blind transformer, k={k}', input_name,
                                  paired(main, blind, input_name, k), above_zero))
        for k in cfg.K_VALUES[1:]:
            rows.append(check_row(f'In-context minus labels-only average, k={k}', input_name,
                                  paired(main, labels, input_name, k), above_zero))
            if ridge is not None:
                rows.append(check_row(f'In-context minus ridge refit, k={k}', input_name,
                                      paired(main, ridge, input_name, k), above_zero))

        # Controls use the first three draws, so the comparison uses the same draws.
        same_draws = session_scores(scores, max_draw=3)
        shuffled = session_scores(scores, control='shuffle_labels', max_draw=3)
        gain = paired(same_draws, same_draws, input_name, 20, 0).mean()
        lost = paired(same_draws, shuffled, input_name, 20)
        rows.append(check_row('H3: shuffling labels loses at least half of the k=20 gain', input_name, lost,
                              lambda mean, low, high: gain > 0 and mean >= 0.5 * gain))
        rows.append(check_row('C2: ON context beats OFF context, k=20', input_name,
                              paired(same_draws, session_scores(scores, control='off_context', max_draw=3),
                                     input_name, 20), above_zero))
        rows.append(check_row('C3: hiding the query signal lowers R2, k=20', input_name,
                              paired(same_draws, session_scores(scores, control='no_query_signal', max_draw=3),
                                     input_name, 20), above_zero))
    return pd.DataFrame(rows)


def off_on_table(scores, split):
    """Stimulation loss and the share of it recovered, against each session's same-day OFF session."""
    main = session_scores(scores).reset_index()
    off_lookup = C.splits()['off_reference']
    off = main[main.group == 'no_stim'].set_index(['session', 'k']).r2
    rows = []
    for input_name in INPUT_TITLES:
        frame = main[(main.input == input_name) & (main.group != 'no_stim')]
        for k in cfg.K_VALUES:
            on = frame[frame.k == k].set_index('session').r2
            on_zero = frame[frame.k == 0].set_index('session').r2
            off_zero = pd.Series({s: off[(off_lookup[s], 0)] for s in on.index})
            off_same = pd.Series({s: off[(off_lookup[s], k)] for s in on.index})
            loss = off_zero - on_zero
            gap = off_same - on
            gap_low, gap_high = bootstrap_interval(gap.to_numpy())
            # fraction_recovered is the proposal's metric. It credits any gain from context, including
            # gains that OFF sessions get too. gap_closed compares ON and OFF at the same k, so it
            # only counts improvement specific to stimulation.
            rows.append(dict(input=input_name, k=k, r2_on=on.mean(), r2_off_same_k=off_same.mean(),
                             gap_off_minus_on=gap.mean(), gap_ci_low=gap_low, gap_ci_high=gap_high,
                             gap_closed=1 - gap.mean() / loss.mean(), loss_at_k0=loss.mean(),
                             loss_ci_low=bootstrap_interval(loss.to_numpy())[0],
                             loss_ci_high=bootstrap_interval(loss.to_numpy())[1],
                             fraction_recovered=(on.mean() - on_zero.mean()) / loss.mean()))
    return pd.DataFrame(rows)


def spread_labels(axis, entries, gap):
    """Direct labels at the right edge, nudged apart so they never overlap."""
    entries = sorted(entries, key=lambda entry: entry[0])
    placed = []
    for y, text in entries:
        y = max(y, placed[-1] + gap) if placed else y
        placed.append(y)
        axis.annotate(text, xy=(1, y), xycoords=('axes fraction', 'data'), xytext=(6, 0),
                      textcoords='offset points', va='center', fontsize=8, color=style.INK_SECONDARY,
                      annotation_clip=False)


def main_figure(summary, split):
    figure, axes = style.plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    positions = {k: i for i, k in enumerate(cfg.K_VALUES)}
    short = dict(in_context='In-context', blind='Blind', ridge='Ridge refit', labels_only='Labels-only')
    span = summary.r2.max() - summary.r2.min()
    for axis, (input_name, title) in zip(axes, INPUT_TITLES.items()):
        entries = []
        for method, label, color in METHODS:
            part = summary[(summary.method == method) & (summary.input == input_name)].sort_values('k')
            if part.empty:
                continue
            x = part.k.map(positions).to_numpy()
            axis.fill_between(x, part.ci_low, part.ci_high, color=color, alpha=0.14, linewidth=0)
            axis.plot(x, part.r2, color=color, marker='o', markersize=5, label=label,
                      markeredgecolor=style.SURFACE, markeredgewidth=1.5)
            entries.append((part.r2.iloc[-1], f'{short[method]} {part.r2.iloc[-1]:.3f}'))
        spread_labels(axis, entries, span * 0.07)
        axis.axhline(0, color=style.AXIS, linewidth=0.8)
        axis.set_xticks(range(len(cfg.K_VALUES)), cfg.K_VALUES)
        axis.set_xlabel('Labeled stimulation trials in context (k)')
        axis.set_title(title)
    axes[0].set_ylabel('Velocity R² on query trials')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.07))
    figure.tight_layout(w_pad=7)
    figure.savefig(os.path.join(cfg.RESULTS_DIR, f'fig_main_{split}.png'))
    style.plt.close(figure)


def controls_figure(scores, split):
    conditions = (('In-context', 'in_context', 'none'), ('Shuffled context labels', 'in_context', 'shuffle_labels'),
                  ('OFF trials as context', 'in_context', 'off_context'), ('Blind transformer', 'blind', 'none'),
                  ('Query signal hidden', 'in_context', 'no_query_signal'))
    figure, axes = style.plt.subplots(1, 2, figsize=(11, 3.6), sharex=True, sharey=True)
    for axis, (input_name, title) in zip(axes, INPUT_TITLES.items()):
        means, lows, highs = [], [], []
        for _, model, control in conditions:
            frame = stim_only(session_scores(scores, model=model, control=control, max_draw=3), input_name)
            values = frame[frame.k == 20].r2.to_numpy()
            low, high = bootstrap_interval(values)
            means.append(values.mean()), lows.append(low), highs.append(high)
        y = np.arange(len(conditions))[::-1]
        axis.barh(y, means, height=0.5, color=style.BLUE)
        axis.errorbar(means, y, xerr=[np.subtract(means, lows), np.subtract(highs, means)], fmt='none',
                      ecolor=style.INK_SECONDARY, elinewidth=1, capsize=2)
        for row, high, mean in zip(y, highs, means):
            axis.text(max(high, 0) + 0.004, row, f'{mean:.3f}', va='center', fontsize=8, color=style.INK_SECONDARY)
        axis.set_yticks(y, [c[0] for c in conditions])
        axis.axvline(0, color=style.AXIS, linewidth=0.8)
        axis.grid(axis='y', visible=False)
        axis.set_title(title)
        axis.set_xlabel('Velocity R² at k = 20')
    figure.tight_layout()
    figure.savefig(os.path.join(cfg.RESULTS_DIR, f'fig_controls_{split}.png'))
    style.plt.close(figure)


def training_figure():
    histories = {kind: [pd.read_csv(p) for p in sorted(glob.glob(
        os.path.join(cfg.RUNS_DIR, f'final_{kind}_seed*', 'history.csv')))] for kind in ('in_context', 'blind')}
    if not histories['in_context']:
        return
    figure, axes = style.plt.subplots(1, 2, figsize=(11, 3.6))
    for run in histories['in_context']:
        axes[0].plot(run.step, run.train_loss, color=style.BLUE, linewidth=1, alpha=0.8)
        axes[0].plot(run.step, run.holdout_loss, color=style.ORANGE, linewidth=1, alpha=0.8)
    axes[0].plot([], [], color=style.BLUE, label='Training episodes')
    axes[0].plot([], [], color=style.ORANGE, label='Held-out blocks (simulated shift)')
    axes[0].set(title='In-context loss, 5 seeds', xlabel='Training step', ylabel='MSE (scaled velocity)')
    axes[0].legend()
    for kind, color, label in (('in_context', style.BLUE, 'In-context'), ('blind', style.ORANGE, 'Blind')):
        for run in histories[kind]:
            axes[1].plot(run.step, run.val_r2, color=color, linewidth=1, alpha=0.8)
        axes[1].plot([], [], color=color, label=label)
    axes[1].set(title='Validation-day R², 5 seeds', xlabel='Training step', ylabel='Mean R² (k = 2 to 20)')
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(os.path.join(cfg.RESULTS_DIR, 'fig_training_curves.png'))
    style.plt.close(figure)


def main(split):
    style.apply_style()
    scores = pd.read_csv(os.path.join(cfg.RESULTS_DIR, f'scores_{split}.csv'), keep_default_na=False)
    summary = summary_table(dict(in_context=session_scores(scores), blind=session_scores(scores, model='blind'),
                                 ridge=ridge_scores(split), labels_only=labels_only_scores(scores)))
    checks = hypothesis_table(scores, split)
    off_on = off_on_table(scores, split)
    summary.to_csv(os.path.join(cfg.RESULTS_DIR, f'summary_{split}.csv'), index=False)
    checks.to_csv(os.path.join(cfg.RESULTS_DIR, f'checks_{split}.csv'), index=False)
    off_on.to_csv(os.path.join(cfg.RESULTS_DIR, f'off_vs_on_{split}.csv'), index=False)
    main_figure(summary, split)
    controls_figure(scores, split)
    training_figure()

    print(summary.pivot_table(index=['input', 'k'], columns='method', values='r2').round(4).to_string())
    print(checks.round(4).to_string())
    print(off_on.round(4).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    main(parser.parse_args().split)
