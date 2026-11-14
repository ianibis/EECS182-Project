"""Summarise training runs (plan 5.4): validation R2 by k at the best checkpoint and at the last step.

Run from ICL/:
  .venv/bin/python scripts/summarize_runs.py [--brief] [runs/<name> ...]
Default: every run. Globs such as runs/reg_* are fine (log files are skipped).
--brief prints one line per run, e.g. for  watch -n 30 .venv/bin/python scripts/summarize_runs.py --brief runs/reg_*
The best checkpoint is the one with the highest selection metric: mean B R2 at k = 5 and 20.
"""
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(run_dir):
    """(name, train rows, validation rows), or None if the run has no metrics.csv / no steps yet."""
    path = os.path.join(run_dir, 'metrics.csv')
    if not os.path.isfile(path):
        return None                                   # e.g. runs/<name>.log matched by a glob
    metrics = pd.read_csv(path)
    if metrics.empty:
        return None
    return os.path.basename(os.path.normpath(run_dir)), metrics[metrics.split == 'train'], metrics[metrics.split != 'train']


def selection_curve(val, split='B'):
    """Mean ICL R2 at k = 5 and 20, per validation step."""
    rows = val[(val.split == split) & (val.method == 'icl') & (val.k.isin([5, 20]))]
    return rows.groupby('step').value.mean()


def brief(run_dir):
    loaded = load(run_dir)
    if loaded is None:
        return None
    name, train, val = loaded
    row = dict(run=name, step=int(train.step.max()), loss=round(train.value.iloc[-1], 4))
    b, real = selection_curve(val, 'B'), selection_curve(val, 'real')
    if len(b):
        row.update(B_now=round(b.iloc[-1], 3), B_best=round(b.max(), 3), best_step=int(b.idxmax()))
    if len(real):
        row.update(real_now=round(real.iloc[-1], 3), real_best=round(real.max(), 3))
    return row


def summarize(run_dir):
    loaded = load(run_dir)
    if loaded is None:
        return None
    name, train, val = loaded
    last_loss = f'{train.value.iloc[-1]:.4f}'
    if val.empty:
        print(f'== {name}: step {train.step.max()}, train loss {last_loss}, no validation yet\n')
        return None
    selection = selection_curve(val)
    best_step = int(selection.idxmax())
    print(f'== {name}: step {train.step.max()}, train loss {last_loss}, '
          f'best selection {selection.max():.3f} at step {best_step} (last {selection.iloc[-1]:.3f})')
    views = [('best', best_step)]
    if (val.split == 'real').any():
        views.append(('last real-val step', int(val[val.split == 'real'].step.max())))
    for label, step in views:
        rows = val[val.step == step]
        table = rows.pivot_table(index=['split', 'method'], columns='k', values='value')
        print(f'  {label} (step {step}):')
        print('\n'.join('    ' + line for line in table.round(3).to_string().splitlines()))
    print()
    return dict(run=name, best_step=best_step, best_selection=float(selection.max()))


if __name__ == '__main__':
    args = sys.argv[1:]
    short = '--brief' in args
    runs = [a for a in args if a != '--brief'] or \
        sorted(os.path.dirname(p) for p in glob.glob(os.path.join(ROOT, 'runs', '*', 'metrics.csv')))
    if short:
        rows = [r for r in (brief(run) for run in runs) if r]
        print(pd.DataFrame(rows).to_string(index=False) if rows else 'no runs with metrics yet')
    else:
        results = [r for r in (summarize(run) for run in runs) if r]
        if len(results) > 1:
            print(pd.DataFrame(results).sort_values('best_selection', ascending=False).to_string(index=False))
