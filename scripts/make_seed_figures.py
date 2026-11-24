"""Plots for 4.2.3 (and the tones appendix): validation curves and the seed spread.

Replaces the old ambiguous two-panel "baseline / synthetic tones" figure, which mixed
the tones experiment into the seeds section. Now:
  fig_val_curves.png   -- two panels, both about the five BASELINE runs only:
      left  = validation R2 after every training epoch, one line per run
      right = dot plot of the five final scores with mean and sd, the 0.011 claim
  fig_tones_curves.png -- the same left-panel style for the five tone-augmented runs,
      used in the appendix where tones are actually introduced.

Writes both into reports/figures/.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, 'task4', 'training_logs')
FIGDIR = os.path.join(ROOT, 'reports', 'figures')
SEEDS = [0, 1, 2, 3, 4]

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False})


def load(method):
    curves, bests, dots = [], [], []
    for seed in SEEDS:
        d = json.load(open(os.path.join(LOGS, f'{method}_seed{seed}.json')))
        log = d['training_log']
        curves.append(([e['epoch'] for e in log], [e['val_r2'] for e in log]))
        bests.append(d['best_val_r2'])
        dots.append((d['best_epoch'], d['best_val_r2']))
    return curves, np.array(bests), dots


def curves_panel(ax, curves, dots, bests, color, title, seed_labels=True):
    for (ep, r2), seed in zip(curves, range(len(curves))):
        ax.plot(ep, r2, lw=1.05, color=color, alpha=0.8,
                label=f'run with seed {seed}' if seed_labels else None)
    ax.plot([], [], 'o', ms=5, color='#c62828',
            label='epoch this run kept (early stopping)')
    mean, sd = bests.mean(), bests.std(ddof=1)
    ax.axhline(mean, color='#333333', lw=1, ls='--',
               label=f'mean of the five kept scores = {mean:.3f}')
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel('training epoch')
    ax.set_xlim(0)
    return mean, sd


def val_curves_figure():
    curves, bests, dots = load('eegnet')
    mean, sd = bests.mean(), bests.std(ddof=1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.9, 3.35),
                                   gridspec_kw={'width_ratios': [3, 2]})
    curves_panel(ax1, curves, dots, bests, '#2f5d8a',
                 'Each line is one full training run')
    ax1.set_ylabel('validation R² (150 held-off trials)')
    ax1.legend(fontsize=7, loc='lower right', framealpha=0.9)
    ax1.fill_between([0, 110], mean - sd, mean + sd, color='#333333', alpha=0.10, lw=0)

    ax2.scatter(np.full(5, 0) + 0.07 * (np.arange(5) - 2), bests, s=34,
                color='#2f5d8a', zorder=3)
    ax2.errorbar(0, mean, yerr=sd, fmt='_', ms=40, mew=1.5, color='black', capsize=6)
    ax2.set_xticks([])
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_title('The five kept scores, side by side')
    ax2.set_ylabel('best validation R²')
    ax2.annotate(f'mean {mean:.3f}', (0.34, mean), fontsize=8, va='center',
                 color='#333333')
    ax2.annotate(f'sd {sd:.3f}', (0.34, mean + sd), fontsize=8, va='bottom',
                 color='#333333')
    ax2.annotate(f'sd {sd:.3f}', (0.34, mean - sd), fontsize=8, va='top',
                 color='#333333')

    fig.text(0.5, 0.008,
             'Five runs of one configuration (500 ms kernel, dropout 0.25) on the same 560 '
             'trials; only the seed differs. Right: dots are the same five final scores; the '
             'bar is their mean, ±1 sd.',
             ha='center', fontsize=7.6, color='#57606a')
    fig.suptitle('How far the five seeds drift apart: validation R² during training, and the '
                 'five final scores', fontsize=9.8, y=1.0)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(os.path.join(FIGDIR, 'fig_val_curves.png'), dpi=150)
    plt.close(fig)
    return mean, sd


def tones_curves_figure():
    curves, bests, dots = load('eegnet_stimulation_tones')
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    mean, sd = curves_panel(ax, curves, dots, bests, '#d9772a',
                            'Each line is one full training run (tones in training)')
    ax.set_ylabel('validation R² (150 held-off trials)')
    ax.legend(fontsize=7, loc='lower right', framealpha=0.9)
    ax.fill_between([0, 75], mean - sd, mean + sd, color='#333333', alpha=0.10, lw=0)
    ax.axhline(0.1357, color='#2f5d8a', lw=1.2, ls=':')
    ax.annotate('no-tone runs: mean 0.136', (4, 0.1395), fontsize=7.6, color='#2f5d8a')
    fig.text(0.5, 0.01, 'Validation has no tones; the model is trained to ignore a signal '
             'absent there.', ha='center', fontsize=7.4, color='#57606a')
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(os.path.join(FIGDIR, 'fig_tones_curves.png'), dpi=150)
    plt.close(fig)
    return mean, sd


if __name__ == '__main__':
    m, s = val_curves_figure()
    tm, ts = tones_curves_figure()
    print(f'baseline {m:.4f} ± {s:.4f} | tones {tm:.4f} ± {ts:.4f}')
