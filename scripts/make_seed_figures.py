"""Plots for 4.2.3 (and the tones appendix): validation curves and the seed spread.

Replaces the old ambiguous two-panel "baseline / synthetic tones" figure, which mixed
the tones experiment into the seeds section. Now:
  fig_val_curves.png   -- two panels, validation R2 after every training epoch:
      left  = the five stimulation-off baseline runs
      right = the five tone-augmented (stimulation-facing) runs, with the
              stimulation-off mean as a dotted reference
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
RUN_COLORS = ['#2f6fdb', '#e07b00', '#2e8b57', '#c62828', '#7b4fa8']

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


def mean_and_running(curves, window=5):
    """Seed mean at each common epoch, plus its trailing running average."""
    n = min(len(r2) for _, r2 in curves)
    epochs = np.asarray(curves[0][0][:n], dtype=float)
    mean = np.asarray([r2[:n] for _, r2 in curves], dtype=float).mean(axis=0)
    if len(mean) < window:
        return epochs, mean, np.array([]), np.array([])
    running = np.convolve(mean, np.ones(window) / window, mode='valid')
    return epochs, mean, epochs[window - 1:], running


def curves_panel(ax, curves, dots, bests, color, title, label_runs=False):
    # One colour per run when the runs are named in the legend; a shared colour keeps the
    # panel quiet when the legend is deliberately collapsed to a single "individual seeds".
    for index, (epochs, r2) in enumerate(curves):
        ax.plot(epochs, r2, lw=0.85,
                color=RUN_COLORS[index] if label_runs else color,
                alpha=0.8 if label_runs else 0.35,
                label=f'run with seed {index}' if label_runs else
                      ('individual seeds' if index == 0 else None))
    epochs, mean, run_epochs, running = mean_and_running(curves, 5)
    ax.plot(epochs, mean, lw=2.6, color='black', label='epoch-wise mean (seeds 0-4)')
    if len(run_epochs):
        ax.plot(run_epochs, running, lw=1.4, color='#333333', ls='--',
                label='5-epoch running mean')
    ax.plot([dot[0] for dot in dots], [dot[1] for dot in dots], 'o', ms=4.5,
            color='#c62828', ls='none', label='kept checkpoint')
    mean_best, sd = bests.mean(), bests.std(ddof=1)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel('training epoch')
    ax.set_xlim(0)
    return mean_best, sd


def val_curves_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True)

    curves, bests, dots = load('eegnet')
    mean, sd = bests.mean(), bests.std(ddof=1)
    curves_panel(axes[0], curves, dots, bests, '#2f5d8a',
                 '(a) stimulation-off training', label_runs=True)
    axes[0].set_ylabel('validation R² (150 held-off clean trials)')

    tone_curves, tone_bests, tone_dots = load('eegnet_stimulation_tones')
    curves_panel(axes[1], tone_curves, tone_dots, tone_bests, '#d9772a',
                 '(b) stimulation-facing, tone-augmented', label_runs=True)
    axes[1].axhline(mean, color='#2f5d8a', lw=1.1, ls=':',
                    label='stimulation-off mean (0.136)')

    for ax in axes:
        ax.set_xlim(0, 105)
        ax.set_ylim(-0.01, 0.16)

    handles, labels = axes[0].get_legend_handles_labels()
    handles += axes[1].get_legend_handles_labels()[0][-1:]
    labels += axes[1].get_legend_handles_labels()[1][-1:]
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=6.6,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 0.98))
    fig.savefig(os.path.join(FIGDIR, 'fig_val_curves.png'), dpi=150)
    plt.close(fig)
    return mean, sd


def tones_curves_figure():
    curves, bests, dots = load('eegnet_stimulation_tones')
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    mean, sd = curves_panel(ax, curves, dots, bests, '#d9772a',
                            'Tone-augmented runs: one colour per run', label_runs=True)
    ax.set_ylabel('validation R² (150 held-off trials)')
    ax.fill_between([0, 75], mean - sd, mean + sd, color='#333333', alpha=0.10, lw=0)
    ax.axhline(0.1357, color='#2f5d8a', lw=1.2, ls=':')
    ax.annotate('no-tone runs: mean 0.136', (12, 0.1385), fontsize=7.2, color='#2f5d8a')
    ax.set_ylim(-0.01, 0.15)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=6.6,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.28, 1, 0.94))
    fig.savefig(os.path.join(FIGDIR, 'fig_tones_curves.png'), dpi=150)
    plt.close(fig)
    return mean, sd


if __name__ == '__main__':
    m, s = val_curves_figure()
    tm, ts = tones_curves_figure()
    print(f'baseline {m:.4f} ± {s:.4f} | tones {tm:.4f} ± {ts:.4f}')
