"""Draw the causal temporal convolution on one 1D LFP trial.

Emits reports/figures/causal_convolution.png. The top panel shows the full 2 s
trial with the 499 zero samples on the left and kernels at the first and last
output positions. The bottom panel zooms on the left edge so the padding and
the first three kernel positions are visible.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import EECS182_NHP_data_loader as L

OUT = os.path.join(ROOT, 'reports', 'figures', 'causal_convolution.png')
SESSION = 'Butters-250520-095355'
K1 = 500          # temporal kernel length in samples (500 ms at 1 kHz)
T = 2000          # one trial, 2.0 s at 1 kHz
PAD = K1 - 1      # left-only padding = 499 samples


def load_trial():
    paths = L.Paths(
        root=os.path.join(ROOT, 'data'),
        export_dir=os.path.join(ROOT, 'data', 'export_outputs', 'h5_export_outputs'),
        trial_dir=os.path.join(ROOT, 'data', 'export_outputs', 'trials'),
        kin_dir=os.path.join(ROOT, 'data', 'kinematics_v3', 'bins50ms'),
    )
    session = L.load_session(SESSION, paths, channels='fixed')
    L.normalize(session)
    trial = session.included.iloc[0]
    start = np.searchsorted(session.t, trial.t_go - 0.5)
    signal = session.lfp[0, start:start + T].astype(float)
    # A light 10 ms moving average keeps the hand-drawn line readable.
    signal = np.convolve(signal, np.ones(10) / 10, mode='same')
    return signal


def window_bar(ax, t, y, colour, label):
    ax.add_patch(Rectangle((t - PAD, y - 0.12), K1, 0.24, facecolor=colour,
                           edgecolor=colour, alpha=0.22, lw=0.8))
    ax.plot([t - PAD, t], [y, y], color=colour, lw=1.8)
    ax.plot([t - PAD, t - PAD], [y - 0.18, y + 0.18], color=colour, lw=1.0)
    ax.plot([t, t], [y - 0.18, y + 0.18], color=colour, lw=1.0)
    ax.text(t - PAD + 6, y + 0.25, label, ha='left', va='bottom',
            fontsize=7.8, color=colour, fontweight='bold')


def draw():
    signal = load_trial()
    xpad = np.concatenate([np.zeros(PAD), signal])
    x = np.arange(-PAD, T)
    pad_lo = min(-2.5, np.nanpercentile(xpad, 0.5))
    pad_hi = max(2.5, np.nanpercentile(xpad, 99.5))

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10.0, 5.6), gridspec_kw={'height_ratios': [1.0, 1.2]})

    # ---------------- full trial ----------------
    top.axvspan(-PAD, 0, color='#d9dde2', alpha=0.8, lw=0)
    top.plot(x, xpad, color='#1f2328', lw=0.85)
    top.axvline(0, color='#6e7781', lw=0.9, ls='--')
    top.set_xlim(-PAD - 10, T + 10)
    top.set_ylim(pad_lo - 3.4, pad_hi + 0.5)
    top.set_ylabel('LFP (z)')
    top.set_xlabel('sample (1 kHz; 0 = first LFP sample)')
    top.set_title('(a) one 2 s trial, left-padded before the first sample', loc='left')
    top.annotate('', xy=(-PAD, pad_hi - 0.2), xytext=(0, pad_hi - 0.2),
                 arrowprops=dict(arrowstyle='<->', color='#57606a', lw=1.0))
    top.text(-PAD / 2, pad_hi - 0.05, f'{PAD} zero samples', ha='center',
             fontsize=7.5, color='#57606a')
    top.text(6, pad_hi - 0.05, 'signal starts', ha='left', fontsize=7.5,
             color='#57606a')
    window_bar(top, 0, pad_lo - 1.2, '#c62828', 'y[0]')
    window_bar(top, T - 1, pad_lo - 2.2, '#2f6fdb', f'y[{T - 1}]')

    # ---------------- left-edge zoom ----------------
    left, right = -PAD - 20, 760
    keep = (x >= left) & (x <= right)
    bottom.axvspan(-PAD, 0, color='#d9dde2', alpha=0.8, lw=0)
    bottom.plot(x[keep], xpad[keep], color='#1f2328', lw=0.9)
    bottom.axvline(0, color='#6e7781', lw=0.9, ls='--')
    bottom.set_xlim(left, right)
    bottom.set_ylim(pad_lo - 5.2, pad_hi + 0.5)
    bottom.set_xlabel('sample')
    bottom.set_ylabel('padded input')
    bottom.set_title('(b) zoom on the left edge: the first three kernel positions',
                     loc='left')
    bottom.text(-PAD / 2, pad_hi - 0.05, f'{PAD} zeros', ha='center',
                fontsize=7.5, color='#57606a')
    bottom.text(6, pad_hi - 0.05, 'x[0]', ha='left', fontsize=7.5,
                color='#57606a')
    window_bar(bottom, 0, pad_lo - 1.5, '#c62828', 'y[0]')
    window_bar(bottom, 250, pad_lo - 3.0, '#e07b00', 'y[250]')
    window_bar(bottom, 499, pad_lo - 4.5, '#2f8f4e', 'y[499]')
    bottom.text(520, pad_lo - 1.0,
                'y[0]:   499 zeros + x[0]\n'
                'y[250]: 249 zeros + x[0..250]\n'
                'y[499]:   0 zeros + x[0..499]',
                ha='left', va='top', fontsize=7.2, color='#1f2328',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='#d0d7de', alpha=0.95))

    fig.suptitle(
        f'Causal temporal convolution: K1={K1} samples, left-only padding '
        f'K1-1={PAD};  y[t] = sum_k w[k] x_pad[t+k]',
        fontsize=10)
    fig.text(0.5, 0.005,
             'x_pad[s] = 0 for s < 0. Each kernel window ends at its output index t, '
             'so no output sees a future sample.', ha='center', fontsize=8,
             color='#57606a')
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print('wrote', OUT)


if __name__ == '__main__':
    draw()
