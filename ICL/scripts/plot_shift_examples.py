"""Example figure for the shift generator (plan 5.1.1 and 5.1.2).

Rows 1-2: one channel of one clean OFF trial next to a harmonic artifact (row 1) and a real
PARRM-template artifact (row 2), each at the real level (amp = 1) and at 5x.
Row 3: median-over-channels PSD of the 40 training trials: clean, and shifted with each kind at 5x.
Neural gain drift, dropout and trial jitter are switched off to isolate the artifact.

Run from ICL/:  .venv/bin/python scripts/plot_shift_examples.py
"""
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import welch

from icl import config as C
from icl.cache import load_cached
from icl.shifts import TemplateBank, apply_shift, sample_shift, shift_rms

SESSION = 'Butters-250602-100529'           # training no-stim session
OUT = os.path.join(C.RESULTS, 'figs', 'shift_examples.png')
INK, INK_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
AMP_COLORS = {1.0: '#6da7ec', 5.0: '#1c5cab'}                 # one-hue ordinal ramp for amplitude
KIND_COLORS = {'clean': INK_2, 'harmonic': '#1c5cab', 'template': '#eb6834'}
CHANNEL = 0

data = load_cached(SESSION)
clean = torch.from_numpy(data['lfp'][data['part'] == 'train'])          # (40, 35, 3500) microvolts
trial_ids = data['trial'][data['part'] == 'train']
t = np.arange(C.N_SAMPLES) / C.FS - C.PRE_S                              # seconds from go cue
SHOWN = (t >= -1.0) & (t <= 0.5)
TRIAL = int(clean[:, CHANNEL, SHOWN].abs().amax(dim=1).argmin())         # a trial without a large transient
bank = TemplateBank.load(('train',))


def isolate(shift):
    return replace(shift, neural_gain=torch.ones(C.N_CHANNELS), drop=torch.zeros(C.N_CHANNELS, dtype=torch.bool),
                   trial_jitter=0.0)


def best_template_shift(am):
    """Template shift built from the highest-SNR training template at this frequency (no mixing)."""
    i = int(np.argmax(bank.snr[am]))
    single = TemplateBank(waves={am: bank.waves[am][i:i + 1]}, sessions={am: [bank.sessions[am][i]]},
                          snr={am: [bank.snr[am][i]]}, rms_uv={am: [bank.rms_uv[am][i]]})
    rng = np.random.default_rng(am)
    shift = sample_shift(rng, bank=single, kind='template', f_am=C.AM_FREQS_HZ[am], amp=1.0)
    return isolate(shift), bank.snr[am][i]


plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK_2, 'axes.labelcolor': INK, 'xtick.color': INK_2,
                     'ytick.color': INK_2, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(3, 2, figsize=(10, 9.4), facecolor=SURFACE)

for column, am in enumerate((3, 20)):
    harmonic = isolate(sample_shift(np.random.default_rng(am), kind='harmonic', f_am=C.AM_FREQS_HZ[am], amp=1.0))
    template, snr = best_template_shift(am)
    titles = {'harmonic': f'{am} Hz AM, harmonic artifact',
              'template': f'{am} Hz AM, template of {template.source[0][8:]} (SNR {snr:.1f})'}

    for row, (kind, base) in enumerate((('harmonic', harmonic), ('template', template))):
        ax = axes[row, column]
        ax.plot(t[SHOWN], clean[TRIAL, CHANNEL, SHOWN], color=KIND_COLORS['clean'], lw=1.0,
                label=f'clean LFP (OFF trial {int(trial_ids[TRIAL])})')
        for amp in (1.0, 5.0):
            _, art = apply_shift(clean[TRIAL:TRIAL + 1], replace(base, amp=amp), np.random.default_rng(0),
                                 return_artifact=True)
            ax.plot(t[SHOWN], art[0, CHANNEL, SHOWN], color=AMP_COLORS[amp], lw=1.4,
                    label=f'artifact alone, amp = {amp:g} ({shift_rms(replace(base, amp=amp)):.1f} µV RMS)')
        limit = 1.15 * float(clean[TRIAL, CHANNEL, SHOWN].abs().max())
        ax.set_ylim(-limit, 1.6 * limit)                 # headroom above the traces for the legend
        ax.set_xlabel('time from go cue (s)')
        ax.set_ylabel(f'channel {int(data["channels"][CHANNEL])} (µV)')
        ax.set_title(titles[kind], loc='left', color=INK, fontsize=10)
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.legend(frameon=False, fontsize=8, loc='upper left')

    ax = axes[2, column]
    for kind, shift in (('harmonic', harmonic), ('template', template), ('clean', None)):   # clean on top
        x = clean if shift is None else apply_shift(clean, replace(shift, amp=5.0), np.random.default_rng(1))
        freqs, power = welch(x.numpy(), fs=C.FS, nperseg=2000, axis=-1)          # (trials, channels, f)
        median = np.median(power.mean(axis=0), axis=0)
        keep = (freqs >= 1) & (freqs <= 50)
        label = 'clean' if shift is None else f'{kind} artifact, amp = 5'
        ax.loglog(freqs[keep], median[keep], color=KIND_COLORS[kind], lw=1.0 if shift is None else 1.8, label=label)
    for harmonic_number in range(1, 4):
        if C.AM_FREQS_HZ[am] * harmonic_number <= 50:
            ax.axvline(C.AM_FREQS_HZ[am] * harmonic_number, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel('frequency (Hz); vertical lines at the AM frequency and harmonics')
    ax.set_ylabel('median PSD over channels (µV²/Hz)')
    ax.set_title(f'{am} Hz AM, all 40 training trials', loc='left', color=INK, fontsize=10)
    ax.grid(axis='y', which='major', color=GRID, lw=0.6)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=8, loc='lower left')

fig.suptitle(f'Simulated stimulation artifact on {SESSION} (no-stim, training split)', x=0.01, ha='left',
             color=INK, fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor=SURFACE)
print('saved', OUT)
