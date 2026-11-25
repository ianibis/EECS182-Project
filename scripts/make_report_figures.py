"""Regenerate the task 4 figures that plot scores or raw artifact traces.

Each figure states in its own axis label what is averaged, so a reader never has to
reconstruct it from the caption:

  fig_perday_r2.png     per-day R2 for clean and stimulated input, three methods
  fig_ensemble.png      clean-input R2 for each seed, the mean of the five, and the ensemble
  fig_r2_vs_rmse.png    (a) measured and predicted vx over a 10 s window of one session,
                        (b) predicted against measured for that session with per-axis R2,
                        (c) per-day velocity RMSE against the constant-mean predictor
  fig_artifact_3hz.png  two rows: one second of raw LFP, then the cycle fold
  fig_artifact_20hz.png the same for a 20 Hz day, high-passed at 5 Hz in the fold

R2 is the average of the two per-axis scores,
    R2 = (R2_vx + R2_vy) / 2,
computed by task4/utils.py r2_per_target over every scored 50 ms bin, then averaged over
seeds, then over the sessions of a day, then over the five test days.

Run from the repository root:

    .venv/bin/python scripts/make_report_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'task4'))
import utils as U  # noqa: E402

FIGDIR = os.path.join(ROOT, 'reports', 'figures')
SCORES = os.path.join(ROOT, 'task4_results', 'task4_2_eegnet_off.csv')
SPLITS = json.load(open(os.path.join(ROOT, 'splits.json')))

SEEDS = [0, 1, 2, 3, 4]
COLORS = {'single seed': '#2f6fdb', '5-seed ensemble': '#e07b00',
          'EEGNet + synth. freq.': '#2e8b57'}
SEED_COLORS = ['#2f6fdb', '#e07b00', '#2e8b57', '#c62828', '#7b4fa8']
R2_LABEL = '$R^2$ (average of the $v_x$ and $v_y$ scores)'

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False})


def load_scores():
    frame = pd.read_csv(SCORES)
    frame['day'] = frame.session.str.split('-').str[1]
    return frame


def per_day(frame):
    return {label: frame[frame.method == method].groupby('day')[['r2_ave', 'r2_off_ave']].mean()
            for label, method in (('single seed', 'eegnet'),
                                  ('5-seed ensemble', 'eegnet_ensemble'),
                                  ('EEGNet + synth. freq.', 'eegnet_synthetic_frequencies'))}


# --------------------------------------------------------------------------- figure 4

def perday_figure():
    frame = load_scores()
    days = sorted(frame.day.unique())
    series = per_day(frame)
    width = 0.26
    x = np.arange(len(days))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, column, title in zip(axes, ('r2_off_ave', 'r2_ave'),
                                 ('(a) clean input (stimulation off)',
                                  '(b) stimulated input (stimulation on)')):
        for index, (label, table) in enumerate(series.items()):
            ax.bar(x + (index - 1) * width, table.loc[days, column].to_numpy(), width,
                   label=label, color=COLORS[label], edgecolor='white', linewidth=0.4)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(days)
        ax.set_xlabel('held-out test day')
        ax.set_title(title)
    axes[0].set_ylabel(R2_LABEL)
    axes[0].annotate('0 = the constant-mean predictor', (0.02, 0.03),
                     xycoords='axes fraction', fontsize=7.2, color='#57606a')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7.8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIGDIR, 'fig_perday_r2.png'), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- figure 5

def ensemble_figure():
    frame = load_scores()
    days = sorted(frame.day.unique())
    single = frame[frame.method == 'eegnet']
    ens = frame[frame.method == 'eegnet_ensemble'].groupby('day').r2_off_ave.mean().loc[days]
    mean_of_runs = single.groupby(['day', 'seed']).r2_off_ave.mean().unstack().loc[days]

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    x = np.arange(len(days))
    markers = ['o', 's', '^', 'v', 'D']
    for index, seed in enumerate(SEEDS):
        ax.scatter(x, mean_of_runs[seed].to_numpy(), s=30, marker=markers[index],
                   color=SEED_COLORS[index], edgecolor='white', linewidth=0.4,
                   label=f'run with seed {seed}', zorder=3)
    ax.plot(x, mean_of_runs.mean(axis=1), color='#333333', lw=1.4, zorder=2,
            label='mean of the five runs (trend line)')
    ax.plot(x, ens, color='black', lw=1.0, ls='--', marker='D', ms=6, zorder=4,
            label='5-seed ensemble (mean of the predictions)')
    ax.axhline(0, color='black', lw=0.8, label='0 = the constant-mean predictor')

    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.set_xlim(-0.5, len(days) - 0.5)
    ax.set_xlabel('held-out test day')
    ax.set_ylabel(R2_LABEL)
    ax.set_title('Clean input only (stimulation off)')
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 1, 0, 3, 4, 5, 6, 7]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=6.8,
               frameon=False)
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    fig.savefig(os.path.join(FIGDIR, 'fig_ensemble.png'), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- figure 6

def example_session_predictions():
    """Seed-0 predictions on clean input, for the best held-out day."""
    import torch

    frame = load_scores()
    best_day = frame[frame.method == 'eegnet'].groupby('day').r2_off_ave.mean().idxmax()
    session = sorted(s for s in U.sessions_in_split('test', U.STIMULATION_ON)
                     if s.split('-')[1] == best_day)[0]
    _, control_dataset = U.stimulation_on_test_pairs()[session]

    model = U.CausalEEGNet(temporal_filters_n=8, depth_multiplier=2, kernel_samples=500,
                           separable_samples=62, dropout=0.25)
    state = torch.load(os.path.join(U.CHECKPOINT_DIR, 'task4_2_eegnet_seed0.pt'),
                       map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    prediction = U.predict(model.eval(), control_dataset['features'])
    control_velocity = control_dataset['targets'][control_dataset['valid_bins']]
    predicted_velocity = prediction[control_dataset['valid_bins']]
    return session, control_velocity, predicted_velocity


def r2rmse_figure():
    session, targets, single = example_session_predictions()
    session_r2 = float(np.nanmean(U.r2_per_target(targets, single)))
    seconds = np.arange(len(targets)) * U.BIN_S

    fig, (trace_ax, scatter_ax, rmse_ax) = plt.subplots(
        1, 3, figsize=(9.8, 3.5), gridspec_kw={'width_ratios': [1.3, 1, 1.15]})

    # (a) a 10 s window, which holds about three reaches and stays legible
    window = seconds <= 10.0
    trace_ax.plot(seconds[window], targets[window, 0], color='#1f2328', lw=1.3,
                  label='measured $v_x$')
    trace_ax.plot(seconds[window], single[window, 0], color='#2f6fdb', lw=1.3,
                  label='predicted $v_x$')
    trace_ax.set_xlabel('time within one session (s)')
    trace_ax.set_ylabel('$v_x$ (cm/s)')
    trace_ax.set_title('(a) first 10 s of one held-out session', fontsize=9.5)
    trace_ax.legend(fontsize=7.2, frameon=False, ncol=2, loc='upper center',
                    bbox_to_anchor=(0.5, -0.22))

    # (b) predicted against measured for the whole session, with the per-axis scores
    per_axis = U.r2_per_target(targets, single)
    for column, (label, colour) in enumerate((('$v_x$', '#2f6fdb'), ('$v_y$', '#d9772a'))):
        scatter_ax.scatter(targets[:, column], single[:, column], s=9, alpha=0.5,
                           color=colour, lw=0,
                           label=f'{label}: $R^2$ {per_axis[column]:+.2f}')
    span = float(np.nanpercentile(np.abs(targets), 99.5)) * 1.15
    scatter_ax.plot([-span, span], [-span, span], color='#9aa0a6', lw=1.0, ls='--',
                    label='predicted = measured')
    scatter_ax.axhline(0, color='#c8ccd0', lw=0.8)
    scatter_ax.axvline(0, color='#c8ccd0', lw=0.8)
    scatter_ax.set_xlim(-span, span)
    scatter_ax.set_ylim(-span, span)
    scatter_ax.set_xlabel('measured velocity (cm/s)')
    scatter_ax.set_ylabel('predicted velocity (cm/s)')
    scatter_ax.set_title(f'(b) every scored bin, mean $R^2$ {session_r2:+.2f}', fontsize=9.5)
    scatter_ax.legend(fontsize=7.0, frameon=False, loc='upper left')

    # (c) per-day velocity RMSE against the constant-mean predictor, as in the day table
    days = sorted(load_scores().day.unique())
    rmse = {'mean predictor': [65.0, 61.0, 68.0, 63.0, 65.0],
            '5-seed ensemble': [65.6, 58.0, 63.2, 61.6, 64.5],
            'EEGNet + synth. freq.': [63.4, 58.8, 65.5, 62.0, 64.9]}
    colours = {'mean predictor': '#7a828a', '5-seed ensemble': '#2f6fdb',
               'EEGNet + synth. freq.': '#2e8b57'}
    x = np.arange(len(days))
    width = 0.26
    for index, (label, values) in enumerate(rmse.items()):
        rmse_ax.bar(x + (index - 1) * width, values, width, label=label,
                    color=colours[label], edgecolor='white', linewidth=0.4)
    rmse_ax.set_xticks(x)
    rmse_ax.set_xticklabels(days, rotation=30, fontsize=7.5)
    rmse_ax.set_xlabel('held-out test day')
    rmse_ax.set_ylabel('velocity RMSE (cm/s)')
    rmse_ax.set_ylim(50, 76)
    rmse_ax.set_title('(c) velocity error per day', fontsize=9.5)
    rmse_ax.legend(fontsize=6.8, frameon=False, loc='upper left', ncol=1)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(FIGDIR, 'fig_r2_vs_rmse.png'), dpi=200)
    plt.close(fig)
    return session, session_r2


# ------------------------------------------------------------------------ figures 7, 8

def highpass(signal, fs, cutoff_hz):
    """Subtract a boxcar mean of one cutoff period, which keeps content above cutoff_hz."""
    width = int(round(fs / cutoff_hz)) | 1
    kernel = np.ones(width) / width
    pad = width // 2
    return signal - np.convolve(np.pad(signal, (pad, pad), mode='reflect'), kernel,
                                mode='valid')


def folded_cycle(t, x, period):
    """Median and interquartile range of x at each phase of the modulation period.

    The median and IQR are used rather than the mean and standard deviation because a
    session holds a few transients of several thousand z. A mean over cycles is set by
    those transients, and the periodic component a decoder has to survive disappears
    underneath them.
    """
    bins = int(round(period * 1000))
    index = np.clip((np.mod(t, period) / period * bins).astype(int), 0, bins - 1)
    order = np.argsort(index, kind='stable')
    index, values = index[order], x[order]
    edges = np.searchsorted(index, np.arange(bins + 1))
    middle = np.full(bins, np.nan)
    low = np.full(bins, np.nan)
    high = np.full(bins, np.nan)
    for b in range(bins):
        block = values[edges[b]:edges[b + 1]]
        if block.size:
            middle[b] = np.median(block)
            low[b], high[b] = np.percentile(block, [25, 75])
    phase_ms = (np.arange(bins) + 0.5) / bins * period * 1000
    return phase_ms, middle, low, high


def artifact_panel(stimulated, frequency, out, highpass_hz=None):
    recording = U.lab_loader.load_session(stimulated, U.LOADER_PATHS, with_lfp=True,
                                          manifest=U.MANIFEST_TABLE)
    U.lab_loader.normalize(recording)
    reference = U.lab_loader.load_session(SPLITS['off_reference'][stimulated],
                                          U.LOADER_PATHS, with_lfp=True,
                                          manifest=U.MANIFEST_TABLE)
    U.lab_loader.normalize(reference)

    channel = int(np.argmax(np.nanmax(np.abs(recording.lfp), axis=1)))
    signal, clean = recording.lfp[channel], reference.lfp[channel]
    fs = 1 / np.median(np.diff(recording.t))
    on = recording.stim_on.astype(bool)
    measured = float(recording.attrs.get('am_freq_measured_hz', frequency) or frequency)
    period_s = 1.0 / measured

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 5.8))

    # (a) one second centred on the largest excursion, so no trace touches an edge
    t0 = recording.t[int(np.argmax(np.abs(signal)))] - 0.5
    on_window = (recording.t >= t0) & (recording.t <= t0 + 1.0)
    off_window = (reference.t >= t0) & (reference.t <= t0 + 1.0)
    top.plot(reference.t[off_window] - t0, clean[off_window][:on_window.sum()],
             color='#6b7280', lw=0.9, ls='--', label='unstimulated (same day)')
    top.plot(recording.t[on_window] - t0, signal[on_window], color='#c62828', lw=1.0,
             label='stimulated')
    low, high = np.nanmin(signal[on_window]), np.nanmax(signal[on_window])
    margin = 0.12 * (high - low)
    top.set_ylim(low - margin, high + margin)
    top.set_xlim(0, 1)
    top.set_xlabel('time (s)')
    top.set_ylabel('LFP (z)')
    top.set_title(f'(a) raw LFP, channel {channel}, one second', loc='left')
    top.legend(fontsize=7, frameon=False, loc='upper right')

    # (b) the whole stimulation recording folded at the modulation period
    trace = signal if highpass_hz is None else highpass(signal, fs, highpass_hz)
    phase_ms, middle, low, high = folded_cycle(recording.t[on], trace[on], period_s)
    clean_trace = clean if highpass_hz is None else highpass(clean, fs, highpass_hz)
    clean_phase, clean_middle, _, _ = folded_cycle(reference.t, clean_trace, period_s)

    bottom.fill_between(phase_ms, low, high, color='#c62828', alpha=0.18, lw=0)
    bottom.plot(phase_ms, middle, color='#c62828', lw=1.3, label='stimulated')
    bottom.plot(clean_phase, clean_middle, color='#6b7280', lw=0.9, ls='--',
                label='unstimulated (same day)')
    bottom.axhline(0, color='#c8ccd0', lw=0.8)
    bottom.set_xlim(0, 1000 / measured)
    low_y = float(np.nanmin(low))
    high_y = float(np.nanmax(high))
    margin = 0.05 * (high_y - low_y)
    bottom.set_ylim(low_y - margin, high_y + margin)
    bottom.set_xlabel('phase within one stimulation period (ms)')
    bottom.set_ylabel('LFP (z)' + (f', above {highpass_hz:g} Hz' if highpass_hz else ''))
    bottom.set_title(f'(b) median over {1000 / measured:.0f} ms cycles, '
                     f'band is the 25-75% range', loc='left')
    bottom.legend(fontsize=7, frameon=False, loc='upper right')

    fig.suptitle(f'{frequency:g} Hz AM · {stimulated} · channel {channel}', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, out), dpi=200)
    plt.close(fig)
    return channel, float(np.nanmax(middle) - np.nanmin(middle))


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    perday_figure()
    ensemble_figure()
    print('wrote per-day and ensemble figures')
    session, r2 = r2rmse_figure()
    print(f'wrote r2-vs-rmse figure from {session}, clean R2 {r2:+.3f}')
    channel, spread = artifact_panel('Butters-250613-110147', 3.0, 'fig_artifact_3hz.png')
    print(f'3 Hz  channel {channel}, folded spread {spread:.2f} z')
    channel, spread = artifact_panel('Butters-250616-102853', 20.0, 'fig_artifact_20hz.png',
                                     highpass_hz=5)
    print(f'20 Hz channel {channel}, folded spread {spread:.2f} z')


if __name__ == '__main__':
    main()
