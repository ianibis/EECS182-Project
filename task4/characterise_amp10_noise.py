"""Characterise the variance of the added 10x term, assuming it is 0-centred.

The semi-synthetic recording is `recorded + (scale - 1) * artifact` with scale = 10, so the
term the decoder did not have to cope with is

    n_c(t) = (scale - 1) * artifact_c(t)

on channel c, zero outside stimulation segments. This script treats n as a noise term,
removes its per-channel mean (the "0-centred" assumption), and asks what is left:

    dc            per-channel mean of n; an intercept absorbs it, so it is not the problem
    ac_rms        standard deviation of n - mean(n), the part that cannot be absorbed
    between_phase variance of the phase-conditioned mean waveform: the periodic part
    within_phase  mean of the phase-conditioned variances: amplitude drift around the template
    harmonic      fraction of the AC variance at k * f0 for k = 1..10
    neural_rms    standard deviation of the cleaned LFP over the same samples
    snr           ac variance over neural variance, per channel then median
    mad_ratio     MAD of the 10x recording over the MAD of the recorded one, per channel

A pure sine or a fixed periodic waveform has within_phase ~ 0: its variance is entirely
between-phase, i.e. deterministic given the AM phase, not random. Reporting a single
Gaussian sigma for it would therefore be right about the magnitude and wrong about the
structure, which is what the decoder cares about.

    python task4/characterise_amp10_noise.py
    python task4/characterise_amp10_noise.py --split val

Writes task4/results/amp10_noise_variance.csv (per session),
task4/results/amp10_noise_summary.csv (per frequency) and results/fig_amp10_noise.png.
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils as U  # noqa: E402

PHASE_BINS = 64      # bins around one AM cycle
HARMONICS = 10       # k = 1..HARMONICS at k * f0
DECIMATE = 10        # 1 kHz -> 100 Hz is far above the 20 Hz artifact


def analyse(session):
    with h5py.File(os.path.join(U.DATA_DIR, 'export_outputs', 'h5_export_outputs',
                               f'{session}.h5')) as export:
        keep = export['channel_keep_fixed'][:].astype(bool)
        on = export['stim/on'][:].astype(bool)[::DECIMATE]
        recorded = np.asarray(export['lfp'][keep][:, ::DECIMATE], np.float64)
    with h5py.File(os.path.join(U.PARRM_DIR, f'{session}.h5')) as cleaned:
        artifact = np.asarray(cleaned['artifact'][keep][:, ::DECIMATE], np.float64)
        clean = np.asarray(cleaned['lfp'][keep][:, ::DECIMATE], np.float64)
        phase = np.asarray(cleaned['stim/phase_model'][::DECIMATE], np.float64)

    noise = (U.SEMISYNTHETIC_SCALE - 1.0) * artifact
    if not on.any():
        raise ValueError(f'{session}: no stimulation-on samples')
    noise, clean, phase, recorded = noise[:, on], clean[:, on], phase[on], recorded[:, on]

    # 0-centred: drop the per-channel mean, which any intercept already absorbs.
    dc = noise.mean(axis=1)
    centred = noise - dc[:, None]
    ac_variance = centred.var(axis=1)

    # Split the AC variance into the periodic part and the drift around it.
    wrapped = np.mod(phase, 2 * np.pi) / (2 * np.pi)
    which = np.minimum((wrapped * PHASE_BINS).astype(int), PHASE_BINS - 1)
    counts = np.bincount(which, minlength=PHASE_BINS).astype(float)
    means = np.stack([np.bincount(which, weights=centred[channel], minlength=PHASE_BINS)
                      / np.maximum(counts, 1) for channel in range(centred.shape[0])])
    between = means.var(axis=1)
    within = np.stack([np.bincount(which, weights=(centred[channel] - means[channel][which]) ** 2,
                                  minlength=PHASE_BINS) / np.maximum(counts, 1)
                       for channel in range(centred.shape[0])]).mean(axis=1)

    # Harmonic content: the artifact repeats at f0, so project onto k * f0.
    sample_rate = 1000.0 / DECIMATE
    f0 = float(np.median(np.abs(np.diff(phase)))) * sample_rate / (2 * np.pi)
    harmonic_variance = np.zeros(centred.shape[0])
    if f0 > 0.1:
        time = np.arange(centred.shape[1]) / sample_rate
        for k in range(1, HARMONICS + 1):
            angle = 2 * np.pi * k * f0 * time
            basis = np.stack([np.cos(angle), np.sin(angle)])
            coefficients = (centred @ basis.T) / centred.shape[1] * 2
            harmonic_variance += np.square(coefficients).sum(axis=1) / 2

    neural_variance = clean.var(axis=1)
    mad_recorded = np.median(np.abs(recorded - np.median(recorded, axis=1, keepdims=True)), axis=1)
    mad_amplified = np.median(np.abs(recorded + noise - np.median(recorded + noise, axis=1,
                                                                  keepdims=True)), axis=1)

    # What reaches the decoder: average the term into the 50 ms bins the models see.
    # A 20 Hz artifact has exactly one period per bin, so the bin mean keeps only the
    # slow gain drift; a 3 Hz artifact sweeps 1/6.7 of a cycle per bin, so most of it
    # survives as within-trial structure.
    samples_per_bin = max(int(round(U.BIN_S * sample_rate)), 1)
    usable = (centred.shape[1] // samples_per_bin) * samples_per_bin
    binned = centred[:, :usable].reshape(centred.shape[0], -1, samples_per_bin).mean(axis=2)
    binned_variance = binned.var(axis=1)
    if binned.shape[1] > 1:
        centred_bins = binned - binned.mean(axis=1, keepdims=True)
        lag1 = ((centred_bins[:, 1:] * centred_bins[:, :-1]).mean(axis=1)
                / np.maximum(binned_variance, 1e-30))
    else:
        lag1 = np.zeros(centred.shape[0])

    return dict(
        session=session, am_freq=float(U.SESSION_TABLE.loc[session, 'am_freq_hz']),
        channels=int(centred.shape[0]), samples=int(centred.shape[1]),
        f0_hz=f0,
        dc_rms=float(np.sqrt(np.mean(dc ** 2))),
        ac_rms=float(np.sqrt(np.median(ac_variance))),
        ac_rms_p10=float(np.sqrt(np.percentile(ac_variance, 10))),
        ac_rms_p90=float(np.sqrt(np.percentile(ac_variance, 90))),
        between_phase_fraction=float(np.median(between / np.maximum(ac_variance, 1e-30))),
        within_phase_fraction=float(np.median(within / np.maximum(ac_variance, 1e-30))),
        harmonic_fraction=float(np.median(harmonic_variance / np.maximum(ac_variance, 1e-30))),
        top5_channel_share=float(np.sort(ac_variance)[-5:].sum() / ac_variance.sum()),
        binned_variance_fraction=float(np.median(binned_variance / np.maximum(ac_variance, 1e-30))),
        binned_lag1_autocorrelation=float(np.median(lag1)),
        binned_rms=float(np.sqrt(np.median(binned_variance))),
        binned_snr_db=float(10 * np.log10(np.median(binned_variance)
                                          / np.median(neural_variance))),
        neural_rms=float(np.sqrt(np.median(neural_variance))),
        snr_db=float(10 * np.log10(np.median(ac_variance) / np.median(neural_variance))),
        mad_ratio=float(np.median(mad_amplified / mad_recorded)),
    )


def figure(table, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Attach the EEGNet ON score so the characterisation can be read against the failure.
    scores = pd.read_parquet(os.path.join(U.RESULTS_DIR, 'results_eegnet.parquet'))
    eegnet = (scores[scores.method == U.EEGNET_METHOD].groupby('session').r2_ave.mean()
              .rename('eegnet_r2'))
    table = table.join(eegnet, on='session')

    figure_axes, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    for frequency, colour in ((3.0, '#c44e52'), (20.0, '#4c72b0')):
        part = table[table.am_freq == frequency]
        if part.empty:
            continue
        label = f'{frequency:g} Hz AM'
        axes[0].scatter(part.binned_rms, part.eegnet_r2, color=colour, label=label)
        axes[1].scatter(part.binned_variance_fraction, part.binned_lag1_autocorrelation,
                        color=colour, label=label)
    axes[0].axhline(0, color='black', lw=0.8)
    axes[0].set_xlabel('RMS of the 10x term after 50 ms binning')
    axes[0].set_ylabel('EEGNet query $R^2$')
    axes[0].set_title('Variance that reaches the decoder', fontsize=9)
    axes[1].set_xlabel('fraction of AC variance kept by 50 ms binning')
    axes[1].set_ylabel('lag-1 autocorrelation of the binned term')
    axes[1].set_title('3 Hz stays oscillatory, 20 Hz collapses to drift', fontsize=9)
    for axis in axes:
        axis.legend(fontsize=8)
    figure_axes.tight_layout()
    figure_axes.savefig(path, dpi=150)
    plt.close(figure_axes)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', choices=('test', 'val'), default='test')
    args = parser.parse_args()

    sessions = U.sessions_in_split(args.split, U.STIMULATED_ON)
    rows = []
    for session in sessions:
        rows.append(analyse(session))
        print(f'{session}: done', flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(U.TASK4_RESULTS, 'amp10_noise_variance.csv'), index=False)

    summary = (table.groupby('am_freq')
               [['dc_rms', 'ac_rms', 'ac_rms_p10', 'ac_rms_p90', 'between_phase_fraction',
                 'within_phase_fraction', 'harmonic_fraction', 'top5_channel_share',
                 'binned_variance_fraction', 'binned_lag1_autocorrelation',
                 'binned_rms', 'binned_snr_db', 'neural_rms', 'snr_db', 'mad_ratio']]
               .median().round(4).reset_index())
    summary.to_csv(os.path.join(U.TASK4_RESULTS, 'amp10_noise_summary.csv'), index=False)

    figure(table, os.path.join(U.RESULTS_DIR, 'fig_amp10_noise.png'))

    print('\nper session:')
    print(table.round(4).to_string(index=False))
    print('\nmedian by AM frequency:')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
