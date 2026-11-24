"""Task 5.1.3: does the simulated artifact look like the real one?

Compares the real PARRM artifact estimate with the generator's output on TRAINING-day
stimulation sessions only, so this check cannot leak anything about held-out days.

Writes task5_results/simulation_check.csv and task5_results/fig_simulation_check.png.
Run from the repository root:  python task5/task5_check_simulation.py
"""

import os
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task3_common as C
import task5_data as data
import task5_plot_style as style
from task5_shift import TemplateLibrary, apply_shift, sample_shift

FREQUENCIES = np.fft.rfftfreq(cfg.N_SAMPLES, 1 / 1000)
HANN = np.hanning(cfg.N_SAMPLES)


def power_spectrum(signal):
    """Mean power spectrum over trials and channels of a (trials, channels, samples) array."""
    centered = signal - signal.mean(axis=2, keepdims=True)
    power = np.abs(np.fft.rfft(centered * HANN, axis=2)) ** 2 / (1000 * (HANN ** 2).sum())
    return power.mean(axis=(0, 1))


def channel_correlation(signal):
    """Mean absolute correlation between channels, averaged over trials."""
    values = []
    for trial in signal[:40]:
        matrix = np.corrcoef(trial)
        values.append(np.nanmean(np.abs(matrix[np.triu_indices_from(matrix, k=1)])))
    return float(np.mean(values))


def harmonic_prominence(spectrum, frequency):
    """Power at the AM frequency divided by the median power of nearby bins."""
    index = int(np.argmin(np.abs(FREQUENCIES - frequency)))
    near = np.abs(FREQUENCIES - frequency) < max(2.0, frequency * 0.4)
    away = np.abs(FREQUENCIES - frequency) > 1.0
    return float(spectrum[max(index - 1, 0):index + 2].max() / np.median(spectrum[near & away]))


def non_harmonic_fraction(spectrum, frequency):
    """Share of 0.8-150 Hz power that lies away from the AM harmonics."""
    distance = np.abs(FREQUENCIES / frequency - np.round(FREQUENCIES / frequency)) * frequency
    harmonic = (distance < 0.85) & (FREQUENCIES >= frequency - 1)
    band = (FREQUENCIES >= 0.8) & (FREQUENCIES <= 150)
    return float(spectrum[band & ~harmonic].sum() / spectrum[band].sum())


def real_artifacts(library, is_3hz):
    names = [n for n, three in zip(library.names, library.is_3hz.tolist())
             if three == is_3hz and C.has_cm_velocity(n)]
    return np.concatenate([data.load_session_arrays(n)['artifact'][:12] for n in names])


def simulated_artifacts(library, config, is_3hz, n_trials=240):
    generator = torch.Generator().manual_seed(11)
    output = []
    collected = 0
    while collected < n_trials:
        shift = sample_shift(library, config, 16, 1, generator)
        zero = torch.zeros(16, 1, cfg.N_CHANNELS, cfg.N_SAMPLES)
        wave = apply_shift(zero, shift, library, config)[:, 0]
        chosen = library.is_3hz[shift['template_index']] == is_3hz
        output.append(wave[chosen].numpy())
        collected += int(chosen.sum())
    return np.concatenate(output)[:n_trials]


def main():
    style.apply_style()
    library = TemplateLibrary('cpu')
    base = replace(cfg.ShiftConfig(), amplitude_min=1.0, amplitude_max=1.0, use_dropout=False)
    rows, panels = [], []

    for is_3hz, label, frequency in ((True, '3 Hz', 2.997), (False, '20 Hz', 19.891)):
        real = real_artifacts(library, is_3hz)
        unit = simulated_artifacts(library, base, is_3hz)
        real_rms, unit_rms = float(np.sqrt((real ** 2).mean())), float(np.sqrt((unit ** 2).mean()))
        match = real_rms / unit_rms                     # amplitude at which simulation matches real RMS
        matched = unit * match
        real_power, sim_power = power_spectrum(real), power_spectrum(matched)
        rows.append(dict(am=label, real_rms_uv=real_rms, simulated_rms_at_amplitude_1_uv=unit_rms,
                         amplitude_matching_real=match, amplitude_for_10x=10 * match,
                         real_peak_prominence=harmonic_prominence(real_power, frequency),
                         simulated_peak_prominence=harmonic_prominence(sim_power, frequency),
                         real_non_harmonic_fraction=non_harmonic_fraction(real_power, frequency),
                         simulated_non_harmonic_fraction=non_harmonic_fraction(sim_power, frequency),
                         real_channel_correlation=channel_correlation(real),
                         simulated_channel_correlation=channel_correlation(matched)))
        panels.append((label, real_power, sim_power))

    table = pd.DataFrame(rows)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    table.to_csv(os.path.join(cfg.RESULTS_DIR, 'simulation_check.csv'), index=False)
    print(table.T.to_string(float_format=lambda v: f'{v:.3f}'))

    figure, axes = style.plt.subplots(1, 2, figsize=(10, 3.8))
    shown = (FREQUENCIES >= 1) & (FREQUENCIES <= 150)
    for axis, (label, real_power, sim_power) in zip(axes, panels):
        axis.loglog(FREQUENCIES[shown], real_power[shown], color=style.BLUE, label='Real PARRM artifact')
        axis.loglog(FREQUENCIES[shown], sim_power[shown], color=style.ORANGE, label='Simulated (RMS matched)')
        axis.set_title(f'{label} stimulation')
        axis.set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Artifact power (µV²/Hz)')
    axes[0].legend(loc='lower left')
    figure.tight_layout()
    figure.savefig(os.path.join(cfg.RESULTS_DIR, 'fig_simulation_check.png'))


if __name__ == '__main__':
    main()
