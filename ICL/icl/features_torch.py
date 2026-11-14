"""LMP and band-power features on cached trial windows, in torch (plan 5.1.3).

Reproduces EECS182_NHP_features.compute_session on the cached windows, so features can be
recomputed after a simulated shift. For bin j of the cache, the features use samples up to and
including the last sample of the bin (sample 50 j + 49), exactly like the precomputed files.
"""
import numpy as np
import torch

import EECS182_NHP_features as F
from icl import config as C

FEATURE_NAMES = F.FEATURE_NAMES     # ['lmp', 'pow_1_4', ..., 'pow_130_200']


def _windows(x, n, first_bin, n_bins):
    """Windows of n samples ending at the last sample of bins first_bin .. first_bin + n_bins - 1."""
    start = C.SAMPLES_PER_BIN * (first_bin + 1) - n
    if start < 0:
        raise ValueError(f'a {n}-sample window needs bins >= {n // C.SAMPLES_PER_BIN - 1}')
    return x[..., start:start + (n_bins - 1) * C.SAMPLES_PER_BIN + n].unfold(-1, n, C.SAMPLES_PER_BIN)


def trial_features(x, fs, first_bin=C.TARGET_START_BIN, n_bins=C.N_TARGET_BINS):
    """x: (..., channels, samples) cached LFP in microvolts -> (..., n_bins, channels, 8).

    Feature 0 is the LMP (mean of the last 256 ms, microvolts); 1-7 are log10 mean PSD in the
    seven bands of EECS182_NHP_features (Hann window, line-noise ranges excluded).
    fs is the session's recorded sampling rate (cache meta 'fs'); it decides which FFT bins
    fall inside each band, exactly as in the precomputed features.
    """
    out = []
    out.append(_windows(x, int(round(C.LMP_WIN_S * fs)), first_bin, n_bins).mean(dim=-1))
    for low, high, window_s in F.BANDS:
        n = int(round(window_s * fs))
        window = torch.as_tensor(np.hanning(n), dtype=x.dtype, device=x.device)
        frequencies = np.fft.rfftfreq(n, 1 / fs)
        selected = (frequencies >= low) & (frequencies <= high)
        for start, stop in F.LINE_EXCLUDE:
            selected &= ~((frequencies >= start) & (frequencies <= stop))
        scale = 2.0 / (fs * float(np.sum(np.hanning(n) ** 2)))
        spectrum = torch.fft.rfft(_windows(x, n, first_bin, n_bins) * window, dim=-1)
        power = spectrum[..., torch.as_tensor(np.flatnonzero(selected), device=x.device)].abs().pow(2) * scale
        out.append(torch.log10(power.mean(dim=-1) + 1e-12))
    features = torch.stack(out, dim=-1)              # (..., channels, n_bins, 8)
    return features.transpose(-3, -2)                # (..., n_bins, channels, 8)


def add_lags(features, n_lags):
    """(..., bins, C, 8) with history -> (..., bins - n_lags, (n_lags + 1) * C * 8), lag 0 first."""
    flat = features.flatten(-2)
    n = flat.shape[-2] - n_lags
    return torch.cat([flat[..., n_lags - lag:n_lags - lag + n, :] for lag in range(n_lags + 1)], dim=-1)
