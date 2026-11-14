"""Simulated stimulation shifts for training the in-context decoder (plan 5.1).

A Shift is drawn once per episode and applied to every trial in that episode, context
and query alike. It has three parts:

* an artifact locked to the AM cycle (~3 or ~20 Hz) with its own gain and phase on each
  channel; each trial starts at a random point of the AM cycle, because stimulation runs
  continuously and is not synchronised with the go cue;
* a per-channel gain drift applied to the neural signal;
* channel dropout (dropped channels read zero).

Everything is in microvolts and applied before normalisation, as real stimulation would be.
`amp` scales the artifact relative to the real level (amp = 1 gives the median RMS of the
PARRM templates in the training sessions), so the same generator also serves the
semi-synthetic amplitude sweep.

Artifact waveforms come in two kinds: 'harmonic' (a random sum of AM harmonics, 5.1.1) and
'template' (a real PARRM template from a training-split stim session, 5.1.2). 'mix' picks
one of the two per episode. Templates of validation and test sessions live in a separate
bank that is only used for semi-synthetic evaluation.
"""
import json
import os
from dataclasses import dataclass, field, replace

import h5py
import numpy as np
import torch

from icl import config as C

N_PHASE = 256    # phase grid for one AM cycle: point j is phase 2*pi*j/N_PHASE, 0 = envelope peak


@dataclass
class ShiftConfig:
    am_freqs_hz: tuple = (C.AM_FREQS_HZ[3], C.AM_FREQS_HZ[20])
    kind: str = 'mix'                   # 'harmonic', 'template' or 'mix'
    p_template: float = 0.5             # in 'mix', probability of a template artifact
    # amp = 1, the real artifact level on the fixed 35 channels: the median training-template RMS.
    # 5.1.3 (results/amp_calibration.json) found the real AM-bin excess power matches each session's
    # own template at scale kappa ~ 1 (median 0.97 at 3 Hz, 1.03 at 20 Hz; per-session 0-2.7,
    # covered by amp_range).
    base_rms_uv: dict = field(default_factory=lambda: {3: 2.87, 20: 1.75})
    p_null: float = 0.1                 # probability of no artifact (amp = 0)
    amp_range: tuple = (0.25, 10.0)     # log-uniform, in multiples of the real level
    n_harm_range: tuple = (1, 5)        # harmonic kind: number of harmonics, inclusive
    harm_decay_range: tuple = (0.3, 0.8)
    chan_gain_sd: float = 0.5           # harmonic kind: log-normal sd of the per-channel gain
    chan_phase_sd: float = 0.3          # harmonic kind: rad, per-channel phase offset per harmonic
    p_template_mix: float = 0.5         # template kind: probability of mixing two templates
    template_chan_gain_sd: float = 0.2  # template kind: log-normal sd of the per-channel gain
    neural_gain_sd: float = 0.1         # log-normal sd of the per-channel neural gain drift
    max_drop_frac: float = 0.1          # dropout probability per channel ~ U(0, max_drop_frac)
    trial_jitter_range: tuple = (0.0, 0.6)  # per episode: sd of the relative amplitude across trials


@dataclass
class Shift:
    f_am: float
    wave: torch.Tensor          # (C, N_PHASE): artifact over one AM cycle at amp = 1 (microvolts)
    amp: float
    neural_gain: torch.Tensor   # (C,)
    drop: torch.Tensor          # (C,) bool
    trial_jitter: float
    kind: str = 'harmonic'
    source: tuple = ()          # template kind: session(s) the waveform came from

    @property
    def am_key(self):
        return 3 if self.f_am < 10 else 20

    def to(self, device):
        return replace(self, wave=self.wave.to(device), neural_gain=self.neural_gain.to(device),
                       drop=self.drop.to(device))


def null_shift(n_channels=C.N_CHANNELS, f_am=C.AM_FREQS_HZ[3]):
    """A shift that leaves the signal unchanged."""
    return Shift(f_am=f_am, wave=torch.zeros(n_channels, N_PHASE), amp=0.0,
                 neural_gain=torch.ones(n_channels), drop=torch.zeros(n_channels, dtype=torch.bool),
                 trial_jitter=0.0, kind='null')


def harmonic_wave(rng, n_channels, rms_uv, n_harm, decay, chan_gain_sd, chan_phase_sd):
    """One AM cycle per channel: g_c * sum_h decay^(h-1) cos(h phi + phi_h + delta_ch), RMS = rms_uv."""
    phi = 2 * np.pi * np.arange(N_PHASE) / N_PHASE
    harmonics = np.arange(1, n_harm + 1)
    amplitude = decay ** (harmonics - 1)
    phase = rng.uniform(0, 2 * np.pi, n_harm)[None] + rng.normal(0, chan_phase_sd, (n_channels, n_harm))
    gain = rng.lognormal(0, chan_gain_sd, n_channels)
    wave = np.einsum('h,chp->cp', amplitude,
                     np.cos(harmonics[None, :, None] * phi[None, None] + phase[..., None]))
    wave *= gain[:, None]
    wave *= rms_uv / np.sqrt(np.mean(wave ** 2))
    return wave.astype(np.float32)


def periodic_resample(values, phases, n_out=N_PHASE):
    """Trigonometric interpolation of a periodic signal onto n_out points at phase 2*pi*j/n_out.

    values (..., P) are samples at the uniform phases `phases` (rad, any start). Harmonics that
    either grid cannot hold (k >= min(P, n_out) / 2) are dropped.
    """
    values = np.asarray(values, np.float64)
    n_in = values.shape[-1]
    k = np.arange(min(n_in, n_out) // 2)
    coef = np.fft.rfft(values, axis=-1)[..., k] / n_in
    position = (2 * np.pi * np.arange(n_out) / n_out - phases[0]) * n_in / (2 * np.pi)   # in input bins
    basis = np.exp(2j * np.pi * np.outer(k, position) / n_in)                            # (K, n_out)
    weight = np.where(k == 0, 1.0, 2.0)
    return np.real((coef * weight) @ basis)


def _scalar(value):
    return np.asarray(value).ravel()[0].item()


@dataclass
class TemplateBank:
    """PARRM templates on the fixed 35 channels, resampled to N_PHASE points per AM cycle.

    TemplateBank.load(('train',)) is the training bank; ('val', 'test') is the evaluation bank.
    """
    waves: dict       # {3: (n, C, N_PHASE) float32, 20: ...}, microvolts
    sessions: dict    # {3: [session, ...], 20: [...]}
    snr: dict         # {3: [template_snr, ...], 20: [...]}
    rms_uv: dict      # {3: [RMS over the 35 channels, ...], 20: [...]}

    @classmethod
    def load(cls, splits=('train',), min_snr=1.5, path=None, splits_path=None):
        with open(splits_path or C.SPLITS) as file:
            split_json = json.load(file)
        split_of = {s: part for part in split_json['sessions'] for s in split_json['sessions'][part]}
        waves, sessions, snr, rms = ({3: [], 20: []} for _ in range(4))
        with h5py.File(path or C.TEMPLATE_LIBRARY, 'r') as file:
            for session in sorted(file):
                group = file[session]
                if split_of.get(session) not in splits or _scalar(group.attrs['template_snr']) < min_snr:
                    continue
                key = 3 if _scalar(group.attrs['am_freq_hz']) < 10 else 20
                ids = group['channel_ids'][:].astype(int)
                rows = [int(np.flatnonzero(ids == channel)[0]) for channel in C.FIXED_CHANNELS]
                mean = group['mean'][:][rows]
                waves[key].append(periodic_resample(mean, group['phase_bins'][:]).astype(np.float32))
                sessions[key].append(session)
                snr[key].append(_scalar(group.attrs['template_snr']))
                rms[key].append(float(np.sqrt(np.mean(mean.astype(np.float64) ** 2))))
        waves = {key: np.stack(value) if value else np.zeros((0, len(C.FIXED_CHANNELS), N_PHASE), np.float32)
                 for key, value in waves.items()}
        return cls(waves=waves, sessions=sessions, snr=snr, rms_uv=rms)

    def __len__(self):
        return sum(len(value) for value in self.sessions.values())


def template_wave(rng, bank, key, cfg):
    """A real template (sometimes a convex mix of two) with per-channel gain jitter, RMS = base_rms_uv."""
    waves = bank.waves[key]
    if not len(waves):
        raise ValueError(f'template bank has no {key} Hz templates')
    i = int(rng.integers(len(waves)))
    wave, source = waves[i].astype(np.float64), (bank.sessions[key][i],)
    if len(waves) > 1 and rng.random() < cfg.p_template_mix:
        j = int(rng.choice([index for index in range(len(waves)) if index != i]))
        alpha = rng.uniform()
        wave = alpha * wave + (1 - alpha) * waves[j]
        source += (bank.sessions[key][j],)
    wave = wave * rng.lognormal(0, cfg.template_chan_gain_sd, wave.shape[0])[:, None]
    wave *= cfg.base_rms_uv[key] / np.sqrt(np.mean(wave ** 2))
    return wave.astype(np.float32), source


def sample_shift(rng, cfg=None, bank=None, n_channels=C.N_CHANNELS, f_am=None, amp=None, kind=None):
    """Draw one Shift. f_am, amp and kind can be fixed, e.g. for tests and amplitude sweeps.

    Template shifts ('template', or 'mix' when it picks a template) need a TemplateBank.
    """
    cfg = cfg or ShiftConfig()
    kind = kind or cfg.kind
    if kind == 'mix':
        kind = 'template' if rng.random() < cfg.p_template else 'harmonic'
    if kind not in ('harmonic', 'template'):
        raise ValueError(f'unknown shift kind {kind}')
    if kind == 'template' and bank is None:
        raise ValueError('template shifts need a TemplateBank')

    f_am = float(rng.choice(cfg.am_freqs_hz)) if f_am is None else float(f_am)
    key = 3 if f_am < 10 else 20
    if amp is None:
        null = rng.random() < cfg.p_null
        amp = 0.0 if null else float(np.exp(rng.uniform(*np.log(cfg.amp_range))))
    if kind == 'harmonic':
        wave = harmonic_wave(
            rng, n_channels, cfg.base_rms_uv[key],
            n_harm=int(rng.integers(cfg.n_harm_range[0], cfg.n_harm_range[1] + 1)),
            decay=rng.uniform(*cfg.harm_decay_range),
            chan_gain_sd=cfg.chan_gain_sd, chan_phase_sd=cfg.chan_phase_sd)
        source = ()
    else:
        wave, source = template_wave(rng, bank, key, cfg)
    neural_gain = np.exp(rng.normal(0, cfg.neural_gain_sd, n_channels))
    drop = rng.random(n_channels) < rng.uniform(0, cfg.max_drop_frac)
    return Shift(f_am=f_am, wave=torch.from_numpy(wave), amp=float(amp),
                 neural_gain=torch.from_numpy(neural_gain.astype(np.float32)),
                 drop=torch.from_numpy(drop), trial_jitter=float(rng.uniform(*cfg.trial_jitter_range)),
                 kind=kind, source=source)


def artifact(shift, n_trials, n_samples, rng, device='cpu', fs=C.FS, dtype=torch.float32):
    """Artifact (N, C, T) for n_trials trials, each starting at a random point of the AM cycle."""
    start = torch.as_tensor(rng.random(n_trials), dtype=torch.float64, device=device)
    jitter = 1 + shift.trial_jitter * rng.standard_normal(n_trials)
    t = torch.arange(n_samples, dtype=torch.float64, device=device) / fs
    position = ((shift.f_am * t[None] + start[:, None]) % 1.0) * N_PHASE      # (N, T)
    i0 = position.floor().long() % N_PHASE
    weight = (position - position.floor()).to(dtype)[:, None]                  # (N, 1, T)
    wave = shift.wave.to(device=device, dtype=dtype)
    lower = wave[:, i0].transpose(0, 1)                                         # (N, C, T)
    upper = wave[:, (i0 + 1) % N_PHASE].transpose(0, 1)
    scale = torch.as_tensor(shift.amp * jitter, dtype=dtype, device=device)[:, None, None]
    return scale * (lower + (upper - lower) * weight)


def apply_shift(x, shift, rng, fs=C.FS, return_artifact=False):
    """x: (N, C, T) clean microvolts -> shifted microvolts (same shape, device and dtype)."""
    n_trials, n_channels, n_samples = x.shape
    art = artifact(shift, n_trials, n_samples, rng, x.device, fs, x.dtype)
    gain = shift.neural_gain.to(device=x.device, dtype=x.dtype)[None, :, None]
    drop = shift.drop.to(x.device)[None, :, None]
    y = (x * gain + art).masked_fill(drop, 0.0)
    return (y, art) if return_artifact else y


def shift_rms(shift):
    """RMS of the artifact at this shift's amplitude, over channels and one AM cycle (microvolts)."""
    return float(shift.amp * shift.wave.pow(2).mean().sqrt())
