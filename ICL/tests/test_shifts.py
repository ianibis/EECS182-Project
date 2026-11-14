"""Unit tests for the shift generator with harmonic artifacts (plan 5.1.1). No data needed."""
from dataclasses import replace
from functools import partial

import numpy as np
import pytest
import torch
from scipy.signal import welch

from icl import config as C
from icl.shifts import (N_PHASE, ShiftConfig, apply_shift, harmonic_wave, null_shift, periodic_resample,
                        sample_shift, shift_rms)

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
harmonic = partial(sample_shift, kind='harmonic')


def clean(n_trials=4, n_samples=C.N_SAMPLES, seed=0, device='cpu'):
    generator = torch.Generator().manual_seed(seed)
    return (30 * torch.randn(n_trials, C.N_CHANNELS, n_samples, generator=generator)).to(device)


@pytest.mark.parametrize('device', DEVICES)
def test_null_shift_is_identity(device):
    x = clean(device=device)
    y = apply_shift(x, null_shift(), np.random.default_rng(0))
    torch.testing.assert_close(y, x)


def test_identity_when_amp_zero_and_no_gain_or_drop():
    rng = np.random.default_rng(1)
    shift = harmonic(rng, amp=0.0)
    shift = replace(shift, neural_gain=torch.ones(C.N_CHANNELS), drop=torch.zeros(C.N_CHANNELS, dtype=torch.bool))
    x = clean()
    torch.testing.assert_close(apply_shift(x, shift, rng), x)


@pytest.mark.parametrize('rms', [0.5, 2.0])
def test_wave_rms_matches_request(rms):
    wave = harmonic_wave(np.random.default_rng(2), C.N_CHANNELS, rms, n_harm=4, decay=0.6,
                         chan_gain_sd=0.5, chan_phase_sd=0.3)
    assert np.sqrt(np.mean(wave.astype(np.float64) ** 2)) == pytest.approx(rms, rel=1e-5)


@pytest.mark.parametrize('am', [3, 20])
@pytest.mark.parametrize('device', DEVICES)
def test_psd_peak_at_am_frequency(am, device):
    rng = np.random.default_rng(3)
    f_am = C.AM_FREQS_HZ[am]
    shift = replace(harmonic(rng, f_am=f_am, amp=1.0), drop=torch.zeros(C.N_CHANNELS, dtype=torch.bool))
    x = torch.zeros(1, C.N_CHANNELS, 20000, device=device)
    y = apply_shift(x, shift, rng)[0].cpu().numpy()
    freqs, power = welch(y, fs=C.FS, nperseg=8000)
    peak = freqs[np.argmax(np.median(power, axis=0))]
    assert abs(peak - f_am) < 0.25


def test_artifact_rms_matches_amp():
    rng = np.random.default_rng(4)
    shift = replace(harmonic(rng, amp=3.0), trial_jitter=0.0)
    _, art = apply_shift(torch.zeros(2, C.N_CHANNELS, 20000), shift, rng, return_artifact=True)
    assert art.pow(2).mean().sqrt().item() == pytest.approx(shift_rms(shift), rel=0.02)
    assert shift_rms(shift) == pytest.approx(3.0 * ShiftConfig().base_rms_uv[shift.am_key], rel=1e-4)


def test_same_shift_gives_same_channel_profile_in_every_trial():
    """Trials differ only in the AM start phase: per-channel RMS profile is shared."""
    rng = np.random.default_rng(5)
    shift = replace(harmonic(rng, f_am=C.AM_FREQS_HZ[20], amp=1.0), trial_jitter=0.0)
    _, art = apply_shift(torch.zeros(6, C.N_CHANNELS, 20000), shift, rng, return_artifact=True)
    rms = art.pow(2).mean(dim=2).sqrt()                          # (trials, channels)
    torch.testing.assert_close(rms, rms[:1].expand_as(rms), rtol=0.02, atol=1e-6)
    assert not torch.allclose(art[0], art[1])                    # but the phase differs


def test_trial_jitter_changes_trial_amplitude():
    rng = np.random.default_rng(6)
    shift = replace(harmonic(rng, amp=1.0), trial_jitter=0.3)
    _, art = apply_shift(torch.zeros(20, C.N_CHANNELS, 20000), shift, rng, return_artifact=True)
    trial_rms = art.pow(2).mean(dim=(1, 2)).sqrt()
    assert (trial_rms.std() / trial_rms.mean()).item() > 0.1


def test_dropped_channels_are_zero_and_others_are_not():
    rng = np.random.default_rng(7)
    shift = harmonic(rng, amp=1.0)
    shift = replace(shift, drop=torch.arange(C.N_CHANNELS) % 7 == 0)
    y = apply_shift(clean(), shift, rng)
    assert (y[:, shift.drop] == 0).all()
    assert (y[:, ~shift.drop].abs().amax(dim=2) > 0).all()


def test_same_seed_reproduces():
    def run(seed):
        rng = np.random.default_rng(seed)
        return apply_shift(clean(), harmonic(rng), rng)
    torch.testing.assert_close(run(11), run(11))
    assert not torch.allclose(run(11), run(12))


def test_sampled_parameters_stay_in_range():
    cfg = ShiftConfig()
    rng = np.random.default_rng(8)
    shifts = [harmonic(rng, cfg) for _ in range(2000)]
    amps = np.array([s.amp for s in shifts])
    assert {s.f_am for s in shifts} == set(cfg.am_freqs_hz)
    assert np.mean(amps == 0) == pytest.approx(cfg.p_null, abs=0.03)
    assert amps[amps > 0].min() >= cfg.amp_range[0] and amps.max() <= cfg.amp_range[1]
    drop_frac = np.mean([s.drop.float().mean().item() for s in shifts])
    assert 0 < drop_frac < cfg.max_drop_frac


def test_trial_jitter_is_drawn_per_episode_in_range():
    cfg = ShiftConfig()
    rng = np.random.default_rng(9)
    jitters = np.array([harmonic(rng, cfg).trial_jitter for _ in range(500)])
    assert jitters.min() >= cfg.trial_jitter_range[0] and jitters.max() <= cfg.trial_jitter_range[1]
    assert jitters.std() > 0.1


def test_template_kind_without_bank_is_an_error():
    with pytest.raises(ValueError):
        sample_shift(np.random.default_rng(0), kind='template')


@pytest.mark.parametrize('n_in', [50, 334])
def test_periodic_resample_aligns_phase_zero_to_grid_point_zero(n_in):
    """Bin-centre phases from -pi to pi (as in the template library) map onto 2*pi*j/N_PHASE."""
    phases = -np.pi + (np.arange(n_in) + 0.5) * 2 * np.pi / n_in
    values = np.cos(phases) + 0.5 * np.sin(3 * phases)
    grid = 2 * np.pi * np.arange(N_PHASE) / N_PHASE
    np.testing.assert_allclose(periodic_resample(values, phases), np.cos(grid) + 0.5 * np.sin(3 * grid), atol=1e-10)


def test_periodic_resample_round_trip():
    rng = np.random.default_rng(10)
    phases = -np.pi + (np.arange(50) + 0.5) * 2 * np.pi / 50
    k = np.arange(1, 20)
    values = (rng.normal(size=(3, k.size, 1)) * np.cos(k[:, None] * phases + rng.uniform(0, 6, (k.size, 1)))).sum(axis=1)
    up = periodic_resample(values, phases, 256)
    back = periodic_resample(up, 2 * np.pi * np.arange(256) / 256 - 2 * np.pi, 50)   # same grid, shifted by a full cycle
    np.testing.assert_allclose(periodic_resample(values, phases, 50), back, atol=1e-10)
