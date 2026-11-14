"""Template-based shifts (plan 5.1.2). Needs template_library.h5 and the trial cache."""
import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.signal import welch

from icl import config as C
from icl.shifts import ShiftConfig, TemplateBank, apply_shift, periodic_resample, sample_shift, shift_rms

pytestmark = pytest.mark.skipif(not os.path.exists(C.TEMPLATE_LIBRARY), reason='template library not found')


@pytest.fixture(scope='module')
def split_of():
    with open(C.SPLITS) as file:
        splits = json.load(file)
    return {s: part for part in splits['sessions'] for s in splits['sessions'][part]}


@pytest.fixture(scope='module')
def train_bank():
    return TemplateBank.load(('train',))


@pytest.fixture(scope='module')
def eval_bank():
    return TemplateBank.load(('val', 'test'))


def sessions(bank):
    return [s for key in (3, 20) for s in bank.sessions[key]]


def test_training_bank_only_has_training_sessions(train_bank, eval_bank, split_of):
    assert {split_of[s] for s in sessions(train_bank)} == {'train'}
    assert {split_of[s] for s in sessions(eval_bank)} <= {'val', 'test'}
    assert not set(sessions(train_bank)) & set(sessions(eval_bank))


def test_bank_sizes_and_snr_cut(train_bank, eval_bank):
    assert {k: len(v) for k, v in train_bank.sessions.items()} == {3: 12, 20: 15}
    assert {k: len(v) for k, v in eval_bank.sessions.items()} == {3: 7, 20: 11}
    assert min(min(v) for v in train_bank.snr.values()) >= 1.5
    assert 'Butters-250529-102830' not in sessions(train_bank)          # SNR 1.46
    assert 'Butters-250606-104043' not in sessions(eval_bank)           # SNR 0.97


def test_fixed_channels_match_the_cache():
    from icl.cache import load_cached, load_index
    if not os.path.exists(os.path.join(C.cache_dir(), 'index.json')):
        pytest.skip('run `python -m icl.cache` first')
    session = next(iter(load_index()))
    assert tuple(load_cached(session)['channels']) == C.FIXED_CHANNELS


def test_resampling_keeps_template_rms(train_bank):
    for key in (3, 20):
        resampled = np.sqrt(np.mean(train_bank.waves[key].astype(np.float64) ** 2, axis=(1, 2)))
        np.testing.assert_allclose(resampled, train_bank.rms_uv[key], rtol=0.01)


def test_resampled_template_matches_original_shape():
    import h5py
    with h5py.File(C.TEMPLATE_LIBRARY, 'r') as file:
        group = file['Butters-250523-100308']                      # 20 Hz, 50 phase bins
        mean, phases = group['mean'][:], group['phase_bins'][:]
    up = periodic_resample(mean, phases, 256)
    back = periodic_resample(up, 2 * np.pi * np.arange(256) / 256, 50)
    original_grid = periodic_resample(mean, phases, 50)
    np.testing.assert_allclose(back, original_grid, atol=1e-4 * np.abs(mean).max())


@pytest.mark.parametrize('am', [3, 20])
def test_template_shift_has_requested_rms_and_peak(train_bank, am):
    rng = np.random.default_rng(am)
    shift = sample_shift(rng, bank=train_bank, kind='template', f_am=C.AM_FREQS_HZ[am], amp=2.0)
    assert shift.kind == 'template' and shift.source
    assert shift_rms(shift) == pytest.approx(2.0 * ShiftConfig().base_rms_uv[am], rel=1e-4)
    shift = replace(shift, drop=torch.zeros(C.N_CHANNELS, dtype=torch.bool), trial_jitter=0.0)
    y = apply_shift(torch.zeros(1, C.N_CHANNELS, 20000), shift, rng)[0].numpy()
    freqs, power = welch(y, fs=C.FS, nperseg=8000)
    assert abs(freqs[np.argmax(np.median(power, axis=0))] - C.AM_FREQS_HZ[am]) < 0.25


def test_template_shift_uses_only_matching_frequency(train_bank):
    rng = np.random.default_rng(0)
    for am in (3, 20):
        for _ in range(50):
            shift = sample_shift(rng, bank=train_bank, kind='template', f_am=C.AM_FREQS_HZ[am])
            assert set(shift.source) <= set(train_bank.sessions[am])


def test_mix_draws_both_kinds_and_sometimes_mixes_templates(train_bank):
    rng = np.random.default_rng(1)
    shifts = [sample_shift(rng, bank=train_bank) for _ in range(600)]
    template = [s for s in shifts if s.kind == 'template']
    assert abs(len(template) / len(shifts) - 0.5) < 0.08
    two = np.mean([len(s.source) == 2 for s in template])
    assert abs(two - ShiftConfig().p_template_mix) < 0.1
