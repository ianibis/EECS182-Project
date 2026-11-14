"""Checks the trial cache (plan 5.2a) against the loader and splits.json."""
import os

import numpy as np
import pytest

import EECS182_NHP_data_loader as L
from icl import config as C
from icl.cache import load_cached, load_index, target_bins

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(C.cache_dir(), 'index.json')),
                                reason='run `python -m icl.cache` first')

TRAIN_OFF = 'Butters-250602-100529'      # training no-stim session
VAL_STIM = 'Butters-250606-104043'       # validation 3 Hz session


@pytest.fixture(scope='module')
def splits():
    return L.load_splits(C.paths())


def test_index_has_every_session_with_cm_velocity():
    index = load_index()
    assert len(index) == 67
    assert not [s for s in index if '-250711-' in s]
    assert all(meta['n_trials'] == 50 for meta in index.values())


@pytest.mark.parametrize('session', [TRAIN_OFF, VAL_STIM])
def test_cache_matches_loader_window(session):
    """Cache bins 40..79 equal the loader's standard window (t_go - 0.5 to + 1.5 s)."""
    data = load_cached(session)
    paths = C.paths()
    sess = L.load_session(session, paths)
    kin = L.load_kinematics(session, paths)
    examples = L.trial_examples(sess, kin, align='t_go', pre=0.5, post=1.5)
    assert len(examples) == len(data['trial'])
    window = target_bins()
    samples = slice(window.start * C.SAMPLES_PER_BIN, window.stop * C.SAMPLES_PER_BIN)
    for i, example in enumerate(examples):
        assert example['trial'] == data['trial'][i]
        assert example['bin_id'][0] == data['bin0'][i] + C.TARGET_START_BIN
        np.testing.assert_array_equal(example['lfp'].reshape(C.N_CHANNELS, -1), data['lfp'][i][:, samples])
        np.testing.assert_array_equal(example['kin_valid'], data['valid'][i, window])
        np.testing.assert_array_equal(np.nan_to_num(example['kin']), np.nan_to_num(data['kin'][i, window]))


def test_shapes_and_no_nan():
    data = load_cached(TRAIN_OFF)
    assert data['lfp'].shape == (50, C.N_CHANNELS, C.N_SAMPLES)
    assert data['kin'].shape == (50, C.N_CACHE_BINS, 2)
    assert data['valid'].shape == (50, C.N_CACHE_BINS)
    assert np.isfinite(data['lfp']).all()
    assert np.isfinite(data['kin'][data['valid']]).all()


def test_parts_follow_splits(splits):
    train = load_cached(TRAIN_OFF)
    assert set(train['trial'][train['part'] == 'val']) == set(splits['train_val_trials'][TRAIN_OFF])
    assert (train['part'] == 'train').sum() == 40

    val = load_cached(VAL_STIM)
    calibration = splits['calibration'][VAL_STIM]
    assert set(val['trial'][val['part'] == 'pool']) == set(calibration['pool'])
    assert set(val['trial'][val['part'] == 'query']) == set(calibration['query'])


def test_pool_trials_precede_query_trials():
    data = load_cached(VAL_STIM)
    assert data['t_go'][data['part'] == 'pool'].max() < data['t_go'][data['part'] == 'query'].min()


def test_stats_are_plausible():
    stats = C.load_stats()
    assert stats['n_sessions'] == 13
    assert all(10 < sd < 150 for sd in stats['vel_std'])
