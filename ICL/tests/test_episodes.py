"""Episode sampler and evaluation batches (plan 5.2b). Needs the trial cache."""
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

import icl.episodes as E
from icl import config as C
from icl.cache import load_cached
from icl.shifts import ShiftConfig

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(C.cache_dir(), 'index.json')),
                                reason='run `python -m icl.cache` first')

VAL_STIM = 'Butters-250606-104043'
NO_SHIFT = ShiftConfig(kind='harmonic', p_null=1.0, neural_gain_sd=0.0, max_drop_frac=0.0, trial_jitter_range=(0, 0))


@pytest.fixture(scope='module')
def train_store():
    return E.TrialStore(E.sessions_where('train', 0), parts=['train'])


@pytest.fixture(scope='module')
def splits():
    return E.load_splits()


def reference_window(session, trial_ids, offsets, pool_count):
    """Numpy re-implementation of window -> pool robust z, straight from the cache."""
    data = load_cached(session)
    row = {int(t): i for i, t in enumerate(data['trial'])}
    windows = []
    for t, o in zip(trial_ids, offsets):
        first = (C.TARGET_START_BIN + o - C.MODEL_HISTORY_BINS) * C.SAMPLES_PER_BIN
        windows.append(data['lfp'][row[int(t)]][:, first:first + C.MODEL_SAMPLES])
    x = np.stack(windows).astype(np.float64)
    pool = x[:pool_count].transpose(1, 0, 2).reshape(C.N_CHANNELS, -1)
    lower_median = lambda a: np.quantile(a, 0.5, axis=1, method='lower')    # torch.median's convention
    median = lower_median(pool)
    scale = 1.4826 * lower_median(np.abs(pool - median[:, None]))
    return (x - median[:, None]) / scale[:, None], data, row


def test_training_store_is_training_no_stim_only(train_store):
    index = E.load_index()
    assert len(train_store) == 13
    assert all(index[s]['split'] == 'train' and index[s]['stim'] == 0 for s in train_store.sessions)
    assert all(set(train_store.part[s]) == {'train'} for s in train_store.sessions)


def test_sample_shapes_and_query_outside_pool(train_store):
    sampler = E.EpisodeSampler(train_store, E.EpisodeConfig(shift=replace(NO_SHIFT)), seed=0)
    batch = sampler.sample(4)
    n = sampler.cfg.n_pool + sampler.cfg.n_extra
    assert batch['x'].shape == (4, n, C.N_CHANNELS, C.MODEL_SAMPLES)
    assert batch['y'].shape == (4, n, C.N_TARGET_BINS, 2)
    assert batch['valid'].shape == (4, n, C.N_TARGET_BINS)
    for meta in batch['meta']:
        assert len(set(meta['trial'])) == n                          # no trial twice
        assert meta['trial'][-1] not in meta['trial'][:sampler.cfg.n_pool]


def test_without_shift_x_is_the_pool_normalised_cache_window(train_store):
    cfg = E.EpisodeConfig(shift=NO_SHIFT, max_jitter_bins=0, clip=1e9)
    batch = E.EpisodeSampler(train_store, cfg, seed=1).sample(2)
    for b, meta in enumerate(batch['meta']):
        expected, _, _ = reference_window(meta['session'], meta['trial'], meta['offsets'], cfg.n_pool)
        np.testing.assert_allclose(batch['x'][b].numpy(), expected, rtol=1e-4, atol=1e-4)


def test_jitter_keeps_lfp_and_velocity_aligned(train_store):
    cfg = E.EpisodeConfig(shift=NO_SHIFT, clip=1e9)
    batch = E.EpisodeSampler(train_store, cfg, seed=2).sample(3)
    scaler = E.VelocityScaler()
    offsets = np.concatenate([m['offsets'] for m in batch['meta']])
    assert offsets.min() >= -C.MAX_JITTER_BINS and offsets.max() <= C.MAX_JITTER_BINS and offsets.std() > 3
    for b, meta in enumerate(batch['meta']):
        expected_x, data, row = reference_window(meta['session'], meta['trial'], meta['offsets'], cfg.n_pool)
        np.testing.assert_allclose(batch['x'][b].numpy(), expected_x, rtol=1e-4, atol=1e-4)
        for i, (t, o) in enumerate(zip(meta['trial'], meta['offsets'])):
            bins = slice(C.TARGET_START_BIN + o, C.TARGET_START_BIN + o + C.N_TARGET_BINS)
            valid = data['valid'][row[int(t)], bins]
            np.testing.assert_array_equal(batch['valid'][b, i].numpy(), valid)
            decoded = scaler.decode(batch['y'][b, i]).numpy()
            np.testing.assert_allclose(decoded[valid], data['kin'][row[int(t)], bins][valid], rtol=1e-5, atol=1e-4)
            assert (batch['y'][b, i][~torch.from_numpy(valid)] == 0).all()


def test_one_shift_per_episode(train_store, monkeypatch):
    calls = []
    original = E.sample_shift
    monkeypatch.setattr(E, 'sample_shift', lambda *a, **k: calls.append(1) or original(*a, **k))
    E.EpisodeSampler(train_store, E.EpisodeConfig(shift=replace(ShiftConfig(), kind='harmonic')), seed=3).sample(5)
    assert len(calls) == 5


def test_clipping(train_store):
    batch = E.EpisodeSampler(train_store, E.EpisodeConfig(shift=replace(ShiftConfig(), kind='harmonic'), clip=3.0),
                             seed=4).sample(2)
    assert batch['x'].abs().max().item() <= 3.0


def test_eval_batch_follows_splits(splits):
    store = E.TrialStore([VAL_STIM])
    calibration = splits['calibration'][VAL_STIM]
    for k, draw in ((0, 0), (5, 3), (20, 9)):
        batch = E.eval_batch(store, VAL_STIM, k, draw, splits)
        assert batch['x'].shape == (30, k + 1, C.N_CHANNELS, C.MODEL_SAMPLES)
        assert batch['query_trials'] == calibration['query']
        assert batch['context_trials'] == (calibration['subsets'][str(k)][draw] if k else [])
        assert not set(batch['context_trials']) & set(batch['query_trials'])


def test_eval_normalisation_comes_from_the_pool_only(splits):
    store = E.TrialStore([VAL_STIM])
    calibration = splits['calibration'][VAL_STIM]
    batch = E.eval_batch(store, VAL_STIM, 2, 0, splits, clip=1e9)
    expected, _, _ = reference_window(VAL_STIM, calibration['pool'] + calibration['query'],
                                      [0] * 50, pool_count=20)
    query_expected = expected[20:]
    np.testing.assert_allclose(batch['x'][:, -1].numpy(), query_expected, rtol=1e-4, atol=1e-4)


def test_stim_sessions_get_no_simulated_artifact():
    """train_stim: training-day stim sessions keep their real artifact; no simulated one on top."""
    stim = E.sessions_where('train', 1)[:2]
    store = E.TrialStore(E.sessions_where('train', 0)[:2] + stim, parts=['train'])
    batch = E.EpisodeSampler(store, E.EpisodeConfig(shift=replace(ShiftConfig(), kind='harmonic', p_null=0.0)), seed=4).sample(40)
    amps = {m['session'] in stim: [] for m in batch['meta']}
    for m in batch['meta']:
        amps[m['session'] in stim].append(m['shift'].amp)
    assert all(a == 0.0 for a in amps[True]) and all(a > 0 for a in amps[False])
