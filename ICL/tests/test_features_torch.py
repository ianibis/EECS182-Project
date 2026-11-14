"""features_torch must reproduce the precomputed features (plan 5.1.3)."""
import os

import h5py
import numpy as np
import pytest
import torch

from icl import config as C
from icl.cache import load_cached
from icl.features_torch import add_lags, trial_features

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(C.cache_dir(), 'index.json')),
                                reason='run `python -m icl.cache` first')


@pytest.mark.parametrize('session', ['Butters-250602-100529', 'Butters-250606-104043'])
@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_matches_precomputed_export_none(session, device):
    data = load_cached(session)
    first_bin = 19            # earliest bin with a full 1 s window (decoded bins 30+ have 9 lags after it)
    n_bins = C.N_CACHE_BINS - first_bin
    x = torch.from_numpy(data['lfp']).to(device)
    ours = trial_features(x, data['meta']['fs'], first_bin, n_bins).cpu().numpy()    # (trials, bins, C, 8)

    with h5py.File(os.path.join(C.DATA, 'features', 'export_none', f'{session}.h5'), 'r') as file:
        bin_id = file['bin_id'][:]
        reference = np.concatenate([file['lmp'][:][:, None], file['logpow'][:]], axis=1)   # (C, 8, B)
    position = {b: i for i, b in enumerate(bin_id)}
    for trial in range(len(data['trial'])):
        ids = data['bin0'][trial] + np.arange(first_bin, C.N_CACHE_BINS)
        expected = reference[:, :, [position[b] for b in ids]].transpose(2, 0, 1)            # (bins, C, 8)
        np.testing.assert_allclose(ours[trial, ..., 0], expected[..., 0], atol=2e-3, rtol=1e-4)   # LMP, uV
        np.testing.assert_allclose(ours[trial, ..., 1:], expected[..., 1:], atol=1e-3)           # log10 power


def test_add_lags_puts_lag_zero_first():
    features = torch.arange(5 * 2 * 1, dtype=torch.float32).reshape(5, 2, 1)   # bins 0..4, 2 channels
    lagged = add_lags(features, n_lags=2)                                       # bins 2..4
    assert lagged.shape == (3, 6)
    torch.testing.assert_close(lagged[0], torch.tensor([4., 5., 2., 3., 0., 1.]))
