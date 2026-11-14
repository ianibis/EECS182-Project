"""The in-context decoder must never see a trial's own labels or future LFP (plan 5.3.1)."""
import pytest
import torch
import torch.nn as nn

from icl import config as C
from icl.encoder import EEGNetEncoder, WindowEncoder, token_windows
from icl.model import ICLDecoder, masked_mse

N_TRIALS, BINS = 4, C.N_TARGET_BINS


class TinyEncoder(WindowEncoder):
    """Test encoder: linear map of the whole 500 ms window, so every sample can matter."""
    def __init__(self, d=128):
        super().__init__()
        self.proj = nn.Linear(C.N_CHANNELS * C.TOKEN_WIN, d)

    def forward(self, w):
        return self.proj(w.flatten(1))


@pytest.fixture(scope='module')
def setup():
    torch.manual_seed(0)
    model = ICLDecoder(TinyEncoder(), max_trials=N_TRIALS).double().eval()
    x = torch.randn(1, N_TRIALS, C.N_CHANNELS, C.MODEL_SAMPLES, dtype=torch.float64)
    y = torch.randn(1, N_TRIALS, BINS, 2, dtype=torch.float64)
    valid = torch.ones(1, N_TRIALS, BINS, dtype=torch.bool)
    with torch.no_grad():
        base = model(x, y, valid)
    return model, x, y, valid, base


def test_output_shape(setup):
    model, x, y, valid, base = setup
    assert base.shape == (1, N_TRIALS, BINS, 2)


@pytest.mark.parametrize('j', range(N_TRIALS))
def test_labels_only_reach_later_trials(setup, j):
    model, x, y, valid, base = setup
    y2 = y.clone()
    y2[:, j] += 5.0
    with torch.no_grad():
        out = model(x, y2, valid)
    torch.testing.assert_close(out[:, :j + 1], base[:, :j + 1])          # trials <= j unchanged
    if j < N_TRIALS - 1:
        assert not torch.allclose(out[:, j + 1:], base[:, j + 1:])       # later trials do use them


@pytest.mark.parametrize('bin_index', [0, 17, BINS - 1])
def test_lfp_is_causal_within_a_trial(setup, bin_index):
    model, x, y, valid, base = setup
    j = 2
    last_sample = C.MODEL_SAMPLES - (BINS - 1 - bin_index) * C.SAMPLES_PER_BIN - 1    # last sample of this bin
    x2 = x.clone()
    x2[:, j, :, last_sample] += 10.0
    with torch.no_grad():
        out = model(x2, y, valid)
    torch.testing.assert_close(out[:, :j], base[:, :j])                         # earlier trials unchanged
    torch.testing.assert_close(out[:, j, :bin_index], base[:, j, :bin_index])   # earlier bins unchanged
    assert not torch.allclose(out[:, j, bin_index], base[:, j, bin_index])      # this bin sees it


def test_invalid_labels_are_ignored(setup):
    model, x, y, valid, base = setup
    valid2 = valid.clone()
    valid2[:, 0, 10:20] = False
    y_a, y_b = y.clone(), y.clone()
    y_a[:, 0, 10:20] = float('nan')
    y_b[:, 0, 10:20] = 123.0
    with torch.no_grad():
        torch.testing.assert_close(model(x, y_a, valid2), model(x, y_b, valid2))


def test_token_windows_end_at_bin_ends():
    x = torch.arange(C.MODEL_SAMPLES, dtype=torch.float32).expand(1, C.N_CHANNELS, -1)
    w = token_windows(x)
    assert w.shape == (1, BINS, C.N_CHANNELS, C.TOKEN_WIN)
    assert w[0, 0, 0, 0].item() == (C.MODEL_HISTORY_BINS - 9) * C.SAMPLES_PER_BIN   # 500 ms per token
    first_target_bin = C.MODEL_HISTORY_BINS
    for b in (0, BINS - 1):
        assert w[0, b, 0, -1].item() == (first_target_bin + b) * C.SAMPLES_PER_BIN + C.SAMPLES_PER_BIN - 1


def test_parameter_count_is_about_one_million():
    model = ICLDecoder(EEGNetEncoder(), max_trials=21)
    n = sum(p.numel() for p in model.parameters())
    assert 0.9e6 < n < 1.2e6                             # backbone ~1.0M + EEGNet ~33k


def test_masked_mse_ignores_invalid_bins():
    pred = torch.zeros(1, 1, 3, 2)
    y = torch.tensor([[[[1.0, 1.0], [float('nan'), 0.0], [1.0, 1.0]]]])
    valid = torch.tensor([[[True, False, True]]])
    assert masked_mse(pred, y, valid).item() == pytest.approx(1.0)


@pytest.mark.parametrize('bin_index', [0, 23, BINS - 1])
def test_eegnet_tokens_are_causal(bin_index):
    """In eval mode (fixed BatchNorm), changing a sample only affects tokens of bins ending at or after it."""
    torch.manual_seed(1)
    encoder = EEGNetEncoder().double().eval()
    x = torch.randn(2, C.N_CHANNELS, C.MODEL_SAMPLES, dtype=torch.float64)
    last_sample = C.MODEL_SAMPLES - (BINS - 1 - bin_index) * C.SAMPLES_PER_BIN - 1
    x2 = x.clone()
    x2[0, :, last_sample] += 10.0
    with torch.no_grad():
        a, b = encoder.tokens(x, BINS), encoder.tokens(x2, BINS)
    torch.testing.assert_close(b[:, :bin_index], a[:, :bin_index])
    torch.testing.assert_close(b[1], a[1])                                   # other trial untouched
    assert not torch.allclose(b[0, bin_index], a[0, bin_index])
