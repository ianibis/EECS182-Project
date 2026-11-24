"""Unit tests for the improved decoders. They run on the CPU.

Run from the repository root:  CUDA_VISIBLE_DEVICES= python task5/test_task5_improve.py
"""

import torch

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
from task5_calibrated import CalibratedDecoder
from task5_improve import STEP1_SETTINGS, Setting, TimingDecoder, build_model, jitter_trials
from task5_model import count_parameters


def make_inputs(k=4):
    torch.manual_seed(1)
    context = torch.randn(1, k, cfg.N_CHANNELS, cfg.N_SAMPLES)
    velocity = torch.randn(1, k, cfg.N_BINS, 2)
    valid = torch.ones(1, k, cfg.N_BINS, dtype=torch.bool)
    query = torch.randn(1, 1, cfg.N_CHANNELS, cfg.N_SAMPLES)
    keep = torch.ones(1, cfg.N_CHANNELS, dtype=torch.bool)
    return context, velocity, valid, query, keep


@torch.no_grad()
def predict(model, context, velocity, valid, query, keep):
    context_features = model.encode(context, keep)
    query_features = model.encode(query, keep)[:, 0]
    return model.decode(context_features, velocity, valid, query_features)


def make_timing_model(timing='shift_gain'):
    torch.manual_seed(0)
    return TimingDecoder(cfg.ModelConfig(), timing).eval()


def test_untrained_timing_decoder_equals_calibrated_decoder():
    timing = make_timing_model()
    torch.manual_seed(0)
    plain = CalibratedDecoder(cfg.ModelConfig()).eval()
    plain.load_state_dict({key: value for key, value in timing.state_dict().items() if 'to_timing' not in key})
    inputs = make_inputs()
    assert torch.allclose(predict(timing, *inputs), predict(plain, *inputs), atol=1e-5)


def test_timing_prediction_is_causal():
    model = make_timing_model()
    torch.nn.init.normal_(model.to_timing.weight, std=0.1)
    context, velocity, valid, query, keep = make_inputs()
    first_changed_bin = 21
    changed = query.clone()
    changed[..., (cfg.HISTORY_BINS + first_changed_bin) * cfg.BIN_SAMPLES:] += 5.0
    before = predict(model, context, velocity, valid, query, keep)
    after = predict(model, context, velocity, valid, changed, keep)
    assert torch.allclose(before[:, :first_changed_bin], after[:, :first_changed_bin], atol=1e-5)
    assert not torch.allclose(before[:, first_changed_bin:], after[:, first_changed_bin:], atol=1e-5)


@torch.no_grad()
def test_warped_template_moves_a_peak_by_the_shift():
    model = make_timing_model('shift')
    template = torch.zeros(1, cfg.N_BINS, 2)
    template[0, 20] = 1.0
    features = torch.zeros(1, 1, cfg.N_BINS, 32)
    wanted_shift = 3.0
    model.to_timing.bias[0] = torch.atanh(torch.tensor(wanted_shift / 10.0))
    warped = model.warped_template(template, features)
    assert warped[0, 0, :, 0].argmax().item() == 23


def test_jitter_keeps_signal_and_labels_paired():
    n_trials = 32
    bin_of_sample = torch.arange(cfg.N_SAMPLES).div(cfg.BIN_SAMPLES, rounding_mode='floor') - cfg.HISTORY_BINS
    lfp = bin_of_sample.float()[None, None].expand(n_trials, cfg.N_CHANNELS, -1)
    labels = torch.arange(cfg.N_BINS).float()[None, :, None].expand(n_trials, -1, 2)
    valid = torch.ones(n_trials, cfg.N_BINS, dtype=torch.bool)
    generator = torch.Generator().manual_seed(0)
    shifted_lfp, shifted_labels, shifted_valid = jitter_trials(lfp, labels, valid, 2, generator)

    last_sample_of_bin = (cfg.HISTORY_BINS + torch.arange(cfg.N_BINS) + 1) * cfg.BIN_SAMPLES - 1
    signal_at_bins = shifted_lfp[:, 0, last_sample_of_bin]
    assert torch.equal(signal_at_bins[shifted_valid], shifted_labels[..., 0][shifted_valid])
    assert (~shifted_valid).any()


def test_small_settings_have_fewer_parameters():
    small = next(setting for setting in STEP1_SETTINGS if setting.small)
    assert count_parameters(build_model(small)) < count_parameters(build_model(Setting('full'))) / 3


if __name__ == '__main__':
    tests = [value for name, value in list(globals().items()) if name.startswith('test_')]
    for test in tests:
        test()
        print(f'PASS  {test.__name__}')
    print(f'{len(tests)}/{len(tests)} tests passed')
