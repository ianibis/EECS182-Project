"""Unit tests for the calibrated decoder. They run on the CPU.

Run from the repository root:  CUDA_VISIBLE_DEVICES= python task5/test_task5_calibrated.py
"""

import torch

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
from task5_calibrated import N_FEATURES, CalibratedDecoder


def make_model():
    torch.manual_seed(0)
    return CalibratedDecoder(cfg.ModelConfig()).eval()


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


def test_output_shape():
    model = make_model()
    context, velocity, valid, query, keep = make_inputs()
    assert model.encode(context, keep).shape == (1, 4, cfg.N_BINS, N_FEATURES)
    assert predict(model, context, velocity, valid, query, keep).shape == (1, cfg.N_BINS, 2)


@torch.no_grad()
def test_no_context_uses_the_global_decoder():
    model = make_model()
    torch.nn.init.normal_(model.global_readout)
    *_, query, keep = make_inputs()
    features = model.encode(query, keep)[:, 0]
    empty = features[:, :0, None].expand(-1, 0, cfg.N_BINS, -1)
    output = model.decode(empty, torch.zeros(1, 0, cfg.N_BINS, 2), torch.zeros(1, 0, cfg.N_BINS, dtype=torch.bool),
                          features)
    assert torch.allclose(output, model.typical[None] + features @ model.global_readout[None])


def test_query_prediction_is_causal():
    model = make_model()
    context, velocity, valid, query, keep = make_inputs()
    first_changed_bin = 21
    changed = query.clone()
    changed[..., (cfg.HISTORY_BINS + first_changed_bin) * cfg.BIN_SAMPLES:] += 5.0
    before = predict(model, context, velocity, valid, query, keep)
    after = predict(model, context, velocity, valid, changed, keep)
    assert torch.allclose(before[:, :first_changed_bin], after[:, :first_changed_bin], atol=1e-5)
    assert not torch.allclose(before[:, first_changed_bin:], after[:, first_changed_bin:], atol=1e-5)


def test_breaking_the_signal_label_pairing_changes_the_prediction():
    model = make_model()
    context, velocity, valid, query, keep = make_inputs()
    rotated = context[:, torch.roll(torch.arange(context.shape[1]), 1)]
    before = predict(model, context, velocity, valid, query, keep)
    after = predict(model, rotated, velocity, valid, query, keep)
    assert (before - after).abs().max() > 1e-4


def test_invalid_context_bins_are_ignored():
    model = make_model()
    context, velocity, valid, query, keep = make_inputs()
    valid = valid.clone()
    valid[:, :, 10:20] = False
    changed = velocity.clone()
    changed[:, :, 10:20] = 100.0
    before = predict(model, context, velocity, valid, query, keep)
    after = predict(model, context, changed, valid, query, keep)
    assert torch.allclose(before, after, atol=1e-5)


if __name__ == '__main__':
    tests = [value for name, value in list(globals().items()) if name.startswith('test_')]
    for test in tests:
        test()
        print(f'PASS  {test.__name__}')
    print(f'{len(tests)}/{len(tests)} tests passed')
