"""Unit tests for the Task 5 generator, sampler and model.

Run from the repository root:  python task5/test_task5.py
"""

from dataclasses import replace

import numpy as np
import torch

import task5_config as cfg
from task5_episodes import Episode, OffTrialBank, sample_episode
from task5_model import InContextDecoder, TokenEncoder, count_parameters
from task5_shift import TemplateLibrary, apply_shift, artifact_waveform, normalize_from_pool, sample_shift

device = 'cuda' if torch.cuda.is_available() else 'cpu'
OFF_OPTIONS = dict(use_slow_modulation=False, use_trial_gain=False, use_nonlinearity=False,
                   use_shared_component=False, use_gain_drift=False, use_dropout=False, use_residual=False)


def generator(seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def test_templates_only_from_training_days(library):
    import task3_common as C
    held_out = set(C.sessions('val')) | set(C.sessions('test'))
    assert not set(library.names) & held_out, 'a held-out template is in the training library'
    assert not set(library.names) & set(cfg.POOR_TEMPLATES), 'a poor template is in the library'
    assert library.is_3hz.any() and (~library.is_3hz).any(), 'need both AM frequencies'


def test_shift_is_reproducible(library):
    config = cfg.ShiftConfig()
    clean = torch.randn(2, 3, cfg.N_CHANNELS, cfg.N_SAMPLES, device=device)
    first = apply_shift(clean, sample_shift(library, config, 2, 3, generator(5)), library, config)
    second = apply_shift(clean, sample_shift(library, config, 2, 3, generator(5)), library, config)
    assert torch.equal(first, second), 'the same seed must give the same shift'


def test_each_option_can_be_switched_off(library):
    clean = torch.randn(2, 3, cfg.N_CHANNELS, cfg.N_SAMPLES, device=device)
    plain = cfg.ShiftConfig(**OFF_OPTIONS)
    shift = sample_shift(library, cfg.ShiftConfig(), 2, 3, generator(1))
    baseline = apply_shift(clean, shift, library, plain)
    for option in OFF_OPTIONS:
        changed = apply_shift(clean, shift, library, replace(plain, **{option: True}))
        assert not torch.allclose(changed, baseline), f'{option} has no effect'


def test_artifact_peaks_at_am_frequency(library):
    config = cfg.ShiftConfig(period_jitter=0.0, **OFF_OPTIONS)
    for wanted_3hz in (True, False):
        for _ in range(20):
            shift = sample_shift(library, config, 1, 1, generator(_))
            if bool(library.is_3hz[shift['template_index'][0]]) == wanted_3hz:
                break
        wave = artifact_waveform(library, shift, 4000)[0, 0].cpu().numpy()
        spectrum = np.abs(np.fft.rfft(wave - wave.mean(axis=1, keepdims=True), axis=1)).mean(axis=0)
        peak = np.fft.rfftfreq(4000, 1 / 1000)[spectrum.argmax()]
        frequency = float(shift['frequency'][0])
        multiple = peak / frequency
        assert abs(multiple - round(multiple)) < 0.15 and round(multiple) >= 1, (
            f'spectral peak {peak} Hz is not a harmonic of {frequency} Hz')


def test_normalization_ignores_the_query():
    x = torch.randn(2, 21, cfg.N_CHANNELS, cfg.N_SAMPLES, device=device)
    changed = x.clone()
    changed[:, -1] += 1000.0
    a = normalize_from_pool(x, cfg.POOL_SIZE)[:, :cfg.POOL_SIZE]
    b = normalize_from_pool(changed, cfg.POOL_SIZE)[:, :cfg.POOL_SIZE]
    assert torch.equal(a, b), 'query trial changed the normalization'


def test_episode_shapes_and_shared_shift(bank, library):
    k = 5
    episode = sample_episode(bank, library, cfg.ShiftConfig(), 4, k, generator(2))
    assert episode.context_lfp.shape == (4, k, cfg.N_CHANNELS, cfg.N_SAMPLES)
    assert episode.context_velocity.shape == (4, k, cfg.N_BINS, 2)
    assert episode.query_lfp.shape == (4, cfg.N_CHANNELS, cfg.N_SAMPLES)
    # A dropped channel must be dropped in the context and in the query alike.
    for row in range(4):
        dropped = ~episode.channel_keep[row]
        if dropped.any():
            assert episode.context_lfp[row][:, dropped].abs().max() < 1e3
            spread_context = episode.context_lfp[row][:, dropped].std()
            spread_query = episode.query_lfp[row][dropped].std()
            assert spread_context < 1e-3 and spread_query < 1e-3, 'context and query do not share dropout'


def test_query_is_never_in_the_pool(bank):
    rows = bank.draw_rows(64, generator(3), use_holdout_query=False)
    for row in rows:
        assert len(set(row.tolist())) == len(row), 'repeated trial in an episode'
    held = bank.draw_rows(64, generator(4), use_holdout_query=True)
    fit = set(torch.cat(bank.fit_rows).tolist())
    assert all(int(r[-1]) not in fit for r in held), 'hold-out query came from the fitting trials'


def fake_episode(k, batch=2):
    return Episode(
        context_lfp=torch.randn(batch, k, cfg.N_CHANNELS, cfg.N_SAMPLES, device=device),
        context_velocity=torch.randn(batch, k, cfg.N_BINS, 2, device=device),
        context_valid=torch.ones(batch, k, cfg.N_BINS, dtype=torch.bool, device=device),
        query_lfp=torch.randn(batch, cfg.N_CHANNELS, cfg.N_SAMPLES, device=device),
        query_velocity=torch.zeros(batch, cfg.N_BINS, 2, device=device),
        query_valid=torch.ones(batch, cfg.N_BINS, dtype=torch.bool, device=device),
        channel_keep=torch.ones(batch, cfg.N_CHANNELS, dtype=torch.bool, device=device),
        shift={},
    )


def test_model_shape_and_size(model):
    for k in (0, 2):
        assert model(fake_episode(k)).shape == (2, cfg.N_BINS, 2)
    size = count_parameters(model)
    assert 0.7e6 < size < 1.4e6, f'{size:,} parameters; the proposal says about 1M'
    print(f'    model parameters: {size:,}')


def test_encoder_is_causal_with_500ms_history():
    encoder = TokenEncoder(cfg.ModelConfig()).to(device).eval()
    keep = torch.ones(1, cfg.N_CHANNELS, dtype=torch.bool, device=device)
    x = torch.randn(1, cfg.N_CHANNELS, cfg.N_SAMPLES, device=device)
    base = encoder(x, keep)
    bin_index = 20
    end = (cfg.HISTORY_BINS + bin_index + 1) * cfg.BIN_SAMPLES   # first sample after this token's bin

    future = x.clone()
    future[..., end:] += 5.0
    assert torch.allclose(encoder(future, keep)[0, :bin_index + 1], base[0, :bin_index + 1]), 'encoder sees the future'

    distant = x.clone()
    distant[..., :end - 500] += 5.0
    assert torch.allclose(encoder(distant, keep)[0, bin_index], base[0, bin_index], atol=1e-5), (
        'token sees more than 500 ms of history')


def test_query_prediction_is_causal(model):
    episode = fake_episode(2, batch=1)
    base = model(episode)
    t = 20
    end = (cfg.HISTORY_BINS + t + 1) * cfg.BIN_SAMPLES
    changed = replace(episode, query_lfp=episode.query_lfp.clone())
    changed.query_lfp[..., end:] += 5.0
    after = model(changed)
    assert torch.allclose(after[0, :t + 1], base[0, :t + 1], atol=1e-4), 'prediction used future query signal'
    assert not torch.allclose(after[0, t + 1:], base[0, t + 1:], atol=1e-4), 'later bins ignore the query signal'


def test_context_labels_change_the_prediction(model):
    episode = fake_episode(2, batch=1)
    changed = replace(episode, context_velocity=episode.context_velocity + 3.0)
    assert not torch.allclose(model(episode), model(changed), atol=1e-4), 'context labels are ignored'


if __name__ == '__main__':
    library = TemplateLibrary(device)
    bank = OffTrialBank(device)
    model = InContextDecoder(cfg.ModelConfig()).to(device).eval()
    print(f'{len(library.names)} templates ({int(library.is_3hz.sum())} at 3 Hz), {bank.lfp.shape[0]} OFF trials')
    tests = [(test_templates_only_from_training_days, (library,)), (test_shift_is_reproducible, (library,)),
             (test_each_option_can_be_switched_off, (library,)), (test_artifact_peaks_at_am_frequency, (library,)),
             (test_normalization_ignores_the_query, ()), (test_episode_shapes_and_shared_shift, (bank, library)),
             (test_query_is_never_in_the_pool, (bank,)), (test_model_shape_and_size, (model,)),
             (test_encoder_is_causal_with_500ms_history, ()), (test_query_prediction_is_causal, (model,)),
             (test_context_labels_change_the_prediction, (model,))]
    failed = 0
    with torch.no_grad():
        for test, arguments in tests:
            try:
                test(*arguments)
                print(f'PASS  {test.__name__}')
            except AssertionError as error:
                failed += 1
                print(f'FAIL  {test.__name__}: {error}')
    print(f'{len(tests) - failed}/{len(tests)} tests passed')
    raise SystemExit(failed > 0)
