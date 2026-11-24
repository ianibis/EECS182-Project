"""Task 5.1: simulate a stimulation artifact on stimulation-OFF trials.

One *shift* is sampled per episode and applied to all of that episode's trials:

    shifted = channel_gain * (clean_lfp + artifact),   with dropped channels set to 0

    artifact = amplitude * per_channel_gain * slow_envelope * trial_gain
               * (nonlinear(template(AM phase) + shared_component) + non_periodic_residual)

Only templates from TRAINING-day stimulation sessions are used, so nothing about
the validation or test artifacts can leak into training.
"""

import math

import h5py
import numpy as np
import torch

import task5_config as cfg
import task3_common as C


class TemplateLibrary:
    """Real PARRM artifact templates, resampled to a common number of phase bins."""

    def __init__(self, device):
        manifest = C.manifest()
        names = [s for s in C.sessions('train', 'stim')
                 if s not in cfg.POOR_TEMPLATES and int(manifest.loc[s, 'test_eligible']) == 1]
        keep_ids = C.L.load_session(names[0], C.PATHS, with_lfp=False, manifest=C.loader_manifest()).channels

        templates, frequencies = [], []
        with h5py.File(f'{C.PATHS.parrm_dir}/template_library.h5', 'r') as file:
            for name in names:
                group = file[name]
                rows = np.flatnonzero(np.isin(group['channel_ids'][:].astype(int), keep_ids))
                templates.append(self._resample(group['mean'][:][rows]))
                frequencies.append(float(np.ravel(group.attrs['am_freq_hz'])[0]))

        self.names = names
        self.templates = torch.as_tensor(np.stack(templates), dtype=torch.float32, device=device)  # (N, 35, bins)
        self.frequencies = torch.as_tensor(frequencies, dtype=torch.float32, device=device)
        self.is_3hz = self.frequencies < 10
        self.device = device

    @staticmethod
    def _resample(template):
        """Periodic linear interpolation of (channels, phase bins) to TEMPLATE_BINS."""
        n_old = template.shape[1]
        old = np.arange(n_old + 1) / n_old
        new = np.arange(cfg.TEMPLATE_BINS) / cfg.TEMPLATE_BINS
        closed = np.concatenate([template, template[:, :1]], axis=1)
        return np.stack([np.interp(new, old, channel) for channel in closed])


def _uniform(low, high, shape, generator, device):
    return low + (high - low) * torch.rand(shape, generator=generator, device=device)


def sample_shift(library, config, n_episodes, n_trials, generator):
    """Draw the random parameters of one shift per episode."""
    device = library.device
    n_channels = library.templates.shape[1]
    uniform = lambda low, high, shape: _uniform(low, high, shape, generator, device)
    normal = lambda shape: torch.randn(shape, generator=generator, device=device)

    # Choose 3 Hz or 20 Hz with equal chance, then one real template of that frequency.
    use_3hz = torch.rand(n_episodes, generator=generator, device=device) < 0.5
    three, twenty = library.is_3hz.nonzero()[:, 0], (~library.is_3hz).nonzero()[:, 0]
    pick_three = three[torch.randint(len(three), (n_episodes,), generator=generator, device=device)]
    pick_twenty = twenty[torch.randint(len(twenty), (n_episodes,), generator=generator, device=device)]
    template_index = torch.where(use_3hz, pick_three, pick_twenty)

    log_amplitude = uniform(math.log(config.amplitude_min), math.log(config.amplitude_max), (n_episodes,))
    dropout_rate = uniform(0.0, config.dropout_max, (n_episodes, 1)) if config.use_dropout else \
        torch.zeros((n_episodes, 1), device=device)

    return dict(
        template_index=template_index,
        amplitude=log_amplitude.exp(),
        frequency=library.frequencies[template_index]
        * (1 + uniform(-config.period_jitter, config.period_jitter, (n_episodes,))),
        trial_phase=uniform(0.0, 1.0, (n_episodes, n_trials)),      # stimulation runs freely across trials
        channel_gain=(config.channel_gain_sigma * normal((n_episodes, n_channels))).exp(),
        slow_frequency=uniform(0.1, 1.0, (n_episodes,)),
        slow_phase=uniform(0.0, 2 * math.pi, (n_episodes, n_trials)),
        trial_gain=(config.trial_gain_sigma * normal((n_episodes, n_trials))).exp(),
        nonlinearity=uniform(0.0, config.nonlinearity_max, (n_episodes,)),
        shared=uniform(0.0, config.shared_max, (n_episodes,)),
        recording_gain=(config.gain_sigma * normal((n_episodes, n_channels))).exp(),
        drift=uniform(-config.drift_max, config.drift_max, (n_episodes, n_channels)),
        channel_keep=torch.rand((n_episodes, n_channels), generator=generator, device=device) >= dropout_rate,
        residual=uniform(config.residual_min, config.residual_max, (n_episodes,)),
        noise_seed=int(torch.randint(2 ** 31 - 1, (1,), generator=generator, device=device)),
    )


def draw_from_pool(pool, library, config, n_episodes, n_trials, generator):
    """Task 5.7: use one of a fixed set of distinct shifts.

    A shift's template, gains, amplitude and frequency come from the pool. What differs from
    one recording to the next (stimulation phase, trial gain, residual noise) is drawn fresh.
    """
    fresh_each_episode = ('trial_phase', 'slow_phase', 'trial_gain', 'noise_seed')
    chosen = torch.randint(len(pool['template_index']), (n_episodes,), generator=generator, device=library.device)
    fresh = sample_shift(library, config, n_episodes, n_trials, generator)
    return {name: fresh[name] if name in fresh_each_episode else pool[name][chosen] for name in pool}


def residual_noise(shape, seed, shared_fraction, device):
    """Unit-RMS 1/f noise of shape (B, T, C, S), mostly common across channels.

    The seed is part of the shift, so applying the same shift twice gives the same result.
    """
    n_episodes, n_trials, n_channels, n_samples = shape
    generator = torch.Generator(device=device).manual_seed(seed)
    frequencies = torch.fft.rfftfreq(n_samples, 1 / 1000, device=device)
    amplitude = 1.0 / frequencies.clamp_min(0.5)
    amplitude[0] = 0.0

    def colored(size):
        real = torch.randn((*size, len(frequencies)), generator=generator, device=device)
        imaginary = torch.randn((*size, len(frequencies)), generator=generator, device=device)
        wave = torch.fft.irfft(torch.complex(real, imaginary) * amplitude, n=n_samples, dim=-1)
        return wave / wave.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-9)

    common = colored((n_episodes, n_trials, 1))
    separate = colored((n_episodes, n_trials, n_channels))
    return math.sqrt(shared_fraction) * common + math.sqrt(1 - shared_fraction) * separate


def artifact_waveform(library, shift, n_samples):
    """Evaluate each episode's template at the AM phase of every sample: (B, T, C, S)."""
    device = library.device
    templates = library.templates[shift['template_index']]  # (B, C, bins)
    n_episodes, n_channels, n_bins = templates.shape
    n_trials = shift['trial_phase'].shape[1]

    seconds = torch.arange(n_samples, device=device, dtype=torch.float32) / 1000.0
    phase = (shift['frequency'][:, None, None] * seconds[None, None, :]
             + shift['trial_phase'][:, :, None]) % 1.0
    position = phase.reshape(n_episodes, -1) * n_bins
    lower = position.floor().long() % n_bins
    upper = (lower + 1) % n_bins
    weight = (position - position.floor())[:, None, :]

    expand = lambda index: index[:, None, :].expand(-1, n_channels, -1)
    values = (1 - weight) * templates.gather(2, expand(lower)) + weight * templates.gather(2, expand(upper))
    return values.reshape(n_episodes, n_channels, n_trials, n_samples).permute(0, 2, 1, 3)


def apply_shift(lfp, shift, library, config):
    """Apply one shift per episode to clean LFP of shape (B, T, C, S), in microvolts."""
    n_episodes, n_trials, _, n_samples = lfp.shape
    device = lfp.device
    artifact = artifact_waveform(library, shift, n_samples)

    if config.use_shared_component:
        common = artifact.mean(dim=2, keepdim=True)
        artifact = artifact + shift['shared'][:, None, None, None] * common
    if config.use_nonlinearity:
        size = artifact.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
        artifact = artifact + shift['nonlinearity'][:, None, None, None] * artifact * artifact.abs() / size
    if config.use_residual:
        size = artifact.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        noise = residual_noise(artifact.shape, shift['noise_seed'], config.residual_shared, device)
        artifact = artifact + shift['residual'][:, None, None, None] * size * noise

    gain = shift['amplitude'][:, None, None, None] * shift['channel_gain'][:, None, :, None]
    if config.use_trial_gain:
        gain = gain * shift['trial_gain'][:, :, None, None]
    if config.use_slow_modulation:
        seconds = torch.arange(n_samples, device=device, dtype=torch.float32) / 1000.0
        wave = torch.sin(2 * math.pi * shift['slow_frequency'][:, None, None] * seconds[None, None, :]
                         + shift['slow_phase'][:, :, None])
        gain = gain * (1 + config.slow_modulation_depth * wave)[:, :, None, :]

    output = lfp + gain * artifact

    if config.use_gain_drift:
        progress = torch.linspace(-0.5, 0.5, n_trials, device=device)[None, :, None]   # position in the session
        recording = shift['recording_gain'][:, None, :] * (1 + shift['drift'][:, None, :] * progress)
        output = output * recording[:, :, :, None]
    if config.use_dropout:
        output = output * shift['channel_keep'][:, None, :, None]
    return output


def normalize_from_pool(lfp, n_pool):
    """Per-channel robust z-score of (B, T, C, S), with statistics from the first n_pool trials only."""
    n_episodes, _, n_channels, _ = lfp.shape
    pool = lfp[:, :n_pool, :, ::10].permute(0, 2, 1, 3).reshape(n_episodes, n_channels, -1)
    center = pool.median(dim=2).values
    scale = 1.4826 * (pool - center[:, :, None]).abs().median(dim=2).values
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    normalized = (lfp - center[:, None, :, None]) / scale[:, None, :, None]
    return normalized.clamp(-cfg.CLIP_Z, cfg.CLIP_Z)
