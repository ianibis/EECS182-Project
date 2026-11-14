"""Episodes for training and evaluating the in-context decoder (plan 5.2b).

Training episode: one session, one simulated Shift, n_pool + n_extra trials drawn without
replacement from the session's training trials. The first n_pool trials form the context
chain and set the normalisation; the extra trials are queries outside the pool, as at test
time. Each trial's decoded window is shifted by a random offset of up to +-max_jitter_bins,
so a token's position inside the window carries no information about time since the go cue.

Evaluation batch: one held-out session, the splits.json context draw for (k, draw) and every
query trial, each query preceded by the same k context trials; standard windows, and the
normalisation set by the session's 20 calibration-pool trials (unlabelled), whatever k is.

Both paths share cut_windows() and normalise(), so training and evaluation see identical
preprocessing: shift (microvolts) -> window -> per-channel robust z from the pool -> clip.
"""
import json
from dataclasses import dataclass, field

import numpy as np
import torch

from icl import config as C
from icl.cache import load_cached, load_index
from icl.shifts import ShiftConfig, apply_shift, sample_shift


@dataclass
class EpisodeConfig:
    n_pool: int = 20                      # context chain; also sets the normalisation
    n_extra: int = 1                      # queries outside the pool
    max_jitter_bins: int = C.MAX_JITTER_BINS
    clip: float = 10.0                    # robust z-scores are clipped to +-clip
    shift: ShiftConfig = field(default_factory=ShiftConfig)


def sessions_where(split, stim):
    return [s for s, meta in load_index().items() if meta['split'] == split and meta['stim'] == stim]


class TrialStore:
    """Cached trials of a set of sessions, held as tensors on one device."""

    def __init__(self, sessions, parts=None, device='cpu'):
        self.device = device
        self.lfp, self.kin, self.valid, self.trial, self.part, self.meta = {}, {}, {}, {}, {}, {}
        for session in sessions:
            data = load_cached(session)
            keep = np.isin(data['part'], parts) if parts else np.ones(len(data['part']), bool)
            self.lfp[session] = torch.from_numpy(data['lfp'][keep]).to(device)          # (n, C, 4500) uV
            self.kin[session] = torch.from_numpy(data['kin'][keep]).to(device)          # (n, 90, 2) cm/s
            self.valid[session] = torch.from_numpy(data['valid'][keep]).to(device)      # (n, 90)
            self.trial[session] = data['trial'][keep]
            self.part[session] = data['part'][keep]
            self.meta[session] = data['meta']
        self.sessions = list(sessions)

    def __len__(self):
        return len(self.sessions)


def cut_windows(lfp, kin, valid, offsets):
    """Model windows for each trial: lfp (n, C, 4500) -> (n, C, MODEL_SAMPLES); kin -> (n, 40, 2).

    offsets (n,) shift each trial's decoded window by whole bins (0 = standard window).
    """
    offsets = torch.as_tensor(offsets, device=lfp.device)
    first_target = C.TARGET_START_BIN + offsets                                          # (n,)
    sample = (first_target - C.MODEL_HISTORY_BINS)[:, None] * C.SAMPLES_PER_BIN \
        + torch.arange(C.MODEL_SAMPLES, device=lfp.device)                               # (n, S)
    x = torch.gather(lfp, 2, sample[:, None, :].expand(-1, lfp.shape[1], -1))
    bins = first_target[:, None] + torch.arange(C.N_TARGET_BINS, device=lfp.device)       # (n, 40)
    y = torch.gather(kin, 1, bins[..., None].expand(-1, -1, kin.shape[-1]))
    v = torch.gather(valid, 1, bins)
    return x, y, v


def robust_stats(x):
    """Per-channel median and 1.4826 * MAD over trials and samples: x (n, C, S) -> (C,), (C,)."""
    flat = x.transpose(0, 1).reshape(x.shape[1], -1)
    median = flat.median(dim=1).values
    scale = 1.4826 * (flat - median[:, None]).abs().median(dim=1).values
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))          # e.g. dropped channels
    return median, scale


def normalise(x, median, scale, clip):
    return ((x - median[:, None]) / scale[:, None]).clamp(-clip, clip)


class VelocityScaler:
    """Velocity in cm/s <-> model units, using the training statistics in stats.json."""

    def __init__(self, device='cpu'):
        stats = C.load_stats()
        self.mean = torch.tensor(stats['vel_mean'], device=device)
        self.std = torch.tensor(stats['vel_std'], device=device)

    def encode(self, y, valid):
        return torch.where(valid[..., None], (y - self.mean) / self.std, torch.zeros_like(y))

    def decode(self, y):
        return y * self.std + self.mean


class EpisodeSampler:
    """Training (or synthetic validation) episodes. Stim sessions in the store keep their real artifact
    and get no simulated one (neural gain drift and channel dropout only)."""

    def __init__(self, store, cfg=None, bank=None, seed=0):
        self.store, self.cfg, self.bank = store, cfg or EpisodeConfig(), bank
        self.rng = np.random.default_rng(seed)
        self.scaler = VelocityScaler(store.device)
        index = load_index()
        self.stim = {s for s in store.sessions if index[s]['stim']}   # real artifact: no simulated one on top
        n_needed = self.cfg.n_pool + self.cfg.n_extra
        too_small = [s for s in store.sessions if len(store.trial[s]) < n_needed]
        if too_small:
            raise ValueError(f'{too_small} have fewer than {n_needed} trials')

    def sample(self, batch_size, rng=None):
        """Returns x (B, n, C, S), y (B, n, 40, 2) in model units, valid (B, n, 40) and metadata."""
        rng = rng or self.rng
        cfg, store = self.cfg, self.store
        n = cfg.n_pool + cfg.n_extra
        xs, ys, vs, meta = [], [], [], []
        for _ in range(batch_size):
            session = store.sessions[int(rng.integers(len(store)))]
            index = rng.choice(len(store.trial[session]), n, replace=False)
            amp = 0.0 if session in self.stim else None
            shift = sample_shift(rng, cfg.shift, bank=self.bank, amp=amp).to(store.device)
            lfp = apply_shift(store.lfp[session][index], shift, rng)
            offsets = rng.integers(-cfg.max_jitter_bins, cfg.max_jitter_bins + 1, n)
            x, y, v = cut_windows(lfp, store.kin[session][index], store.valid[session][index], offsets)
            median, scale = robust_stats(x[:cfg.n_pool])
            xs.append(normalise(x, median, scale, cfg.clip))
            ys.append(self.scaler.encode(y, v))
            vs.append(v)
            meta.append(dict(session=session, trial=store.trial[session][index], offsets=offsets, shift=shift))
        return dict(x=torch.stack(xs), y=torch.stack(ys), valid=torch.stack(vs), meta=meta)


def load_splits():
    with open(C.SPLITS) as file:
        return json.load(file)


def eval_batch(store, session, k, draw, splits, shift=None, rng=None, clip=10.0, scaler=None):
    """Every query trial of a held-out session, each preceded by the same k context trials.

    Context = splits['calibration'][session]['subsets'][str(k)][draw] (k = 0: no context).
    shift (optional) is applied to every trial of the session, e.g. for semi-synthetic tests.
    Returns x (Q, k + 1, C, S), y (Q, k + 1, 40, 2) in model units, valid, and the trial ids.
    """
    calibration = splits['calibration'][session]
    position = {int(t): i for i, t in enumerate(store.trial[session])}
    pool = [position[t] for t in calibration['pool']]
    query = [position[t] for t in calibration['query']]
    context = [position[t] for t in calibration['subsets'][str(k)][draw]] if k > 0 else []

    lfp = store.lfp[session]
    if shift is not None:
        lfp = apply_shift(lfp, shift.to(store.device), rng or np.random.default_rng(0))
    x, y, v = cut_windows(lfp, store.kin[session], store.valid[session], np.zeros(len(lfp), int))
    median, scale = robust_stats(x[pool])
    x = normalise(x, median, scale, clip)
    y = (scaler or VelocityScaler(store.device)).encode(y, v)

    sequence = torch.tensor([context + [q] for q in query], device=store.device)        # (Q, k + 1)
    return dict(x=x[sequence], y=y[sequence], valid=v[sequence],
                context_trials=[int(store.trial[session][i]) for i in context],
                query_trials=[int(store.trial[session][i]) for i in query])
