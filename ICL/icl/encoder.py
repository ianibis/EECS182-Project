"""Token windows and token encoders for the in-context decoder (plan 5.3.2)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from icl import config as C


def token_windows(x, n_tok=C.N_TARGET_BINS, win=C.TOKEN_WIN, hop=C.SAMPLES_PER_BIN):
    """x: (..., C, T) -> (..., n_tok, C, win). Window b ends at the last sample of the b-th of
    the last n_tok bins, so token b only sees LFP up to the end of its own bin."""
    start = x.shape[-1] - (n_tok - 1) * hop - win
    if start < 0:
        raise ValueError(f'need at least {(n_tok - 1) * hop + win} samples, got {x.shape[-1]}')
    return x[..., start:].unfold(-1, win, hop).movedim(-2, -3)



class WindowEncoder(nn.Module):
    """Encoder that maps each 500 ms window independently: subclasses define forward((M, C, win)) -> (M, d)."""

    def tokens(self, x, n_tok):
        """x: (N, C, T) -> (N, n_tok, d), token b from the window ending at the end of the b-th last bin."""
        windows = token_windows(x, n_tok)
        return self(windows.flatten(0, 1)).view(x.shape[0], n_tok, -1)


class EEGNetEncoder(nn.Module):
    """Streaming EEGNet (Lawhern et al. 2018): one d-dimensional token per 50 ms bin.

    Block 1 runs once over the whole trial: spatial filters (35 channels -> F1 * D) followed by a
    causal temporal conv (kernel k1) on each spatial signal -> BN -> ELU. EEGNet applies the
    temporal conv first and the spatial (depthwise) conv second; both are linear, so the order
    can be swapped, which makes the temporal conv run on F1 * D signals instead of 35 * F1
    (about 10x faster). Differences from the original: no BatchNorm between the two linear
    filters, and each spatial signal gets its own temporal filter (a superset of EEGNet's tied
    pairs). Each token then takes block 1's output over the 500 ms window ending at its bin and
    applies block 2 (AvgPool 4 -> Dropout -> separable conv -> BN -> ELU -> AvgPool 8 -> Dropout)
    and a Linear layer. Strictly causal.

    BatchNorm as in the original: a per-window norm such as GroupNorm would divide out the
    window's overall power, which carries the movement signal (beta power drops during reaching).
    """

    def __init__(self, n_channels=C.N_CHANNELS, win=C.TOKEN_WIN, d=128, F1=8, D=2, F2=16, k1=64, k2=16,
                 dropout=0.25):
        super().__init__()
        self.k1, self.win = k1, win
        self.spatial = nn.Conv1d(n_channels, F1 * D, 1, bias=False)
        self.temporal = nn.Conv1d(F1 * D, F1 * D, k1, groups=F1 * D, bias=False)   # causal: left-padded
        self.bn1 = nn.BatchNorm1d(F1 * D)
        self.separable_depth = nn.Conv1d(F1 * D, F1 * D, k2, padding=k2 // 2, groups=F1 * D, bias=False)
        self.separable_point = nn.Conv1d(F1 * D, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(F2)
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            n_flat = self.block2(torch.zeros(2, F1 * D, win)).shape[1]
        self.proj = nn.Linear(n_flat, d)

    def trunk(self, x):
        """Block 1 over whole trials: (N, C, T) -> (N, F1 * D, T), causal in time."""
        x = F.pad(self.spatial(x), (self.k1 - 1, 0))
        return F.elu(self.bn1(self.temporal(x)))

    def block2(self, w):
        """(M, F1 * D, win) -> (M, flat)."""
        w = self.dropout(F.avg_pool1d(w, 4))
        w = F.elu(self.bn3(self.separable_point(self.separable_depth(w))))
        return self.dropout(F.avg_pool1d(w, 8)).flatten(1)

    def tokens(self, x, n_tok):
        """x: (N, C, T) -> (N, n_tok, d)."""
        windows = token_windows(self.trunk(x), n_tok, win=self.win)          # (N, n_tok, F1 * D, win)
        return self.proj(self.block2(windows.flatten(0, 1))).view(x.shape[0], n_tok, -1)


class FeatureTokenEncoder(WindowEncoder):
    """Fallback / debugging encoder: LMP + 7 log band powers per channel within the window -> MLP.

    Band power uses a Hann-windowed FFT of the whole 500 ms window (2 Hz resolution), so the
    1-4 Hz band is coarse; the LMP is the mean of the last 256 ms.
    """
    BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 50), (70, 110), (130, 200))

    def __init__(self, n_channels=C.N_CHANNELS, n_samples=C.TOKEN_WIN, d=128, hidden=256, fs=C.FS):
        super().__init__()
        freqs = np.fft.rfftfreq(n_samples, 1 / fs)
        masks = np.stack([(freqs >= lo) & (freqs <= hi) for lo, hi in self.BANDS]).astype(np.float32)
        self.register_buffer('band_masks', torch.from_numpy(masks / masks.sum(axis=1, keepdims=True)))
        self.register_buffer('window', torch.hann_window(n_samples, periodic=False))
        self.n_lmp = int(round(C.LMP_WIN_S * fs))
        self.mlp = nn.Sequential(nn.Linear(n_channels * 8, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x):                      # x: (N, C, samples)
        power = torch.fft.rfft(x * self.window, dim=-1).abs().pow(2)
        log_power = torch.log10(power @ self.band_masks.T + 1e-6)                # (N, C, 7)
        lmp = x[..., -self.n_lmp:].mean(dim=-1, keepdim=True)                     # (N, C, 1)
        return self.mlp(torch.cat([lmp, log_power], dim=-1).flatten(1))
