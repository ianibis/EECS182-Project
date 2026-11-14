"""In-context decoder (plan 5.3.1), following the prompt format of Garg et al. (2022).

A prompt is a sequence of trials. Each trial contributes 40 signal tokens (one per 50 ms bin,
each encoding the preceding 500 ms of LFP) followed by 40 label tokens (velocity of each bin):

    X_1 Y_1 X_2 Y_2 ... X_n-1 Y_n-1 X_n

With the ordinary causal mask, signal token t of trial j sees every earlier trial's signals and
labels and its own trial's signals up to bin t, but never its own labels. Reading predictions at
every signal token therefore gives, in one pass, the prediction for each trial with 0, 1, ...,
n - 1 labelled trials in context (Garg's every-prefix training).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from icl import config as C


class ICLDecoder(nn.Module):
    def __init__(self, encoder, d=128, n_layer=4, n_head=4, n_bins=C.N_TARGET_BINS, max_trials=21, n_out=2,
                 dropout=0.0):
        super().__init__()
        self.encoder = encoder                    # .tokens((N, C, T), n_tok) -> (N, n_tok, d)
        self.n_bins = n_bins
        self.label_in = nn.Linear(n_out + 1, d)   # [vx, vy, valid]
        self.type_emb = nn.Embedding(2, d)        # 0 = signal token, 1 = label token
        self.bin_emb = nn.Embedding(n_bins, d)    # position of the bin inside its trial window
        self.backbone = GPT2Model(GPT2Config(
            vocab_size=1, bos_token_id=0, eos_token_id=0, n_positions=2 * max_trials * n_bins, n_embd=d,
            n_layer=n_layer, n_head=n_head, resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout,
            use_cache=False))
        self.read_out = nn.Linear(d, n_out)

    def forward(self, x, y, valid):
        """x: (B, n, channels, samples) normalised LFP; y: (B, n, 40, n_out); valid: (B, n, 40) bool.

        Returns (B, n, 40, n_out): predictions for every bin of every trial; trial j's prediction
        uses the labels of trials < j only. The last trial's labels are never read.
        """
        batch, n_trials = y.shape[:2]
        bins = self.n_bins
        signal = self.encoder.tokens(x.flatten(0, 1), bins).view(batch, n_trials, bins, -1)   # (B, n, 40, d)
        position = self.bin_emb.weight
        signal = signal + position + self.type_emb.weight[0]
        v = valid[..., None].to(signal.dtype)
        label = self.label_in(torch.cat([torch.nan_to_num(y) * v, v], dim=-1)) + position + self.type_emb.weight[1]
        tokens = torch.stack([signal, label], dim=2).reshape(batch, 2 * n_trials * bins, -1)
        tokens = tokens[:, :(2 * n_trials - 1) * bins]                            # drop the last trial's labels
        hidden = self.backbone(inputs_embeds=tokens).last_hidden_state
        hidden = F.pad(hidden, (0, 0, 0, bins)).view(batch, n_trials, 2, bins, -1)[:, :, 0]
        return self.read_out(hidden)


def masked_mse(pred, y, valid):
    """Mean squared error over valid bins (labels may be NaN where invalid)."""
    mask = valid[..., None].expand_as(pred)
    return ((pred - torch.nan_to_num(y)) ** 2)[mask].mean()
