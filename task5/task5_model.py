"""Task 5.3: EEGNet-style token encoder and causal in-context transformer.

Token order for k context trials and one query trial (40 time bins each):

    [signal, movement] x 40   for context trial 1
    ...
    [signal, movement] x 40   for context trial k
    [signal] x 40             for the query trial

The mask is causal, so a query prediction at time t sees all the context plus the
query's own signal up to t. It never sees the query's movement.
"""

import torch
import torch.nn.functional as F
from torch import nn

import task5_config as cfg


def causal_conv(layer, x):
    """Run a time convolution that only looks at the current and earlier samples."""
    return layer(F.pad(x, (layer.kernel_size[1] - 1, 0)))


class FilterNorm(nn.Module):
    """Layer normalization across filters, separately at every time step.

    EEGNet's BatchNorm depends on the rest of the batch, and GroupNorm averages over
    the whole trial, which would let early tokens see future samples. This norm does neither.
    """

    def __init__(self, n_filters):
        super().__init__()
        self.norm = nn.LayerNorm(n_filters)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class TokenEncoder(nn.Module):
    """EEGNet-style encoder: one token per 50 ms, each seeing at most the past 500 ms.

    Receptive field: 125 samples (temporal) + 63 x 5 (separable) + 49 (pooling) = 489 samples.
    """

    def __init__(self, config):
        super().__init__()
        depth = config.temporal_filters * config.depth_multiplier
        self.temporal = nn.Conv2d(1, config.temporal_filters, (1, config.temporal_kernel), bias=False)
        self.norm1 = FilterNorm(config.temporal_filters)
        self.spatial = nn.Conv2d(config.temporal_filters, depth, (cfg.N_CHANNELS, 1),
                                 groups=config.temporal_filters, bias=False)
        self.norm2 = FilterNorm(depth)
        self.pool1 = nn.AvgPool2d((1, 5))
        self.separable_depth = nn.Conv2d(depth, depth, (1, config.separable_kernel), groups=depth, bias=False)
        self.separable_point = nn.Conv2d(depth, config.separable_filters, 1, bias=False)
        self.norm3 = FilterNorm(config.separable_filters)
        self.pool2 = nn.AvgPool2d((1, 10))
        self.dropout = nn.Dropout(config.encoder_dropout)
        self.to_token = nn.Linear(config.separable_filters + cfg.N_CHANNELS, config.d_model)

    def forward(self, lfp, channel_keep):
        """(N, C, S) signal and (N, C) channel mask -> (N, 40, d_model) tokens."""
        x = lfp[:, None]
        x = self.norm1(causal_conv(self.temporal, x))
        x = F.gelu(self.norm2(self.spatial(x)))
        x = self.dropout(self.pool1(x))
        x = self.separable_point(causal_conv(self.separable_depth, x))
        x = F.gelu(self.norm3(x))
        x = self.dropout(self.pool2(x))
        x = x.squeeze(2).transpose(1, 2)[:, cfg.HISTORY_BINS:]      # keep the 40 scored bins
        mask = channel_keep.float()[:, None, :].expand(-1, x.shape[1], -1)
        return self.to_token(torch.cat([x, mask], dim=2))


class InContextDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blind = config.blind
        self.encoder = TokenEncoder(config)
        self.movement = nn.Linear(3, config.d_model)                # vx, vy, valid flag
        self.time_embedding = nn.Embedding(cfg.N_BINS, config.d_model)
        self.trial_embedding = nn.Embedding(cfg.POOL_SIZE + 1, config.d_model)
        self.type_embedding = nn.Embedding(3, config.d_model)       # context signal, context movement, query
        layer = nn.TransformerEncoderLayer(config.d_model, config.n_heads, config.ff_size, config.dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, 2)

    def encode(self, lfp, channel_keep):
        """Encode (B, T, C, S) trials into (B, T, 40, d_model) signal tokens."""
        n_episodes, n_trials = lfp.shape[:2]
        if self.blind:      # the labels-only control model never sees brain signal or channel dropout
            lfp = torch.zeros_like(lfp)
            channel_keep = torch.ones_like(channel_keep)
        flat = lfp.reshape(n_episodes * n_trials, *lfp.shape[2:])
        keep = channel_keep[:, None].expand(-1, n_trials, -1).reshape(n_episodes * n_trials, -1)
        return self.encoder(flat, keep).reshape(n_episodes, n_trials, cfg.N_BINS, -1)

    def decode(self, context_tokens, context_velocity, context_valid, query_tokens):
        """Run the transformer on encoded tokens and return (B, 40, 2) query velocities."""
        n_episodes, k = context_tokens.shape[:2]
        device = query_tokens.device
        times = self.time_embedding(torch.arange(cfg.N_BINS, device=device))

        parts = []
        if k > 0:
            slots = self.trial_embedding(torch.arange(k, device=device))[None, :, None, :]
            base = times[None, None, :, :] + slots
            signal = context_tokens + base + self.type_embedding.weight[0]
            moves = torch.cat([context_velocity * context_valid[..., None], context_valid[..., None].float()], dim=3)
            movement = self.movement(moves) + base + self.type_embedding.weight[1]
            pairs = torch.stack([signal, movement], dim=3)              # (B, k, 40, 2, d)
            parts.append(pairs.reshape(n_episodes, k * cfg.N_BINS * 2, -1))

        query_slot = self.trial_embedding.weight[cfg.POOL_SIZE]
        parts.append(query_tokens + times[None, :, :] + query_slot + self.type_embedding.weight[2])
        tokens = torch.cat(parts, dim=1)

        mask = nn.Transformer.generate_square_subsequent_mask(tokens.shape[1], device=device, dtype=torch.bool)
        output = self.transformer(tokens, mask=mask, is_causal=True)
        return self.head(self.norm(output[:, -cfg.N_BINS:]))

    def forward(self, episode):
        """Predict scaled query velocity for a batch of episodes."""
        keep = episode.channel_keep
        query_tokens = self.encode(episode.query_lfp[:, None], keep)[:, 0]
        k = episode.context_lfp.shape[1]
        if k > 0:
            context_tokens = self.encode(episode.context_lfp, keep)
        else:
            context_tokens = query_tokens[:, :0, None].expand(-1, 0, cfg.N_BINS, -1)
        return self.decode(context_tokens, episode.context_velocity, episode.context_valid, query_tokens)


def masked_mse(prediction, target, valid):
    """Mean squared error over valid bins only."""
    return (prediction - target).square().mean(dim=2)[valid].mean()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
