"""Task 5.2: build training episodes the same way the model is used at test time.

An episode takes one training session and one simulated shift. It draws a 20-trial
pool plus a separate query trial, applies the shift to all of them, normalizes from
the pool (never from the query), and uses the first k pool trials as labeled context.
"""

from dataclasses import dataclass

import numpy as np
import torch

import task5_config as cfg
import task5_data as data
from task5_shift import apply_shift, draw_from_pool, normalize_from_pool, sample_shift


@dataclass
class Episode:
    context_lfp: torch.Tensor       # (B, k, C, S)
    context_velocity: torch.Tensor  # (B, k, 40, 2), scaled
    context_valid: torch.Tensor     # (B, k, 40)
    query_lfp: torch.Tensor         # (B, C, S)
    query_velocity: torch.Tensor    # (B, 40, 2), scaled
    query_valid: torch.Tensor       # (B, 40)
    channel_keep: torch.Tensor      # (B, C)
    shift: dict


class OffTrialBank:
    """All usable stimulation-OFF training trials, held in GPU memory."""

    def __init__(self, device):
        lfp, velocity, valid, fit_rows, holdout_rows = [], [], [], [], []
        offset = 0
        for session in data.usable_sessions('train', 'no_stim'):
            arrays = data.load_session_arrays(session)
            fit = data.part_mask(session, arrays['trial'], 'train')
            lfp.append(arrays['lfp'])
            velocity.append(arrays['velocity'])
            valid.append(arrays['valid'])
            fit_rows.append(torch.as_tensor(np.flatnonzero(fit) + offset, device=device))
            holdout_rows.append(torch.as_tensor(np.flatnonzero(~fit) + offset, device=device))
            offset += len(fit)

        velocity = np.concatenate(velocity)
        valid = np.concatenate(valid)
        fit_all = torch.cat(fit_rows).cpu().numpy()
        fit_velocity = velocity[fit_all][valid[fit_all]]

        # The scale comes from training trials only and is applied to all data.
        self.velocity_scale = torch.as_tensor(fit_velocity.std(axis=0), dtype=torch.float32, device=device)
        self.lfp = torch.as_tensor(np.concatenate(lfp), device=device)
        self.velocity = torch.as_tensor(np.nan_to_num(velocity), device=device) / self.velocity_scale
        self.valid = torch.as_tensor(valid, device=device)
        self.fit_rows = fit_rows
        self.holdout_rows = holdout_rows
        self.device = device

    def draw_rows(self, n_episodes, generator, use_holdout_query):
        """Pick 20 pool trials and one query trial from one session, per episode."""
        rows = []
        sessions = torch.randint(len(self.fit_rows), (n_episodes,), generator=generator, device=self.device)
        for session in sessions.tolist():
            fit = self.fit_rows[session]
            order = fit[torch.randperm(len(fit), generator=generator, device=self.device)]
            if use_holdout_query:
                holdout = self.holdout_rows[session]
                query = holdout[torch.randint(len(holdout), (1,), generator=generator, device=self.device)]
                rows.append(torch.cat([order[:cfg.POOL_SIZE], query]))
            else:
                rows.append(order[:cfg.POOL_SIZE + 1])
        return torch.stack(rows)    # (B, 21); the last column is the query


def sample_episode(bank, library, shift_config, batch_size, k, generator, use_holdout_query=False,
                   shift_pool=None):
    """Draw one batch of episodes that all have k context trials.

    With shift_pool (Task 5.7), each episode uses one of a fixed set of distinct shifts.
    """
    rows = bank.draw_rows(batch_size, generator, use_holdout_query)
    if shift_pool is None:
        shift = sample_shift(library, shift_config, batch_size, rows.shape[1], generator)
    else:
        shift = draw_from_pool(shift_pool, library, shift_config, batch_size, rows.shape[1], generator)
    shifted = apply_shift(bank.lfp[rows], shift, library, shift_config)
    normalized = normalize_from_pool(shifted, cfg.POOL_SIZE)

    return Episode(
        context_lfp=normalized[:, :k],
        context_velocity=bank.velocity[rows[:, :k]],
        context_valid=bank.valid[rows[:, :k]],
        query_lfp=normalized[:, -1],
        query_velocity=bank.velocity[rows[:, -1]],
        query_valid=bank.valid[rows[:, -1]],
        channel_keep=shift['channel_keep'],
        shift=shift,
    )
