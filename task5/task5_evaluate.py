"""Task 5.5 / 5.6: score a trained model on held-out sessions, with the context controls.

A session is scored as in Task 3: pool the valid bins of its 30 query trials, compute
R² for vx and vy, and average them. Context trials come from the draws in splits.json.

Conditions:  'recorded' (real ON or OFF trials)   'amp10' (PARRM artifact amplified 10x)
Controls:    None                real context
             'shuffle_labels'    context velocities shuffled across trials and time
             'off_context'       context taken from the same-day stimulation-OFF session
             'no_query_signal'   the query's brain signal is replaced by zeros
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task3_common as C
import task5_data as data
from task5_shift import normalize_from_pool


@dataclass
class HeldOutSession:
    session: str
    condition: str
    pool_trials: np.ndarray
    pool_lfp: torch.Tensor          # (20, C, S), normalized
    pool_velocity: torch.Tensor     # (20, 40, 2), scaled
    pool_valid: torch.Tensor        # (20, 40)
    query_lfp: torch.Tensor         # (30, C, S), normalized
    query_velocity: np.ndarray      # (30, 40, 2), cm/s
    query_valid: np.ndarray         # (30, 40)


def load_held_out(session, condition, velocity_scale, device):
    """Load a validation or test session, normalized from its calibration pool only."""
    arrays = data.load_session_arrays(session)
    lfp = arrays['lfp']
    if condition == 'amp10':
        lfp = lfp + 9.0 * arrays['artifact']    # the construction in semisynthetic/README.md
    elif condition != 'recorded':
        raise ValueError(f'unknown condition {condition}')

    in_pool = data.part_mask(session, arrays['trial'], 'pool')
    in_query = data.part_mask(session, arrays['trial'], 'query')
    ordered = np.concatenate([lfp[in_pool], lfp[in_query]])
    normalized = normalize_from_pool(torch.as_tensor(ordered, device=device)[None], int(in_pool.sum()))[0]
    n_pool = int(in_pool.sum())
    pool_velocity = torch.as_tensor(np.nan_to_num(arrays['velocity'][in_pool]), device=device) / velocity_scale
    return HeldOutSession(
        session=session, condition=condition, pool_trials=arrays['trial'][in_pool],
        pool_lfp=normalized[:n_pool], pool_velocity=pool_velocity,
        pool_valid=torch.as_tensor(arrays['valid'][in_pool], device=device),
        query_lfp=normalized[n_pool:], query_velocity=arrays['velocity'][in_query],
        query_valid=arrays['valid'][in_query],
    )


def context_positions(held, k, draw):
    """Positions inside the pool of the k context trials of one draw from splits.json."""
    if k == 0:
        return np.array([], dtype=int)
    trials = C.splits()['calibration'][held.session]['subsets'][str(k)][draw]
    lookup = {int(t): i for i, t in enumerate(held.pool_trials)}
    return np.array([lookup[int(t)] for t in trials], dtype=int)


def n_draws(k, requested=10):
    """k = 0 has no draw to vary, and k = 20 always uses the entire pool."""
    return 1 if k in (0, cfg.POOL_SIZE) else requested


def session_r2(prediction, velocity, valid):
    """Task 3 score: R² per axis over all valid bins, and their mean."""
    r2_vx, r2_vy = C.r2(velocity[valid], prediction[valid])
    return float((r2_vx + r2_vy) / 2), float(r2_vx), float(r2_vy)


def labels_only_prediction(held, positions, baseline, velocity_scale):
    """The no-brain baseline for this context set (see task5_baselines.py), in cm/s."""
    velocity = (held.pool_velocity[positions] * velocity_scale).cpu().numpy()
    valid = held.pool_valid[positions].cpu().numpy()
    return baseline.predict(held.session, velocity, valid, held.query_velocity.shape)


@torch.no_grad()
def predict_queries(model, held, positions, control=None, off_session=None, seed=0, chunk=10):
    """Predict velocity (scaled) for all query trials with one context set."""
    device = held.query_lfp.device
    keep = torch.ones(1, cfg.N_CHANNELS, dtype=torch.bool, device=device)
    source = off_session if control == 'off_context' else held
    k = len(positions)

    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        if k:
            context_tokens = model.encode(source.pool_lfp[positions][None], keep).float()
            velocity = source.pool_velocity[positions].clone()
            valid = source.pool_valid[positions]
            if control == 'shuffle_labels':
                generator = torch.Generator(device=device).manual_seed(seed)
                order = torch.randperm(k * cfg.N_BINS, generator=generator, device=device)
                velocity = velocity.reshape(-1, 2)[order].reshape(k, cfg.N_BINS, 2)
                valid = valid.reshape(-1)[order].reshape(k, cfg.N_BINS)
        outputs = []
        for start in range(0, len(held.query_lfp), chunk):
            query = held.query_lfp[start:start + chunk]
            if control == 'no_query_signal':
                query = torch.zeros_like(query)
            n = len(query)
            query_tokens = model.encode(query[:, None], keep.expand(n, -1))[:, 0]
            if k:
                out = model.decode(context_tokens.expand(n, -1, -1, -1), velocity[None].expand(n, -1, -1, -1),
                                   valid[None].expand(n, -1, -1), query_tokens)
            else:
                empty = query_tokens[:, :0, None].expand(-1, 0, cfg.N_BINS, -1)
                out = model.decode(empty, torch.zeros(n, 0, cfg.N_BINS, 2, device=device),
                                   torch.zeros(n, 0, cfg.N_BINS, dtype=torch.bool, device=device), query_tokens)
            outputs.append(out.float())
    return torch.cat(outputs)


def evaluate(model, sessions, conditions, k_values, velocity_scale, baseline, controls=(None,), draws=10,
             control_draws=3):
    """Score the model on every session, condition, k, draw and control. Returns a tidy table."""
    device = velocity_scale.device
    rows = []
    model.eval()
    off_cache = {}
    for session in sessions:
        is_stim = int(C.manifest().loc[session, 'stim']) == 1
        for condition in conditions:
            if condition == 'amp10' and not is_stim:
                continue
            held = load_held_out(session, condition, velocity_scale, device)
            off_session = None
            if is_stim and 'off_context' in controls:
                off_name = C.splits()['off_reference'][session]
                if off_name not in off_cache:
                    off_cache[off_name] = load_held_out(off_name, 'recorded', velocity_scale, device)
                off_session = off_cache[off_name]

            for k in k_values:
                for draw in range(n_draws(k, draws)):
                    positions = context_positions(held, k, draw)
                    labels_only = labels_only_prediction(held, positions, baseline, velocity_scale)
                    base_r2, _, _ = session_r2(labels_only, held.query_velocity, held.query_valid)
                    for control in controls:
                        if control is not None and k == 0 and control != 'no_query_signal':
                            continue
                        if control is not None and draw >= control_draws:   # controls use the first draws
                            continue
                        if control == 'off_context':
                            if off_session is None:
                                continue
                            use = context_positions(off_session, k, draw)
                        else:
                            use = positions
                        scaled = predict_queries(model, held, use, control, off_session, seed=draw)
                        prediction = (scaled * velocity_scale).cpu().numpy()
                        r2, r2_vx, r2_vy = session_r2(prediction, held.query_velocity, held.query_valid)
                        rows.append(dict(session=session, group=C.manifest().loc[session, 'condition'],
                                         input=condition, k=k, draw=draw, control=control or 'none',
                                         r2=r2, r2_vx=r2_vx, r2_vy=r2_vy, r2_labels_only=base_r2,
                                         neural_gain=r2 - base_r2))
    return pd.DataFrame(rows)
