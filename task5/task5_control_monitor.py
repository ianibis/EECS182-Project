"""Control C4: can the stimulation monitor alone predict movement?

A recurrent model can learn the average trajectory just by counting time steps, so its
R² on its own proves nothing. We therefore train two GRUs on the monitor features:

    matched      each trial receives its own monitor signal
    mismatched   each trial receives the monitor signal of another trial in the session

Timing is available to both. Only the matched model could use a true link between
stimulation and movement, so  leak = R²(matched) - R²(mismatched).

Run from the repository root (CPU only):
    python task5/task5_control_monitor.py --split val
    python task5/task5_control_monitor.py --split test
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task3_common as C
import task5_data as data

L = C.L
SEEDS = (1, 2, 3, 4, 5)
EPOCHS = 60
BATCH_SIZE = 64


def monitor_features(session, norm_part):
    """Five monitor features per bin: envelope mean and SD, sin and cos of AM phase, fraction on."""
    arrays = data.load_session_arrays(session)
    recording = L.load_session(session, C.PATHS, with_lfp=False, manifest=C.loader_manifest())
    starts = np.searchsorted(recording.t, arrays['first_bin'][:, None] * L.BIN_S
                             + np.arange(cfg.N_BINS) * L.BIN_S - 0.5 / recording.fs)
    index = starts[..., None] + np.arange(cfg.BIN_SAMPLES)
    index = np.clip(index, 0, len(recording.t) - 1)

    envelope = np.nan_to_num(recording.stim_envelope[index])
    used = data.part_mask(session, arrays['trial'], norm_part)
    center = np.median(envelope[used])
    scale = 1.4826 * np.median(np.abs(envelope[used] - center)) or 1.0
    envelope = np.clip((envelope - center) / scale, -cfg.CLIP_Z, cfg.CLIP_Z)

    phase = recording.stim_phase[index[..., -1]]
    features = np.stack([envelope.mean(axis=2), envelope.std(axis=2), np.nan_to_num(np.sin(phase)),
                         np.nan_to_num(np.cos(phase)), recording.stim_on[index].mean(axis=2)], axis=2)
    return features.astype(np.float32), arrays


def collect(sessions, part, norm_part, mismatched):
    """Stack trials of several sessions, optionally pairing each trial with another trial's monitor."""
    X, Y, valid, owner = [], [], [], []
    for session in sessions:
        features, arrays = monitor_features(session, norm_part)
        mask = data.part_mask(session, arrays['trial'], part)
        chosen = features[mask]
        X.append(np.roll(chosen, 1, axis=0) if mismatched else chosen)
        Y.append(np.nan_to_num(arrays['velocity'][mask]))
        valid.append(arrays['valid'][mask])
        owner += [session] * int(mask.sum())
    return np.concatenate(X), np.concatenate(Y), np.concatenate(valid), np.array(owner)


class MonitorGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(5, 64, batch_first=True)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        return self.head(self.gru(x)[0])


def fit(X, Y, valid, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scale = Y[valid].std(axis=0)
    x, y, v = torch.as_tensor(X), torch.as_tensor(Y / scale, dtype=torch.float32), torch.as_tensor(valid)
    model = MonitorGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(EPOCHS):
        order = rng.permutation(len(x))
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[rows]) - y[rows]).square().mean(dim=2)[v[rows]].mean()
            loss.backward()
            optimizer.step()
    return model.eval(), scale


def run(split):
    train_sessions = data.usable_sessions('train', 'stim')
    held_sessions = data.usable_sessions(split, 'stim')
    rows = []
    for name, mismatched in (('matched', False), ('mismatched', True)):
        X, Y, valid, _ = collect(train_sessions, 'train', 'train', mismatched)
        X_held, Y_held, valid_held, owner = collect(held_sessions, 'query', 'pool', mismatched)
        for seed in SEEDS:
            model, scale = fit(X, Y, valid, seed)
            with torch.no_grad():
                prediction = model(torch.as_tensor(X_held)).numpy() * scale
            for session in held_sessions:
                mask = owner == session
                r2 = C.r2(Y_held[mask][valid_held[mask]], prediction[mask][valid_held[mask]]).mean()
                rows.append(dict(session=session, group=C.manifest().loc[session, 'condition'],
                                 monitor=name, seed=seed, r2=float(r2)))
            print(f'{name} seed {seed}: mean R2 {np.mean([r["r2"] for r in rows[-len(held_sessions):]]):+.4f}',
                  flush=True)

    table = pd.DataFrame(rows)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    table.to_csv(os.path.join(cfg.RESULTS_DIR, f'control_monitor_{split}.csv'), index=False)
    by_session = table.groupby(['session', 'monitor']).r2.mean().unstack()
    leak = by_session.matched - by_session.mismatched
    print(f'\nMatched {by_session.matched.mean():+.4f}   mismatched {by_session.mismatched.mean():+.4f}   '
          f'leak {leak.mean():+.4f}  (session range {leak.min():+.4f} to {leak.max():+.4f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    torch.set_num_threads(4)
    run(parser.parse_args().split)
