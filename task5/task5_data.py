"""Cut every session into fixed-size trial arrays and cache them for Task 5.

Each cached session holds, for its included trials:

    lfp       (trials, 35, 2450)  recorded LFP in microvolts
    artifact  (trials, 35, 2450)  PARRM artifact estimate (stimulation sessions only)
    velocity  (trials, 40, 2)     Index velocity in cm/s
    valid     (trials, 40)        True where the velocity and the 500 ms of signal are usable
    trial     (trials,)           trial numbers, used to look up splits.json

Run once from the repository root:  python task5/task5_data.py
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import task5_config as cfg
import task3_common as C

L = C.L


def cut_windows(signal, times, fs, first_bins):
    """Cut (trials, channels, N_SAMPLES) from a continuous (channels, time) signal."""
    output = np.full((len(first_bins), signal.shape[0], cfg.N_SAMPLES), np.nan, np.float32)
    start_bins = first_bins - cfg.HISTORY_BINS
    starts = np.searchsorted(times, start_bins * L.BIN_S - 0.5 / fs)
    for row, start in enumerate(starts):
        if start >= 0 and start + cfg.N_SAMPLES <= signal.shape[1]:
            output[row] = signal[:, start:start + cfg.N_SAMPLES]
    return output


def build_session(session):
    """Build and save the cached arrays of one session."""
    path = os.path.join(cfg.CACHE_DIR, f'{session}.npz')
    if os.path.exists(path):
        return session, 'exists'

    is_stim = int(C.manifest().loc[session, 'stim']) == 1
    source = 'parrm' if is_stim else 'export'
    parrm = L.load_session(session, C.PATHS, source=source, with_artifact=is_stim,
                           manifest=C.loader_manifest())
    recorded = L.load_session(session, C.PATHS, source='export', manifest=C.loader_manifest())

    rows = C.trial_rows(session, 'all')
    first_bins = np.floor((rows.t_go.to_numpy() - cfg.PRE_S) / L.BIN_S + 1e-9).astype(np.int64)
    lfp = cut_windows(recorded.lfp, recorded.t, recorded.fs, first_bins)

    kinematics = C.kinematics(session)
    velocity = np.stack([kinematics.reindex(np.arange(b, b + cfg.N_BINS))[list(cfg.KIN_COLS)].to_numpy(np.float32)
                         for b in first_bins])
    has_signal = np.isfinite(lfp).all(axis=(1, 2))
    valid = np.isfinite(velocity).all(axis=2) & has_signal[:, None]

    arrays = dict(lfp=np.nan_to_num(lfp), velocity=velocity, valid=valid,
                  trial=rows.trial.to_numpy(), first_bin=first_bins)
    if is_stim:
        arrays['artifact'] = np.nan_to_num(cut_windows(parrm.artifact, parrm.t, parrm.fs, first_bins))

    temporary = path + '.tmp.npz'
    np.savez(temporary, **arrays)
    os.replace(temporary, path)
    return session, f'{len(first_bins)} trials, {valid.mean():.0%} valid bins'


def load_session_arrays(session):
    """Load one cached session as a dict of numpy arrays."""
    with np.load(os.path.join(cfg.CACHE_DIR, f'{session}.npz')) as file:
        return {name: file[name] for name in file.files}


def part_mask(session, trials, part):
    """Boolean mask over `trials` for a split part ('train', 'val', 'pool' or 'query')."""
    wanted = C.trial_rows(session, part).trial.to_numpy()
    return np.isin(trials, wanted)


def usable_sessions(split, condition=None):
    """Sessions of a split that have velocity in cm/s (this drops July 11)."""
    return [s for s in C.sessions(split, condition) if C.has_cm_velocity(s)]


if __name__ == '__main__':
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)
    names = list(C.manifest().session)
    with ProcessPoolExecutor(max_workers=8) as pool:
        for index, (name, message) in enumerate(pool.map(build_session, names), 1):
            print(f'{index:2d}/{len(names)} {name}: {message}', flush=True)
    print('DONE')
