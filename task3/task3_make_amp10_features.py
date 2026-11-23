"""Feature caches for the 10x PARRM-artifact semisynthetic ON recordings.

For every stimulation-ON session (validation and test days by default), the 35
decoding channels are rebuilt as

    LFP_10x = LFP_recorded + 9 * PARRM_artifact

(the same construction as semisynthetic/EECS182_make_semisynthetic_h5.py), and
LMP + 7 band powers are computed exactly as for the recorded features
(EECS182_NHP_features.compute_session). Only the feature cache is written, to
data/features/parrm_amp10/<session>.h5; no amplified HDF5 recording is saved.
The recording reference is used (no CAR), matching export_none.

Run (from the repository root):
      python task3/task3_make_amp10_features.py              (val + test ON sessions)
      python task3/task3_make_amp10_features.py --split test
"""

import argparse
import os
import time

import numpy as np

import task3_common as C

L, F = C.L, C.F
VARIANT = 'parrm_amp10'
SCALE = 10.0
TOLERANCE = 0.03  # maximum relative RMS error of recorded = PARRM-cleaned + artifact


def amp10_recording(session, scale=SCALE):
    """The session's recording with the 35 decoding channels replaced by
    recorded + (scale - 1) x PARRM artifact. Returns (Session, reconstruction error)."""
    recorded = C.load_recording(session, 'export_none')
    parrm = L.load_session(session, C.PATHS, source='parrm', reference='none', with_artifact=True,
                           manifest=C.loader_manifest())
    x = recorded.lfp.astype(np.float64)
    a = parrm.artifact.astype(np.float64)
    error = float(np.sqrt(np.sum((x - parrm.lfp - a) ** 2) / np.sum(x ** 2)))
    if error > TOLERANCE:
        raise ValueError(f'{session}: recorded != cleaned + artifact ({error:.2%} relative RMS)')
    recorded.lfp = (x + (scale - 1.0) * a).astype(recorded.lfp.dtype)
    recorded.cache = {}
    return recorded, error


def compute_amp10(session, scale=SCALE):
    recorded, error = amp10_recording(session, scale)

    # Same bins, LMP and band powers as F.compute_session.
    first_bin = int(np.ceil(recorded.t[0] / L.BIN_S - 1e-9))
    last_bin = int(np.floor(recorded.t[-1] / L.BIN_S + 1e-9))
    bin_ids = np.arange(first_bin, last_bin)
    ends = np.searchsorted(recorded.t, (bin_ids + 1) * L.BIN_S - 0.5 / recorded.fs) - 1
    lmp = L.bin_lmp(recorded, bin_ids)
    logpow = np.stack([F._band_power(recorded.lfp, ends, recorded.fs, window, low, high)
                       for low, high, window in F.BANDS], axis=1)
    feats = dict(session=session, bin_id=bin_ids, lmp=lmp, logpow=logpow,
                 channels=recorded.channels, fs=recorded.fs,
                 source=f'semisynthetic_parrm_artifact_x{scale:g}', reference='none')
    return feats, error


def main(split, overwrite):
    start = time.time()
    folder = F.features_dir(C.PATHS, VARIANT)
    os.makedirs(folder, exist_ok=True)
    parts = ('val', 'test') if split == 'val-test' else (split,)
    todo = [s for part in parts for s in C.sessions(part, 'stim')]
    for i, session in enumerate(todo, 1):
        filename = os.path.join(folder, f'{session}.h5')
        if os.path.exists(filename) and not overwrite:
            print(f'{i:2d}/{len(todo)} {session}: exists')
            continue
        feats, error = compute_amp10(session)
        F.save_session(feats, filename)
        print(f'{i:2d}/{len(todo)} {session}: written (PARRM reconstruction error {error:.3%})', flush=True)
    print(f'Done in {time.time() - start:.0f} s. Features in {os.path.relpath(folder, C.REPO)}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LMP + band-power features of the 10x PARRM-artifact ON recordings.')
    parser.add_argument('--split', choices=('val-test', 'val', 'test', 'train'), default='val-test')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    main(args.split, args.overwrite)
