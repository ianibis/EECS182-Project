"""Cache trial windows of raw LFP and Index velocity for the in-context decoder (plan 5.2a).

Each included trial is cut on the 50 ms TDT grid from t_go - 2.5 s to t_go + 2.0 s:
90 bins = 4500 samples per channel, in microvolts and not normalised; kin/valid cover the
same 90 bins. The standard decoded window is bins 40..79 (t_go - 0.5 to t_go + 1.5 s).
The margins let training shift the decoded window by up to +-0.5 s while keeping 500 ms of
encoder history, and give the ridge baseline enough history for 9-lag band-power features.

Each trial is labelled with its role from splits.json: 'train' / 'val' in training
sessions, 'pool' / 'query' in validation and test sessions.

Run from ICL/:  .venv/bin/python -m icl.cache [--source export|parrm] [--overwrite]
"""
import argparse
import json
import os
import time

import numpy as np

import EECS182_NHP_data_loader as L
from icl import config as C


def trial_parts(session, trials, splits):
    if session in splits['calibration']:
        pool = set(splits['calibration'][session]['pool'])
        query = set(splits['calibration'][session]['query'])
        parts = ['pool' if t in pool else 'query' if t in query else 'other' for t in trials]
    else:
        held_out = set(splits['train_val_trials'][session])
        parts = ['val' if t in held_out else 'train' for t in trials]
    assert 'other' not in parts, f'{session}: trial in neither pool nor query'
    return np.array(parts)


def cache_session(session, paths, manifest, source='export'):
    """Return (arrays, meta) for one session, or None when it has no cm velocity."""
    kin = L.load_kinematics(session, paths)
    if kin is None or kin.attrs.get('calibration') is None:
        return None
    sess = L.load_session(session, paths, source=source, manifest=manifest)
    examples = L.trial_examples(sess, kin, align='t_go', pre=C.PRE_S, post=C.POST_S, kin_cols=C.KIN_COLS)

    lfp = np.stack([e['lfp'].reshape(e['lfp'].shape[0], -1) for e in examples]).astype(np.float32)
    assert lfp.shape[1:] == (C.N_CHANNELS, C.N_SAMPLES), f'{session}: lfp shape {lfp.shape}'
    assert np.isfinite(lfp).all(), f'{session}: NaN in cached LFP'
    arrays = dict(
        lfp=lfp,
        kin=np.stack([e['kin'] for e in examples]).astype(np.float32),        # (n, 90, 2)
        valid=np.stack([e['kin_valid'] for e in examples]),                   # (n, 90)
        trial=np.array([e['trial'] for e in examples], np.int64),
        block=np.array([int(e['row'].task_block) for e in examples], np.int64),
        t_go=np.array([float(e['row'].t_go) for e in examples]),
        bin0=np.array([int(e['bin_id'][0]) for e in examples], np.int64),
        channels=np.asarray(sess.channels, np.int64),
    )
    info = sess.info
    meta = dict(session=session, source=sess.source, condition=str(info.condition), stim=int(info.stim),
                am_freq_hz=int(info.am_freq_hz), date=str(info.date), fs=float(sess.fs),
                n_trials=int(len(examples)))
    return arrays, meta


def save(path, arrays, meta):
    temporary = path[:-len('.npz')] + '.tmp.npz'
    np.savez(temporary, **arrays, meta=json.dumps(meta))
    os.replace(temporary, path)


def target_bins(offset=0):
    """Cache bins of the decoded window shifted by `offset` bins (0 = standard window)."""
    start = C.TARGET_START_BIN + offset
    return slice(start, start + C.N_TARGET_BINS)


def load_cached(session, source='export'):
    """Load one cached session: lfp (n, 35, 4500), kin (n, 90, 2), valid (n, 90), part (n,), ..."""
    with np.load(os.path.join(C.cache_dir(source), f'{session}.npz')) as file:
        out = {key: file[key] for key in file.files if key != 'meta'}
        out['meta'] = json.loads(str(file['meta']))
    return out


def load_index(source='export'):
    with open(os.path.join(C.cache_dir(source), 'index.json')) as file:
        return json.load(file)


def compute_stats(index, source='export'):
    """Velocity mean/sd from training trials of training no-stim sessions (the ICL training data)."""
    velocities, robust_sd = [], []
    for session, meta in index.items():
        if meta['split'] != 'train' or meta['stim'] != 0:
            continue
        data = load_cached(session, source)
        train = data['part'] == 'train'
        window = target_bins()
        velocities.append(data['kin'][train, window][data['valid'][train, window]])
        lfp = data['lfp'][train][..., window.start * C.SAMPLES_PER_BIN:window.stop * C.SAMPLES_PER_BIN]
        lfp = lfp.transpose(1, 0, 2).reshape(C.N_CHANNELS, -1)
        median = np.median(lfp, axis=1, keepdims=True)
        robust_sd.append(1.4826 * np.median(np.abs(lfp - median), axis=1))
    velocity = np.concatenate(velocities)
    return dict(
        vel_mean=velocity.mean(axis=0).tolist(), vel_std=velocity.std(axis=0).tolist(),
        n_velocity_bins=int(len(velocity)), n_sessions=len(velocities),
        lfp_robust_sd_uv_median=float(np.median(np.concatenate(robust_sd))),
        definition='training trials of training-split no-stim sessions; standard window, valid bins only',
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='export', choices=['export', 'parrm'])
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    paths = C.paths()
    manifest, splits = L.load_manifest(paths), L.load_splits(paths)
    split_of = {s: part for part in ('train', 'val', 'test') for s in splits['sessions'][part]}
    out_dir = C.cache_dir(args.source)
    os.makedirs(out_dir, exist_ok=True)

    index, skipped = {}, []
    start = time.time()
    for row in manifest.itertuples():
        session = row.session
        if args.source == 'parrm' and int(row.stim) == 0:
            continue                       # no-stim sessions have no PARRM file; use the export cache
        path = os.path.join(out_dir, f'{session}.npz')
        if os.path.exists(path) and not args.overwrite:
            index[session] = load_cached(session, args.source)['meta']
            continue
        result = cache_session(session, paths, manifest, args.source)
        if result is None:
            skipped.append(session)
            continue
        arrays, meta = result
        meta['split'] = split_of[session]
        arrays['part'] = trial_parts(session, arrays['trial'].tolist(), splits)
        save(path, arrays, meta)
        index[session] = meta
        print(f'{len(index):2d}  {session}  {meta["split"]:5s}  {meta["condition"]:7s}  {meta["n_trials"]} trials')

    with open(os.path.join(out_dir, 'index.json'), 'w') as file:
        json.dump(index, file, indent=1)
    stats = compute_stats(index, args.source) if args.source == 'export' else {}
    if stats:
        with open(os.path.join(out_dir, 'stats.json'), 'w') as file:
            json.dump(stats, file, indent=1)
    print(f'cached {len(index)} sessions in {time.time() - start:.0f} s; skipped (no cm velocity): {skipped}')
    print('stats:', stats)


if __name__ == '__main__':
    main()
