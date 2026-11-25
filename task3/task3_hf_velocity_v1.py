"""Test whether 200-400 Hz LFP power improves horizontal-velocity decoding.

For OFF training and validation sessions only, compute one causal 200-400 Hz
power feature per channel from the exported ~1-kHz LFP; cache it separately;
then compare its contribution using the unchanged v7 residual ridge decoder and
sequence-specific time baseline. No original feature files are modified.

By default, adds the new band to the existing eight LMP/power features. The
--mode hf-only option evaluates the new band alone. Lambda is chosen on the
three validation OFF days by full-prediction R2 gain, as in v7; no test-day
recordings or stimulation-ON sessions are used.

The exported LFP was low-pass filtered and downsampled before this script sees
it. Its 200-400 Hz content is therefore only a first-stage HF test; native-rate
LFP4/RSn1 would be preferable for a subsequent spike-band analysis.

Run from the project folder:
  python task3/task3_hf_velocity_v1.py --mode augmented --lags 19 --lambda-max 1e8
  python task3/task3_hf_velocity_v1.py --mode hf-only --lags 19 --lambda-max 1e8
"""

import argparse
import os
from functools import lru_cache

import h5py
import numpy as np
import pandas as pd

import task3_diag_levels_v7 as D
from task3_paths import GRU_RESULTS

C = D.C
F = C.F
L = C.L
RAW_LOAD = F.load_features
BAND = (200, 400)
WINDOW_S = 0.256
NEW_NAME = 'pow_200_400'
CACHE_DIR = GRU_RESULTS / 'feature_cache' / 'hf200_400_off'


def hf_path(session):
    original = os.path.join(C.PATHS.root, 'features', 'hf200_400_off', f'{session}.h5')
    if os.path.isfile(original):
        return original
    return os.path.join(CACHE_DIR, f'{session}.h5')


def prepare_session(session, overwrite=False):
    path = hf_path(session)
    if os.path.exists(path) and not overwrite:
        return False
    path = os.path.join(CACHE_DIR, f'{session}.h5')
    recording = L.load_session(session, C.PATHS, channels='fixed', source='export',
                               reference='none', manifest=C.manifest().reset_index(drop=True))
    fs = float(recording.fs)
    if fs <= 2 * BAND[1]:
        raise ValueError(f'{session}: LFP sample rate {fs:g} Hz is too low for {BAND} Hz')
    first_bin = int(np.ceil(recording.t[0] / L.BIN_S - 1e-9))
    last_bin = int(np.floor(recording.t[-1] / L.BIN_S + 1e-9))
    bin_ids = np.arange(first_bin, last_bin)
    ends = np.searchsorted(recording.t, (bin_ids + 1) * L.BIN_S - 0.5 / fs) - 1
    power = F._band_power(recording.lfp, ends, fs, WINDOW_S, *BAND)
    os.makedirs(CACHE_DIR, exist_ok=True)
    temporary = path + '.tmp'
    with h5py.File(temporary, 'w') as f:
        f.create_dataset('bin_id', data=bin_ids)
        f.create_dataset('channel_ids', data=recording.channels)
        f.create_dataset('logpow', data=power, compression='gzip', compression_opts=1)
        f.attrs['fs_hz'] = fs
        f.attrs['band_hz'] = BAND
        f.attrs['window_s'] = WINDOW_S
        f.attrs['reference'] = 'none'
        f.attrs['source'] = 'exported LFP (already low-pass filtered to ~500 Hz)'
        f.attrs['method'] = 'causal Hann-windowed log10 PSD, matching EECS182_NHP_features._band_power'
    os.replace(temporary, path)
    print(f'  saved {session}: {len(bin_ids)} bins, {len(recording.channels)} channels, fs={fs:.2f} Hz')
    return True


@lru_cache(maxsize=32)
def read_hf(session):
    with h5py.File(hf_path(session), 'r') as f:
        return f['bin_id'][:], f['channel_ids'][:], f['logpow'][:]


def load_augmented(session, paths=None, variant='export_none', normalize=True, pad=1.0, trials=None):
    if variant != 'export_none':
        raise ValueError('HF diagnostic uses the export_none reference only')
    feat = RAW_LOAD(session, paths, variant=variant, normalize=normalize, pad=pad, trials=trials)
    bin_ids, channels, power = read_hf(session)
    if not np.array_equal(bin_ids, feat['bin_id']):
        raise ValueError(f'{session}: HF bins do not match original feature bins')
    if not np.array_equal(channels, feat['channel_ids']):
        raise ValueError(f'{session}: HF channels do not match original feature channels')
    extra = power.T.astype(np.float32)  # (bins, channels)
    if normalize:
        start, stop = feat['norm']['window']
        in_span = (bin_ids * L.BIN_S >= start) & ((bin_ids + 1) * L.BIN_S <= stop)
        if not in_span.any():
            raise ValueError(f'{session}: empty HF normalization window')
        med = np.nanmedian(extra[in_span], axis=0)
        scale = 1.4826 * np.nanmedian(np.abs(extra[in_span] - med), axis=0)
        scale[~(scale > 0)] = 1.0
        extra = (extra - med) / scale
    return dict(feat, X=np.concatenate((feat['X'], extra[:, :, None]), axis=2),
                feature_names=[*feat['feature_names'], NEW_NAME])


def run(mode, lags, lambda_max, overwrite, prepare_only):
    train = [s for s in C.sessions('train', 'no_stim') if C.has_cm_velocity(s)]
    val = C.sessions('val', 'no_stim')
    sessions = list(dict.fromkeys([*train, *val]))
    if len(train) < 2 or len(val) < 1:
        raise SystemExit('Expected OFF training and validation sessions in splits.json')
    print(f'HF 200-400 Hz OFF diagnostic: {len(train)} train sessions, {len(val)} validation sessions')
    print('Preparing causal power from exported ~1-kHz LFP; no test or ON sessions are accessed.')
    missing = []
    for session in sessions:
        if not os.path.isfile(os.path.join(C.PATHS.export_dir, f'{session}.h5')):
            missing.append(f'raw export {session}')
        if not os.path.isfile(os.path.join(C.PATHS.root, 'features', 'export_none', f'{session}.h5')):
            missing.append(f'base features {session}')
    if missing:
        raise SystemExit('Missing required files:\n  ' + '\n  '.join(missing))
    for idx, session in enumerate(sessions, 1):
        wrote = prepare_session(session, overwrite=overwrite)
        if not wrote:
            print(f'  {idx}/{len(sessions)} {session}: HF cache already exists')
    if prepare_only:
        print('Preparation complete; no models fitted.')
        return

    C.KIN_COLS = ('Index_vx_cm',)
    D.TIME_PROFILE = 'sequence'
    D.NORM = 'own'
    D.TRIAL_END_PAD = None
    C.REFERENCES = ('export_none',)
    C.LAMBDAS = np.logspace(0, np.log10(lambda_max),
                            int(round(2 * np.log10(lambda_max))) + 1)
    F.load_features = load_augmented
    names = ([NEW_NAME] if mode == 'hf-only' else [*F.FEATURE_NAMES, NEW_NAME])
    feature_set = f'hf200_400_{mode.replace("-", "_")}'
    C.FEATURE_SETS[feature_set] = names
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    profile = D.fit_time_profile(train, C.REFERENCES[0], feature_set, lags)
    D.use_window_arrays(profile)
    decoder = D.fit_off_decoder_gain(feature_set, lags, profile)
    D.print_lambda_search_gain(decoder['search'])
    model = decoder['model']
    rows = [dict(level=level, session=s,
                 **D.score(model, profile, s, 'export_none', part, feature_set, lags, True))
            for level, names, part in [('train', train, 'train'), ('held_out', train, 'val'),
                                       ('val_days', val, 'query')] for s in names]
    results = pd.DataFrame(rows)
    results['reference'] = 'export_none'
    results['lambda'] = decoder['lam']
    prefix = f'task3_hf_velocity_{mode.replace("-", "_")}_lags{lags}'
    results.to_csv(os.path.join(C.RESULTS_DIR, f'{prefix}_sessions.csv'), index=False)
    decoder['search'].to_csv(os.path.join(C.RESULTS_DIR, f'{prefix}_lambda.csv'), index=False)
    summary = results.groupby('level', sort=False).agg(
        n_sessions=('session', 'size'), n_bins=('n_bins', 'mean'),
        r2=('r2', 'mean'), r2_time=('r2_time', 'mean'), gain=('r2_gain', 'mean'))
    print('\nNeural contribution to vx (mean R2 across OFF sessions):')
    print(summary.to_string(float_format=lambda v: f'{v:.5f}'))
    if lags == 19:
        print('\nExisting zero-HF benchmark (v7, export_none, 19 lags): '
              'held_out gain +0.0050; val_days gain +0.0033.')
        print('Compare bin counts: held_out ~336/session, val_days ~1004/session.')
    print('All results saved under results/task3/. No test days were inspected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('augmented', 'hf-only'), default='augmented')
    parser.add_argument('--lags', type=int, default=19)
    parser.add_argument('--lambda-max', type=float, default=1e8)
    parser.add_argument('--overwrite-hf', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.mode, args.lags, args.lambda_max, args.overwrite_hf, args.prepare_only)
