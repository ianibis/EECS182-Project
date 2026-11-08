"""Extract LMP and band-power features from the Butters LFP recordings.

Features are computed every 50 ms on the TDT clock, using only samples at or
before each bin's end. The saved values are unnormalized; load_features() applies
session or calibration-pool normalization when they are read.
"""

import argparse
import os
import h5py
import numpy as np
import pandas as pd

import EECS182_NHP_data_loader as L

BANDS = [(1, 4, 1.0), (4, 8, 0.5), (8, 13, 0.5), (13, 30, 0.256), (30, 50, 0.256),
         (70, 110, 0.256), (130, 200, 0.256)]
LINE_EXCLUDE = [(50, 70), (110, 130), (170, 190)]
FEATURE_NAMES = ['lmp'] + [f'pow_{lo}_{hi}' for lo, hi, _ in BANDS]
CHUNK = 500


def features_dir(paths, variant):
    return os.path.join(paths.root, 'features', variant)


def _band_power(lfp, ends, fs, win_s, lo, hi):
    n = int(round(win_s * fs))
    window = np.hanning(n).astype(np.float32)
    frequencies = np.fft.rfftfreq(n, 1 / fs)
    selected = (frequencies >= lo) & (frequencies <= hi)
    for start, stop in LINE_EXCLUDE:
        selected &= ~((frequencies >= start) & (frequencies <= stop))

    scale = 2.0 / (fs * np.sum(window ** 2))
    power = np.full((lfp.shape[0], len(ends)), np.nan, np.float32)
    usable = np.flatnonzero(ends - n + 1 >= 0)

    for start in range(0, len(usable), CHUNK):
        indices = usable[start:start + CHUNK]
        samples = np.stack([lfp[:, end - n + 1:end + 1] for end in ends[indices]], axis=1)
        spectrum = np.abs(np.fft.rfft(samples * window, axis=2)) ** 2 * scale
        power[:, indices] = np.log10(spectrum[:, :, selected].mean(axis=2) + 1e-12)
    return power


def compute_session(session, paths, source='export', reference='none', manifest=None):
    """Compute LMP and seven power bands for every complete 50 ms bin.

    Windows that extend before the recording contain NaN.
    """
    recording = L.load_session(session, paths, source=source, reference=reference, manifest=manifest)
    first_bin = int(np.ceil(recording.t[0] / L.BIN_S - 1e-9))
    last_bin = int(np.floor(recording.t[-1] / L.BIN_S + 1e-9))
    bin_ids = np.arange(first_bin, last_bin)
    ends = np.searchsorted(recording.t, (bin_ids + 1) * L.BIN_S - 0.5 / recording.fs) - 1

    lmp = L.bin_lmp(recording, bin_ids)
    logpow = np.stack([
        _band_power(recording.lfp, ends, recording.fs, window, low, high)
        for low, high, window in BANDS
    ], axis=1)
    return dict(session=session, bin_id=bin_ids, lmp=lmp, logpow=logpow,
                channels=recording.channels, fs=recording.fs,
                source=recording.source, reference=reference)


def save_session(F, fn):
    temporary = fn + '.tmp'
    with h5py.File(temporary, 'w') as file:
        file['bin_id'] = F['bin_id']
        file['t_bin'] = F['bin_id'] * L.BIN_S
        file['lmp'] = F['lmp']
        file['logpow'] = F['logpow']
        file['channel_ids'] = F['channels']
        file.attrs['session'] = F['session']
        file.attrs['source'] = F['source']
        file.attrs['reference'] = F['reference']
        file.attrs['lfp_fs'] = F['fs']
        file.attrs['bin_s'] = L.BIN_S
        file.attrs['bands_hz'] = np.array([[lo, hi] for lo, hi, _ in BANDS], float)
        file.attrs['band_window_s'] = np.array([window for _, _, window in BANDS], float)
        file.attrs['line_exclude_hz'] = np.array(LINE_EXCLUDE, float)
        file.attrs['lmp_window_s'] = L.LMP_WIN_S
        file.attrs['definition'] = ('bin j covers [0.05 j, 0.05 (j + 1)) s on the TDT clock; every feature uses '
                                    'only samples up to the end of its bin; lmp is the mean LFP over the '
                                    'preceding 256 ms (microvolts); logpow is log10 of the mean Hann-windowed '
                                    'PSD (microvolts^2/Hz) over the band, window ending at the bin end')
        file.attrs['created_by'] = 'butters_features_v1.py'
    os.replace(temporary, fn)


def compute_all(paths=None, variants=('export_none',), sessions=None, overwrite=False):
    """Save features for the requested sessions, skipping files already present."""
    paths = paths or L.Paths()
    manifest = L.load_manifest(paths)
    sessions = list(manifest.session) if sessions is None else sessions

    for variant in variants:
        source, reference = variant.split('_')
        directory = features_dir(paths, variant)
        os.makedirs(directory, exist_ok=True)

        for index, session in enumerate(sessions):
            filename = os.path.join(directory, f'{session}.h5')
            if os.path.exists(filename) and not overwrite:
                continue
            try:
                features = compute_session(session, paths, source, reference, manifest)
                save_session(features, filename)
                print(f'{variant}  {index + 1:2d}/{len(sessions)}  {session}')
            except Exception as error:
                print(f'{variant}  {index + 1:2d}/{len(sessions)}  {session}  ERROR {error}')


def load_features(session, paths=None, variant='export_none', normalize=True, pad=1.0, trials=None):
    """Load features as X (bins, channels, 8), optionally normalized.

    Normalization uses each feature's median and 1.4826 times its MAD. By default
    it uses included trials from the whole session. For held-out sessions, pass
    the calibration-pool trial rows through trials to exclude query data.
    """
    paths = paths or L.Paths()
    filename = os.path.join(features_dir(paths, variant), f'{session}.h5')
    with h5py.File(filename, 'r') as file:
        bin_ids = file['bin_id'][:]
        X = np.concatenate([file['lmp'][:][:, None, :], file['logpow'][:]], axis=1)
        channel_ids = file['channel_ids'][:]

    X = np.transpose(X, (2, 0, 1)).astype(np.float32)
    norm = None
    if normalize:
        selected_trials = pd.read_csv(os.path.join(paths.trial_dir, f'{session}_trials.csv'))
        selected_trials = selected_trials[selected_trials.include == 1] if trials is None else trials
        start = selected_trials.t_go.min() - pad
        stop = selected_trials.t_reward.max() + pad
        within = (bin_ids * L.BIN_S >= start) & ((bin_ids + 1) * L.BIN_S <= stop)
        median = np.nanmedian(X[within], axis=0)
        scale = 1.4826 * np.nanmedian(np.abs(X[within] - median), axis=0)
        scale[~(scale > 0)] = 1.0
        X = (X - median) / scale
        norm = dict(median=median, scale=scale, window=(start, stop))

    return dict(session=session, bin_id=bin_ids, X=X, channel_ids=channel_ids,
                feature_names=FEATURE_NAMES, variant=variant, norm=norm)


def trial_arrays(feat, trials, kin=None, align='t_go', pre=0.5, post=1.5, n_lags=0,
                 kin_cols=('Index_vx_cm', 'Index_vy_cm')):
    """Cut trial windows, adding n_lags earlier bins of feature history.

    Returns X, Y (or None), a valid-bin mask, and bin IDs. Missing features
    or kinematics are NaN and excluded by the mask.
    """
    positions = pd.Series(np.arange(len(feat['bin_id'])), index=feat['bin_id'])
    n_bins = int(round((pre + post) / L.BIN_S))
    X_trials, Y_trials, valid_trials, trial_bins = [], [], [], []
    flattened = feat['X'].reshape(len(feat['bin_id']), -1)

    for _, trial in trials.iterrows():
        first_bin = int(np.floor((trial[align] - pre) / L.BIN_S + 1e-9))
        bin_ids = np.arange(first_bin, first_bin + n_bins)
        history = []

        for lag in range(n_lags + 1):
            positions_for_lag = positions.reindex(bin_ids - lag).to_numpy()
            samples = np.full((n_bins, flattened.shape[1]), np.nan, np.float32)
            present = ~np.isnan(positions_for_lag)
            samples[present] = flattened[positions_for_lag[present].astype(int)]
            history.append(samples)

        X = np.concatenate(history, axis=1)
        valid = np.isfinite(X).all(axis=1)
        Y = None
        if kin is not None:
            Y = kin.reindex(bin_ids)[list(kin_cols)].to_numpy(np.float32)
            valid &= np.isfinite(Y).all(axis=1)

        X_trials.append(X)
        Y_trials.append(Y)
        valid_trials.append(valid)
        trial_bins.append(bin_ids)

    Y = np.stack(Y_trials) if kin is not None else None
    return np.stack(X_trials), Y, np.stack(valid_trials), np.stack(trial_bins)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=L.DEFAULT_ROOT)
    parser.add_argument('--variants', nargs='+', default=['export_none'])
    parser.add_argument('--sessions', nargs='*', default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    compute_all(L.Paths(root=args.root), args.variants, args.sessions, args.overwrite)
