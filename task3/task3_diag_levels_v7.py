"""Task 3.1 diagnostic: at which step is decoding accuracy lost?

Fits the same OFF decoder as task3_1_ridge.py (same training trials, same lambda and reference
chosen on the validation days), then scores it at four levels, from easiest to hardest:
  train      the trials it was fit on (training-day OFF sessions)
  held_out   the held-out block of those same sessions (same days, unseen trials; normalized
             like the training trials, over the whole session)
  val_days   query trials of the validation-day OFF sessions (used to choose lambda, so slightly optimistic)
  test_days  query trials of the test-day OFF sessions (the 3.1 OFF number)
R² is computed as in 3.1: over all valid bins of a session, per target column, then averaged over
the columns, then over sessions.

--target velocity (default) decodes Index vx and vy in cm/s, as in 3.1; --target vx decodes
horizontal velocity alone. --target speed decodes Index speed in cm/s, and --target position
Index x and y in screen cm (lambda and reference are chosen again for each). Position, like velocity,
is NaN where the hand is off the screen plane.
Band power and LMP can track whether and how fast the hand moves without tracking its direction; if
speed decodes much better than velocity, the features carry little direction information.

Time-only baseline (r2_time): predicts each bin from its position in the trial window alone, as the
mean target at that position over the training trials (the same trials the decoder is fit on). It is
scored on exactly the bins the decoder is scored on. Because the window is locked to the go cue, a
stereotyped reach is partly predictable from time alone; r2_gain = r2 - r2_time is what the LFP adds.
For velocity and position, r2_time_vx, r2_time_vy (or _x, _y) give the baseline per axis.
--time-profile sequence fits the baseline separately for each target sequence (manifest column
'sequence') and uses each session's own sequence. The two sequences, 1-9-4-3-7 and 3-7-6-1-9, have the
same vertical path and opposite horizontal ones, so a baseline pooled over both cannot predict vx;
the sequence baseline is the fair one for velocity and position, and with --residual it is the one the
LFP decoder must improve on.

--top N keeps only the N channel-feature pairs whose values (no history) correlate most with the
targets on the training trials, mean |r| over the target columns, as Flint et al. 2012 selected
features before their decoder; with --residual the correlation is with the residual. The selection is
made once per reference, from the training trials only, and the history is then added to the kept
pairs.

--spatial-pca replaces, for each feature, the channels with their principal components across
channels, fit on the training trials (no history) and each scaled to unit variance there; --pca-drop K
removes the first K components (K = 1 removes the component shared by all channels, a data-driven
version of a common average). Without the scaling a full rotation would change nothing, since ridge
predictions do not depend on rotating the inputs; scaled, the small, channel-specific components are
no longer shrunk away before the shared one. The share of variance in the first component is printed
for each feature.

--per-channel fits and scores one decoder per channel, using that channel's features alone, plus one
with all channels, and reports the test-day R2 of each (and r2_gain against the time-only baseline).
It uses the first reference only (export_none unless --feature-tag), skips the other levels, and
writes task3_diag_levels_<target>_perchannel_..._lags<N>.csv. With the channels nearly identical in
beta, most channels should score alike; a channel well above the rest on vx would point to local
direction information that the shared signal hides.

--features takes lmp, bands or lmp+bands, or a comma-separated list of feature names, e.g.
--features lmp,pow_1_4,pow_70_110,pow_130_200. --lambda-max extends the lambda grid (half-decade
steps from 1) up to the given value, for when the chosen lambda is at the top of the grid.

--residual fits the LFP decoder to what the time-only baseline misses (target minus the mean at that
position), with lambda and reference chosen on the validation days as before, and predicts
time-only baseline + LFP decoder. r2 is then the combined prediction, and r2_gain > 0 means the LFP
carries trial-specific movement information beyond the go-locked timing. In this mode lambda and
reference are chosen to maximize validation R2 gain of the full prediction over time-only, not
R2 of the residual alone. Both scores and full-prediction R2 are printed for each lambda.

--window go (default) scores 0.5 s before the go cue to 1.5 s after it, as in 3.1: rest, reaction
time and the first reach. --window trial scores 0.5 s before the go cue to 0.5 s after reward, so all
five reaches; each trial's bins after reward + 0.5 s are dropped. The window is cut at a fixed length
(--post, seconds after the go cue; by default the 95th percentile of reward minus go cue, plus 0.5 s),
so the longest trials are truncated; the count is printed. Lambda, the reference and the time-only
baseline are all refit for the window used.

Levels test_on_am3hz and test_on_am20hz score the same OFF-trained decoder on the query trials of the
test-day stimulation sessions, normalized from their own calibration pool as in 3.1. A table of each
ON session against its same-day OFF session follows the summary.

--norm off z-scores the test-day stimulation sessions with their same-day OFF session's statistics
(median and 1.4826 x MAD over the OFF session's calibration-pool span) instead of their own, so the
features are not recalibrated before decoding; everything else, including the OFF sessions, is
normalized as before. It also prints how far each feature of each ON session sits from its OFF
session in those units (median over the ON session's trial span, then over channels). A shift
concentrated in the bands that hold the stimulation frequency points to the stimulation; a similar
shift in every band points to differences between the two recordings.

--features lmp uses the LMP alone (with lmp+bands and bands, the three-way ablation).

--feature-tag TAG reads features from features/export_none_<TAG>/ (computed for all 64 channels with
EECS182_NHP_features.py --channels all --tag TAG) instead of features/export_none/ and
features/export_car/; the common average reference is not used then. --channels picks the channel
set from those files: fixed (the 35 kept channels), pickup (fixed plus the stimulation-pickup channels
11 and 23), clean_canonical (fixed plus 9, 12, 14, 24 and 63, canonical channels that looked healthy
in the channel audit), all (every channel in the file), or a comma-separated list of channel numbers.

--smooth HZ low-pass filters the targets before fitting and scoring: each session's continuous 50 ms
series is gap-filled by linear interpolation, filtered in the frequency domain with zero phase
(gain 1 up to HZ, raised-cosine taper to 0 at 1.5 x HZ, ends padded by reflection), and the gaps are
put back. Lambda is chosen on the smoothed targets. r2 is scored against the smoothed targets;
r2_raw scores the same predictions against the raw targets, which says whether smoothing the
training target predicts the actual movement better. Choose the cutoff on val_days r2_raw only.

Reads the Box project layout; task3_common.py is not edited, its paths are replaced below. The
project folder is the one that holds task3/:
  EECS182_data_manifest.csv, splits.json, EECS182_NHP_data_loader.py, EECS182_NHP_features.py
  export_outputs/h5_export_outputs/<session>.h5   (touch data, read by the loader for the cm calibration)
  export_outputs/trials/<session>_trials.csv
  features/<variant>/<session>.h5                  (export_none and export_car are both needed)
  kinematics_v3/bins50ms/<session>_bins50ms.csv
Results go to results/task3/ in the project folder, as for the other Task 3 scripts.

v7: --target vx; with --residual, choose lambda/reference by full-prediction validation R2 gain
rather than residual-only R2 (including --per-channel), and print/save all three validation metrics.
Adds _selectgain to residual output names to preserve prior results; other fitting unchanged.
v6: --spatial-pca and --pca-drop.
v5: --features accepts a list of feature names; --time-profile sequence; per-axis time-only R2;
--lambda-max; --top; --per-channel.
v4 (current numbering; the file sent before as v11): --norm.
Earlier changes, in the old numbering:
v11: the butters_features_v1 fallback is removed; the manifest is read as EECS182_data_manifest.csv; without
--feature-tag, feature files are read as in v9 (no channel selection).
v10: --feature-tag and --channels.
v9: --target position.
v8: test-day ON levels with the ON vs same-day OFF table, --features lmp, --smooth.
v7: --window trial (all five reaches) and --post.
v6: --residual.
v5: time-only baseline (r2_time, r2_gain) next to the decoder at every level.
v4: --target speed.
v3: export files are read from export_outputs/h5_export_outputs/. Before fitting, every file the run
reads is checked for every session it uses, and all missing files are listed at once.
v2: Box project layout instead of the repository's data/ layout.

Run (from the project folder):  python task3/task3_diag_levels_v7.py --target vx --lags 19 --time-profile sequence --residual --lambda-max 1e8
Writes results/task3/task3_diag_levels_<features>_lags<N>.csv for velocity (same name as v3) and
task3_diag_levels_speed_<features>_lags<N>.csv for speed, one row per session and level; with
--residual, '_residual' and '_selectgain' are added before '_lags', with --window trial, '_trialwin', and with
--smooth, '_smooth<HZ>Hz', with --channels other than fixed, '_ch-<set>', and with --norm off, '_normoff'.
"""

import argparse
import contextlib
import io
import os
import sys
import time
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import task3_common as C  # noqa: E402
from task3_paths import DATA_ROOT, metadata_path

MANIFEST = metadata_path(DATA_ROOT, 'EECS182_data_manifest.csv')
TARGETS = {'velocity': ('Index_vx_cm', 'Index_vy_cm'), 'vx': ('Index_vx_cm',),
           'speed': ('Index_speed_cm',),
           'position': ('Index_x_cm', 'Index_y_cm')}
SHORT = {'Index_vx_cm': 'vx', 'Index_vy_cm': 'vy', 'Index_speed_cm': 'speed', 'Index_x_cm': 'x', 'Index_y_cm': 'y'}
END_PAD = 0.5
TRIAL_END_PAD = None
SMOOTH_HZ = None
NORM = 'own'
NORM_PAD = 1.0
ON_SESSIONS = set()
TIME_PROFILE = 'pooled'
SELECTED = {}
PCA = {}
C.FEATURE_SETS['lmp'] = ['lmp']
RAW_KINEMATICS = C.kinematics
RAW_LOAD_FEATURES = C.F.load_features
FIXED_CHANNELS = [21, 22, *range(26, 39), *range(41, 46), 47, 48, 49, *range(51, 56), *range(57, 63), 64]
CHANNEL_SETS = {'fixed': FIXED_CHANNELS, 'pickup': sorted(FIXED_CHANNELS + [11, 23]),
                'clean_canonical': sorted(FIXED_CHANNELS + [9, 12, 14, 24, 63]), 'all': None}

SESSION_FILES = {
    'export .h5': lambda s: os.path.join(C.PATHS.export_dir, f'{s}.h5'),
    'trial table': lambda s: os.path.join(C.PATHS.trial_dir, f'{s}_trials.csv'),
    'kinematics bins': lambda s: os.path.join(C.PATHS.kin_dir, f'{s}_bins50ms.csv'),
    **{f'features/{v}': (lambda s, v=v: os.path.join(C.PATHS.root, 'features', v, f'{s}.h5')) for v in C.REFERENCES},
}


@lru_cache(maxsize=None)
def box_manifest():
    return pd.read_csv(MANIFEST).set_index('session', drop=False)


C.manifest = box_manifest


def use_channels(channels):
    """Make every feature load keep only the given channels (None = all in the file), in file order."""
    def load_features(*args, **kwargs):
        feat = RAW_LOAD_FEATURES(*args, **kwargs)
        ids = np.asarray(feat['channel_ids']).ravel().astype(int)
        if channels is None:
            return feat
        missing = sorted(set(channels) - set(ids))
        if missing:
            raise SystemExit(f'{feat["session"]}: channels {missing} are not in the feature file '
                             f'({len(ids)} channels); compute it with --channels all')
        keep = np.isin(ids, channels)
        return dict(feat, X=feat['X'][:, keep, :], channel_ids=ids[keep])
    C.F.load_features = load_features


def parse_channels(text):
    if text in CHANNEL_SETS:
        return CHANNEL_SETS[text]
    try:
        return sorted({int(c) for c in text.split(',')})
    except ValueError:
        raise SystemExit(f'--channels: {text!r} is not a set name ({", ".join(CHANNEL_SETS)}) or a list of numbers')


def session_groups():
    """Sessions this run reads: training-day, validation-day and test-day OFF sessions, and the
    test-day stimulation sessions by condition."""
    S = C.splits()
    train = [s for s in C.sessions('train', 'no_stim') if C.has_cm_velocity(s)]
    test_stim = C.sessions('test', 'stim')
    test_off = sorted({S['off_reference'][s] for s in test_stim})
    test_on = {c: [s for s in test_stim if C.manifest().loc[s, 'condition'] == c] for c in ('am3hz', 'am20hz')}
    return train, C.sessions('val', 'no_stim'), test_off, test_on


def lowpass_series(x, cutoff, fs):
    """Zero-phase low-pass of a series with gaps: interpolate the gaps, filter, put the gaps back."""
    finite = np.isfinite(x)
    if finite.sum() < 2:
        return x.copy()
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[finite], x[finite])
    pad = min(len(x) - 1, int(np.ceil(2 * fs / cutoff)))
    padded = np.concatenate([filled[pad:0:-1], filled, filled[-2:-pad - 2:-1]]) if pad > 0 else filled
    f = np.fft.rfftfreq(len(padded), 1 / fs)
    gain = 0.5 - 0.5 * np.cos(np.pi * np.clip((1.5 * cutoff - f) / (0.5 * cutoff), 0, 1))
    mean = padded.mean()
    y = np.fft.irfft(np.fft.rfft(padded - mean) * gain, len(padded)) + mean
    y = y[pad:pad + len(x)]
    y[~finite] = np.nan
    return y


@lru_cache(maxsize=None)
def smoothed_kinematics(session):
    """Kinematics with every target column low-pass filtered; raw copies kept as <column>_raw."""
    K = RAW_KINEMATICS(session)
    if K is None:
        return None
    K = K.reindex(np.arange(K.index.min(), K.index.max() + 1))
    for col in sorted({c for cols in TARGETS.values() for c in cols}):
        if col in K:
            K[f'{col}_raw'] = K[col]
            K[col] = lowpass_series(K[col].to_numpy(np.float64), SMOOTH_HZ, 1 / C.L.BIN_S)
    return K


def check_files():
    missing = []
    if not os.path.exists(MANIFEST):
        missing.append(MANIFEST)
    if not os.path.exists(metadata_path(DATA_ROOT, 'splits.json')):
        missing.append(f'splits.json in {ROOT}')
    if missing:
        raise SystemExit('Not found:\n  ' + '\n  '.join(missing))

    train, val, test_off, test_on = session_groups()
    names = sorted({*train, *val, *test_off, *[s for group in test_on.values() for s in group]})
    lines = []
    for kind, path_of in SESSION_FILES.items():
        absent = [s for s in names if not os.path.exists(path_of(s))]
        if absent:
            lines.append(f'{kind}: {len(absent)} of {len(names)} sessions missing, expected like '
                         f'{path_of(absent[0])}\n    ' + ', '.join(absent))
    if lines:
        raise SystemExit('Missing files:\n  ' + '\n  '.join(lines))
    print(f'All files found for the {len(names)} sessions this run reads.')


def arrays_with_position(session, variant, part, feature_set, n_lags):
    """C.session_arrays(), plus each bin's position in its trial window and, with --smooth, the raw
    targets of the same bins; with --window trial, bins after each trial's reward + END_PAD are dropped."""
    if NORM == 'off' and session in ON_SESSIONS:
        feat = off_normalized(session, variant)
    else:
        norm_rows = C.trial_rows(session, 'pool') if session in C.splits()['calibration'] else None
        feat = C.F.load_features(session, C.PATHS, variant=variant, normalize=True, trials=norm_rows)
    keep = [feat['feature_names'].index(name) for name in C.FEATURE_SETS[feature_set]]
    feat = dict(feat, X=np.clip(feat['X'][:, :, keep], -C.CLIP_Z, C.CLIP_Z),
                feature_names=C.FEATURE_SETS[feature_set])
    if variant in SELECTED:
        flat = feat['X'].reshape(len(feat['bin_id']), -1)[:, SELECTED[variant]]
        feat = dict(feat, X=flat[:, :, None])
    if variant in PCA:
        components = [(feat['X'][:, :, f] - mean) @ weights for f, (mean, weights, _) in enumerate(PCA[variant])]
        feat = dict(feat, X=np.stack(components, axis=2))
    rows = C.trial_rows(session, part)
    raw_cols = [f'{c}_raw' for c in C.KIN_COLS] if SMOOTH_HZ else []
    X, Y, valid, bins = C.F.trial_arrays(feat, rows, kin=C.kinematics(session), align=C.ALIGN, pre=C.PRE,
                                         post=C.POST, n_lags=n_lags, kin_cols=[*C.KIN_COLS, *raw_cols])
    if TRIAL_END_PAD is not None:
        last = np.ceil((rows.t_reward.to_numpy() + TRIAL_END_PAD) / C.L.BIN_S - 1e-9)
        valid = valid & (bins < last[:, None])
    position = np.broadcast_to(np.arange(valid.shape[1]), valid.shape)
    Y = Y[valid]
    n = len(C.KIN_COLS)
    return X[valid], Y[:, :n], position[valid], (Y[:, n:] if SMOOTH_HZ else None)


def off_normalized(session, variant):
    """Features of a test-day stimulation session z-scored with the statistics of its same-day OFF
    session: median and 1.4826 x MAD over the bins from the OFF session's first calibration-pool go cue
    minus NORM_PAD to its last calibration-pool reward plus NORM_PAD, as load_features computes them."""
    off = C.splits()['off_reference'][session]
    on_feat = C.F.load_features(session, C.PATHS, variant=variant, normalize=False)
    off_feat = C.F.load_features(off, C.PATHS, variant=variant, normalize=False)
    if not np.array_equal(np.asarray(on_feat['channel_ids']), np.asarray(off_feat['channel_ids'])):
        raise SystemExit(f'{session} and {off} have different channels in features/{variant}/')
    rows = C.trial_rows(off, 'pool')
    a, b = rows.t_go.min() - NORM_PAD, rows.t_reward.max() + NORM_PAD
    ids = off_feat['bin_id']
    in_span = (ids * C.L.BIN_S >= a) & ((ids + 1) * C.L.BIN_S <= b)
    X_off = off_feat['X'][in_span]
    median = np.nanmedian(X_off, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(X_off - median), axis=0)
    scale[~(scale > 0)] = 1.0
    return dict(on_feat, X=(on_feat['X'] - median) / scale,
                norm=dict(median=median, scale=scale, window=(a, b), source=off))


def feature_shift(session, variant):
    """Median over the session's trial span, then over channels, of each feature in OFF-session z units."""
    feat = off_normalized(session, variant)
    rows = C.trial_rows(session, 'all')
    a, b = rows.t_go.min() - NORM_PAD, rows.t_reward.max() + NORM_PAD
    ids = feat['bin_id']
    in_span = (ids * C.L.BIN_S >= a) & ((ids + 1) * C.L.BIN_S <= b)
    shift = np.nanmedian(np.nanmedian(feat['X'][in_span], axis=0), axis=0)
    return dict(zip(feat['feature_names'], shift))


def parse_features(text):
    """A feature-set name, or a comma-separated list of feature names registered as a new set."""
    if text in C.FEATURE_SETS:
        return text
    names = [name.strip() for name in text.split(',') if name.strip()]
    unknown = [name for name in names if name not in C.F.FEATURE_NAMES]
    if unknown or not names:
        raise SystemExit(f'--features: {", ".join(unknown) if unknown else repr(text)} not known; use '
                         f'{", ".join(sorted(C.FEATURE_SETS))} or a comma-separated list of {", ".join(C.F.FEATURE_NAMES)}')
    key = '-'.join(names)
    C.FEATURE_SETS[key] = names
    return key


def sequence_of(session):
    if TIME_PROFILE == 'pooled':
        return 'all'
    manifest = C.manifest()
    if 'sequence' not in manifest.columns:
        raise SystemExit('--time-profile sequence needs a sequence column in the manifest')
    return str(manifest.loc[session, 'sequence'])


def fit_time_profile(sessions, variant, feature_set, n_lags):
    """Mean target at each position of the trial window over the training trials, for all of them
    together or, with --time-profile sequence, separately for each target sequence."""
    groups = {}
    for s in sessions:
        Y, P = arrays_with_position(s, variant, 'train', feature_set, n_lags)[1:3]
        groups.setdefault(sequence_of(s), []).append((Y, P))
    n_pos = int(round((C.PRE + C.POST) / C.L.BIN_S))
    profiles = {}
    for key, parts in groups.items():
        Y = np.concatenate([y for y, _ in parts])
        P = np.concatenate([p for _, p in parts])
        profile = np.tile(Y.mean(axis=0), (n_pos, 1))
        for p in np.unique(P):
            profile[p] = Y[P == p].mean(axis=0)
        profiles[key] = profile
    if TIME_PROFILE == 'sequence':
        print('Time-only baseline per sequence, from ' + ', '.join(
            f'{key} ({len(parts)} sessions)' for key, parts in sorted(groups.items())))
    return profiles


def profile_values(profiles, session, P):
    key = sequence_of(session)
    if key not in profiles:
        raise SystemExit(f'{session}: no training session has sequence {key}, so there is no time-only baseline for it')
    return profiles[key][P]


def select_features(sessions, feature_set, top_n, profile, residual):
    """Keep, per reference, the top_n channel-feature pairs most correlated with the targets (no history)."""
    for variant in C.REFERENCES:
        Xs, Ys = [], []
        for s in sessions:
            X, Y, P, _ = arrays_with_position(s, variant, 'train', feature_set, 0)
            Xs.append(X)
            Ys.append(Y - profile_values(profile, s, P) if residual else Y)
        X, Y = np.concatenate(Xs).astype(np.float64), np.concatenate(Ys).astype(np.float64)
        Xc, Yc = X - X.mean(axis=0), Y - Y.mean(axis=0)
        denom = np.sqrt((Xc ** 2).sum(axis=0))[:, None] * np.sqrt((Yc ** 2).sum(axis=0))[None, :]
        with np.errstate(invalid='ignore', divide='ignore'):
            r = (Xc.T @ Yc) / denom
        strength = np.nan_to_num(np.abs(r).mean(axis=1), nan=-1.0)
        n_pairs = X.shape[1]
        if top_n >= n_pairs:
            raise SystemExit(f'--top {top_n} keeps all {n_pairs} channel-feature pairs; use a smaller number')
        chosen = np.sort(np.argsort(-strength)[:top_n])
        SELECTED[variant] = chosen
        names = C.FEATURE_SETS[feature_set]
        counts = pd.Series([names[i % len(names)] for i in chosen]).value_counts()
        print(f'  {variant}: kept {top_n} of {n_pairs} channel-feature pairs (mean |r| {strength[chosen].min():.3f} '
              f'to {strength[chosen].max():.3f}); per feature: '
              + ', '.join(f'{name} {counts.get(name, 0)}' for name in names))


def fit_spatial_pca(sessions, feature_set, drop):
    """Per reference and feature, principal components across channels on the training trials, each scaled
    to unit variance, with the first `drop` removed."""
    names = C.FEATURE_SETS[feature_set]
    for variant in C.REFERENCES:
        X = np.concatenate([arrays_with_position(s, variant, 'train', feature_set, 0)[0] for s in sessions])
        X = X.reshape(len(X), -1, len(names)).astype(np.float64)
        if drop >= X.shape[1]:
            raise SystemExit(f'--pca-drop {drop} removes all {X.shape[1]} components')
        params, shares = [], []
        for f in range(len(names)):
            mean = X[:, :, f].mean(axis=0)
            eigval, eigvec = np.linalg.eigh(np.cov(X[:, :, f] - mean, rowvar=False))
            order = np.argsort(eigval)[::-1]
            eigval, eigvec = np.maximum(eigval[order], 1e-12), eigvec[:, order]
            params.append((mean, eigvec[:, drop:] / np.sqrt(eigval[drop:]), eigval / eigval.sum()))
            shares.append(eigval[0] / eigval.sum())
        PCA[variant] = params
        print(f'  {variant}: {X.shape[1] - drop} of {X.shape[1]} components per feature; first component\'s share '
              'of variance: ' + ', '.join(f'{n} {v:.2f}' for n, v in zip(names, shares)))


def fit_off_decoder_gain(feature_set, n_lags, profile):
    """Select lambda/reference by validation gain of time-only + residual ridge over time-only."""
    train = [s for s in C.sessions('train', 'no_stim') if C.has_cm_velocity(s)]
    val = C.sessions('val', 'no_stim')
    print(f'Features: {feature_set} ({", ".join(C.FEATURE_SETS[feature_set])}), '
          f'history: {n_lags} earlier bins ({(n_lags + 1) * 50} ms including the current bin)')
    print(f'Training on {len(train)} OFF sessions (held-out block excluded); '
          f'choosing lambda on {len(val)} validation-day OFF sessions using full-prediction R2 gain.')

    rows, fitters = [], {}
    for variant in C.REFERENCES:
        X, Y_residual = C.stack_arrays(train, variant, 'train', n_lags, feature_set=feature_set)
        fitters[variant] = C.RidgeFitter(X, Y_residual)
        print(f'  {variant}: {X.shape[0]:,} training bins x {X.shape[1]:,} inputs')
        del X, Y_residual
        held = {}
        for session in val:
            Xv, Yv, P, _ = arrays_with_position(session, variant, 'query', feature_set, n_lags)
            baseline = profile_values(profile, session, P)
            held[session] = (Xv, Yv, baseline, C.r2(Yv, baseline).mean())
        for lam in C.LAMBDAS:
            model = fitters[variant].fit(lam)
            full_scores, gains, residual_scores = {}, {}, {}
            for session, (Xv, Yv, baseline, time_r2) in held.items():
                residual_prediction = model.predict(Xv)
                full_r2 = C.r2(Yv, baseline + residual_prediction).mean()
                full_scores[session] = full_r2
                gains[session] = full_r2 - time_r2
                residual_scores[session] = C.r2(Yv - baseline, residual_prediction).mean()
            rows.append(dict(feature_set=feature_set, n_lags=n_lags, reference=variant, lam=lam,
                             val_r2_mean=np.mean(list(full_scores.values())),
                             val_r2_gain_mean=np.mean(list(gains.values())),
                             val_r2_residual_mean=np.mean(list(residual_scores.values())),
                             **{f'val_r2_{s}': v for s, v in full_scores.items()}))

    search = pd.DataFrame(rows)
    best = search.loc[search.val_r2_gain_mean.idxmax()]
    search['chosen'] = (search.reference == best.reference) & (search.lam == best.lam)
    return dict(model=fitters[best.reference].fit(best.lam), reference=best.reference,
                lam=float(best.lam), search=search)


def print_lambda_search_gain(search):
    """Display complete-prediction validation score, neural gain and residual-only score."""
    for metric, label in (('val_r2_gain_mean', 'Full-prediction R2 gain over time-only (selection criterion)'),
                          ('val_r2_mean', 'Full-prediction R2'),
                          ('val_r2_residual_mean', 'Residual-only R2 (diagnostic, not used for selection)')):
        print(f'\nValidation {label}, mean over OFF sessions, query trials:')
        print(search.pivot(index='lam', columns='reference', values=metric)
              .to_string(float_format=lambda v: f'{v:8.4f}'))
    for variant, group in search.groupby('reference'):
        row = group.loc[group.val_r2_gain_mean.idxmax()]
        edge = ' (at the edge of the grid!)' if row.lam in (C.LAMBDAS[0], C.LAMBDAS[-1]) else ''
        print(f'  best for {variant}: lambda = {row.lam:g}, gain = {row.val_r2_gain_mean:.4f}{edge}')
    chosen = search[search.chosen].iloc[0]
    print(f'Chosen: reference = {chosen.reference}, lambda = {chosen.lam:g}, '
          f'validation full-prediction gain = {chosen.val_r2_gain_mean:.4f}')


def per_channel_check(channel_list, base_channels, test_off, feature_set, n_lags, profile, residual, path):
    """Fit and score one decoder per channel, plus one with base_channels (None = all in the file)."""
    C.REFERENCES = C.REFERENCES[:1]
    print(f'\nOne decoder per channel ({len(channel_list)} channels, reference {C.REFERENCES[0]}), '
          f'scored on the {len(test_off)} test-day OFF sessions:')
    rows = []
    for ch in [None, *channel_list]:
        use_channels(base_channels if ch is None else [ch])
        with contextlib.redirect_stdout(io.StringIO()):
            decoder = (fit_off_decoder_gain(feature_set, n_lags, profile) if residual
                       else C.fit_off_decoder(n_lags, feature_set=feature_set))
        scores = pd.DataFrame([score(decoder['model'], profile, s, decoder['reference'], 'query', feature_set,
                                     n_lags, residual) for s in test_off])
        row = dict(channel='all' if ch is None else int(ch), lam=decoder['lam'],
                   **scores.drop(columns='n_bins').mean().to_dict())
        rows.append(row)
        cols = [c for c in ('r2', 'r2_vx', 'r2_vy', 'r2_gain') if c in row]
        print(f'  channel {row["channel"]!s:>4}: ' + ', '.join(f'{c} {row[c]:.3f}' for c in cols)
              + f' (lambda {row["lam"]:g})')
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    single = table[table.channel != 'all']
    print('\nSingle channels, summary over channels:')
    shown = single.drop(columns=['channel', 'lam', *[c for c in single if c.startswith('r2_time')]])
    print(shown.describe(percentiles=[0.25, 0.5, 0.75])
          .loc[['min', '25%', '50%', '75%', 'max']].to_string(float_format=lambda v: f'{v:7.3f}'))
    time_cols = [c for c in table if c.startswith('r2_time')]
    print('Time-only baseline: ' + ', '.join(f'{c} {table.iloc[0][c]:.3f}' for c in time_cols))
    print('All channels together: ' + ', '.join(f'{c} {table.iloc[0][c]:.3f}'
                                                 for c in ('r2', 'r2_vx', 'r2_vy', 'r2_gain') if c in table))
    print(f'Per-channel rows in {os.path.relpath(path, ROOT)}')


def use_window_arrays(profile=None):
    """Make the shared fitting code use arrays_with_position(), and with a profile, see the target
    minus the time-only baseline."""
    def window_arrays(session, variant, part, n_lags=C.N_LAGS, hf=False, *, feature_set='bands'):
        X, Y, P, _ = arrays_with_position(session, variant, part, feature_set, n_lags)
        return X, (Y - profile_values(profile, session, P)) if profile is not None else Y
    C.session_arrays = window_arrays


def set_window(window, post, sessions):
    """Set C.POST and TRIAL_END_PAD for the window; print the trial lengths it is based on."""
    global TRIAL_END_PAD
    lengths = np.concatenate([(r.t_reward - r.t_go).to_numpy() for r in (C.trial_rows(s, 'all') for s in sessions)])
    print(f'Trial length (reward minus go cue) over {len(lengths)} trials: median {np.median(lengths):.1f} s, '
          f'95th percentile {np.percentile(lengths, 95):.1f} s, max {lengths.max():.1f} s')
    if window == 'trial':
        TRIAL_END_PAD = END_PAD
        if post is None:
            post = np.ceil((np.percentile(lengths, 95) + END_PAD) / C.L.BIN_S) * C.L.BIN_S
        cut = int((lengths + END_PAD > post).sum())
        print(f'Window: {C.PRE} s before the go cue to reward + {END_PAD} s, cut at {post:.2f} s after the go cue '
              f'({cut} of {len(lengths)} trials truncated)')
    elif post is not None:
        print(f'Window: {C.PRE} s before the go cue to {post:.2f} s after it')
    if post is not None:
        C.POST = float(post)


def score(model, profile, session, variant, part, feature_set, n_lags, residual):
    X, Y, P, Y_raw = arrays_with_position(session, variant, part, feature_set, n_lags)
    baseline = profile_values(profile, session, P)
    prediction = model.predict(X) + (baseline if residual else 0.0)
    per_column = C.r2(Y, prediction)
    per_column_time = C.r2(Y, baseline)
    r2_time = per_column_time.mean()
    out = dict(r2=per_column.mean(), **{f'r2_{SHORT[c]}': v for c, v in zip(C.KIN_COLS, per_column)},
               r2_time=r2_time, r2_gain=per_column.mean() - r2_time, n_bins=len(X))
    if len(C.KIN_COLS) > 1:
        out.update({f'r2_time_{SHORT[c]}': v for c, v in zip(C.KIN_COLS, per_column_time)})
    if Y_raw is not None:
        r2_raw, r2_time_raw = C.r2(Y_raw, prediction).mean(), C.r2(Y_raw, baseline).mean()
        out.update(r2_raw=r2_raw, r2_time_raw=r2_time_raw, r2_gain_raw=r2_raw - r2_time_raw)
    return out


def main(feature_set, n_lags, target, residual, window, post, smooth, feature_tag, channel_set, norm,
         time_profile, lambda_max, top_n, per_channel, spatial_pca, pca_drop):
    global SMOOTH_HZ, NORM, TIME_PROFILE
    start = time.time()
    C.KIN_COLS = TARGETS[target]
    channels = parse_channels(channel_set)
    if feature_tag:
        C.REFERENCES = (f'export_none_{feature_tag}',)
        use_channels(channels)
    elif channel_set != 'fixed':
        raise SystemExit('--channels other than fixed needs --feature-tag (the default features hold only the 35 fixed channels)')
    print(f'Project folder: {ROOT}')
    print(f'Features from {", ".join(os.path.join("features", v) for v in C.REFERENCES)}; channels: {channel_set}'
          + (f' ({len(channels)})' if channels is not None else ' (all in the file)'))
    print(f'Target: {target} ({", ".join(C.KIN_COLS)})'
          + (', LFP decoder fit to the residual of the time-only baseline' if residual else '')
          + (f', smoothed below {smooth:g} Hz' if smooth else ''))
    NORM = norm
    TIME_PROFILE = time_profile
    if lambda_max:
        C.LAMBDAS = np.logspace(0, np.log10(lambda_max), int(round(2 * np.log10(lambda_max))) + 1)
        print(f'Lambda grid: {len(C.LAMBDAS)} values from 1 to {lambda_max:g}')
    if norm == 'off':
        print('Test-day stimulation sessions are z-scored with their same-day OFF session\'s statistics, '
              'not their own')
    check_files()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    if smooth:
        SMOOTH_HZ = float(smooth)
        C.kinematics = smoothed_kinematics
    train, val, test_off, test_on = session_groups()
    ON_SESSIONS.update(s for group in test_on.values() for s in group)
    set_window(window, post, sorted({*train, *val, *test_off, *[s for g in test_on.values() for s in g]}))
    if window == 'trial':
        use_window_arrays()
    profile = fit_time_profile(train, C.REFERENCES[0], feature_set, n_lags)
    if residual:
        use_window_arrays(profile)
    prefix = 'task3_diag_levels_' if target == 'velocity' else f'task3_diag_levels_{target}_'
    extra = (('_residual' if residual else '') + ('_trialwin' if window == 'trial' else '')
             + (f'_smooth{smooth:g}Hz' if smooth else '')
             + (f'_ch-{channel_set.replace(",", "-")}' if channel_set != 'fixed' else '')
             + ('_normoff' if norm == 'off' else '')
             + ('_seqtime' if time_profile == 'sequence' else '')
             + (f'_lammax{lambda_max:.0e}'.replace('+', '') if lambda_max else '')
             + ('_selectgain' if residual else '')
             + (f'_top{top_n}' if top_n else '')
             + ((f'_pcadrop{pca_drop}' if pca_drop else '_pca') if spatial_pca else ''))
    tag = C.run_tag(n_lags, feature_set=feature_set).replace('_lags', f'{extra}_lags')
    if sum(map(bool, (per_channel, top_n, spatial_pca))) > 1:
        raise SystemExit('--per-channel, --top and --spatial-pca cannot be combined')
    if pca_drop and not spatial_pca:
        raise SystemExit('--pca-drop needs --spatial-pca')
    if per_channel:
        base_channels = channels if feature_tag else None
        file_channels = base_channels if base_channels is not None else list(np.asarray(
            RAW_LOAD_FEATURES(train[0], C.PATHS, variant=C.REFERENCES[0], normalize=False)['channel_ids']).ravel().astype(int))
        per_channel_check(file_channels, base_channels, test_off, feature_set, n_lags, profile, residual,
                          os.path.join(C.RESULTS_DIR, f'{prefix}perchannel_{tag}.csv'))
        print(f'\nDone in {time.time() - start:.0f} s.')
        return
    if top_n:
        print(f'Selecting features on the training trials ({"residual" if residual else "targets"}):')
        select_features(train, feature_set, top_n, profile, residual)
    if spatial_pca:
        print('Spatial PCA on the training trials:')
        fit_spatial_pca(train, feature_set, pca_drop)
    if (top_n or spatial_pca) and not residual and window != 'trial':
        use_window_arrays()
    decoder = fit_off_decoder_gain(feature_set, n_lags, profile) if residual else C.fit_off_decoder(n_lags, feature_set=feature_set)
    if residual:
        print_lambda_search_gain(decoder['search'])
    else:
        C.print_lambda_search(decoder['search'])
    model, reference = decoder['model'], decoder['reference']
    if residual:
        decoder['search'].to_csv(os.path.join(C.RESULTS_DIR, f'{prefix}{tag}_lambda_search.csv'), index=False)
    levels = [('train', train, 'train'), ('held_out', train, 'val'),
              ('val_days', val, 'query'), ('test_days', test_off, 'query'),
              *[(f'test_on_{c}', names, 'query') for c, names in test_on.items() if names]]

    rows = [dict(level=level, session=s, **score(model, profile, s, reference, part, feature_set, n_lags, residual))
            for level, names, part in levels for s in names]
    results = pd.DataFrame(rows)
    results['reference'], results['lam'] = reference, decoder['lam']
    results['window'], results['post_s'] = window, C.POST
    results.to_csv(os.path.join(C.RESULTS_DIR, f'{prefix}{tag}.csv'), index=False)

    per_column = {name: (name, 'mean') for c in C.KIN_COLS for name in (f'r2_{SHORT[c]}', f'r2_time_{SHORT[c]}')} \
        if len(C.KIN_COLS) > 1 else {}
    summary = results.groupby('level', sort=False).agg(
        n_sessions=('session', 'size'), r2=('r2', 'mean'), **per_column,
        r2_min=('r2', 'min'), r2_max=('r2', 'max'), r2_time=('r2_time', 'mean'), r2_gain=('r2_gain', 'mean'),
        **({'r2_raw': ('r2_raw', 'mean'), 'r2_time_raw': ('r2_time_raw', 'mean'),
            'r2_gain_raw': ('r2_gain_raw', 'mean')} if smooth else {}),
        bins_per_session=('n_bins', 'mean'))
    summary['bins_per_session'] = summary.bins_per_session.round().astype(int)
    model_name = 'Time-only baseline + LFP decoder' if residual else 'OFF decoder'
    print(f'\n{model_name} for {target} ({reference}, lambda = {decoder["lam"]:g}) at each level '
          f'(mean over sessions of R2{" averaged over " + " and ".join(SHORT[c] for c in C.KIN_COLS) if len(C.KIN_COLS) > 1 else ""}):')
    print(summary.to_string(float_format=lambda v: f'{v:7.3f}'))
    print('r2_time: time-only baseline (mean target at each position of the window); r2_gain = r2 - r2_time.'
          + (' r2_raw, r2_time_raw, r2_gain_raw: the same predictions scored against the raw targets.' if smooth else ''))

    S = C.splits()
    on = results[results.level.str.startswith('test_on_')]
    if len(on):
        off = results[results.level == 'test_days'].set_index('session')
        cols = ['r2_raw' if smooth else 'r2', 'r2_gain_raw' if smooth else 'r2_gain']
        pairs = pd.DataFrame([dict(session=r.session, condition=r.level.replace('test_on_', ''),
                                   r2_on=r[cols[0]], r2_off=off.loc[S['off_reference'][r.session], cols[0]],
                                   gain_on=r[cols[1]], gain_off=off.loc[S['off_reference'][r.session], cols[1]])
                              for _, r in on.iterrows()])
        pairs['on_minus_off'] = pairs.r2_on - pairs.r2_off
        print(f'\nTest-day ON sessions against their same-day OFF session ({cols[0]}; gain = over the time-only baseline):')
        print(pairs.to_string(index=False, float_format=lambda v: f'{v:7.3f}'))
        print(pairs.groupby('condition')[['r2_on', 'r2_off', 'on_minus_off', 'gain_on', 'gain_off']].mean()
              .to_string(float_format=lambda v: f'{v:7.3f}'))
        if norm == 'off':
            shifts = pd.DataFrame([dict(session=s, condition=c, **feature_shift(s, reference))
                                   for c, names in test_on.items() for s in names])
            print('\nFeatures of each ON session relative to its same-day OFF session (median z, OFF-session units):')
            print(shifts.to_string(index=False, float_format=lambda v: f'{v:6.2f}'))
            print(shifts.drop(columns='session').groupby('condition').mean()
                  .to_string(float_format=lambda v: f'{v:6.2f}'))
    print(f'\nDone in {time.time() - start:.0f} s. Per-session rows in {os.path.relpath(C.RESULTS_DIR, ROOT)}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 3.1 diagnostic: the OFF decoder scored on its training '
                                                 'trials, held-out blocks, validation days and test days.')
    parser.add_argument('--features', default='lmp+bands',
                        help='lmp, bands, lmp+bands (default), or a comma-separated list of feature names')
    parser.add_argument('--lags', type=int, default=C.N_LAGS,
                        help=f'earlier 50 ms bins of feature history per bin (default: {C.N_LAGS})')
    parser.add_argument('--target', choices=sorted(TARGETS), default='velocity',
                        help='velocity (Index vx and vy, as in 3.1; default), vx (horizontal velocity only), '
                             'speed (Index speed), or position (Index x and y)')
    parser.add_argument('--residual', action='store_true',
                        help='fit the LFP decoder to what the time-only baseline misses, and score both together')
    parser.add_argument('--window', choices=('go', 'trial'), default='go',
                        help='go: 0.5 s before to 1.5 s after the go cue, as in 3.1 (default); '
                             'trial: 0.5 s before the go cue to 0.5 s after reward (all five reaches)')
    parser.add_argument('--post', type=float, default=None,
                        help='seconds after the go cue at which the window is cut '
                             '(default: 1.5 for go; 95th percentile of trial length + 0.5 for trial)')
    parser.add_argument('--smooth', type=float, default=None,
                        help='low-pass the targets below this frequency in Hz before fitting and scoring')
    parser.add_argument('--feature-tag', default='',
                        help='read features/export_none_<TAG>/ (all-channel features) instead of the default folders')
    parser.add_argument('--channels', default='fixed',
                        help='fixed (default), pickup, clean_canonical, all, or a comma-separated list of channels')
    parser.add_argument('--time-profile', choices=('pooled', 'sequence'), default='pooled',
                        help='time-only baseline over all training sessions (default) or per target sequence')
    parser.add_argument('--lambda-max', type=float, default=None,
                        help='extend the lambda grid up to this value (default: 1e6)')
    parser.add_argument('--top', type=int, default=None,
                        help='keep only the N channel-feature pairs most correlated with the targets')
    parser.add_argument('--per-channel', action='store_true',
                        help='one decoder per channel, scored on the test days')
    parser.add_argument('--spatial-pca', action='store_true',
                        help='replace channels by their principal components across channels, scaled to unit variance')
    parser.add_argument('--pca-drop', type=int, default=0,
                        help='with --spatial-pca, remove the first K components (1 = the shared one)')
    parser.add_argument('--norm', choices=('own', 'off'), default='own',
                        help='own: test-day stimulation sessions z-scored with their own calibration pool, as in '
                             '3.1 (default); off: with their same-day OFF session\'s statistics')
    args = parser.parse_args()
    main(parse_features(args.features), args.lags, args.target, args.residual, args.window, args.post, args.smooth,
         args.feature_tag, args.channels, args.norm, args.time_profile, args.lambda_max, args.top,
         args.per_channel, args.spatial_pca, args.pca_drop)
