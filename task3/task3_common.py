"""Shared helpers for Task 3 (problem size check).

Holds the data paths, the trial selection rules from splits.json, the
(input, velocity) arrays built from the precomputed features, a closed-form
ridge regression, and R². Tasks 3.1-3.3 (and the PARRM + ridge baseline in
4.3) import from here so every script makes the same choices.
"""

import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache

import h5py
import numpy as np
import pandas as pd

TASK_DIR = os.path.dirname(os.path.abspath(__file__))  # .../task3
REPO = os.path.dirname(TASK_DIR)                         # repository root
if REPO not in sys.path:  # the shared loader and features scripts live at the repository root
    sys.path.insert(0, REPO)

import EECS182_NHP_data_loader as L  # noqa: E402
import EECS182_NHP_features as F     # noqa: E402

from task3_paths import DATA_ROOT, data_paths, metadata_path

RESULTS_DIR = os.path.join(REPO, 'task3_results')
PATHS = data_paths(DATA_ROOT)
DATA = PATHS.root
META = str(metadata_path(DATA_ROOT, 'splits.json').parent)

# Trial window and feature history: 0.5 s before the go cue to 1.5 s after it
# (40 bins of 50 ms), each bin also seeing the 19 bins before (1 s including the
# current bin, as in the GRU). Earlier runs used 9; pass --lags 9 to reproduce them.
ALIGN, PRE, POST, N_LAGS = 't_go', 0.5, 1.5, 19
KIN_COLS = ('Index_vx_cm', 'Index_vy_cm')  # fingertip velocity, cm/s

# Normalized features are clipped to ±CLIP_Z so a single artifact cannot
# dominate the linear decoder (see Notes.md, Task 3.1: channel 64 in
# 250616-102853 reaches z ≈ 5300). Clean data stays well inside this range.
CLIP_Z = 10.0

# Decoder inputs for every channel: LMP + the 7 band powers. (A band-power-only
# decoder was dropped: it scored lower, validation R² 0.021 vs. 0.035 with 9 lags.)
FEATURES = list(F.FEATURE_NAMES)  # ['lmp', 'pow_1_4', ..., 'pow_130_200']
FEATURE_SET = 'lmp+bands'         # label written to the result files
FEATURE_SETS = {
    'bands': [name for name in FEATURES if name.startswith('pow_')],
    'lmp+bands': FEATURES,
}


# ---------------------------------------------------------------- metadata

@lru_cache(maxsize=None)
def manifest():
    return pd.read_csv(metadata_path(DATA_ROOT, 'EECS182_data_manifest.csv')).set_index('session', drop=False)


@lru_cache(maxsize=None)
def splits():
    with open(metadata_path(DATA_ROOT, 'splits.json')) as file:
        return json.load(file)


def sessions(split, condition=None):
    """Sessions in a split ('train', 'val', 'test'), optionally of one condition
    ('no_stim', 'am3hz', 'am20hz', or 'stim' for either AM frequency)."""
    names = splits()['sessions'][split]
    if condition is None:
        return list(names)
    conditions = manifest().loc[names, 'condition']
    if condition == 'stim':
        return [s for s in names if conditions[s] != 'no_stim']
    return [s for s in names if conditions[s] == condition]


def has_cm_velocity(session):
    """False for the July 11 sessions, which have no touch data and so no cm/s velocity."""
    return bool(manifest().loc[session, 'touch_available'])


# ---------------------------------------------------------------- trials

@lru_cache(maxsize=None)
def _included_trials(session):
    trials = pd.read_csv(os.path.join(PATHS.trial_dir, f'{session}_trials.csv'))
    return trials[trials.include == 1]


def trial_rows(session, part):
    """Included trials of one session for a split part.

    'train' / 'val': training sessions, without / only the held-out block.
    'pool' / 'query': validation and test sessions, blocks 1-2 / blocks 3-5.
    'all': every included trial.
    A list of trial numbers (e.g. a k-trial draw from splits.json) selects those trials.
    """
    included = _included_trials(session)
    if not isinstance(part, str):
        return included[included.trial.isin(list(part))]
    if part == 'all':
        return included
    if part in ('train', 'val'):
        held_out = included.trial.isin(splits()['train_val_trials'][session])
        return included[held_out] if part == 'val' else included[~held_out]
    if part in ('pool', 'query'):
        return included[included.trial.isin(splits()['calibration'][session][part])]
    raise ValueError(f'unknown part {part}')


@lru_cache(maxsize=None)
def kinematics(session):
    return L.load_kinematics(session, PATHS)


# ---------------------------------------------------------------- recordings and the 200-400 Hz band

def loader_manifest():
    """The manifest in the form the data loader expects (a 'session' column, default index)."""
    return manifest().reset_index(drop=True)


def load_recording(session, variant):
    """The LFP (35 fixed channels, recording reference) behind a feature variant:
    'export_none' (recorded), 'parrm_none' (PARRM-cleaned) or 'parrm_amp10' (10x artifact)."""
    if variant == 'parrm_amp10':
        from task3_make_amp10_features import amp10_recording  # imports this module
        return amp10_recording(session)[0]
    source = {'export_none': 'export', 'parrm_none': 'parrm'}[variant]
    return L.load_session(session, PATHS, source=source, reference='none', manifest=loader_manifest())


# The GRU's ninth feature per channel: log power in 200-400 Hz over a 256 ms window ending at
# each bin's end (task3_gru_core.py, BAND and WINDOW), computed and normalized the same way.
HF_BAND, HF_WINDOW = (200, 400), 0.256
HF_DIRS = {'export_none': 'hf200_400_export', 'parrm_none': 'hf200_400_parrm',
           'parrm_amp10': 'hf200_400_amp10'}


def hf_power(session, variant):
    """200-400 Hz log power, (bins, channels), with its bin and channel IDs.
    Computed from the recording and cached in <DATA>/features/<HF_DIRS[variant]>/ the first time."""
    path = os.path.join(DATA, 'features', HF_DIRS[variant], f'{session}.h5')
    if not os.path.exists(path):
        rec = load_recording(session, variant)
        bin_ids = np.arange(int(np.ceil(rec.t[0] / L.BIN_S - 1e-9)), int(np.floor(rec.t[-1] / L.BIN_S + 1e-9)))
        ends = np.searchsorted(rec.t, (bin_ids + 1) * L.BIN_S - 0.5 / rec.fs) - 1
        power = F._band_power(rec.lfp, ends, rec.fs, HF_WINDOW, *HF_BAND)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path + '.tmp', 'w') as file:
            file.create_dataset('bin_id', data=bin_ids)
            file.create_dataset('channel_ids', data=rec.channels)
            file.create_dataset('logpow', data=power, compression='gzip')
            file.attrs['band_hz'] = HF_BAND
            file.attrs['window_s'] = HF_WINDOW
        os.replace(path + '.tmp', path)
        print(f'    made {os.path.relpath(path, DATA)}', flush=True)
    with h5py.File(path, 'r') as file:
        return file['bin_id'][:], file['channel_ids'][:], file['logpow'][:].T.astype(np.float32)


# ---------------------------------------------------------------- arrays

def session_arrays(session, variant, part, n_lags=N_LAGS, hf=False, return_trials=False, *, feature_set=None):
    """Decoder inputs and target velocities for one session's trials.

    Returns X (n, d) and Y (n, 2) with one row per 50 ms bin that has both
    finite features and finite velocity. d = (n_lags + 1) * channels * 8
    features (LMP + 7 band powers; 9 with hf=True, adding 200-400 Hz power as
    in the GRU): each bin sees its own features plus those of the n_lags bins
    before it.

    Features are z-scored within the session, then clipped to ±CLIP_Z.
    Validation and test sessions take the statistics from their calibration
    pool (blocks 1-2) only, so the query trials are never used for anything
    but the final R².

    With return_trials=True, also returns each row's trial number, so subsets
    of trials can be taken without cutting the windows again.
    """
    norm_rows = trial_rows(session, 'pool') if session in splits()['calibration'] else None
    feat = F.load_features(session, PATHS, variant=variant, normalize=True, trials=norm_rows)
    if feature_set is None:
        assert feat['feature_names'] == FEATURES
    else:
        names = FEATURE_SETS[feature_set]
        indices = [feat['feature_names'].index(name) for name in names]
        feat = dict(feat, X=feat['X'][:, :, indices], feature_names=names)
    X = feat['X']
    if hf:
        bin_ids, channels, power = hf_power(session, variant)
        if not (np.array_equal(bin_ids, feat['bin_id']) and np.array_equal(channels, feat['channel_ids'])):
            raise ValueError(f'{session}: 200-400 Hz bins or channels differ from the other features')
        start, stop = feat['norm']['window']  # same normalization window and rule as the other features
        inside = (bin_ids * L.BIN_S >= start) & ((bin_ids + 1) * L.BIN_S <= stop)
        median = np.nanmedian(power[inside], axis=0)
        spread = 1.4826 * np.nanmedian(np.abs(power[inside] - median), axis=0)
        spread[~(spread > 0)] = 1.0
        X = np.concatenate([X, ((power - median) / spread)[:, :, None]], axis=2)
    feat = dict(feat, X=np.clip(X, -CLIP_Z, CLIP_Z))  # NaN stays NaN

    rows = trial_rows(session, part)
    X, Y, valid, _ = F.trial_arrays(feat, rows, kin=kinematics(session),
                                    align=ALIGN, pre=PRE, post=POST, n_lags=n_lags, kin_cols=KIN_COLS)
    if return_trials:
        trial_ids = np.repeat(rows.trial.to_numpy()[:, None], valid.shape[1], axis=1)
        return X[valid], Y[valid], trial_ids[valid]
    return X[valid], Y[valid]


def stack_arrays(session_list, variant, part, n_lags=N_LAGS, hf=False, *, feature_set=None):
    """session_arrays() for several sessions, concatenated."""
    options = {} if feature_set is None else {'feature_set': feature_set}
    pairs = [session_arrays(s, variant, part, n_lags=n_lags, hf=hf, **options)
             for s in session_list]
    return np.concatenate([x for x, _ in pairs]), np.concatenate([y for _, y in pairs])


# ---------------------------------------------------------------- ridge and R²

@dataclass
class RidgeModel:
    W: np.ndarray  # (d, 2)
    b: np.ndarray  # (2,)
    lam: float

    def predict(self, X):
        return X @ self.W + self.b


class RidgeFitter:
    """Ridge regression Y ≈ X W + b, minimizing ||Y - X W - b||² + λ ||W||².

    The intercept b is not penalized. X'X is computed once, so refitting for
    each λ only costs one linear solve: W = (X'X + λI)⁻¹ X'Y on centered data.
    """

    def __init__(self, X, Y, chunk=4096):
        self.x_mean = X.mean(axis=0, dtype=np.float64)
        self.y_mean = Y.mean(axis=0, dtype=np.float64)
        d = X.shape[1]
        self.gram = np.zeros((d, d))
        self.xty = np.zeros((d, Y.shape[1]))
        for i in range(0, len(X), chunk):  # in chunks, so no full float64 copy of X
            Xc = X[i:i + chunk].astype(np.float64) - self.x_mean
            self.gram += Xc.T @ Xc
            self.xty += Xc.T @ (Y[i:i + chunk].astype(np.float64) - self.y_mean)
        self.n_samples = len(X)

    def fit(self, lam):
        W = np.linalg.solve(self.gram + lam * np.eye(len(self.gram)), self.xty)
        return RidgeModel(W=W, b=self.y_mean - self.x_mean @ W, lam=float(lam))


def ridge_path(X, Y, lambdas):
    """The same ridge fit as RidgeFitter, for every λ in lambdas at once.

    One eigendecomposition serves every λ. With fewer rows than inputs (one
    session: ~1,300-2,000 bins vs. 2,800+ inputs) it uses the n x n matrix
    X Xᵀ = Q E Qᵀ (X, Y centered) and the equivalent form
    W(λ) = Xᵀ (X Xᵀ + λI)⁻¹ Y = Xᵀ Q (E + λ)⁻¹ Qᵀ Y; otherwise the d x d
    matrix Xᵀ X. Returns a list of RidgeModel.
    """
    x_mean = X.mean(axis=0, dtype=np.float64)
    y_mean = Y.mean(axis=0, dtype=np.float64)
    Xc = X.astype(np.float64) - x_mean
    Yc = Y.astype(np.float64) - y_mean
    if len(Xc) <= Xc.shape[1]:
        e, Q = np.linalg.eigh(Xc @ Xc.T)
        weights = lambda q: Xc.T @ (Q @ q)
        qty = Q.T @ Yc
    else:
        e, Q = np.linalg.eigh(Xc.T @ Xc)
        weights = lambda q: Q @ q
        qty = Q.T @ (Xc.T @ Yc)
    e = np.maximum(e, 0.0)  # remove tiny negative rounding errors
    models = []
    for lam in lambdas:
        W = weights(qty / (e + lam)[:, None])
        models.append(RidgeModel(W=W, b=y_mean - x_mean @ W, lam=float(lam)))
    return models


# Penalties for refitting a trained decoder on a few labeled trials (SameDayAdapter):
# 1 to 10^8 in half decades, then inf = keep the trained decoder unchanged.
MUS = np.concatenate([np.logspace(0, 8, 17), [np.inf]])


class SameDayAdapter:
    """Refits a trained ridge model on a few labeled bins from one session.

    With X, Y centered on those bins, W = W0 + D where D minimizes
    ||(Y - X W0) - X D||² + mu ||D||², i.e. D = Xᵀ (X Xᵀ + mu I)⁻¹ (Y - X W0).
    The intercept is refit on the same bins. With fewer bins than inputs this
    uses the eigendecomposition X Xᵀ = Q E Qᵀ, computed once for every mu.
    Used for same-day OFF calibration (task3_1_ridge_gru_matched.py) and for
    the traditional pipeline's refit on k stimulation trials (task3_2_parrm.py).
    """

    def __init__(self, X, Y):
        self.x_mean = X.mean(axis=0, dtype=np.float64)
        self.y_mean = Y.mean(axis=0, dtype=np.float64)
        self.Xc = X.astype(np.float64) - self.x_mean
        self.Yc = Y.astype(np.float64) - self.y_mean
        e, self.Q = np.linalg.eigh(self.Xc @ self.Xc.T)
        self.e = np.maximum(e, 0.0)

    def fit(self, base, mu):
        if np.isinf(mu):
            return base
        residual = self.Yc - self.Xc @ base.W
        W = base.W + self.Xc.T @ (self.Q @ ((self.Q.T @ residual) / (self.e + mu)[:, None]))
        return RidgeModel(W=W, b=self.y_mean - self.x_mean @ W, lam=base.lam)


def r2(Y, Y_pred):
    """R² for each velocity axis (vx, vy).

    1 - (squared error of the prediction) / (squared error of always predicting
    the mean of Y). 1 is perfect, 0 is no better than the mean, < 0 is worse.
    """
    ss_res = ((Y - Y_pred) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------- the OFF decoder (3.1, reused by 3.2)

REFERENCES = ('export_none', 'export_car')     # recording reference vs. common average reference
LAMBDAS = np.logspace(0, 6, 13)                # 1 to 10^6, half-decade steps
PARRM_VARIANT = {'export_none': 'parrm_none', 'export_car': 'parrm_car'}  # cleaned counterpart


def run_tag(n_lags, *, feature_set=None):
    """File-name tag for one setting, e.g. 'lmp_bands_lags9'."""
    name = FEATURE_SET if feature_set is None else feature_set
    return f"{name.replace('+', '_')}_lags{n_lags}"


def fit_off_decoder(n_lags=N_LAGS, *, feature_set=None):
    """Train the OFF decoder and choose λ and the reference on the validation days.

    Fits on the 'train' trials of the training-day no-stim sessions (July 11
    excluded: no cm/s velocity; each session's held-out block in
    splits.json's train_val_trials is not used). Validation is the 3
    validation days: every (reference, λ) is scored by the mean R² over the
    query trials (blocks 3-5) of the validation-day no-stim sessions, and the
    best one is refit. Returns dict(model, reference, lam, search).
    """
    train_sessions = [s for s in sessions('train', 'no_stim') if has_cm_velocity(s)]
    val_sessions = sessions('val', 'no_stim')
    print(f'Features: LMP + 7 band powers ({", ".join(FEATURES)}), '
          f'history: {n_lags} earlier bins ({(n_lags + 1) * 50} ms including the current bin)')
    print(f'Training on {len(train_sessions)} OFF sessions (training days, held-out block excluded); '
          f'choosing lambda on {len(val_sessions)} validation-day OFF sessions.')

    options = {} if feature_set is None else {'feature_set': feature_set}
    rows, fitters = [], {}
    for variant in REFERENCES:
        X, Y = stack_arrays(train_sessions, variant, 'train', n_lags, **options)
        fitters[variant] = RidgeFitter(X, Y)
        print(f'  {variant}: {X.shape[0]:,} training bins x {X.shape[1]:,} inputs')
        del X, Y
        val = {s: session_arrays(s, variant, 'query', n_lags, **options) for s in val_sessions}
        for lam in LAMBDAS:
            model = fitters[variant].fit(lam)
            per_session = {s: r2(Yv, model.predict(Xv)).mean() for s, (Xv, Yv) in val.items()}
            rows.append(dict(feature_set=feature_set or FEATURE_SET, n_lags=n_lags, reference=variant, lam=lam,
                             val_r2_mean=np.mean(list(per_session.values())),
                             **{f'val_r2_{s}': v for s, v in per_session.items()}))

    search = pd.DataFrame(rows)
    best = search.loc[search.val_r2_mean.idxmax()]
    search['chosen'] = (search.reference == best.reference) & (search.lam == best.lam)
    return dict(model=fitters[best.reference].fit(best.lam), reference=best.reference,
                lam=float(best.lam), search=search)


def print_lambda_search(search):
    """Print validation R² for every λ and reference, and the choice."""
    print('\nValidation R2 (mean of the validation OFF sessions, query trials) for every lambda:')
    print(search.pivot(index='lam', columns='reference', values='val_r2_mean')
          .to_string(float_format=lambda v: f'{v:8.4f}'))
    for variant, group in search.groupby('reference'):
        row = group.loc[group.val_r2_mean.idxmax()]
        edge = ' (at the edge of the grid!)' if row.lam in (LAMBDAS[0], LAMBDAS[-1]) else ''
        print(f'  best for {variant}: lambda = {row.lam:g}, R2 = {row.val_r2_mean:.4f}{edge}')
    chosen = search[search.chosen].iloc[0]
    print(f'Chosen: reference = {chosen.reference}, lambda = {chosen.lam:g}')


def score_session(model, session, variant, n_lags=N_LAGS, hf=False):
    """R² of a fitted decoder on one session's query trials, blocks 3-5 (mean and per axis)."""
    X, Y = session_arrays(session, variant, 'query', n_lags, hf)
    return score_arrays(model, X, Y)


def score_arrays(model, X, Y):
    """R² of a fitted decoder on given arrays (mean of vx and vy, and per axis)."""
    r2_vx, r2_vy = r2(Y, model.predict(X))
    return dict(r2=(r2_vx + r2_vy) / 2, r2_vx=r2_vx, r2_vy=r2_vy, n_bins=len(X))
