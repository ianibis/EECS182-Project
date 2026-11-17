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

import numpy as np
import pandas as pd

TASK_DIR = os.path.dirname(os.path.abspath(__file__))  # .../task3
REPO = os.path.dirname(TASK_DIR)                         # repository root
if REPO not in sys.path:  # the shared loader and features scripts live at the repository root
    sys.path.insert(0, REPO)

import EECS182_NHP_data_loader as L  # noqa: E402
import EECS182_NHP_features as F     # noqa: E402

DATA = os.path.join(REPO, 'data')
RESULTS_DIR = os.path.join(REPO, 'results', 'task3')

# The loader's defaults assume the original folder names; point it at data/.
PATHS = L.Paths(
    root=DATA,  # features are read from data/features/<variant>/
    export_dir=os.path.join(DATA, 'exports', 'export_h5_LFPs'),
    trial_dir=os.path.join(DATA, 'exports', 'export_trials'),
    parrm_dir=os.path.join(DATA, 'parrm', 'parrm_h5_LFPs'),
    kin_dir=os.path.join(DATA, 'kinematics_bins50ms', 'bins50ms'),
)

# Trial window and feature history (README defaults): 0.5 s before the go cue
# to 1.5 s after it (40 bins of 50 ms), each bin also seeing the 9 bins before.
ALIGN, PRE, POST, N_LAGS = 't_go', 0.5, 1.5, 9
KIN_COLS = ('Index_vx_cm', 'Index_vy_cm')  # fingertip velocity, cm/s

# Normalized features are clipped to ±CLIP_Z so a single artifact cannot
# dominate the linear decoder (see Notes.md, Task 3.1: channel 64 in
# 250616-102853 reaches z ≈ 5300). Clean data stays well inside this range.
CLIP_Z = 10.0

# Decoder inputs for every channel: LMP + the 7 band powers. (A band-power-only
# decoder was dropped: it scored lower, validation R² 0.021 vs. 0.035 with 9 lags.)
FEATURES = list(F.FEATURE_NAMES)  # ['lmp', 'pow_1_4', ..., 'pow_130_200']
FEATURE_SET = 'lmp+bands'         # label written to the result files


# ---------------------------------------------------------------- metadata

@lru_cache(maxsize=None)
def manifest():
    return pd.read_csv(os.path.join(REPO, 'EECS182_data_manifest.csv')).set_index('session', drop=False)


@lru_cache(maxsize=None)
def splits():
    with open(os.path.join(REPO, 'splits.json')) as file:
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
    """
    included = _included_trials(session)
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


# ---------------------------------------------------------------- arrays

def session_arrays(session, variant, part, n_lags=N_LAGS):
    """Decoder inputs and target velocities for one session's trials.

    Returns X (n, d) and Y (n, 2) with one row per 50 ms bin that has both
    finite features and finite velocity. d = (n_lags + 1) * channels * 8
    features (LMP + 7 band powers): each bin sees its own features plus those
    of the n_lags bins before it.

    Features are z-scored within the session, then clipped to ±CLIP_Z.
    Validation and test sessions take the statistics from their calibration
    pool (blocks 1-2) only, so the query trials are never used for anything
    but the final R².
    """
    norm_rows = trial_rows(session, 'pool') if session in splits()['calibration'] else None
    feat = F.load_features(session, PATHS, variant=variant, normalize=True, trials=norm_rows)
    assert feat['feature_names'] == FEATURES
    feat = dict(feat, X=np.clip(feat['X'], -CLIP_Z, CLIP_Z))  # NaN stays NaN

    X, Y, valid, _ = F.trial_arrays(feat, trial_rows(session, part), kin=kinematics(session),
                                    align=ALIGN, pre=PRE, post=POST, n_lags=n_lags, kin_cols=KIN_COLS)
    return X[valid], Y[valid]


def stack_arrays(session_list, variant, part, n_lags=N_LAGS):
    """session_arrays() for several sessions, concatenated."""
    pairs = [session_arrays(s, variant, part, n_lags) for s in session_list]
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


def run_tag(n_lags):
    """File-name tag for one setting, e.g. 'lmp_bands_lags9'."""
    return f"{FEATURE_SET.replace('+', '_')}_lags{n_lags}"


def fit_off_decoder(n_lags=N_LAGS):
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

    rows, fitters = [], {}
    for variant in REFERENCES:
        X, Y = stack_arrays(train_sessions, variant, 'train', n_lags)
        fitters[variant] = RidgeFitter(X, Y)
        print(f'  {variant}: {X.shape[0]:,} training bins x {X.shape[1]:,} inputs')
        del X, Y
        val = {s: session_arrays(s, variant, 'query', n_lags) for s in val_sessions}
        for lam in LAMBDAS:
            model = fitters[variant].fit(lam)
            per_session = {s: r2(Yv, model.predict(Xv)).mean() for s, (Xv, Yv) in val.items()}
            rows.append(dict(feature_set=FEATURE_SET, n_lags=n_lags, reference=variant, lam=lam,
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


def score_session(model, session, variant, n_lags=N_LAGS):
    """R² of a fitted decoder on one session's query trials, blocks 3-5 (mean and per axis)."""
    X, Y = session_arrays(session, variant, 'query', n_lags)
    r2_vx, r2_vy = r2(Y, model.predict(X))
    return dict(r2=(r2_vx + r2_vy) / 2, r2_vx=r2_vx, r2_vy=r2_vy, n_bins=len(X))
