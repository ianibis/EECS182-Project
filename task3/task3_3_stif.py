"""Task 3.3: stimulus-monitor (StiF) sanity check.

Can the stimulation signal alone predict hand velocity? It should not: the
amplitude-modulated stimulation runs on its own clock, independent of the
task. If a decoder fed only StiF features reached a clearly positive R², the
stimulation would be linked to the movement (e.g. through its timing), and a
brain decoder could score on ON trials by reading the stimulation instead of
the brain. Expected result: R² close to 0 (or below).

StiF features per 50 ms bin, from /stim in the session's export file:
  env_mean   mean AM envelope in the bin (z-scored within the session)
  env_sd     SD of the envelope within the bin (z-scored within the session)
  phase_sin, phase_cos   AM phase at the bin end (0 while stimulation is off)
  on_frac    fraction of the bin with the stimulator on
Same trial window, lags, clipping and λ grid as Task 3.1. Trained on the
training-day stimulation sessions, λ chosen on the validation-day stimulation
sessions, scored on the query trials of the test-day stimulation sessions.

Run (from the repository root):  python task3/task3_3_stif.py --lags 9
Writes results/task3/task3_3_lambda_search_lags<N>.csv and
task3_3_test_results_lags<N>.csv.
"""

import argparse
import os
import time
from functools import lru_cache

import h5py
import numpy as np
import pandas as pd

import task3_common as C  # import first: it puts the repository root on sys.path
import EECS182_NHP_features as F  # noqa: E402

STIF_FEATURES = ('env_mean', 'env_sd', 'phase_sin', 'phase_cos', 'on_frac')
NORMALIZED = (0, 1)  # env_mean and env_sd are in uncalibrated monitor units


@lru_cache(maxsize=None)
def _stif_bins(session):
    """Unnormalized StiF features for every 50 ms bin of the recording."""
    with h5py.File(os.path.join(C.PATHS.export_dir, f'{session}.h5'), 'r') as file:
        t = file['t'][:]
        envelope = file['stim/envelope'][:].astype(np.float64)
        on = file['stim/on'][:].astype(np.float64)
        phase = file['stim/phase'][:]

    # Bin j covers [0.05 j, 0.05 (j + 1)) on the TDT clock, as in the loader.
    bin_of_sample = np.floor(t / C.L.BIN_S + 1e-9).astype(np.int64)
    first = bin_of_sample[0]
    index = bin_of_sample - first
    n_bins = index[-1] + 1
    count = np.bincount(index, minlength=n_bins).astype(np.float64)
    count[count == 0] = np.nan  # empty bins (none expected) become NaN

    mean = np.bincount(index, envelope, n_bins) / count
    mean_sq = np.bincount(index, envelope ** 2, n_bins) / count
    sd = np.sqrt(np.maximum(mean_sq - mean ** 2, 0.0))
    on_frac = np.bincount(index, on, n_bins) / count

    last_sample = np.searchsorted(index, np.arange(n_bins), side='right') - 1  # causal: bin end
    phase_end = phase[np.clip(last_sample, 0, len(phase) - 1)]
    off = ~np.isfinite(phase_end)  # phase is NaN while stimulation is off
    phase_sin = np.where(off, 0.0, np.sin(np.where(off, 0.0, phase_end)))
    phase_cos = np.where(off, 0.0, np.cos(np.where(off, 0.0, phase_end)))

    X = np.stack([mean, sd, phase_sin, phase_cos, on_frac], axis=1).astype(np.float32)
    return np.arange(first, first + n_bins), X


def stif_features(session):
    """StiF features as a feature dict for F.trial_arrays: X is (bins, 1, 5).

    env_mean and env_sd are z-scored (median and 1.4826 x MAD) over the task
    interval, using the calibration pool for validation and test sessions, as
    for the LFP features; all features are clipped to ±CLIP_Z.
    """
    bin_ids, X = _stif_bins(session)
    X = X.copy()
    rows = C.trial_rows(session, 'pool' if session in C.splits()['calibration'] else 'all')
    start, stop = rows.t_go.min() - 1.0, rows.t_reward.max() + 1.0
    within = (bin_ids * C.L.BIN_S >= start) & ((bin_ids + 1) * C.L.BIN_S <= stop)
    for j in NORMALIZED:
        median = np.nanmedian(X[within, j])
        scale = 1.4826 * np.nanmedian(np.abs(X[within, j] - median))
        X[:, j] = (X[:, j] - median) / (scale if scale > 0 else 1.0)
    X = np.clip(X, -C.CLIP_Z, C.CLIP_Z)
    return dict(bin_id=bin_ids, X=X[:, None, :], feature_names=list(STIF_FEATURES))


def stif_arrays(session, part, n_lags):
    X, Y, valid, _ = F.trial_arrays(stif_features(session), C.trial_rows(session, part),
                                    kin=C.kinematics(session), align=C.ALIGN, pre=C.PRE, post=C.POST,
                                    n_lags=n_lags, kin_cols=C.KIN_COLS)
    return X[valid], Y[valid]


def main(n_lags):
    start = time.time()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    M = C.manifest()

    train_sessions = [s for s in C.sessions('train', 'stim') if C.has_cm_velocity(s)]
    val_sessions = C.sessions('val', 'stim')
    print(f'StiF features: {", ".join(STIF_FEATURES)}; history: {n_lags} earlier bins '
          f'({(n_lags + 1) * 50} ms including the current bin)')
    print(f'Training on {len(train_sessions)} stimulation sessions (training days, held-out block excluded); '
          f'choosing lambda on {len(val_sessions)} validation-day stimulation sessions.')

    # ---- Train, and choose λ on the validation-day stimulation sessions
    pairs = [stif_arrays(s, 'train', n_lags) for s in train_sessions]
    X, Y = np.concatenate([x for x, _ in pairs]), np.concatenate([y for _, y in pairs])
    fitter = C.RidgeFitter(X, Y)
    print(f'  {X.shape[0]:,} training bins x {X.shape[1]:,} inputs')
    val = {s: stif_arrays(s, 'query', n_lags) for s in val_sessions}

    rows = []
    for lam in C.LAMBDAS:
        model = fitter.fit(lam)
        per_session = {s: C.r2(Yv, model.predict(Xv)).mean() for s, (Xv, Yv) in val.items()}
        rows.append(dict(n_lags=n_lags, lam=lam, val_r2_mean=np.mean(list(per_session.values())),
                         **{f'val_r2_{s}': v for s, v in per_session.items()}))
    search = pd.DataFrame(rows)
    best = search.loc[search.val_r2_mean.idxmax()]
    search['chosen'] = search.lam == best.lam
    search.to_csv(os.path.join(C.RESULTS_DIR, f'task3_3_lambda_search_lags{n_lags}.csv'), index=False)

    print('\nValidation R2 (mean of the validation stimulation sessions, query trials) for every lambda:')
    print(search[['lam', 'val_r2_mean']].to_string(index=False, float_format=lambda v: f'{v:10.4f}'))
    edge = ' (at the edge of the grid!)' if best.lam in (C.LAMBDAS[0], C.LAMBDAS[-1]) else ''
    print(f'Chosen: lambda = {best.lam:g}, R2 = {best.val_r2_mean:.4f}{edge}')
    model = fitter.fit(best.lam)

    # ---- Test days: every stimulation session, query trials
    rows = []
    for session in C.sessions('test', 'stim'):
        Xt, Yt = stif_arrays(session, 'query', n_lags)
        r2_vx, r2_vy = C.r2(Yt, model.predict(Xt))
        rows.append(dict(session=session, date=M.loc[session, 'date'], condition=M.loc[session, 'condition'],
                         r2=(r2_vx + r2_vy) / 2, r2_vx=r2_vx, r2_vy=r2_vy, n_bins=len(Xt),
                         lam=best.lam, n_lags=n_lags))
    results = pd.DataFrame(rows).sort_values(['condition', 'session'])
    results.to_csv(os.path.join(C.RESULTS_DIR, f'task3_3_test_results_lags{n_lags}.csv'), index=False)

    print('\nTest days, query trials, StiF-only decoder (R2 averaged over vx and vy):')
    print(results[['session', 'condition', 'r2', 'r2_vx', 'r2_vy']]
          .to_string(index=False, float_format=lambda v: f'{v:7.3f}'))
    print('\nSummary by condition:')
    print(results.groupby('condition').r2.agg(['size', 'mean', 'std', 'min', 'max'])
          .to_string(float_format=lambda v: f'{v:7.3f}'))

    brain_file = os.path.join(C.RESULTS_DIR, f'task3_1_test_results_{C.run_tag("lmp+bands", n_lags)}.csv')
    if os.path.exists(brain_file):
        brain = pd.read_csv(brain_file).groupby('condition').r2_on.mean()
        print('For comparison, Task 3.1 brain decoder (LMP + band power, same lags) on the same ON trials: '
              + ', '.join(f'{c} {v:.3f}' for c, v in brain.items()))
    print(f'\nDone in {time.time() - start:.0f} s. Results in {os.path.relpath(C.RESULTS_DIR, C.REPO)}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 3.3: ridge decoder from stimulus-monitor (StiF) '
                                                 'features alone, as a sanity check.')
    parser.add_argument('--lags', type=int, default=C.N_LAGS,
                        help=f'earlier 50 ms bins of feature history per bin (default: {C.N_LAGS})')
    main(parser.parse_args().lags)
