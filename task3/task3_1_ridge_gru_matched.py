"""Task 3.1 ridge decoder matched to the GRU's inputs and same-day calibration.

Differences from task3_1_ridge.py, each taken from the GRU (task3_gru_core.py):
- Features: LMP + 7 band powers + 200-400 Hz power per channel (9, not 8).
  The 200-400 Hz power is computed, normalized and clipped like the others.
- History: 19 earlier bins (1 s including the current bin) by default.
- Reference: the recording reference (export_none), as in the GRU; no CAR search.
- Same-day calibration: like the GRU's fine-tuning on the 20 calibration
  trials (blocks 1-2) of the same-day OFF session, the cross-day ridge
  weights W0 are refit on those trials with a penalty on moving away from them:

      min over W, b   ||Y - X W - b||² + mu ||W - W0||²

  The intercept b is refit on the same trials. mu = inf keeps the cross-day
  model unchanged. The adapted model is then scored on the query trials
  (blocks 3-5) of the OFF session and of that day's ON sessions.
  (Not matched: the GRU's per-sequence time baseline and time-in-trial correction.)

Hyperparameters, chosen on the 3 validation days (adapting on each
validation-day OFF session's calibration trials and scoring its query trials):
- same-day model: (lambda, mu) jointly;
- cross-day model (no same-day labels, mu = inf): lambda alone.

Each test-day ON session is scored on the recorded, PARRM-cleaned and 10x
features (as in task3_1_ridge_semisynthetic.py), with both models.

Run (from the repository root):
      python task3/task3_1_ridge_gru_matched.py              (19 lags)
Writes task3_results/task3_1_ridge_gru_matched_search_lags<N>.csv,
task3_1_ridge_gru_matched_test_results_lags<N>.csv (one row per ON session and dataset)
and task3_1_ridge_gru_matched_summary_lags<N>.csv.
The 200-400 Hz caches are made on first use in data/features/hf200_400_*/.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

import task3_common as C
from task3_make_amp10_features import VARIANT as AMP10_VARIANT

REFERENCE = 'export_none'
DATASETS = {'recorded': 'export_none', 'parrm': 'parrm_none', 'amp10': AMP10_VARIANT}
MUS = C.MUS  # 1 to 10^8 in half decades, then no adaptation
N_LAGS = 19
FEATURE_LABEL = 'lmp+bands+hf200_400'


def calibration(session, variant, n_lags):
    """Adapter on a session's 20 calibration trials, after checking the trial counts match the GRU's."""
    n_pool, n_query = len(C.trial_rows(session, 'pool')), len(C.trial_rows(session, 'query'))
    if (n_pool, n_query) != (20, 30):
        raise ValueError(f'{session}: expected 20 calibration and 30 query trials, got {n_pool} and {n_query}')
    return C.SameDayAdapter(*C.session_arrays(session, variant, 'pool', n_lags, hf=True))


def main(n_lags):
    start = time.time()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    M, S = C.manifest(), C.splits()
    train_sessions = [s for s in C.sessions('train', 'no_stim') if C.has_cm_velocity(s)]
    val_sessions = C.sessions('val', 'no_stim')
    print(f'Features: {FEATURE_LABEL} (9 per channel), history: {n_lags} earlier bins, reference: {REFERENCE}')

    # ---- Cross-day fit on the training-day OFF trials, for every lambda
    X, Y = C.stack_arrays(train_sessions, REFERENCE, 'train', n_lags, hf=True)
    print(f'Training on {len(train_sessions)} OFF sessions: {X.shape[0]:,} bins x {X.shape[1]:,} inputs', flush=True)
    fitter = C.RidgeFitter(X, Y)
    del X, Y
    bases = {lam: fitter.fit(lam) for lam in C.LAMBDAS}
    del fitter

    # ---- Validation days: adapt on each OFF session's calibration trials, score its query trials
    rows = []
    for s in val_sessions:
        adapter = calibration(s, REFERENCE, n_lags)
        Xq, Yq = C.session_arrays(s, REFERENCE, 'query', n_lags, hf=True)
        for lam, base in bases.items():
            for mu in MUS:
                rows.append(dict(session=s, lam=lam, mu=mu,
                                 r2=C.score_arrays(adapter.fit(base, mu), Xq, Yq)['r2']))
    per_session = pd.DataFrame(rows)
    search = per_session.pivot_table(index=['lam', 'mu'], columns='session', values='r2').add_prefix('val_r2_')
    search.insert(0, 'val_r2_mean', search.mean(axis=1))
    search = search.reset_index()
    same = search.loc[search.val_r2_mean.idxmax()]
    cross = search[np.isinf(search.mu)].loc[lambda d: d.val_r2_mean.idxmax()]
    search['chosen_same_day'] = (search.lam == same.lam) & (search.mu == same.mu)
    search['chosen_cross_day'] = (search.lam == cross.lam) & np.isinf(search.mu)
    search = search.assign(n_lags=n_lags, features=FEATURE_LABEL, reference=REFERENCE)
    search.to_csv(os.path.join(C.RESULTS_DIR, f'task3_1_ridge_gru_matched_search_lags{n_lags}.csv'), index=False)

    print('\nValidation R2 (mean of the validation OFF sessions, query trials); columns are mu:')
    print(search.pivot(index='lam', columns='mu', values='val_r2_mean')
          .to_string(float_format=lambda v: f'{v:7.3f}'))
    for name, row in (('same-day', same), ('cross-day', cross)):
        edge = (row.lam in (C.LAMBDAS[0], C.LAMBDAS[-1])) or (row.mu in (MUS[0], MUS[-2]))
        print(f'Chosen {name}: lambda = {row.lam:g}, mu = {row.mu:g}, validation R2 = {row.val_r2_mean:.4f}'
              + (' (at the edge of the grid!)' if edge else ''))
    same_base, cross_model = bases[same.lam], bases[cross.lam]

    # ---- Test days
    off_cache, rows = {}, []
    for session in C.sessions('test', 'stim'):
        off = S['off_reference'][session]
        if off not in off_cache:
            adapted = calibration(off, REFERENCE, n_lags).fit(same_base, same.mu)
            Xq, Yq = C.session_arrays(off, REFERENCE, 'query', n_lags, hf=True)
            off_cache[off] = (adapted, C.score_arrays(adapted, Xq, Yq), C.score_arrays(cross_model, Xq, Yq))
        adapted, off_same, off_cross = off_cache[off]
        n_bins = set()
        for dataset, variant in DATASETS.items():
            Xq, Yq = C.session_arrays(session, variant, 'query', n_lags, hf=True)
            on_same, on_cross = C.score_arrays(adapted, Xq, Yq), C.score_arrays(cross_model, Xq, Yq)
            n_bins.add(len(Xq))
            rows.append(dict(
                session=session, date=M.loc[session, 'date'], condition=M.loc[session, 'condition'],
                off_reference=off, dataset=dataset,
                r2_same_day=on_same['r2'], r2_off_same_day=off_same['r2'],
                gap_same_day=on_same['r2'] - off_same['r2'],
                r2_cross_day=on_cross['r2'], r2_off_cross_day=off_cross['r2'],
                gap_cross_day=on_cross['r2'] - off_cross['r2'],
                r2_same_day_vx=on_same['r2_vx'], r2_same_day_vy=on_same['r2_vy'],
                r2_cross_day_vx=on_cross['r2_vx'], r2_cross_day_vy=on_cross['r2_vy'],
                n_bins_on=len(Xq), n_bins_off=off_same['n_bins'],
                lam_same_day=same.lam, mu_same_day=same.mu, lam_cross_day=cross.lam,
                reference=REFERENCE, features=FEATURE_LABEL, n_lags=n_lags))
        assert len(n_bins) == 1, f'{session}: datasets differ in valid bins'
    results = pd.DataFrame(rows).sort_values(['dataset', 'condition', 'session'])
    results.to_csv(os.path.join(C.RESULTS_DIR, f'task3_1_ridge_gru_matched_test_results_lags{n_lags}.csv'),
                   index=False)

    summary = results.groupby(['dataset', 'condition'], as_index=False)[
        ['r2_same_day', 'r2_off_same_day', 'gap_same_day', 'r2_cross_day', 'r2_off_cross_day', 'gap_cross_day']
    ].mean()
    summary.insert(2, 'n_sessions', results.groupby(['dataset', 'condition']).size().to_numpy())
    summary.to_csv(os.path.join(C.RESULTS_DIR, f'task3_1_ridge_gru_matched_summary_lags{n_lags}.csv'), index=False)
    print('\nTest days, session means (query trials, R2 averaged over vx and vy):')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:7.3f}'))
    off_same = [v[1]['r2'] for v in off_cache.values()]
    off_cross = [v[2]['r2'] for v in off_cache.values()]
    print(f'Test-day OFF sessions ({len(off_cache)}): same-day {np.mean(off_same):.3f}, '
          f'cross-day {np.mean(off_cross):.3f}')
    print(f'\nDone in {time.time() - start:.0f} s. Results in {os.path.relpath(C.RESULTS_DIR, C.REPO)}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 3.1 ridge matched to the GRU: 9 features, 19 lags, '
                                                 'same-day calibration on 20 OFF trials.')
    parser.add_argument('--lags', type=int, default=N_LAGS,
                        help=f'earlier 50 ms bins of feature history per bin (default: {N_LAGS})')
    main(parser.parse_args().lags)
