"""Task 3.1 check: how well does ridge decode within a single session?

For each of the 14 no-stim sessions on the training days, fit a ridge decoder
on 4 of the session's 5 task blocks and compute R² on the 5th, once for each
block (leave one block out). Everything else is as in task3_1_ridge.py: LMP +
7 band powers, the same trial window and lags, ±10 clipping, the recording
reference and the same λ grid.

This is the easiest setting for the decoder: the trials it is trained on and
the trials it is scored on come from the same recording, so there is no change
of day. Comparing it with the cross-day R² of task3_1_ridge.py checks that
pipeline, and shows how much of its low R² comes from the change of day.

Details:
- Normalization (median and MAD per channel and feature) uses only the 4
  training blocks of each fold, so the scored block is used for nothing but R².
- R² is computed on the held-out block's 10 trials (vx and vy averaged), then
  averaged over the 5 blocks.
- λ: R² is printed for every λ in the grid. Each session's main R² uses the λ
  that does best on average over the other sessions, so a session never picks
  its own λ. Each session's own best λ is also shown, as an optimistic upper
  bound.
- July 11 (Butters-250711-101424) has no velocity in cm/s (no touch data), so
  it is decoded in pixels/s. Its pixel axes are the camera's, not the screen's,
  so it is listed separately and left out of the averages.

Run (from the repository root):
      python task3/task3_1_ridge_within_session.py             (9 earlier bins of history)
      python task3/task3_1_ridge_within_session.py --lags 19
Writes results/task3/task3_1_within_session_folds_lmp_bands_lags<N>.csv (one row
per session, held-out block and λ) and task3_1_within_session_lmp_bands_lags<N>.csv
(one row per session). Run task3_1_ridge.py with the same --lags first to get the
comparison with the cross-day decoder.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import task3_common as C  # import first: it puts the repository root on sys.path
import EECS182_NHP_features as F  # noqa: E402

PIXEL_COLS = ('Index_vx', 'Index_vy')  # px/s; for July 11, which has no cm/s velocity
PAD = 1.0  # s of padding around each block for the normalization window, as in F.load_features
# The per-fold matrices are small. With all 22 cores of a laptop, the linear algebra
# ran ~10x slower than with 8 threads (the threads spend their time coordinating).
BLAS_THREADS = 8


def cut_trials(session, variant, n_lags):
    """Cut all 50 trials of a session once, from the unnormalized features.

    Returns the features, the trial rows, X (trials, bins, d), Y (trials, bins, 2)
    and the valid-bin mask (trials, bins).
    """
    feat = F.load_features(session, C.PATHS, variant=variant, normalize=False)
    assert feat['feature_names'] == C.FEATURES
    rows = C.trial_rows(session, 'all')
    kin_cols = C.KIN_COLS if C.has_cm_velocity(session) else PIXEL_COLS
    X, Y, valid, _ = F.trial_arrays(feat, rows, kin=C.kinematics(session), align=C.ALIGN, pre=C.PRE,
                                    post=C.POST, n_lags=n_lags, kin_cols=kin_cols)
    return feat, rows, X, Y, valid


def block_statistics(feat, rows):
    """Median and 1.4826 x MAD per channel and feature, over the blocks of the given
    trials only: from 1 s before each block's first go cue to 1 s after its last reward."""
    t = feat['bin_id'] * C.L.BIN_S
    within = np.zeros(len(t), bool)
    for _, block in rows.groupby('task_block'):
        within |= (t >= block.t_go.min() - PAD) & (t + C.L.BIN_S <= block.t_reward.max() + PAD)
    median = np.nanmedian(feat['X'][within], axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(feat['X'][within] - median), axis=0)
    scale[~(scale > 0)] = 1.0
    return median, scale


def normalize(X, median, scale, n_lags):
    """z-score lagged inputs with per-(channel, feature) statistics and clip to ±CLIP_Z,
    the same values session_arrays() gets by normalizing before cutting the trials."""
    median, scale = np.tile(median.ravel(), n_lags + 1), np.tile(scale.ravel(), n_lags + 1)
    return np.clip((X - median) / scale, -C.CLIP_Z, C.CLIP_Z)


def leave_one_block_out(session, variant, n_lags):
    """R² on each held-out block, for every λ. One row per (block, λ)."""
    feat, rows, X, Y, valid = cut_trials(session, variant, n_lags)
    blocks = rows.task_block.to_numpy()
    results = []
    for block in np.unique(blocks):
        held_out = blocks == block
        Xn = normalize(X, *block_statistics(feat, rows[~held_out]), n_lags)
        ok = valid & np.isfinite(Xn).all(axis=2)
        train, test = ok & ~held_out[:, None], ok & held_out[:, None]
        for model in C.ridge_path(Xn[train], Y[train], C.LAMBDAS):
            r2_vx, r2_vy = C.r2(Y[test], model.predict(Xn[test]))
            results.append(dict(session=session, held_out_block=int(block), lam=model.lam,
                                r2=(r2_vx + r2_vy) / 2, r2_vx=r2_vx, r2_vy=r2_vy,
                                n_train_bins=int(train.sum()), n_test_bins=int(test.sum())))
    return results


def main(n_lags, reference):
    start = time.time()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    tag = C.run_tag(n_lags)
    M = C.manifest()

    sessions = C.sessions('train', 'no_stim')
    cm_sessions = [s for s in sessions if C.has_cm_velocity(s)]
    units = {s: 'cm/s' if s in cm_sessions else 'px/s' for s in sessions}
    print(f'Within-session ridge, leave one block out: {len(sessions)} training-day OFF sessions '
          f'({len(cm_sessions)} in cm/s, {len(sessions) - len(cm_sessions)} in px/s).')
    print(f'Features: LMP + 7 band powers, history: {n_lags} earlier bins '
          f'({(n_lags + 1) * 50} ms including the current bin), reference: {reference}')

    rows = []
    with threadpool_limits(BLAS_THREADS):
        for session in sessions:
            rows += leave_one_block_out(session, reference, n_lags)
    folds = pd.DataFrame(rows)
    folds.insert(1, 'velocity_units', folds.session.map(units))
    folds = folds.assign(reference=reference, feature_set=C.FEATURE_SET, n_lags=n_lags)
    folds.to_csv(os.path.join(C.RESULTS_DIR, f'task3_1_within_session_folds_{tag}.csv'), index=False)

    # Mean over the 5 held-out blocks, for every session and λ
    by_session = folds.groupby(['session', 'lam'])[['r2', 'r2_vx', 'r2_vy']].mean()
    grid = by_session.r2.unstack('lam')  # sessions x λ
    grid_mean = grid.loc[cm_sessions].mean()

    summary = []
    for session in sessions:
        lam = grid.loc[[s for s in cm_sessions if s != session]].mean().idxmax()  # chosen on the other sessions
        best = grid.loc[session].idxmax()
        at_lam = folds[(folds.session == session) & (folds.lam == lam)]
        summary.append(dict(session=session, date=M.loc[session, 'date'], velocity_units=units[session],
                            lam=lam, r2=by_session.loc[(session, lam), 'r2'],
                            r2_vx=by_session.loc[(session, lam), 'r2_vx'],
                            r2_vy=by_session.loc[(session, lam), 'r2_vy'],
                            r2_sd_over_blocks=at_lam.r2.std(),
                            best_lam=best, r2_best_lam=grid.loc[session, best],
                            reference=reference, feature_set=C.FEATURE_SET, n_lags=n_lags))
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(C.RESULTS_DIR, f'task3_1_within_session_{tag}.csv'), index=False)
    cm = summary[summary.velocity_units == 'cm/s']

    # The cross-day decoder's choices, if task3_1_ridge.py has been run with the same lags
    search_file = os.path.join(C.RESULTS_DIR, f'task3_1_lambda_search_{tag}.csv')
    test_file = os.path.join(C.RESULTS_DIR, f'task3_1_test_results_{tag}.csv')
    cross_day = None
    if os.path.exists(search_file) and os.path.exists(test_file):
        chosen = pd.read_csv(search_file).query('chosen').iloc[0]
        test_off = pd.read_csv(test_file).drop_duplicates('off_reference').r2_off
        cross_lam = grid_mean.index[np.argmin(np.abs(np.log(grid_mean.index.to_numpy() / chosen.lam)))]
        cross_day = dict(chosen=chosen, test_off=test_off, lam=cross_lam)

    print(f'\nWithin-session R2 for every lambda (mean over the 5 held-out blocks; '
          f'"mean (cm/s)" = mean of the {len(cm_sessions)} sessions in cm/s):')
    px_sessions = [s for s in sessions if s not in cm_sessions]
    table = pd.DataFrame({'mean (cm/s)': grid_mean, **{f'{s} (px/s)': grid.loc[s] for s in px_sessions}})
    table.index = [f'{lam:>9,.0f}' + ('  <- chosen by task3_1_ridge.py' if cross_day and lam == cross_day['lam']
                                      else '') for lam in table.index]
    table.index.name = 'lambda'
    print(table.to_string(float_format=lambda v: f'{v:8.4f}'))

    print('\nPer session (lambda chosen on the other sessions; best lambda = chosen on the scored blocks):')
    print(summary[['session', 'velocity_units', 'lam', 'r2', 'r2_vx', 'r2_vy', 'r2_sd_over_blocks',
                   'best_lam', 'r2_best_lam']]
          .to_string(index=False, float_format=lambda v: f'{v:7.3f}' if abs(v) < 100 else f'{v:,.0f}'))
    print(f'Mean of the {len(cm)} sessions in cm/s: R2 = {cm.r2.mean():.3f} '
          f'(range {cm.r2.min():.3f} to {cm.r2.max():.3f}; vx {cm.r2_vx.mean():.3f}, vy {cm.r2_vy.mean():.3f}); '
          f'with each session\'s best lambda {cm.r2_best_lam.mean():.3f}')

    # ---- Side by side with the cross-day decoder
    if cross_day:
        chosen, test_off = cross_day['chosen'], cross_day['test_off']
        print(f'\nCompared with the cross-day decoder (task3_1_ridge.py: trained on the 13 training-day OFF '
              f'sessions, {chosen.reference}, lambda {chosen.lam:,.0f} chosen on the validation days):')
        print(f'  cross-day, validation days (3 OFF sessions, query trials):  R2 = {chosen.val_r2_mean:.3f}')
        print(f'  cross-day, test days ({len(test_off)} OFF sessions, query trials):        '
              f'R2 = {test_off.mean():.3f}')
        print(f'  within session, leave one block out ({len(cm)} sessions in cm/s): R2 = {cm.r2.mean():.3f} '
              f'(at the cross-day lambda: {grid_mean[cross_day["lam"]]:.3f}; '
              f'each session\'s best lambda: {cm.r2_best_lam.mean():.3f})')
        if chosen.reference != reference:
            print(f'  Note: task3_1_ridge.py chose {chosen.reference}; this run used {reference}.')
    else:
        print(f'\nRun task3_1_ridge.py --lags {n_lags} to add the comparison with the cross-day decoder.')
    print(f'\nDone in {time.time() - start:.0f} s. Results in {os.path.relpath(C.RESULTS_DIR, C.REPO)}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 3.1 check: ridge fit and scored within each training-day '
                                                 'OFF session, leaving out one task block at a time.')
    parser.add_argument('--lags', type=int, default=C.N_LAGS,
                        help=f'earlier 50 ms bins of feature history per bin (default: {C.N_LAGS})')
    parser.add_argument('--reference', choices=C.REFERENCES, default='export_none',
                        help='export_none (recording reference; chosen by task3_1_ridge.py) or export_car')
    args = parser.parse_args()
    main(args.lags, args.reference)
