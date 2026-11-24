"""Task 3.2: the traditional pipeline. PARRM cleaning, then a ridge refit on k stimulation trials.

Part 1 (k = 0): does PARRM cleaning alone recover the accuracy lost to stimulation?
Uses the same OFF decoder as Task 3.1 (same training trials, same λ and
reference chosen on the validation days). No-stim sessions have no PARRM
file, so their PARRM features are identical to the raw ones and the decoder
does not change. Only the test-day stimulation-ON sessions are scored twice:
on raw features (as in 3.1) and on PARRM-cleaned features.

Part 2 (k = 2, 5, 10, 20): the proposal's "traditional pipeline". The OFF
decoder W0 is refit on k labeled, PARRM-cleaned trials of the stimulation-ON
session itself, with a penalty on moving away from W0:

      min over W, b   ||Y - X W - b||² + mu ||W - W0||²

(task3_common.SameDayAdapter). The k trials are the 10 draws per session
stored in splits.json (calibration[session].subsets[k], drawn from blocks
1-2; k = 20 is the whole pool, so its 10 draws are identical). Each refit
model is scored on the same session's query trials (blocks 3-5), whose
labels are used for nothing else. Features are normalized with the
statistics of the 20 calibration-pool trials (features only, no labels),
as for every validation and test session.

mu is chosen separately for each k on the validation-day ON sessions
(mean query R² over their draws), from 1 to 10^8 and inf (= no refit).
λ stays the one chosen for the OFF decoder, so k = 0 is exactly Part 1's
PARRM score.

Run (from the repository root):  python task3/task3_2_parrm.py            (19 lags)
Writes task3_results/
  task3_2_test_results_lmp_bands_lags<N>.csv             k = 0: raw vs. PARRM, per ON session
  task3_2_traditional_mu_search_lmp_bands_lags<N>.csv    validation R² for every k and mu
  task3_2_traditional_test_draws_lmp_bands_lags<N>.csv   test R² per ON session, k and draw
  task3_2_traditional_summary_lmp_bands_lags<N>.csv      per condition and k: mean R², 95% bootstrap
                                                         interval over sessions, fraction of the loss recovered
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

import task3_common as C

KS = (2, 5, 10, 20)  # labeled stimulation trials used for the refit (k = 0 is Part 1)
N_BOOT = 10_000      # bootstrap resamples of sessions for the 95% intervals


def k_draws(session, k):
    """The 10 k-trial draws of a session's calibration pool, from splits.json."""
    return C.splits()['calibration'][session]['subsets'][str(k)]


class RefitSession:
    """One ON session's PARRM-cleaned calibration pool and query trials, cut once."""

    def __init__(self, session, variant, n_lags):
        self.session = session
        self.Xp, self.Yp, self.trials = C.session_arrays(session, variant, 'pool', n_lags, return_trials=True)
        self.Xq, self.Yq = C.session_arrays(session, variant, 'query', n_lags)

    def adapter(self, draw):
        use = np.isin(self.trials, draw)
        return C.SameDayAdapter(self.Xp[use], self.Yp[use]), int(use.sum())

    def score(self, model):
        return C.score_arrays(model, self.Xq, self.Yq)


def bootstrap_interval(values, rng):
    """95% percentile interval of the mean, resampling sessions with replacement."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan, np.nan
    means = values[rng.integers(0, len(values), (N_BOOT, len(values)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def choose_mus(base, clean_variant, n_lags):
    """Validation days: refit on each ON session's k-trial draws for every mu, score its query trials.
    Returns the search table and the chosen mu for each k."""
    val_sessions = C.sessions('val', 'stim')
    print(f'\nChoosing mu for each k on {len(val_sessions)} validation-day ON sessions (PARRM-cleaned).')
    rows = []
    for session in val_sessions:
        data = RefitSession(session, clean_variant, n_lags)
        for k in KS:
            for draw in k_draws(session, k):
                adapter, _ = data.adapter(draw)
                for mu in C.MUS:
                    rows.append(dict(session=session, k=k, mu=mu, r2=data.score(adapter.fit(base, mu))['r2']))
        print(f'  {session} done', flush=True)
    per_draw = pd.DataFrame(rows)
    per_session = per_draw.groupby(['k', 'mu', 'session'], as_index=False).r2.mean()  # mean over draws
    search = per_session.pivot_table(index=['k', 'mu'], columns='session', values='r2').add_prefix('val_r2_')
    search.insert(0, 'val_r2_mean', search.mean(axis=1))
    search = search.reset_index()
    best = search.loc[search.groupby('k').val_r2_mean.idxmax()].set_index('k')
    search['chosen'] = search.mu.to_numpy() == search.k.map(best.mu).to_numpy()

    print('\nValidation R2 (mean of the validation ON sessions and their draws); rows are mu, columns k:')
    print(search.pivot(index='mu', columns='k', values='val_r2_mean').to_string(float_format=lambda v: f'{v:7.4f}'))
    for k, row in best.iterrows():
        edge = ' (at the edge of the grid!)' if row.mu in (C.MUS[0], C.MUS[-2]) else ''
        edge = ' (no refit: mu = inf)' if np.isinf(row.mu) else edge
        print(f'  k = {k:2d}: mu = {row.mu:g}, validation R2 = {row.val_r2_mean:.4f}{edge}')
    return search, best.mu.to_dict()


def main(n_lags):
    start = time.time()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    tag = C.run_tag(n_lags)
    M, S = C.manifest(), C.splits()

    # ---- The 3.1 decoder (trained on OFF trials, λ and reference from the validation days)
    decoder = C.fit_off_decoder(n_lags)
    C.print_lambda_search(decoder['search'])
    model, raw_variant = decoder['model'], decoder['reference']
    clean_variant = C.PARRM_VARIANT[raw_variant]
    if not os.path.isdir(os.path.join(C.DATA, 'features', clean_variant)):
        raise SystemExit(f'Missing data/features/{clean_variant}/. Compute it from the task3 folder with:\n'
                         f'  python -c "import task3_common as C; '
                         f"C.F.compute_all(C.PATHS, variants=['{clean_variant}'])\"")
    print(f'\nScoring ON sessions on raw features ({raw_variant}) and PARRM-cleaned features ({clean_variant}).')

    # ==== Part 1 (k = 0). Test days: ON raw vs. ON cleaned vs. same-day OFF
    off_scores = {}
    rows = []
    for session in C.sessions('test', 'stim'):
        off = S['off_reference'][session]
        if off not in off_scores:
            off_scores[off] = C.score_session(model, off, raw_variant, n_lags)
        raw = C.score_session(model, session, raw_variant, n_lags)
        clean = C.score_session(model, session, clean_variant, n_lags)
        r2_off = off_scores[off]['r2']
        rows.append(dict(
            session=session, date=M.loc[session, 'date'], condition=M.loc[session, 'condition'],
            off_reference=off, r2_off=r2_off, r2_on_raw=raw['r2'], r2_on_parrm=clean['r2'],
            parrm_minus_raw=clean['r2'] - raw['r2'],
            gap_raw=raw['r2'] - r2_off, gap_parrm=clean['r2'] - r2_off,
            r2_on_parrm_vx=clean['r2_vx'], r2_on_parrm_vy=clean['r2_vy'],
            n_bins_on=clean['n_bins'], n_bins_off=off_scores[off]['n_bins'],
            parrm_am_clean_over_raw=M.loc[session, 'parrm_am_clean_over_raw'],
            reference=raw_variant, lam=decoder['lam'], feature_set=C.FEATURE_SET, n_lags=n_lags))
    results = pd.DataFrame(rows).sort_values(['condition', 'session'])
    results.to_csv(os.path.join(C.RESULTS_DIR, f'task3_2_test_results_{tag}.csv'), index=False)

    print('\nTest days, query trials (R2 averaged over vx and vy):')
    print(results[['session', 'condition', 'r2_off', 'r2_on_raw', 'r2_on_parrm', 'parrm_minus_raw']]
          .to_string(index=False, float_format=lambda v: f'{v:7.3f}'))

    print('\nSummary by condition:')
    for condition, group in results.groupby('condition'):
        off, raw, clean = group.r2_off.mean(), group.r2_on_raw.mean(), group.r2_on_parrm.mean()
        lost = off - raw
        recovered = f'{(clean - raw) / lost:.0%} of the ON-OFF gap closed' if lost > 0 else 'no gap to close'
        print(f'  {condition} ({len(group)} sessions): OFF {off:.3f} | ON raw {raw:.3f} | ON PARRM {clean:.3f} '
              f'| PARRM - raw {clean - raw:+.3f} (better in {int((group.parrm_minus_raw > 0).sum())}'
              f'/{len(group)} sessions) | {recovered}')

    # ==== Part 2 (k = 2, 5, 10, 20). PARRM cleaning, then ridge refit on k ON trials
    search, mus = choose_mus(model, clean_variant, n_lags)
    search = search.assign(reference=raw_variant, lam=decoder['lam'], feature_set=C.FEATURE_SET, n_lags=n_lags)
    search.to_csv(os.path.join(C.RESULTS_DIR, f'task3_2_traditional_mu_search_{tag}.csv'), index=False)

    print('\nTest days: refitting on k PARRM-cleaned ON trials, scoring the query trials.')
    k0 = results.set_index('session')
    draw_rows = []
    for session in k0.index:
        data = RefitSession(session, clean_variant, n_lags)
        info = dict(session=session, date=k0.loc[session, 'date'], condition=k0.loc[session, 'condition'],
                    off_reference=k0.loc[session, 'off_reference'])
        sc = data.score(model)  # k = 0: the OFF decoder on PARRM features, as in Part 1
        draw_rows.append(dict(info, k=0, draw=0, mu=np.inf, n_bins_calib=0,
                              r2=sc['r2'], r2_vx=sc['r2_vx'], r2_vy=sc['r2_vy']))
        for k in KS:
            for i, draw in enumerate(k_draws(session, k)):
                adapter, n_bins = data.adapter(draw)
                sc = data.score(adapter.fit(model, mus[k]))
                draw_rows.append(dict(info, k=k, draw=i, mu=mus[k], n_bins_calib=n_bins,
                                      r2=sc['r2'], r2_vx=sc['r2_vx'], r2_vy=sc['r2_vy']))
        print(f'  {session} done', flush=True)
    draws = pd.DataFrame(draw_rows).assign(reference=raw_variant, lam=decoder['lam'],
                                           feature_set=C.FEATURE_SET, n_lags=n_lags)
    draws.to_csv(os.path.join(C.RESULTS_DIR, f'task3_2_traditional_test_draws_{tag}.csv'), index=False)

    # Session R² = mean over that session's draws; then mean and bootstrap interval over sessions
    per_session = draws.groupby(['session', 'condition', 'k'], as_index=False).agg(
        r2=('r2', 'mean'), r2_sd_draws=('r2', 'std'))
    per_session = per_session.join(k0[['r2_off', 'r2_on_raw']], on='session')
    rng = np.random.default_rng(0)
    summary_rows = []
    for condition in sorted(per_session.condition.unique()) + ['all']:
        group = per_session if condition == 'all' else per_session[per_session.condition == condition]
        for k, g in group.groupby('k'):
            low, high = bootstrap_interval(g.r2, rng)
            off, raw, r2_k = g.r2_off.mean(), g.r2_on_raw.mean(), g.r2.mean()
            summary_rows.append(dict(
                condition=condition, k=k, mu=np.inf if k == 0 else mus[k], n_sessions=len(g),
                r2_mean=r2_k, r2_ci_low=low, r2_ci_high=high, mean_sd_over_draws=g.r2_sd_draws.mean(),
                r2_off_mean=off, r2_on_raw_mean=raw, gap_to_off=r2_k - off,
                fraction_of_loss_recovered=(r2_k - raw) / (off - raw) if off > raw else np.nan))
    summary = pd.DataFrame(summary_rows).assign(reference=raw_variant, lam=decoder['lam'],
                                                feature_set=C.FEATURE_SET, n_lags=n_lags)
    summary.to_csv(os.path.join(C.RESULTS_DIR, f'task3_2_traditional_summary_{tag}.csv'), index=False)

    print('\nTraditional pipeline (PARRM + ridge refit on k ON trials), test-day ON sessions:')
    print(summary[['condition', 'k', 'mu', 'n_sessions', 'r2_mean', 'r2_ci_low', 'r2_ci_high',
                   'r2_off_mean', 'r2_on_raw_mean', 'fraction_of_loss_recovered']]
          .to_string(index=False, float_format=lambda v: f'{v:7.3f}'))
    print(f'\nDone in {time.time() - start:.0f} s. Results in {os.path.relpath(C.RESULTS_DIR, C.REPO)}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 3.2: traditional pipeline. The Task 3.1 OFF decoder on '
                                                 'PARRM-cleaned stimulation-ON trials (k = 0), then refit on '
                                                 'k = 2, 5, 10, 20 labeled PARRM-cleaned ON trials.')
    parser.add_argument('--lags', type=int, default=C.N_LAGS,
                        help=f'earlier 50 ms bins of feature history per bin (default: {C.N_LAGS})')
    main(parser.parse_args().lags)
