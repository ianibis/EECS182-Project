"""Task 3.1: how much decoding accuracy does stimulation cost?

1. Train a ridge decoder (LFP features: LMP + 7 band powers per channel ->
   fingertip velocity) on stimulation-OFF trials from the training days
   (40 trials per session; the held-out block in splits.json is not used).
2. Choose the penalty λ, and the reference (recording reference vs. common
   average reference), on the 3 validation days: by R² on the query trials
   (blocks 3-5, 30 trials) of the validation-day OFF sessions. Every λ tried
   is printed and saved.
3. With those choices fixed, report R² on the query trials of every test-day
   stimulation-ON session and of its same-day OFF session.

Run (from the repository root):
      python task3/task3_1_ridge.py                          (9 earlier bins of history)
      python task3/task3_1_ridge.py --lags 19                (1 s of feature history)
Writes results/task3/task3_1_lambda_search_lmp_bands_lags<N>.csv and
task3_1_test_results_lmp_bands_lags<N>.csv.
See task3_1_ridge_within_session.py for the same decoder fit within single sessions.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

import task3_common as C


def main(n_lags):
    start = time.time()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    tag = C.run_tag(n_lags)
    M, S = C.manifest(), C.splits()

    # ---- 1-2. Train on OFF trials; choose λ and reference on the validation days
    decoder = C.fit_off_decoder(n_lags)
    decoder['search'].to_csv(os.path.join(C.RESULTS_DIR, f'task3_1_lambda_search_{tag}.csv'), index=False)
    C.print_lambda_search(decoder['search'])
    model, reference, lam = decoder['model'], decoder['reference'], decoder['lam']

    # ---- 3. Test days: ON sessions vs. their same-day OFF session
    off_scores = {}
    rows = []
    for session in C.sessions('test', 'stim'):
        off = S['off_reference'][session]
        if off not in off_scores:
            off_scores[off] = C.score_session(model, off, reference, n_lags)
        on = C.score_session(model, session, reference, n_lags)
        rows.append(dict(
            session=session, date=M.loc[session, 'date'], condition=M.loc[session, 'condition'],
            off_reference=off, r2_on=on['r2'], r2_off=off_scores[off]['r2'],
            gap_on_minus_off=on['r2'] - off_scores[off]['r2'],
            r2_on_vx=on['r2_vx'], r2_on_vy=on['r2_vy'],
            r2_off_vx=off_scores[off]['r2_vx'], r2_off_vy=off_scores[off]['r2_vy'],
            n_bins_on=on['n_bins'], n_bins_off=off_scores[off]['n_bins'],
            reference=reference, lam=lam, feature_set=C.FEATURE_SET, n_lags=n_lags))
    results = pd.DataFrame(rows).sort_values(['condition', 'session'])
    results.to_csv(os.path.join(C.RESULTS_DIR, f'task3_1_test_results_{tag}.csv'), index=False)

    print('\nTest days, query trials (R2 averaged over vx and vy):')
    print(results[['session', 'condition', 'off_reference', 'r2_on', 'r2_off', 'gap_on_minus_off']]
          .to_string(index=False, float_format=lambda v: f'{v:7.3f}'))

    summary = results.groupby('condition').agg(
        n_sessions=('session', 'size'), mean_r2_on=('r2_on', 'mean'),
        mean_r2_same_day_off=('r2_off', 'mean'), mean_gap=('gap_on_minus_off', 'mean'),
        sd_gap=('gap_on_minus_off', 'std'))
    print('\nSummary by condition:')
    print(summary.to_string(float_format=lambda v: f'{v:7.3f}'))
    off_values = [v['r2'] for v in off_scores.values()]
    print(f'Test-day OFF sessions ({len(off_values)}): mean R2 = {np.mean(off_values):.3f} '
          f'(range {min(off_values):.3f} to {max(off_values):.3f})')
    print(f'\nDone in {time.time() - start:.0f} s. Results in {os.path.relpath(C.RESULTS_DIR, C.REPO)}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 3.1: ridge decoder trained on stimulation-OFF trials, '
                                                 'scored on test-day ON trials and same-day OFF trials.')
    parser.add_argument('--lags', type=int, default=C.N_LAGS,
                        help=f'earlier 50 ms bins of feature history per bin (default: {C.N_LAGS})')
    main(parser.parse_args().lags)
