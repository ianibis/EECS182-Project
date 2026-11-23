"""Task 3.2: does PARRM cleaning recover the accuracy lost to stimulation?

Uses the same OFF decoder as Task 3.1 (same training trials, same λ and
reference chosen on the validation days). No-stim sessions have no PARRM
file, so their PARRM features are identical to the raw ones and the decoder
does not change. Only the test-day stimulation-ON sessions are scored twice:
on raw features (as in 3.1) and on PARRM-cleaned features.

Run (from the repository root):  python task3/task3_2_parrm.py --lags 9
Writes task3_results/task3_2_test_results_lmp_bands_lags<N>.csv.
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

    # ---- Test days: ON raw vs. ON cleaned vs. same-day OFF
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
    print(f'\nDone in {time.time() - start:.0f} s. Results in {os.path.relpath(C.RESULTS_DIR, C.REPO)}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 3.2: the Task 3.1 OFF decoder scored on '
                                                 'PARRM-cleaned stimulation-ON trials.')
    parser.add_argument('--lags', type=int, default=C.N_LAGS,
                        help=f'earlier 50 ms bins of feature history per bin (default: {C.N_LAGS})')
    main(parser.parse_args().lags)
