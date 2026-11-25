"""Task 3.1 ridge decoder on the original and the 10x semisynthetic ON recordings.

Uses the same OFF decoder as task3_1_ridge.py (trained on training-day OFF
trials; λ and reference chosen on the validation-day OFF query trials). Each
test-day stimulation-ON session is scored on the query trials (blocks 3-5) of

    original:   the recorded LFP                         (export_none)
    parrm:      the PARRM-cleaned LFP                    (parrm_none)
    amp10:      recorded LFP + 9 x PARRM artifact        (parrm_amp10, semisynthetic)

next to its same-day OFF session. Trial times and velocity labels are the
same in all three; only the LFP features differ. The amp10 features must
first be made with task3_make_amp10_features.py.

Run (from the repository root):
      python task3/task3_1_ridge_semisynthetic.py --lags 9
Writes task3_results/task3_1_semisynthetic_test_results_lmp_bands_lags<N>.csv.
"""

import argparse
import os
import time

import pandas as pd

import task3_common as C
from task3_make_amp10_features import VARIANT as AMP10_VARIANT

DATASETS = {'original': 'export_none', 'parrm': 'parrm_none', 'amp10': AMP10_VARIANT}


def main(n_lags):
    start = time.time()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    tag = C.run_tag(n_lags)
    M, S = C.manifest(), C.splits()
    test_on = C.sessions('test', 'stim')
    missing = [s for s in test_on if not os.path.exists(
        os.path.join(C.F.features_dir(C.PATHS, AMP10_VARIANT), f'{s}.h5'))]
    if missing:
        raise SystemExit(f'Missing {AMP10_VARIANT} features for {missing}. '
                         f'Run: python task3/task3_make_amp10_features.py --split test')

    decoder = C.fit_off_decoder(n_lags)
    C.print_lambda_search(decoder['search'])
    model, reference = decoder['model'], decoder['reference']
    if reference != 'export_none':
        raise SystemExit(f'Chosen reference is {reference}; the 10x features are built without CAR '
                         f'(export_none), so the comparison would not be like for like.')

    off_scores, rows = {}, []
    for session in test_on:
        off = S['off_reference'][session]
        if off not in off_scores:
            off_scores[off] = C.score_session(model, off, reference, n_lags)
        scores = {name: C.score_session(model, session, variant, n_lags) for name, variant in DATASETS.items()}
        r2_off = off_scores[off]['r2']
        row = dict(session=session, date=M.loc[session, 'date'], condition=M.loc[session, 'condition'],
                   off_reference=off, r2_off=r2_off)
        row.update({f'r2_on_{name}': sc['r2'] for name, sc in scores.items()})
        row.update({f'gap_{name}': sc['r2'] - r2_off for name, sc in scores.items()})
        row['amp10_minus_original'] = scores['amp10']['r2'] - scores['original']['r2']
        for name, sc in scores.items():
            row[f'r2_on_{name}_vx'], row[f'r2_on_{name}_vy'] = sc['r2_vx'], sc['r2_vy']
        row.update(n_bins_on=scores['original']['n_bins'], n_bins_off=off_scores[off]['n_bins'],
                   reference=reference, lam=decoder['lam'], feature_set=C.FEATURE_SET, n_lags=n_lags)
        assert len({sc['n_bins'] for sc in scores.values()}) == 1, f'{session}: datasets differ in valid bins'
        rows.append(row)
    results = pd.DataFrame(rows).sort_values(['condition', 'session'])
    out = os.path.join(C.RESULTS_DIR, f'task3_1_semisynthetic_test_results_{tag}.csv')
    results.to_csv(out, index=False)

    print('\nTest days, query trials (R2 averaged over vx and vy):')
    print(results[['session', 'condition', 'r2_off', 'r2_on_original', 'r2_on_parrm', 'r2_on_amp10',
                   'amp10_minus_original']].to_string(index=False, float_format=lambda v: f'{v:7.3f}'))

    summary = results.groupby('condition').agg(
        n_sessions=('session', 'size'), r2_off=('r2_off', 'mean'), r2_on_original=('r2_on_original', 'mean'),
        r2_on_parrm=('r2_on_parrm', 'mean'), r2_on_amp10=('r2_on_amp10', 'mean'),
        amp10_minus_original=('amp10_minus_original', 'mean'), sd_amp10_minus_original=('amp10_minus_original', 'std'))
    print('\nSession means by condition:')
    print(summary.to_string(float_format=lambda v: f'{v:7.3f}'))
    print(f'\nDone in {time.time() - start:.0f} s. Results in {os.path.relpath(out, C.REPO)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 3.1 ridge decoder scored on recorded, PARRM-cleaned, '
                                                 'and 10x semisynthetic test-day ON trials.')
    parser.add_argument('--lags', type=int, default=C.N_LAGS,
                        help=f'earlier 50 ms bins of feature history per bin (default: {C.N_LAGS})')
    main(parser.parse_args().lags)
