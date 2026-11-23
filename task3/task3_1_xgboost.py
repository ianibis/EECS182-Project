"""Task 3.1 with XGBoost: does a nonlinear decoder do better than ridge?

Same design as task3_1_ridge.py, with gradient-boosted trees instead of ridge:
1. Train on stimulation-OFF trials from the training days (one model for vx,
   one for vy). Same features, window, normalization and ±10 clipping.
2. On the query trials of the validation-day OFF sessions: stop adding trees
   once the validation error stops improving (early stopping), and choose the
   tree depth by mean R². Every depth tried is printed and saved.
3. With those choices fixed, report R² on the query trials of every test-day
   stimulation-ON session and of its same-day OFF session.

Early stopping uses the validation days, not XGBoost's usual random hold-out
of training rows: neighbouring 50 ms bins are near-copies of each other, so a
random hold-out would leak and make the model look better than it is.

Run (from the repository root):
      python task3/task3_1_xgboost.py                        (LMP + band power, 9 lags, CPU)
      python task3/task3_1_xgboost.py --lags 19
      python task3/task3_1_xgboost.py --device cuda          (train on an NVIDIA GPU)
      python task3/task3_1_xgboost.py --depths 2 4 6         (choose which tree depths to try)
GPU and CPU training do not build bit-identical trees, so results can differ
very slightly between the two.
Writes task3_results/task3_1_xgboost_search_lmp_bands_lags<N>.csv and
task3_1_xgboost_test_results_lmp_bands_lags<N>.csv.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import task3_common as C

DEPTHS = (2, 3, 4, 6, 8)  # tree depths tried; the best one on the validation days is kept
PARAMS = dict(
    n_estimators=2000,         # upper limit; early stopping picks the actual number of trees
    early_stopping_rounds=50,  # stop after 50 trees without improvement on the validation days
    learning_rate=0.05,        # how much each new tree is allowed to change the prediction
    subsample=0.8,             # each tree sees a random 80% of the training time steps
    colsample_bytree=0.3,      # ... and a random 30% of the (highly redundant) inputs
    min_child_weight=20,       # each leaf must cover at least ~20 time steps
    tree_method='hist', eval_metric='rmse', random_state=0, n_jobs=-1,
)


class XGBDecoder:
    """One XGBoost model per velocity axis; predict() returns (n, 2) like the ridge decoder."""

    def __init__(self, max_depth, device='cpu'):
        self.max_depth = max_depth
        self.models = [xgb.XGBRegressor(max_depth=max_depth, device=device, **PARAMS) for _ in C.KIN_COLS]

    def fit(self, X, Y, X_val, Y_val):
        for axis, model in enumerate(self.models):
            model.fit(X, Y[:, axis], eval_set=[(X_val, Y_val[:, axis])], verbose=False)
            # Training is the slow part. Predict on the CPU, where our NumPy arrays live; predicting
            # with a GPU model on CPU data works but prints a "mismatched devices" warning.
            model.set_params(device='cpu')
        return self

    @property
    def n_trees(self):
        return [model.best_iteration + 1 for model in self.models]

    def predict(self, X):
        return np.column_stack([model.predict(X, iteration_range=(0, model.best_iteration + 1))
                                for model in self.models])


def _out_of_memory(error):
    text = str(error)
    return 'out of memory' in text or 'bad_malloc' in text or 'Failed to allocate' in text


def main(n_lags, reference, device, depths=DEPTHS):
    start = time.time()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    tag = C.run_tag(n_lags)
    M, S = C.manifest(), C.splits()

    train_sessions = [s for s in C.sessions('train', 'no_stim') if C.has_cm_velocity(s)]
    val_sessions = C.sessions('val', 'no_stim')
    print(f'XGBoost decoder on {device}. Features: LMP + 7 band powers, history: {n_lags} earlier bins '
          f'({(n_lags + 1) * 50} ms including the current bin), reference: {reference}')
    print(f'Training on {len(train_sessions)} OFF sessions (training days, held-out block excluded); '
          f'early stopping and depth chosen on {len(val_sessions)} validation-day OFF sessions.')

    X, Y = C.stack_arrays(train_sessions, reference, 'train', n_lags)
    val = {s: C.session_arrays(s, reference, 'query', n_lags) for s in val_sessions}
    X_val = np.concatenate([x for x, _ in val.values()])
    Y_val = np.concatenate([y for _, y in val.values()])
    print(f'  {X.shape[0]:,} training bins x {X.shape[1]:,} inputs; {X_val.shape[0]:,} validation bins')

    # ---- 1-2. Fit one decoder per depth; early stopping and depth choice on the validation days
    rows, decoders, devices = [], {}, {}
    for depth in depths:
        t0 = time.time()
        devices[depth] = device
        try:
            decoder = XGBDecoder(depth, device).fit(X, Y, X_val, Y_val)
        except xgb.core.XGBoostError as error:
            if not _out_of_memory(error):
                raise
            decoder = None
            if device != 'cpu':
                # Deep trees on many inputs can need more GPU memory than is available.
                # Retry that depth on the CPU.
                print(f'  depth {depth}: GPU out of memory, retrying on the CPU (slower)')
                devices[depth] = 'cpu'
                try:
                    decoder = XGBDecoder(depth, 'cpu').fit(X, Y, X_val, Y_val)
                except xgb.core.XGBoostError as cpu_error:
                    if not _out_of_memory(cpu_error):
                        raise
            if decoder is None:
                # Not enough RAM either: record the depth as skipped instead of losing the whole run.
                print(f'  depth {depth}: out of memory on the CPU too; skipped')
                rows.append(dict(feature_set=C.FEATURE_SET, n_lags=n_lags, reference=reference,
                                 device='skipped: out of memory', max_depth=depth,
                                 fit_seconds=time.time() - t0))
                continue
        decoders[depth] = decoder
        per_session = {s: C.r2(Yv, decoder.predict(Xv)).mean() for s, (Xv, Yv) in val.items()}
        rows.append(dict(feature_set=C.FEATURE_SET, n_lags=n_lags, reference=reference, device=devices[depth],
                         max_depth=depth,
                         n_trees_vx=decoder.n_trees[0], n_trees_vy=decoder.n_trees[1],
                         train_r2=C.r2(Y, decoder.predict(X)).mean(),
                         val_r2_mean=np.mean(list(per_session.values())),
                         **{f'val_r2_{s}': v for s, v in per_session.items()},
                         fit_seconds=time.time() - t0))
        print(f'  depth {depth}: done in {time.time() - t0:.0f} s')
    search = pd.DataFrame(rows)
    depth = int(search.loc[search.val_r2_mean.idxmax(), 'max_depth'])
    search['chosen'] = search.max_depth == depth
    search.to_csv(os.path.join(C.RESULTS_DIR, f'task3_1_xgboost_search_{tag}.csv'), index=False)

    print('\nFor every tree depth: trees kept by early stopping, R2 on the training data itself, '
          'and validation R2 (mean of the validation OFF sessions, query trials):')
    print(search[['max_depth', 'device', 'n_trees_vx', 'n_trees_vy', 'train_r2', 'val_r2_mean']]
          .to_string(index=False, float_format=lambda v: f'{v:8.4f}'))
    print(f'Chosen: max_depth = {depth}')
    model = decoders[depth]

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
            model='xgboost', device=devices[depth], max_depth=depth,
            n_trees_vx=model.n_trees[0], n_trees_vy=model.n_trees[1],
            reference=reference, feature_set=C.FEATURE_SET, n_lags=n_lags))
    results = pd.DataFrame(rows).sort_values(['condition', 'session'])
    results.to_csv(os.path.join(C.RESULTS_DIR, f'task3_1_xgboost_test_results_{tag}.csv'), index=False)

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

    # ---- Side by side with ridge, if the ridge script has been run with the same settings
    ridge_search = os.path.join(C.RESULTS_DIR, f'task3_1_lambda_search_{tag}.csv')
    ridge_test = os.path.join(C.RESULTS_DIR, f'task3_1_test_results_{tag}.csv')
    if os.path.exists(ridge_search) and os.path.exists(ridge_test):
        rs = pd.read_csv(ridge_search)
        rt = pd.read_csv(ridge_test)
        ridge_val = rs[rs.reference == reference].val_r2_mean.max()
        ridge_off = rt.drop_duplicates('off_reference').r2_off.mean()
        print(f'\nCompared with ridge (same features and lags, {reference}):')
        print(f'  validation R2:   ridge {ridge_val:.3f} | XGBoost {search.val_r2_mean.max():.3f}')
        print(f'  test-day OFF R2: ridge {ridge_off:.3f} | XGBoost {np.mean(off_values):.3f}')
        for condition, group in results.groupby('condition'):
            r = rt[rt.condition == condition]
            print(f'  {condition} ON R2:  ridge {r.r2_on.mean():.3f} | XGBoost {group.r2_on.mean():.3f}')
    print(f'\nDone in {time.time() - start:.0f} s. Results in {os.path.relpath(C.RESULTS_DIR, C.REPO)}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 3.1 with an XGBoost decoder trained on '
                                                 'stimulation-OFF trials.')
    parser.add_argument('--lags', type=int, default=C.N_LAGS,
                        help=f'earlier 50 ms bins of feature history per bin (default: {C.N_LAGS})')
    parser.add_argument('--reference', choices=C.REFERENCES, default='export_none',
                        help='export_none (recording reference, best for ridge) or export_car')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu',
                        help='cpu (default, works everywhere) or cuda (NVIDIA GPU)')
    parser.add_argument('--depths', type=int, nargs='+', default=list(DEPTHS),
                        help=f'tree depths to try (default: {" ".join(map(str, DEPTHS))}); deep trees '
                             'with long histories may not fit in memory')
    args = parser.parse_args()
    main(args.lags, args.reference, args.device, tuple(args.depths))
