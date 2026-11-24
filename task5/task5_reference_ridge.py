"""Least-squares reference for Task 5: ridge refit on the same k context trials.

Inputs are the Task 3 features (LMP + 7 band powers, 9 lags). For each session, k and
draw from splits.json, ridge is fit on the k context trials and scored on the 30 query
trials. The penalty is chosen per k on the validation days and then reused on the test days.

Run from the repository root (CPU only):
    python task5/task5_reference_ridge.py --split val
    python task5/task5_reference_ridge.py --split test
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task3_common as C
import task5_data as data
from task5_evaluate import n_draws

LAMBDAS = np.logspace(1, 7, 13)
VARIANTS = {'recorded': 'export_none', 'amp10': 'parrm_amp10'}
ADAPT_K = (2, 5, 10, 20)


def session_features(session, input_name):
    """Pool and query arrays, normalized from the calibration pool and capped at ±10."""
    pool_rows = C.trial_rows(session, 'pool')
    features = C.F.load_features(session, C.PATHS, variant=VARIANTS[input_name], normalize=True, trials=pool_rows)
    features = dict(features, X=np.clip(features['X'], -cfg.CLIP_Z, cfg.CLIP_Z))
    arrays = {}
    for part, rows in (('pool', pool_rows), ('query', C.trial_rows(session, 'query'))):
        X, Y, valid, _ = C.F.trial_arrays(features, rows, kin=C.kinematics(session), align=C.ALIGN,
                                          pre=C.PRE, post=C.POST, n_lags=C.N_LAGS, kin_cols=C.KIN_COLS)
        arrays[part] = (X, Y, valid, rows.trial.to_numpy())
    return arrays


def score_session(session, input_name, lambdas_by_k=None):
    """R² for every k, draw and penalty (or only the chosen penalty per k)."""
    arrays = session_features(session, input_name)
    X_pool, Y_pool, valid_pool, pool_trials = arrays['pool']
    X_query, Y_query, valid_query, _ = arrays['query']
    lookup = {int(t): i for i, t in enumerate(pool_trials)}
    rows = []
    for k in ADAPT_K:
        grid = LAMBDAS if lambdas_by_k is None else [lambdas_by_k[k]]
        for draw in range(n_draws(k)):
            chosen = [lookup[int(t)] for t in C.splits()['calibration'][session]['subsets'][str(k)][draw]]
            mask = valid_pool[chosen]
            models = C.ridge_path(X_pool[chosen][mask], Y_pool[chosen][mask], grid)
            for model in models:
                prediction = model.predict(X_query[valid_query])
                r2_vx, r2_vy = C.r2(Y_query[valid_query], prediction)
                rows.append(dict(session=session, group=C.manifest().loc[session, 'condition'], input=input_name,
                                 k=k, draw=draw, lam=model.lam, r2=(r2_vx + r2_vy) / 2,
                                 r2_vx=r2_vx, r2_vy=r2_vy, method='ridge_refit'))
    return rows


def run(split):
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    chosen_path = os.path.join(cfg.RESULTS_DIR, 'ridge_reference_lambdas.json')
    sessions = data.usable_sessions(split, 'stim')

    if split == 'val':
        rows = [row for session in sessions for name in VARIANTS for row in score_session(session, name)]
        table = pd.DataFrame(rows)
        means = table.groupby(['k', 'lam']).r2.mean().reset_index()
        best = means.loc[means.groupby('k').r2.idxmax()]
        lambdas_by_k = {int(row.k): float(row.lam) for row in best.itertuples()}
        with open(chosen_path, 'w') as file:
            json.dump(lambdas_by_k, file, indent=2)
        table = table.merge(best[['k', 'lam']], on=['k', 'lam'])
    else:
        with open(chosen_path) as file:
            lambdas_by_k = {int(k): v for k, v in json.load(file).items()}
        table = pd.DataFrame([row for session in sessions for name in VARIANTS
                              for row in score_session(session, name, lambdas_by_k)])

    table.to_csv(os.path.join(cfg.RESULTS_DIR, f'ridge_reference_{split}.csv'), index=False)
    print('Chosen penalties:', lambdas_by_k)
    print(table.groupby(['input', 'k']).r2.mean().unstack(0).round(4).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    run(parser.parse_args().split)
