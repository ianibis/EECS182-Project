import matplotlib
matplotlib.use('Agg')


# ---- notebook cell ----
import os
import sys

HERE = os.path.abspath(os.getcwd())
TASK4_DIR = HERE if os.path.basename(HERE) == 'task4' else os.path.join(HERE, 'task4')
TASK3_DIR = os.path.join(os.path.dirname(TASK4_DIR), 'task3')
for path in (TASK4_DIR, TASK3_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import utils
import task3_common  # ridge solver, penalty grid, clipping and PARRM map shared with task 3

print('task 4 helpers:', utils.TASK4_DIR)
print('task 3 helpers:', task3_common.TASK_DIR)
print('draws file    :', utils.DRAWS_PATH)



# ---- notebook cell ----
import numpy as np
import pandas as pd

LAG_COUNT = utils.LAG_COUNT
FEATURE_VARIANTS = ((utils.RAW_VARIANT, 'raw'), (utils.PARRM_VARIANT, 'parrm'))
DRAWS = utils.load_json(utils.DRAWS_PATH)

def clip(windows):
    X, y = windows
    return np.clip(X, -task3_common.CLIP_Z, task3_common.CLIP_Z), y

print('draws:', len(DRAWS['sessions']), 'sessions | ks', DRAWS['ks'],
      '| query definition', DRAWS.get('query_definition'))



# ---- notebook cell ----
X_train, y_train = clip(utils.concat_session_windows(utils.OFF_TRAINING_SESSIONS, 'train', utils.RAW_VARIANT, LAG_COUNT))

validation = [clip(utils.windows_from_valid_bins(utils.load_split_dataset(s, 'query', utils.RAW_VARIANT, LAG_COUNT)))
              for s in utils.sessions_in_split('val', utils.STIMULATED_ON)]

fitter = task3_common.RidgeFitter(X_train, y_train)

scores = {float(lam): np.mean([utils.mean_r2(y, fitter.fit(lam).predict(X)) for X, y in validation])
          for lam in task3_common.LAMBDAS}

alpha = max(scores, key=scores.get)
print('ridge alpha:', alpha, '(from task 3 grid of', len(task3_common.LAMBDAS), ')')
pd.Series(scores).round(4)



# ---- notebook cell ----
def refit_draw(session, k, index, draw, variant, feature_label, alpha):
    X_cal, y_cal = clip(utils.drawn_trial_windows(session, draw, variant, LAG_COUNT))
    X_query, y_query = clip(utils.query_trial_windows(session, draw, variant, LAG_COUNT))
    off = clip(utils.windows_from_valid_bins(utils.load_same_day_off_dataset(session, variant, LAG_COUNT)))
    model = task3_common.ridge_path(X_cal, y_cal, [alpha])[0]  # task 3 solver, one lambda
    return utils.score_window_decoder(model, session, 'ridge', feature_label,
                           (X_query, y_query), off, k=k, draw=index)

rows = []
for session in DRAWS['sessions']:
    for k, index, draw in utils.iter_draws(DRAWS, session):
        for variant, feature_label in FEATURE_VARIANTS:
            rows.append(refit_draw(session, k, index, draw, variant, feature_label, alpha))
    print('scored', session)

results = pd.DataFrame(rows)
results.to_csv(os.path.join(utils.TASK4_RESULTS, 'task4_3_parrm_ridge.csv'), index=False)
print(results.shape)
results.groupby(['feature_variant', 'k'])[['r2_ave', 'r2_off_ave']].mean().round(3)



# ---- notebook cell ----
summary = []
for (variant, k), part in results.groupby(['feature_variant', 'k']):
    groups = [group.r2_ave.to_numpy() for _, group in part.groupby('session')]
    off_groups = [group.r2_off_ave.to_numpy() for _, group in part.groupby('session')]
    low, high = utils.two_level_bootstrap_confidence_interval(groups)
    summary.append(dict(feature_variant=variant, k=k, sessions=len(groups),
                        draws_per_session=len(groups[0]),
                        r2=float(np.mean([group.mean() for group in groups])),
                        ci_low=low, ci_high=high,
                        r2_off=float(np.mean([group.mean() for group in off_groups]))))
summary = pd.DataFrame(summary)
summary.to_csv(os.path.join(utils.TASK4_RESULTS, 'task4_3_summary.csv'), index=False)
summary.round(4)
