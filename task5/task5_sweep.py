"""Task 5.4.1: choose the learning rate, then the warmup, on the validation days.

Stage 1  learning rates at the default warmup (3000 steps each)
Stage 2  other warmups at the best learning rate (3000 steps each)
Stage 3  the best short run is repeated at the full 6000 steps and compared with the
         6000-step diagnostic run, because the ranking at 3000 steps may not hold at full length

Runs one model at a time, and finished runs are reused. The test days are not read.
Writes task5_results/sweep.csv and task5_results/chosen_setting.json.

Run from the repository root:  CUDA_VISIBLE_DEVICES=1 python task5/task5_sweep.py
"""

import json
import os

import pandas as pd

import task5_config as cfg
from task5_train import train

SWEEP_ITERATIONS = 3000     # the 6000-step diagnostic began using context at about 2500 steps
FULL_ITERATIONS = 6000
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 2e-3)   # 2e-3 was added because 1e-3 won at the edge of the first grid
DEFAULT_WARMUP = 300
OTHER_WARMUPS = (100, 1000)
DIAGNOSTIC = dict(name='diag_6000', learning_rate=3e-4, warmup=300, iterations=FULL_ITERATIONS)


def finished_score(name):
    path = os.path.join(cfg.RUNS_DIR, name, 'summary.json')
    if not os.path.exists(path):
        return None
    with open(path) as file:
        return json.load(file)['best_val_r2']


def run(learning_rate, warmup, iterations=SWEEP_ITERATIONS, prefix='sweep'):
    name = f'{prefix}_lr{learning_rate:g}_warmup{warmup}'
    score = finished_score(name)
    if score is None:
        config = cfg.TrainConfig(seed=1, iterations=iterations, learning_rate=learning_rate,
                                 warmup_iterations=warmup, eval_every=250 if iterations < 6000 else 500)
        score = train(name, config)['best_val_r2']
    return dict(name=name, learning_rate=learning_rate, warmup=warmup, iterations=iterations, best_val_r2=score)


def save(rows):
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(cfg.RESULTS_DIR, 'sweep.csv'), index=False)


if __name__ == '__main__':
    rows = []
    for learning_rate in LEARNING_RATES:
        rows.append(run(learning_rate, DEFAULT_WARMUP))
        save(rows)
    best_rate = max(rows, key=lambda row: row['best_val_r2'])['learning_rate']

    for warmup in OTHER_WARMUPS:
        rows.append(run(best_rate, warmup))
        save(rows)
    best_short = max(rows, key=lambda row: row['best_val_r2'])

    # Stage 3: compare the finalists at the length the final models will be trained for.
    full = run(best_short['learning_rate'], best_short['warmup'], FULL_ITERATIONS, prefix='full')
    diagnostic = dict(DIAGNOSTIC, best_val_r2=finished_score(DIAGNOSTIC['name']))
    rows += [full, diagnostic]
    save(rows)

    best = max((full, diagnostic), key=lambda row: row['best_val_r2'])
    with open(os.path.join(cfg.RESULTS_DIR, 'chosen_setting.json'), 'w') as file:
        json.dump(best, file, indent=2)
    print(pd.DataFrame(rows).to_string(index=False))
    print('Chosen:', best)
