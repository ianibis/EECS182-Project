"""Task 5.4.2: train the final models, one at a time, with the setting chosen on validation days.

Trains 5 seeds of the in-context model and 5 seeds of the blind (labels-only) control.
A finished run is skipped, so this script can safely be restarted.

Run from the repository root:  CUDA_VISIBLE_DEVICES=1 python task5/task5_run_final.py
"""

import json
import os

import task5_config as cfg
from task5_train import train

SEEDS = (1, 2, 3, 4, 5)
FINAL_ITERATIONS = 6000


def run_name(kind, seed):
    return f'final_{kind}_seed{seed}'


if __name__ == '__main__':
    with open(os.path.join(cfg.RESULTS_DIR, 'chosen_setting.json')) as file:
        chosen = json.load(file)
    print('Chosen setting:', chosen, flush=True)

    for seed in SEEDS:
        for kind, blind in (('in_context', False), ('blind', True)):
            name = run_name(kind, seed)
            if os.path.exists(os.path.join(cfg.RUNS_DIR, name, 'summary.json')):
                print(f'{name}: already finished', flush=True)
                continue
            train(name, cfg.TrainConfig(seed=seed, iterations=FINAL_ITERATIONS,
                                        learning_rate=chosen['learning_rate'],
                                        warmup_iterations=chosen['warmup'], eval_every=500),
                  model_config=cfg.ModelConfig(blind=blind))
    print('ALL FINAL RUNS DONE', flush=True)
