"""Task 5.5 / 5.6: score the final checkpoints on held-out days and save one tidy table.

Each row is one session, input, k, draw, control, model and seed.
    --split val    uses the validation days
    --split test   uses the test days. Run this only after every choice is final.

Run from the repository root:  CUDA_VISIBLE_DEVICES=1 python task5/task5_score.py --split val
"""

import argparse
import os

import pandas as pd
import torch

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task5_data as data
from task5_baselines import LabelsOnlyBaseline
from task5_evaluate import evaluate
from task5_model import InContextDecoder
from task5_run_final import SEEDS, run_name

CONTROLS = (None, 'shuffle_labels', 'off_context', 'no_query_signal')


def load_model(name, device):
    # These checkpoints are created by task5_train.py in this project.
    saved = torch.load(os.path.join(cfg.RUNS_DIR, name, 'best.pt'), map_location=device, weights_only=False)
    model = InContextDecoder(cfg.ModelConfig(**saved['model_config'])).to(device)
    model.load_state_dict(saved['model'])
    return model.eval(), saved['velocity_scale'].to(device), saved['step']


def main(split, seeds):
    device = 'cuda'
    baseline = LabelsOnlyBaseline()
    sessions = data.usable_sessions(split)
    tables = []
    for seed in seeds:
        for kind in ('in_context', 'blind'):
            model, scale, step = load_model(run_name(kind, seed), device)
            controls = CONTROLS if kind == 'in_context' else (None,)
            table = evaluate(model, sessions, ('recorded', 'amp10'), cfg.K_VALUES, scale, baseline,
                             controls=controls)
            table.insert(0, 'model', kind)
            table.insert(1, 'seed', seed)
            table['checkpoint_step'] = step
            tables.append(table)
            pd.concat(tables).to_csv(os.path.join(cfg.RESULTS_DIR, f'scores_{split}.csv'), index=False)
            print(f'{kind} seed {seed}: {len(table)} rows', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    arguments = parser.parse_args()
    main(arguments.split, arguments.seeds)
