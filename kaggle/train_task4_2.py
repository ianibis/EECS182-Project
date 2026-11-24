"""Task 4.2: train CausalEEGNet on 1 kHz LFP for five seeds and score it.

Reads the repository layout (dataset root -> task4/utils.py) and writes checkpoints,
training logs and the score table to OUT_DIR. Runs unchanged on Kaggle:

    LFP_ROOT=/kaggle/input/eecs182-lfp python train_task4_2.py

Local smoke test (one epoch, one seed, one test session):

    SMOKE=1 python kaggle/train_task4_2.py
"""
import os
import sys
from itertools import islice

import pandas as pd
import torch
from torch import nn
from braindecode.models import EEGNet

HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
INPUTS = ['/kaggle/input'] if os.path.isdir('/kaggle/input') else []


def listing(folder, depth=3):
    out = []
    for root, dirs, files in os.walk(folder):
        level = root[len(folder):].count(os.sep)
        out.append('  ' * level + os.path.basename(root) + '/ ' + ' '.join(sorted(files)[:3]))
        if level >= depth:
            dirs[:] = []
    return out


def find_root():
    """Directory holding task4/utils.py: the repository, or the attached Kaggle dataset."""
    for folder in [os.environ.get('LFP_ROOT')] + INPUTS + [HERE, os.path.dirname(HERE)]:
        if folder and os.path.isfile(os.path.join(folder, 'task4', 'utils.py')):
            return folder
    for folder in INPUTS:  # Kaggle nests datasets below /kaggle/input
        for root, dirs, files in os.walk(folder):
            if os.path.basename(root) == 'task4' and 'utils.py' in files:
                return os.path.dirname(root)
            if root[len(folder):].count(os.sep) >= 6:
                dirs[:] = []
    raise SystemExit('no task4/utils.py; tree:\n' + '\n'.join(sum(
        (listing(folder) for folder in INPUTS or [HERE]), [])))


ROOT = find_root()
OUT = os.environ.get('OUT_DIR') or ('/kaggle/working/task4_2' if INPUTS
                                   else os.path.join(os.getcwd(), 'task4_2_run'))
sys.path.insert(0, os.path.join(ROOT, 'task4'))
import utils as U

print('data root:', ROOT, '| outputs:', OUT, flush=True)

U.CHECKPOINT_DIR = os.path.join(OUT, 'checkpoints')
U.TRAINING_LOG_DIR = os.path.join(OUT, 'training_logs')
for folder in (U.CHECKPOINT_DIR, U.TRAINING_LOG_DIR):
    os.makedirs(folder, exist_ok=True)

# Winning row of task4/results/eegnet_sweep.csv; see the report for why it is not a hard win.
SETTINGS = dict(temporal_filters=8, depth_multiplier=2, kernel_samples=500,
                separable_samples=62, dropout=0.25)
SEEDS = tuple(int(seed) for seed in os.environ.get('SEEDS', '0,1,2,3,4').split(','))
EPOCHS = int(os.environ.get('EPOCHS', U.MAX_TRAINING_EPOCHS))
SMOKE = os.environ.get('SMOKE') == '1'


class CausalEEGNet(nn.Module):
    """braindecode EEGNet, left-only temporal padding, one velocity pair per 50 ms bin."""

    def __init__(self, channels=35, samples=2000, outputs=2, pool_first=2, pool_second=25):
        super().__init__()
        net = EEGNet(n_chans=channels, n_outputs=outputs, n_times=samples, final_conv_length=1,
                     F1=SETTINGS['temporal_filters'], D=SETTINGS['depth_multiplier'],
                     kernel_length=SETTINGS['kernel_samples'],
                     depthwise_kernel_length=SETTINGS['separable_samples'],
                     pool1_kernel_size=pool_first, pool2_kernel_size=pool_second,
                     drop_prob=SETTINGS['dropout'])
        layers = []
        for module in net.children():
            if isinstance(module, nn.Conv2d) and module.padding[1] > 0:
                layers.append(nn.ConstantPad2d((module.kernel_size[1] - 1, 0, 0, 0), 0.0))
                module.padding = (module.padding[0], 0)
            layers.append(module)
        self.layers = nn.Sequential(*layers)

    def forward(self, lfp):
        return self.layers(lfp).transpose(1, 2)


def train_run(dataset, method, validation, test):
    """Train one model per seed and score every test session, plus the seed ensemble."""
    models, rows = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)  # weights, trial order and dropout masks all follow from this
        model, history = U.train_with_early_stopping(CausalEEGNet(), dataset, validation, seed,
                                                     max_epochs=EPOCHS)
        U.save_checkpoint(model, f'task4_2_{method}_seed{seed}')
        U.save_training_log(history, method, seed, U.TRAINING_LOG_DIR)
        models.append(model)
        rows += [U.score_trial_model(model, session, method, 'raw', seed, query, off)
                 for session, (query, off) in test.items()]
        print(f'{method} seed {seed} val R2 {history["best_val_r2"]:+.4f} '
              f'epoch {history["best_epoch"]} of {history["stopped_epoch"]}', flush=True)
    rows += [U.score_ensemble(models, session, f'{method}_ensemble', 'raw', query, off)
             for session, (query, off) in test.items()]
    return rows


def main():
    train = U.off_training_datasets()
    validation = U.off_validation_datasets()
    test = U.stimulated_test_datasets()
    if SMOKE:
        test = dict(islice(test.items(), 1))
    print(f'train {train["features"].shape} val {validation["features"].shape} '
          f'test {len(test)} sessions seeds {SEEDS}', flush=True)

    rows = train_run(train, U.EEGNET_METHOD, validation, test)
    rows += train_run(U.add_stimulation_tones(train, seed=SEEDS[0]), U.EEGNET_TONE_METHOD,
                      validation, test)
    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(OUT, 'task4_2_eegnet_off.csv'), index=False)
    summary = (scores.groupby(['method', 'session'])[['r2_off_ave', 'r2_ave']].mean()
               .groupby('method').mean().round(4))
    summary.to_csv(os.path.join(OUT, 'task4_2_summary.csv'))
    print(f'wrote {len(scores)} rows to {OUT}')
    print(summary.to_string())


if __name__ == '__main__':
    main()
