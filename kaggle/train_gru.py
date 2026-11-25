"""Task 4: GRU decoder — hyperparameter sweep, then five seeds at the winner.

Runs unchanged on Kaggle:

    LFP_ROOT=/kaggle/input/eecs182-lfp python train_gru.py

The GRU reads the cached spectral features (280 numbers per 50 ms bin), not the raw LFP, so
the dataset must carry `data/features/export_none/`. It sees a 40-bin trial and predicts one
velocity pair per bin. Training is on stimulation-off trials only; every stimulated session
is scored with no adaptation.

Two stages, so the reported model is the one the sweep selected:

  1. sweep   12 settings (hidden 64/128/256 x layers 1/2 x dropout 0/0.2), seed 0, ranked by
             stimulation-off validation R2;
  2. final   five seeds at the winning setting, scored on every stimulated test session and
             on each session's matched no-stimulation recording.

Local smoke test (two settings, two seeds, few epochs):

    SMOKE=1 LFP_ROOT=data python kaggle/train_gru.py
"""
import json
import os
import sys
import time
from itertools import islice

import pandas as pd
import torch

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
OUT = os.environ.get('OUT_DIR') or ('/kaggle/working/task4_gru' if INPUTS
                                    else os.path.join(os.getcwd(), 'task4_gru_run'))
sys.path.insert(0, os.path.join(ROOT, 'task4'))
import utils as U  # noqa: E402

os.makedirs(OUT, exist_ok=True)
print('data root:', ROOT, '| outputs:', OUT, '| device:', U.select_device(), flush=True)

SMOKE = os.environ.get('SMOKE') == '1'
SEEDS = tuple(int(seed) for seed in os.environ.get('SEEDS', '0,1,2,3,4').split(','))
MAX_EPOCHS = int(os.environ.get('EPOCHS', U.MAX_TRAINING_EPOCHS))

GRID = [dict(hidden=hidden, layers=layers, dropout=dropout)
        for hidden in (64, 128, 256)
        for layers in (1, 2)
        for dropout in (0.0, 0.2)]


def main():
    train, validation, test = U.load_gru_datasets(
        U.RAW_VARIANT, sorted(U.sessions_in_split('test', U.STIMULATION_ON)))
    inputs = train['features'].shape[-1]
    if SMOKE:
        test = dict(islice(test.items(), 1))
    print(f'train {train["features"].shape} val {validation["features"].shape} '
          f'test {len(test)} sessions | {inputs} features per bin', flush=True)

    grid = GRID[:2] if SMOKE else GRID
    seeds = SEEDS[:2] if SMOKE else SEEDS
    epochs = 20 if SMOKE else MAX_EPOCHS

    # 1. sweep
    sweep = []
    for setting in grid:
        started = time.time()
        torch.manual_seed(seeds[0])
        model = U.GRURegressor(inputs, **setting)
        _, history = U.train_with_early_stopping(model, train, validation, seeds[0],
                                                 max_epochs=epochs)
        sweep.append(dict(**setting, seed=seeds[0],
                          parameters=sum(p.numel() for p in model.parameters()),
                          best_val_r2=history['best_val_r2'],
                          best_epoch=history['best_epoch'],
                          stopped_epoch=history['stopped_epoch'],
                          seconds=round(time.time() - started, 1)))
        print(f'sweep {setting} -> validation R2 {history["best_val_r2"]:.4f} '
              f'({sweep[-1]["seconds"]} s)', flush=True)

    sweep_table = pd.DataFrame(sweep).sort_values('best_val_r2', ascending=False)
    sweep_table.to_csv(os.path.join(OUT, 'gru_sweep.csv'), index=False)
    best = sweep_table.iloc[0]
    config = dict(hidden=int(best.hidden), layers=int(best.layers), dropout=float(best.dropout))
    print('\nsweep winner:', config, 'validation R2', round(float(best.best_val_r2), 4), flush=True)

    # 2. five seeds at the winner
    models, rows, summaries = [], [], []
    for seed in seeds:
        torch.manual_seed(seed)
        model = U.GRURegressor(inputs, **config)
        _, history = U.train_with_early_stopping(model, train, validation, seed,
                                                 max_epochs=epochs)
        models.append(model)
        summaries.append(dict(seed=seed, **config,
                              parameters=sum(p.numel() for p in model.parameters()),
                              best_val_r2=history['best_val_r2'],
                              best_epoch=history['best_epoch'],
                              stopped_epoch=history['stopped_epoch']))
        rows += U.evaluate_gru(model, seed, test, 'raw')
        print(f'seed {seed}: validation R2 {history["best_val_r2"]:.4f} '
              f'at epoch {history["best_epoch"]}', flush=True)

    rows += [U.score_ensemble(models, session, 'gru_ensemble', 'raw', query, off,
                              seed=U.ENSEMBLE_SEED)
             for session, (query, off) in test.items()]

    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(OUT, 'task4_gru_final.csv'), index=False)
    pd.DataFrame(summaries).to_csv(os.path.join(OUT, 'task4_gru_seeds.csv'), index=False)
    with open(os.path.join(OUT, 'task4_gru_config.json'), 'w') as handle:
        json.dump(config, handle, indent=1)

    torch.save(models[0].state_dict(), os.path.join(OUT, 'gru_seed0.pt'))

    per_seed = scores[scores.seed != U.ENSEMBLE_SEED].groupby(['seed', 'am_freq'])[['r2_ave', 'r2_off_ave']].mean()
    ensemble = scores[scores.seed == U.ENSEMBLE_SEED].groupby('am_freq')[['r2_ave', 'r2_off_ave']].mean()
    print('\nper-seed, by frequency:')
    print(per_seed.round(4).to_string())
    print('\n5-seed ensemble, by frequency:')
    print(ensemble.round(4).to_string())
    print(f'\nwrote {len(scores)} rows to {OUT}')


if __name__ == '__main__':
    main()
