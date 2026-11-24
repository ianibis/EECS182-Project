"""Hyperparameter sweep for the task 4 GRU decoder.

Mirrors task4/utils.sweep_eegnet_settings so the two models are compared under the same
protocol: one seed, early stopping on the stimulation-off validation blocks, ranked by
validation R2. Only model-shape knobs are swept; the optimiser, loss and data are fixed.

    python task4/sweep_gru.py
    python task4/sweep_gru.py --seeds 0 1        # repeat each setting over more seeds

Writes task4/results/gru_sweep.csv, sorted by validation R2.
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils as U  # noqa: E402

GRID = [
    dict(hidden=hidden, layers=layers, dropout=dropout)
    for hidden in (64, 128, 256)
    for layers in (1, 2)
    for dropout in (0.0, 0.2)
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--max-epochs', type=int, default=U.MAX_TRAINING_EPOCHS)
    parser.add_argument('--out', default=os.path.join(U.TASK4_RESULTS, 'gru_sweep.csv'))
    args = parser.parse_args()

    train, validation, _ = U.load_gru_datasets(U.RAW_VARIANT, [])
    inputs = train['features'].shape[-1]
    print(f'{len(train["features"])} train trials, {len(validation["features"])} '
          f'validation trials, {inputs} features per bin, {len(GRID)} settings', flush=True)

    rows = []
    for setting in GRID:
        for seed in args.seeds:
            torch.manual_seed(seed)  # the init follows from this, as in the notebook
            started = time.time()
            model = U.GRURegressor(inputs, **setting)
            _, history = U.train_with_early_stopping(model, train, validation, seed,
                                                     max_epochs=args.max_epochs)
            rows.append(dict(**setting, seed=seed,
                             parameters=sum(p.numel() for p in model.parameters()),
                             best_val_r2=history['best_val_r2'],
                             best_epoch=history['best_epoch'],
                             stopped_epoch=history['stopped_epoch'],
                             seconds=round(time.time() - started, 1)))
            print(setting, seed, 'validation R2', round(history['best_val_r2'], 4),
                  f'({rows[-1]["seconds"]} s)', flush=True)

    table = pd.DataFrame(rows).sort_values('best_val_r2', ascending=False).reset_index(drop=True)
    table.to_csv(args.out, index=False)
    print(f'\n{len(table)} runs -> {args.out}')
    print(table.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
