"""Train the GRU at the hyperparameter setting its sweep selected, five seeds, and score it.

The sweep (task4/sweep_gru.py) ranks settings by stimulation-off validation R2; this script
takes the best row and trains the model that is actually reported, so the sweep and the test
table describe the same network. Training uses the same early-stopping protocol as the sweep.

    python task4/train_gru_final.py
    python task4/train_gru_final.py --config hidden=256 --seeds 0 1 2 3 4

Writes task4/results/task4_gru_final.csv (per session, seed and the ensemble) plus the
checkpoints under results/checkpoints/gru_h<hidden>_seed<k>.pt.
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils as U  # noqa: E402

CHECKPOINTS = os.path.join(U.PROJECT_ROOT, 'results', 'checkpoints')


def sweep_winner(sweep_path=None):
    path = sweep_path or os.path.join(U.TASK4_RESULTS, 'gru_sweep.csv')
    table = pd.read_csv(path)
    best = table.sort_values('best_val_r2', ascending=False).iloc[0]
    return dict(hidden=int(best.hidden), layers=int(best.layers), dropout=float(best.dropout)), path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--max-epochs', type=int, default=U.MAX_TRAINING_EPOCHS)
    parser.add_argument('--config', nargs='*', default=None,
                        help='override the sweep winner, e.g. hidden=256 layers=1 dropout=0')
    args = parser.parse_args()

    if args.config:
        config = {key: float(value) if '.' in value else int(value)
                  for key, value in (item.split('=') for item in args.config)}
    else:
        config, sweep_path = sweep_winner()
        print('sweep winner from', os.path.basename(sweep_path), config)
    config = dict(hidden=int(config['hidden']), layers=int(config['layers']),
                  dropout=float(config['dropout']))

    train, validation, test = U.load_gru_datasets(
        U.RAW_VARIANT, sorted(U.sessions_in_split('test', U.STIMULATED_ON)))
    inputs = train['features'].shape[-1]
    os.makedirs(CHECKPOINTS, exist_ok=True)

    models, rows, summaries = [], [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = U.GRURegressor(inputs, **config)
        _, history = U.train_with_early_stopping(model, train, validation, seed,
                                                 max_epochs=args.max_epochs)
        name = f'gru_h{config["hidden"]}_seed{seed}'
        torch.save(model.state_dict(), os.path.join(CHECKPOINTS, f'{name}.pt'))
        models.append(model)
        summaries.append(dict(seed=seed, **config, name=name,
                              parameters=sum(p.numel() for p in model.parameters()),
                              best_val_r2=history['best_val_r2'],
                              best_epoch=history['best_epoch'],
                              stopped_epoch=history['stopped_epoch']))
        rows += U.evaluate_gru(model, seed, test, 'raw')
        print(f'{name}: validation R2 {history["best_val_r2"]:.4f} at epoch '
              f'{history["best_epoch"]}', flush=True)

    rows += [U.score_ensemble(models, s, 'gru_ensemble', 'raw', q, off, seed=U.ENSEMBLE_SEED)
             for s, (q, off) in test.items()]

    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(U.TASK4_RESULTS, 'task4_gru_final.csv'), index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(os.path.join(U.TASK4_RESULTS, 'task4_gru_final_seeds.csv'), index=False)
    with open(os.path.join(U.TASK4_RESULTS, 'task4_gru_final_config.json'), 'w') as handle:
        json.dump(config, handle, indent=1)

    print('\nseed summary:')
    print(summary.round(4).to_string(index=False))
    per_seed = scores[scores.seed != U.ENSEMBLE_SEED].groupby('seed')[['r2_ave', 'r2_off_ave']].mean()
    ensemble = scores[scores.seed == U.ENSEMBLE_SEED][['r2_ave', 'r2_off_ave']].mean()
    print(f'\nper-seed mean ON {per_seed.r2_ave.mean():+.4f} | clean {per_seed.r2_off_ave.mean():.4f}')
    print(f'5-seed ensemble  ON {ensemble.r2_ave:+.4f} | clean {ensemble.r2_off_ave:.4f}')


if __name__ == '__main__':
    main()
