"""Export the first in-context model's predictions for every query trial, so other decoders can be compared
trial by trial on exactly the same context draws.

What is exported: the 5 seeds of the first in-context model (final_in_context_seed1..5), control 'none' only,
every held-out session, input, k and draw scored in scores_{split}.csv. Nothing is trained, nothing is chosen,
and the scores are not changed. Before anything is saved, the session R² is recomputed from the exported
predictions and must match scores_{split}.csv, so the file is known to hold the predictions behind those scores.

One file per split:  data/task5_predictions/first_model_predictions_{split}.npz  (load with numpy.load).
Every array has one entry per row, and a row is one (session, input, k, draw):

    session          (R,)                  session name
    input            (R,)                  'recorded' (real data) or 'amp10' (recorded + 9x PARRM artifact)
    k                (R,)                  number of context trials: 0, 2, 5, 10 or 20
    draw             (R,)                  index into splits.json["calibration"][session]["subsets"][str(k)]
    context_trials   (R, 20)               trial numbers given as context, in the order given; -1 pads to 20
    query_trials     (R, 30)               trial numbers of the query trials (blocks 3-5), ascending
    query_first_bin  (R, 30)               index of each query trial's first bin on the session's 50 ms grid
    prediction       (R, 5, 30, 40, 2)     predicted (vx, vy) in cm/s; axis 1 is the seed, in the order of `seeds`
    velocity         (R, 30, 40, 2)        recorded (vx, vy) in cm/s; NaN where not valid
    valid            (R, 30, 40)           the bins that are scored
    seeds            (5,)                  model seeds

Bins are 50 ms, from 0.5 s before the go cue to 1.5 s after (40 bins). The reported score of a row is, for each
seed: pool the valid bins of the 30 query trials, compute R² for vx and for vy, average the two. The reported
numbers then average over draws, then over seeds, then over sessions (R² is averaged, not the predictions).

Run from the repository root, on one free GPU:
    CUDA_VISIBLE_DEVICES=0 python task5/task5_export_predictions.py --split val
    CUDA_VISIBLE_DEVICES=0 python task5/task5_export_predictions.py --split test
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task3_common as C
import task5_data as data
from task5_evaluate import context_positions, load_held_out, n_draws, predict_queries, session_r2
from task5_run_final import SEEDS, run_name
from task5_score import load_model

OUTPUT_DIR = os.path.join(cfg.REPO, 'data', 'task5_predictions')
N_QUERY = 30
MAX_SCORE_DIFFERENCE = 1e-4


def inputs_of(session):
    """The amplified-artifact input exists only for stimulation sessions."""
    is_stim = int(C.manifest().loc[session, 'stim']) == 1
    return ('recorded', 'amp10') if is_stim else ('recorded',)


def row_keys(sessions):
    """Every (session, input, k, draw) scored with control 'none', in a fixed order."""
    return [(session, input_name, k, draw)
            for session in sessions
            for input_name in inputs_of(session)
            for k in cfg.K_VALUES
            for draw in range(n_draws(k))]


def context_trials(session, k, draw):
    trials = C.splits()['calibration'][session]['subsets'][str(k)][draw] if k else []
    return np.array(list(trials) + [-1] * (cfg.POOL_SIZE - len(trials)), dtype=np.int64)


def predict_all(sessions, device):
    """Predictions in cm/s for every row and seed, and the query trials' recorded velocity."""
    keys = row_keys(sessions)
    prediction = np.full((len(keys), len(SEEDS), N_QUERY, cfg.N_BINS, 2), np.nan, np.float32)
    truth = {}
    for seed_index, seed in enumerate(SEEDS):
        model, scale, _ = load_model(run_name('in_context', seed), device)
        held = None
        for row, (session, input_name, k, draw) in enumerate(keys):
            if held is None or (held.session, held.condition) != (session, input_name):
                held = load_held_out(session, input_name, scale, device)
                truth[session] = (held.query_velocity, held.query_valid)
            scaled = predict_queries(model, held, context_positions(held, k, draw))
            prediction[row, seed_index] = (scaled * scale).cpu().numpy()
        print(f'seed {seed} done', flush=True)
        del model
    return keys, prediction, truth


def query_layout(session):
    """Trial numbers and first bins of the query trials, in the order the model scores them."""
    arrays = data.load_session_arrays(session)
    in_query = data.part_mask(session, arrays['trial'], 'query')
    return arrays['trial'][in_query], arrays['first_bin'][in_query]


def check_against_saved_scores(keys, prediction, truth, split):
    """The exported predictions must give back the R² already saved in scores_{split}.csv."""
    saved = pd.read_csv(os.path.join(cfg.RESULTS_DIR, f'scores_{split}.csv'), keep_default_na=False)
    saved = saved[(saved.model == 'in_context') & (saved.control == 'none')]
    saved = saved.set_index(['seed', 'session', 'input', 'k', 'draw']).sort_index()
    if len(saved) != len(keys) * len(SEEDS):
        raise RuntimeError(f'{len(saved)} saved scores but {len(keys) * len(SEEDS)} exported predictions')

    largest = 0.0
    for row, (session, input_name, k, draw) in enumerate(keys):
        velocity, valid = truth[session]
        for seed_index, seed in enumerate(SEEDS):
            r2, r2_vx, r2_vy = session_r2(prediction[row, seed_index], velocity, valid)
            reference = saved.loc[(seed, session, input_name, k, draw)]
            largest = max(largest, abs(r2 - reference.r2), abs(r2_vx - reference.r2_vx),
                          abs(r2_vy - reference.r2_vy))
    print(f'Check: largest difference from scores_{split}.csv is {largest:.2e}', flush=True)
    if largest > MAX_SCORE_DIFFERENCE:
        raise RuntimeError('the exported predictions do not reproduce the saved scores')


def save(keys, prediction, truth, split):
    sessions = [key[0] for key in keys]
    layouts = {session: query_layout(session) for session in dict.fromkeys(sessions)}
    for session, (trials, _) in layouts.items():
        if trials.tolist() != C.splits()['calibration'][session]['query']:
            raise RuntimeError(f'{session}: query trials are not in the order of splits.json')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f'first_model_predictions_{split}.npz')
    np.savez_compressed(
        path,
        session=np.array(sessions),
        input=np.array([key[1] for key in keys]),
        k=np.array([key[2] for key in keys], dtype=np.int64),
        draw=np.array([key[3] for key in keys], dtype=np.int64),
        context_trials=np.stack([context_trials(session, k, draw) for session, _, k, draw in keys]),
        query_trials=np.stack([layouts[session][0] for session in sessions]).astype(np.int64),
        query_first_bin=np.stack([layouts[session][1] for session in sessions]).astype(np.int64),
        prediction=prediction,
        velocity=np.stack([truth[session][0] for session in sessions]).astype(np.float32),
        valid=np.stack([truth[session][1] for session in sessions]),
        seeds=np.array(SEEDS, dtype=np.int64),
    )
    print(f'Saved {len(keys)} rows to {path} ({os.path.getsize(path) / 1e6:.0f} MB)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    split_name = parser.parse_args().split

    with torch.no_grad():
        all_keys, all_predictions, query_truth = predict_all(data.usable_sessions(split_name), 'cuda')
    for name, (velocity_array, _) in query_truth.items():
        if velocity_array.shape != (N_QUERY, cfg.N_BINS, 2):
            raise RuntimeError(f'{name}: expected {N_QUERY} query trials, got {velocity_array.shape}')
    check_against_saved_scores(all_keys, all_predictions, query_truth, split_name)
    save(all_keys, all_predictions, query_truth, split_name)
    print('DONE', flush=True)
