"""Diagnostic, run after the test days were scored: does the model read the brain signal in its context?

This uses VALIDATION days only and trains nothing. It reuses the 5 final in-context checkpoints.
The context labels are always the real ON labels. Only the context's brain signal changes:

    real         the context trials' own signal
    zero         all zeros. This is unnatural input, so it is read together with the next two
    rotated      each label is paired with the signal of another context trial from the same session
    off_signal   signal from the same-day stimulation-OFF session, so it contains no artifact

How the result is read, fixed before running. A change "matters" at k = 20 if R²(real) - R²(change)
is at least 0.01 and its 95% interval over sessions excludes 0.

    zero does not matter         the context is used only as a template of the movement
    rotated matters              the model uses the trial-by-trial link between signal and movement
    off_signal matters           the model reads something in the ON context signal that OFF lacks
    zero matters, others do not  the model needs a plausible signal, but not its specific content

Run from the repository root:  CUDA_VISIBLE_DEVICES=1 python task5/task5_diagnose_context.py
"""

import os

import numpy as np
import pandas as pd
import torch

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task3_common as C
import task5_analyze as analysis
import task5_data as data
from task5_evaluate import context_positions, load_held_out, n_draws, session_r2
from task5_run_final import SEEDS, run_name
from task5_score import load_model

SPLIT = 'val'
CONDITIONS = ('real', 'zero', 'rotated', 'off_signal')
K_VALUES = (2, 5, 10, 20)
DRAWS = 3               # the same first draws the other context controls use
MIN_DROP = 0.01


def context_signal(condition, held, off_session, positions, k, draw):
    """The (k, C, S) brain signal shown to the model for this condition."""
    if condition == 'real':
        return held.pool_lfp[positions]
    if condition == 'zero':
        return torch.zeros_like(held.pool_lfp[positions])
    if condition == 'rotated':
        return held.pool_lfp[np.roll(positions, 1)]
    return off_session.pool_lfp[context_positions(off_session, k, draw)]


@torch.no_grad()
def predict(model, held, signal, positions, chunk=10):
    """Predict scaled velocity for all query trials, with real labels and the given context signal."""
    device = held.query_lfp.device
    keep = torch.ones(1, cfg.N_CHANNELS, dtype=torch.bool, device=device)
    velocity, valid = held.pool_velocity[positions], held.pool_valid[positions]
    outputs = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        context_tokens = model.encode(signal[None], keep).float()
        for start in range(0, len(held.query_lfp), chunk):
            query = held.query_lfp[start:start + chunk]
            n = len(query)
            query_tokens = model.encode(query[:, None], keep.expand(n, -1))[:, 0]
            output = model.decode(context_tokens.expand(n, -1, -1, -1), velocity[None].expand(n, -1, -1, -1),
                                  valid[None].expand(n, -1, -1), query_tokens)
            outputs.append(output.float())
    return torch.cat(outputs)


def score_all(device='cuda', runs=None):
    """Score each run. `runs` maps a seed to a run name; the default is the 5 final in-context models."""
    runs = runs or {seed: run_name('in_context', seed) for seed in SEEDS}
    sessions = data.usable_sessions(SPLIT, 'stim')
    rows = []
    for seed, name in runs.items():
        model, scale, _ = load_model(name, device)
        for session in sessions:
            off_session = load_held_out(C.splits()['off_reference'][session], 'recorded', scale, device)
            for input_name in ('recorded', 'amp10'):
                held = load_held_out(session, input_name, scale, device)
                for k in K_VALUES:
                    for draw in range(n_draws(k, DRAWS)):
                        positions = context_positions(held, k, draw)
                        for condition in CONDITIONS:
                            signal = context_signal(condition, held, off_session, positions, k, draw)
                            prediction = (predict(model, held, signal, positions) * scale).cpu().numpy()
                            r2, _, _ = session_r2(prediction, held.query_velocity, held.query_valid)
                            rows.append(dict(seed=seed, session=session, group=C.manifest().loc[session, 'condition'],
                                             input=input_name, k=k, draw=draw, condition=condition, r2=r2))
        print(f'seed {seed} done', flush=True)
    return pd.DataFrame(rows)


def check_real_matches_saved_scores(table):
    """The new prediction code must reproduce the saved scores when nothing is changed."""
    saved = pd.read_csv(os.path.join(cfg.RESULTS_DIR, f'scores_{SPLIT}.csv'), keep_default_na=False)
    saved = saved[(saved.model == 'in_context') & (saved.control == 'none') & (saved.group != 'no_stim')]
    real = table[table.condition == 'real']
    merged = real.merge(saved, on=['seed', 'session', 'input', 'k', 'draw'], suffixes=('', '_saved'))
    if len(merged) != len(real):
        raise RuntimeError(f'only {len(merged)} of {len(real)} rows match the saved scores')
    largest = (merged.r2 - merged.r2_saved).abs().max()
    print(f'Check: largest difference from saved scores is {largest:.6f}', flush=True)
    if largest > 1e-3:
        raise RuntimeError('the diagnostic does not reproduce the saved scores')


def summarize(table):
    """Paired drop in R² for each change, averaged over draws and seeds within each session."""
    per_seed = table.groupby(['session', 'input', 'k', 'condition', 'seed']).r2.mean()
    per_session = per_seed.groupby(['session', 'input', 'k', 'condition']).mean().unstack('condition')
    saved = pd.read_csv(os.path.join(cfg.RESULTS_DIR, f'scores_{SPLIT}.csv'), keep_default_na=False)
    labels = analysis.labels_only_scores(saved).reset_index()

    rows = []
    for (input_name, k), part in per_session.groupby(level=['input', 'k']):
        part = part.droplevel(['input', 'k'])
        labels_only = labels[(labels.input == input_name) & (labels.k == k) & labels.session.isin(part.index)].r2.mean()
        for condition in CONDITIONS[1:]:
            drop = part['real'] - part[condition]
            low, high = analysis.bootstrap_interval(drop.to_numpy())
            day_low, day_high = analysis.bootstrap_interval(drop.to_numpy(), [analysis.day_of(s) for s in drop.index])
            rows.append(dict(input=input_name, k=k, condition=condition, n_sessions=len(drop),
                             r2_real=part['real'].mean(), r2_changed=part[condition].mean(),
                             r2_labels_only=labels_only, drop=drop.mean(), ci_low=low, ci_high=high,
                             day_ci_low=day_low, day_ci_high=day_high, sessions_with_drop=int((drop > 0).sum()),
                             matters=bool(drop.mean() >= MIN_DROP and low > 0)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    scores = score_all()
    check_real_matches_saved_scores(scores)
    summary = summarize(scores)
    scores.to_csv(os.path.join(cfg.RESULTS_DIR, f'diagnostic_context_signal_{SPLIT}.csv'), index=False)
    summary.to_csv(os.path.join(cfg.RESULTS_DIR, f'diagnostic_context_signal_summary_{SPLIT}.csv'), index=False)
    print(summary.round(4).to_string())
    print('DONE', flush=True)
