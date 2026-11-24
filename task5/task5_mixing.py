"""Follow-up to the context diagnostic: train the model so that it needs the context's brain signal.

The diagnostic showed that the model decodes from the query's own signal and uses the context only as a
template of the movement. A fixed encoder can do that because every session looks alike to it. Here each
training episode passes all of its trials through one random channel-mixing matrix:

    mixed LFP = M * normalized LFP,    M = (1 - a) * I + a * Q

Q is a random orthogonal matrix and a is drawn uniformly from 0 to alpha_max, so unmixed data stay inside
the training distribution. With a new M in every episode, strongly mixed trials can only be decoded by
working out the mapping from the context.

Development uses VALIDATION days only. The test days are not read by this script.

Pilot decision rule, fixed before running. A setting passes only if all three hold at k = 20:

    A. Mechanism    on validation data with new mixing at alpha_max, zeroing the context signal lowers R²
                    by at least 0.01 and the 95% session interval excludes 0, for both inputs
    B. No collapse  best validation R² on real data is within 0.02 of the seed-1 model without mixing
    C. Real data    on unmixed validation data, zeroing the context signal lowers R² by at least 0.01
                    and the 95% session interval excludes 0, for both inputs

The passing setting with the highest validation R² goes on to 5 seeds. If none passes, no seeds are trained.

Run from the repository root, on one free GPU (replace 0 with its index):
    CUDA_VISIBLE_DEVICES=0 python task5/task5_mixing.py --pilot
    CUDA_VISIBLE_DEVICES=0 python task5/task5_mixing.py --final
"""

import argparse
import json
import os
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import torch

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task5_analyze as analysis
import task5_data as data
import task5_diagnose_context as diagnose
from task5_baselines import LabelsOnlyBaseline
from task5_episodes import OffTrialBank, sample_episode
from task5_evaluate import context_positions, load_held_out, session_r2
from task5_model import InContextDecoder, count_parameters, masked_mse
from task5_run_final import FINAL_ITERATIONS, SEEDS
from task5_score import load_model
from task5_shift import TemplateLibrary
from task5_train import VALIDATION_K, learning_rate, set_seed, validation_score

ALPHA_MAX_VALUES = (0.25, 0.5, 1.0)
PILOT_SEED = 1
CHECK_K = 20
MIXING_DRAWS = 3
MIN_DROP = 0.01
MAX_R2_LOSS = 0.02
INPUTS = ('recorded', 'amp10')
CHOSEN_PATH = os.path.join(cfg.RESULTS_DIR, 'mixing_chosen.json')


def run_name(alpha_max, seed):
    return f'mixing_a{alpha_max:g}_seed{seed}'


def random_mixing(strength, generator):
    """One mixing matrix per entry of `strength`: (B, C, C)."""
    device = strength.device
    n_episodes, n_channels = len(strength), cfg.N_CHANNELS
    gaussian = torch.randn((n_episodes, n_channels, n_channels), generator=generator, device=device)
    q, r = torch.linalg.qr(gaussian)
    q = q * torch.sign(torch.diagonal(r, dim1=1, dim2=2))[:, None, :]   # makes Q uniformly distributed
    identity = torch.eye(n_channels, device=device)[None]
    a = strength[:, None, None]
    return (1 - a) * identity + a * q


def apply_mixing(lfp, mixing, channel_keep):
    """Mix normalized LFP of shape (B, T, C, S). Dropped channels stay at zero."""
    mixed = torch.einsum('bcd,btds->btcs', mixing, lfp)
    return (mixed * channel_keep[:, None, :, None]).clamp(-cfg.CLIP_Z, cfg.CLIP_Z)


def mix_episode(episode, alpha_max, generator):
    """Apply one new mixing matrix to all trials of each episode."""
    n_episodes = len(episode.query_lfp)
    device = episode.query_lfp.device
    strength = alpha_max * torch.rand(n_episodes, generator=generator, device=device)
    mixing = random_mixing(strength, generator)
    keep = episode.channel_keep
    context = episode.context_lfp
    if context.shape[1] > 0:
        context = apply_mixing(context, mixing, keep)
    query = apply_mixing(episode.query_lfp[:, None], mixing, keep)[:, 0]
    return replace(episode, context_lfp=context, query_lfp=query)


def train_with_mixing(name, alpha_max, seed, iterations=FINAL_ITERATIONS, device='cuda'):
    """The same training as task5_train.train, with channel mixing added to every episode."""
    run_dir = os.path.join(cfg.RUNS_DIR, name)
    if os.path.exists(os.path.join(run_dir, 'summary.json')):
        print(f'{name}: already finished', flush=True)
        return
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(cfg.RESULTS_DIR, 'chosen_setting.json')) as file:
        chosen = json.load(file)
    config = cfg.TrainConfig(seed=seed, iterations=iterations, learning_rate=chosen['learning_rate'],
                             warmup_iterations=chosen['warmup'])
    model_config, shift_config = cfg.ModelConfig(), cfg.ShiftConfig()
    set_seed(seed)

    bank, library = OffTrialBank(device), TemplateLibrary(device)
    baseline = LabelsOnlyBaseline()
    generator = torch.Generator(device=device).manual_seed(seed)
    model = InContextDecoder(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    print(f'[{name}] {count_parameters(model):,} parameters, {iterations} iterations', flush=True)

    history, running, best_score, started = [], [], -np.inf, time.time()
    for step in range(1, iterations + 1):
        model.train()
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step - 1, config)
        k = int(torch.randint(config.max_context + 1, (1,), generator=generator, device=device))
        with torch.no_grad():
            episode = sample_episode(bank, library, shift_config, config.batch_size, k, generator)
            episode = mix_episode(episode, alpha_max, generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(episode)
        loss = masked_mse(prediction.float(), episode.query_velocity, episode.query_valid)
        if not torch.isfinite(loss):
            raise RuntimeError(f'[{name}] loss became {loss.item()} at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        running.append(loss.item())

        if step % config.eval_every == 0 or step == iterations:
            model.eval()
            holdout = []
            with torch.no_grad():
                for index in range(10):
                    held_k = VALIDATION_K[index % len(VALIDATION_K)]
                    batch = sample_episode(bank, library, shift_config, config.batch_size, held_k, generator,
                                           use_holdout_query=True)
                    batch = mix_episode(batch, alpha_max, generator)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = model(batch)
                    holdout.append(masked_mse(output.float(), batch.query_velocity, batch.query_valid).item())
            score, gain, context_gain = validation_score(model, bank, baseline)
            row = dict(step=step, train_loss=float(np.mean(running)), holdout_loss=float(np.mean(holdout)),
                       val_r2=float(score), val_neural_gain=float(gain), val_context_gain=float(context_gain),
                       lr=group['lr'], minutes=(time.time() - started) / 60)
            history.append(row)
            running = []
            pd.DataFrame(history).to_csv(os.path.join(run_dir, 'history.csv'), index=False)
            marker = ''
            if score > best_score:
                best_score, marker = score, ' *'
                torch.save(dict(model=model.state_dict(), step=step, val_r2=float(score),
                                velocity_scale=bank.velocity_scale.cpu(), model_config=asdict(model_config),
                                shift_config=asdict(shift_config), train_config=asdict(config),
                                alpha_max=alpha_max), os.path.join(run_dir, 'best.pt'))
            print(f'[{name}] step {step:5d}  train {row["train_loss"]:.4f}  holdout {row["holdout_loss"]:.4f}  '
                  f'val R2 {score:+.4f}  vs labels-only {gain:+.4f}  context gain {context_gain:+.4f}  '
                  f'{row["minutes"]:.1f} min{marker}', flush=True)

    with open(os.path.join(run_dir, 'summary.json'), 'w') as file:
        json.dump(dict(name=name, alpha_max=alpha_max, best_val_r2=float(best_score), **asdict(config)), file, indent=2)


def mixed_context_check(name, alpha, device='cuda'):
    """Gate A: R² per session and input on newly mixed validation data, with real or zeroed context signal."""
    model, scale, _ = load_model(name, device)
    keep = torch.ones(1, cfg.N_CHANNELS, dtype=torch.bool, device=device)
    rows = []
    for session in data.usable_sessions('val', 'stim'):
        for input_name in INPUTS:
            held = load_held_out(session, input_name, scale, device)
            positions = context_positions(held, CHECK_K, 0)
            for draw in range(MIXING_DRAWS):
                generator = torch.Generator(device=device).manual_seed(draw)
                mixing = random_mixing(torch.full((1,), float(alpha), device=device), generator)
                mixed = replace(held, pool_lfp=apply_mixing(held.pool_lfp[None], mixing, keep)[0],
                                query_lfp=apply_mixing(held.query_lfp[None], mixing, keep)[0])
                signals = dict(real=mixed.pool_lfp[positions], zero=torch.zeros_like(mixed.pool_lfp[positions]))
                for condition, signal in signals.items():
                    prediction = (diagnose.predict(model, mixed, signal, positions) * scale).cpu().numpy()
                    r2 = session_r2(prediction, held.query_velocity, held.query_valid)[0]
                    rows.append(dict(session=session, input=input_name, draw=draw, condition=condition, r2=r2))
    table = pd.DataFrame(rows)
    return table.groupby(['session', 'input', 'condition']).r2.mean().unstack('condition')


def evaluate_setting(alpha, reference_r2):
    """Apply the three gates to one trained pilot model."""
    name = run_name(alpha, PILOT_SEED)
    with open(os.path.join(cfg.RUNS_DIR, name, 'summary.json')) as file:
        best_r2 = json.load(file)['best_val_r2']
    mixed = mixed_context_check(name, alpha)
    real = diagnose.summarize(diagnose.score_all(runs={PILOT_SEED: name}))
    real = real[(real.k == CHECK_K) & (real.condition == 'zero')].set_index('input')

    row = dict(alpha_max=alpha, run=name, best_val_r2=best_r2, reference_val_r2=reference_r2)
    gate_a, gate_c = True, True
    for input_name in INPUTS:
        part = mixed.xs(input_name, level='input')
        drop = part['real'] - part['zero']
        low, high = analysis.bootstrap_interval(drop.to_numpy())
        passes_a = drop.mean() >= MIN_DROP and low > 0
        passes_c = real.loc[input_name, 'drop'] >= MIN_DROP and real.loc[input_name, 'ci_low'] > 0
        gate_a, gate_c = gate_a and passes_a, gate_c and passes_c
        row.update({f'mixed_r2_{input_name}': part['real'].mean(), f'mixed_drop_{input_name}': drop.mean(),
                    f'mixed_ci_low_{input_name}': low, f'mixed_ci_high_{input_name}': high,
                    f'real_drop_{input_name}': real.loc[input_name, 'drop'],
                    f'real_ci_low_{input_name}': real.loc[input_name, 'ci_low'],
                    f'real_ci_high_{input_name}': real.loc[input_name, 'ci_high']})
    gate_b = best_r2 >= reference_r2 - MAX_R2_LOSS
    row.update(gate_a_mechanism=bool(gate_a), gate_b_no_collapse=bool(gate_b), gate_c_real_data=bool(gate_c),
               passed=bool(gate_a and gate_b and gate_c))
    return row


def run_pilot():
    for alpha in ALPHA_MAX_VALUES:
        train_with_mixing(run_name(alpha, PILOT_SEED), alpha, PILOT_SEED)
    with open(os.path.join(cfg.RUNS_DIR, f'final_in_context_seed{PILOT_SEED}', 'summary.json')) as file:
        reference_r2 = json.load(file)['best_val_r2']
    table = pd.DataFrame([evaluate_setting(alpha, reference_r2) for alpha in ALPHA_MAX_VALUES])
    table.to_csv(os.path.join(cfg.RESULTS_DIR, 'mixing_pilot.csv'), index=False)
    print(table.round(4).T.to_string())

    passing = table[table.passed]
    if passing.empty:
        if os.path.exists(CHOSEN_PATH):
            os.remove(CHOSEN_PATH)
        print('PILOT RESULT: no setting passed, so no further seeds are trained', flush=True)
        return
    best = passing.loc[passing.best_val_r2.idxmax()]
    with open(CHOSEN_PATH, 'w') as file:
        json.dump(dict(alpha_max=float(best.alpha_max), best_val_r2=float(best.best_val_r2)), file, indent=2)
    print(f'PILOT RESULT: alpha_max {best.alpha_max:g} passed and was chosen', flush=True)


def run_final():
    with open(CHOSEN_PATH) as file:
        alpha = json.load(file)['alpha_max']
    for seed in SEEDS:
        train_with_mixing(run_name(alpha, seed), alpha, seed)
    print('ALL MIXING SEEDS DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--final', action='store_true')
    arguments = parser.parse_args()
    if arguments.pilot:
        run_pilot()
    if arguments.final:
        run_final()
