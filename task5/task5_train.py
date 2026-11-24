"""Task 5.3.3 / 5.4: overfit check and training of the in-context decoder.

Examples, from the repository root (always on GPU 1):
    CUDA_VISIBLE_DEVICES=1 python task5/task5_train.py --overfit
    CUDA_VISIBLE_DEVICES=1 python task5/task5_train.py --name pilot --iterations 1000

Model selection uses only the validation days. The test days are never read here.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

import task5_config as cfg
import task5_data as data
from task5_baselines import LabelsOnlyBaseline
from task5_episodes import OffTrialBank, sample_episode
from task5_evaluate import evaluate
from task5_model import InContextDecoder, count_parameters, masked_mse
from task5_shift import TemplateLibrary, sample_shift

VALIDATION_K = (2, 5, 10, 20)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate(step, config):
    """Linear warmup, then cosine decay to zero."""
    if step < config.warmup_iterations:
        return config.learning_rate * (step + 1) / max(1, config.warmup_iterations)
    progress = (step - config.warmup_iterations) / max(1, config.iterations - config.warmup_iterations)
    return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def overfit_check(device, steps=600):
    """Task 5.3.3: the model must be able to memorize one small batch."""
    set_seed(0)
    bank, library = OffTrialBank(device), TemplateLibrary(device)
    generator = torch.Generator(device=device).manual_seed(0)
    episode = sample_episode(bank, library, cfg.ShiftConfig(), 4, 5, generator)
    model = InContextDecoder(cfg.ModelConfig(dropout=0.0, encoder_dropout=0.0)).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(episode)
        loss = masked_mse(prediction.float(), episode.query_velocity, episode.query_valid)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    pd.DataFrame(dict(step=np.arange(steps), loss=losses)).to_csv(
        os.path.join(cfg.RESULTS_DIR, 'overfit_check.csv'), index=False)
    ratio = losses[-1] / losses[0]
    print(f'Overfit check: loss {losses[0]:.4f} -> {losses[-1]:.4f} ({ratio:.1%} of start); '
          f'{"PASS" if ratio < 0.05 else "FAIL"} (needs < 5%)')
    return ratio < 0.05


def validation_score(model, bank, baseline, draws=3):
    """Mean R² over validation-day ON sessions, recorded and 10x, at k = 2, 5, 10 and 20."""
    sessions = data.usable_sessions('val', 'stim')
    table = evaluate(model, sessions, ('recorded', 'amp10'), (0, *VALIDATION_K), bank.velocity_scale,
                     baseline, draws=draws)
    adapted = table[table.k > 0]        # k = 0 is logged but does not count toward model selection
    context_gain = table[table.k == 20].r2.mean() - table[table.k == 0].r2.mean()
    return adapted.r2.mean(), adapted.neural_gain.mean(), context_gain


def train(name, train_config, model_config=None, shift_config=None, device='cuda', n_shifts=None):
    """Train one model. With n_shifts (Task 5.7), training sees only that many distinct shifts."""
    model_config = model_config or cfg.ModelConfig()
    shift_config = shift_config or cfg.ShiftConfig()
    run_dir = os.path.join(cfg.RUNS_DIR, name)
    os.makedirs(run_dir, exist_ok=True)
    set_seed(train_config.seed)

    bank, library = OffTrialBank(device), TemplateLibrary(device)
    baseline = LabelsOnlyBaseline()
    generator = torch.Generator(device=device).manual_seed(train_config.seed)
    shift_pool = None
    if n_shifts is not None:
        pool_generator = torch.Generator(device=device).manual_seed(train_config.seed + 1000)
        shift_pool = sample_shift(library, shift_config, n_shifts, cfg.POOL_SIZE + 1, pool_generator)
    model = InContextDecoder(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate,
                                  weight_decay=train_config.weight_decay)
    print(f'[{name}] {count_parameters(model):,} parameters, {train_config.iterations} iterations', flush=True)

    history, running, best_score, started = [], [], -np.inf, time.time()
    for step in range(1, train_config.iterations + 1):
        model.train()
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step - 1, train_config)
        k = int(torch.randint(train_config.max_context + 1, (1,), generator=generator, device=device))
        with torch.no_grad():
            episode = sample_episode(bank, library, shift_config, train_config.batch_size, k, generator,
                                     shift_pool=shift_pool)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(episode)
        loss = masked_mse(prediction.float(), episode.query_velocity, episode.query_valid)
        if not torch.isfinite(loss):
            raise RuntimeError(f'[{name}] loss became {loss.item()} at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()
        running.append(loss.item())

        if step % train_config.eval_every == 0 or step == train_config.iterations:
            model.eval()
            holdout = []
            with torch.no_grad():
                for index in range(10):
                    held_k = VALIDATION_K[index % len(VALIDATION_K)]
                    batch = sample_episode(bank, library, shift_config, train_config.batch_size, held_k,
                                           generator, use_holdout_query=True, shift_pool=shift_pool)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = model(batch)
                    holdout.append(masked_mse(output.float(), batch.query_velocity, batch.query_valid).item())
            score, gain, context_gain = validation_score(model, bank, baseline)
            row = dict(step=step, train_loss=float(np.mean(running)), holdout_loss=float(np.mean(holdout)),
                       val_r2=float(score), val_neural_gain=float(gain), val_context_gain=float(context_gain),
                       lr=group['lr'],
                       minutes=(time.time() - started) / 60)
            history.append(row)
            running = []
            pd.DataFrame(history).to_csv(os.path.join(run_dir, 'history.csv'), index=False)
            marker = ''
            if score > best_score:
                best_score, marker = score, ' *'
                torch.save(dict(model=model.state_dict(), step=step, val_r2=float(score),
                                velocity_scale=bank.velocity_scale.cpu(), model_config=asdict(model_config), shift_config=asdict(shift_config),
                                train_config=asdict(train_config)), os.path.join(run_dir, 'best.pt'))
            print(f'[{name}] step {step:5d}  train {row["train_loss"]:.4f}  holdout {row["holdout_loss"]:.4f}  '
                  f'val R2 {score:+.4f}  vs labels-only {gain:+.4f}  context gain (k20-k0) {context_gain:+.4f}  '
                  f'{row["minutes"]:.1f} min{marker}', flush=True)

    summary = dict(name=name, best_val_r2=float(best_score), **asdict(train_config))
    with open(os.path.join(run_dir, 'summary.json'), 'w') as file:
        json.dump(summary, file, indent=2)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--overfit', action='store_true', help='run only the one-batch overfit check')
    parser.add_argument('--name', default='pilot')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=cfg.TrainConfig.iterations)
    parser.add_argument('--lr', type=float, default=cfg.TrainConfig.learning_rate)
    parser.add_argument('--warmup', type=int, default=cfg.TrainConfig.warmup_iterations)
    parser.add_argument('--eval-every', type=int, default=cfg.TrainConfig.eval_every)
    parser.add_argument('--blind', action='store_true', help='train the labels-only control (no brain signal)')
    args = parser.parse_args()

    if args.overfit:
        raise SystemExit(0 if overfit_check('cuda') else 1)
    train(args.name, cfg.TrainConfig(seed=args.seed, iterations=args.iterations, learning_rate=args.lr,
                                     warmup_iterations=args.warmup, eval_every=args.eval_every),
          model_config=cfg.ModelConfig(blind=args.blind))
