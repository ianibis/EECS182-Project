"""Overfit tests for the in-context decoder (plan 5.3.3).

L1  one fixed training episode: the model must fit it (eval-mode R2 > 0.95).
L2  64 fixed training episodes: the model must fit them (eval-mode R2 > 0.9).
L3  in-context test on real LFP with synthetic labels: every bin's label is A_e @ g, where g is
    the LMP averaged over two channel halves (computable from the LFP) and A_e is a random 2x2
    matrix drawn per episode. Without context A_e is unknown (R2 near 0 at k = 0); with k labelled
    trials it can be inferred, so R2 must rise with k. Real velocity is not used for L3 because
    it is barely decodable from this LFP (5.1.3), which would hide whether in-context learning works.

Run from ICL/:  .venv/bin/python scripts/overfit_tests.py [--level L1 L2 L3]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import icl.episodes as E
from icl import config as C
from icl.encoder import EEGNetEncoder
from icl.features_torch import _windows
from icl.model import ICLDecoder, masked_mse
from icl.shifts import TemplateBank

OUT_DIR = os.path.join(C.RESULTS, 'overfit')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
K_REPORT = (0, 2, 5, 10, 20)


def r2(pred, y, valid):
    """R2 over valid bins, per output dimension, averaged."""
    p, t = pred[valid].float(), y[valid].float()
    return float((1 - ((p - t) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)).mean())


def r2_by_k(pred, y, valid):
    return {k: r2(pred[:, k], y[:, k], valid[:, k]) for k in K_REPORT if k < pred.shape[1]}


def synthetic_labels(x, rng):
    """Labels A_e @ g per episode; g = LMP (last 256 ms) averaged over two channel halves, z-scored."""
    lmp = _windows(x, int(round(C.LMP_WIN_S * C.FS)), C.MODEL_HISTORY_BINS, C.N_TARGET_BINS).mean(-1)  # (B, n, C, 40)
    half = x.shape[2] // 2
    g = torch.stack([lmp[:, :, :half].mean(2), lmp[:, :, half:].mean(2)], dim=-1)                       # (B, n, 40, 2)
    g = (g - g.mean(dim=(0, 1, 2))) / g.std(dim=(0, 1, 2))
    A = torch.as_tensor(rng.normal(size=(x.shape[0], 2, 2)) / np.sqrt(2), dtype=x.dtype, device=x.device)
    return torch.einsum('bij,bntj->bnti', A, g)


def make_model(seed):
    torch.manual_seed(seed)
    return ICLDecoder(EEGNetEncoder()).to(DEVICE)


def train(model, get_batch, steps, lr, log_every=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / 200))
    history = []
    model.train()
    start = time.time()
    for step in range(1, steps + 1):
        batch = get_batch(step)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            pred = model(batch['x'], batch['y'], batch['valid'])
        loss = masked_mse(pred.float(), batch['y'], batch['valid'])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if step % log_every == 0 or step == 1:
            history.append((step, loss.item()))
            print(f'  step {step:5d}  loss {loss.item():.4f}  {time.time() - start:.0f} s', flush=True)
    return history


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    preds, ys, vs = [], [], []
    for batch in batches:
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            preds.append(model(batch['x'], batch['y'], batch['valid']).float())
        ys.append(batch['y'])
        vs.append(batch['valid'])
    model.train()
    return torch.cat(preds), torch.cat(ys), torch.cat(vs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', nargs='+', default=['L1', 'L2', 'L3'])
    parser.add_argument('--l3-steps', type=int, default=6000)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    store = E.TrialStore(E.sessions_where('train', 0), parts=['train'], device=DEVICE)
    bank = TemplateBank.load(('train',))
    sampler = E.EpisodeSampler(store, E.EpisodeConfig(), bank=bank, seed=0)
    results = {}

    if 'L1' in args.level:
        print('L1: one fixed episode')
        batch = sampler.sample(1)
        model = make_model(0)
        history = train(model, lambda step: batch, steps=600, lr=1e-3)
        pred, y, v = evaluate(model, [batch])
        results['L1'] = dict(r2=r2(pred, y, v), history=history, passed=r2(pred, y, v) > 0.95)
        print(f"L1 eval R2 {results['L1']['r2']:.3f}  ->  {'PASS' if results['L1']['passed'] else 'FAIL'}")

    if 'L2' in args.level:
        print('L2: 64 fixed episodes')
        fixed = [sampler.sample(16) for _ in range(4)]
        model = make_model(1)
        history = train(model, lambda step: fixed[step % 4], steps=3000, lr=1e-3, log_every=250)
        pred, y, v = evaluate(model, fixed)
        results['L2'] = dict(r2=r2(pred, y, v), r2_by_k=r2_by_k(pred, y, v), history=history,
                             passed=r2(pred, y, v) > 0.9)
        print(f"L2 eval R2 {results['L2']['r2']:.3f}  ->  {'PASS' if results['L2']['passed'] else 'FAIL'}")

    if 'L3' in args.level:
        print('L3: synthetic per-episode linear labels')
        label_rng = np.random.default_rng(100)

        def l3_batch(step, source=sampler, rng=None):
            batch = source.sample(32, rng=rng)
            batch['y'] = synthetic_labels(batch['x'], label_rng)
            batch['valid'] = torch.ones_like(batch['valid'])
            return batch

        model = make_model(2)
        history = train(model, l3_batch, steps=args.l3_steps, lr=3e-4, log_every=250)
        held_rng = np.random.default_rng(12345)
        held = [l3_batch(0, rng=held_rng) for _ in range(4)]                  # 128 fresh episodes
        pred, y, v = evaluate(model, held)
        by_k = r2_by_k(pred, y, v)
        passed = by_k[0] < 0.2 and by_k[10] > 0.6 and by_k[10] > by_k[2] > by_k[0]
        results['L3'] = dict(r2_by_k=by_k, history=history, passed=passed)
        print('L3 held-out R2 by k: ' + '  '.join(f'k={k}: {r:.3f}' for k, r in by_k.items())
              + f"  ->  {'PASS' if passed else 'FAIL'}")

    with open(os.path.join(OUT_DIR, 'overfit_results.json'), 'w') as file:
        json.dump(results, file, indent=1, default=float)


if __name__ == '__main__':
    main()
