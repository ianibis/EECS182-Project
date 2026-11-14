"""Train the in-context decoder (plan 5.4).

Training: episodes from the 13 training no-stim sessions with simulated shifts (training-split
templates and harmonic artifacts); every-prefix MSE over all signal tokens. With train_stim, the
26 training-day stim sessions are added; they already carry a real artifact, so their simulated
shift has no artifact (neural gain drift and channel dropout only).

Validation (R2 by number of context trials k):
  A     held-out block of the training sessions, training-type shifts (k up to 9)
  B     the 3 validation-day no-stim sessions with shifts from validation-session templates,
        context draws from splits.json -- the model-selection metric is mean B R2 at k = 5 and 20
  real  the 6 validation stim sessions, no simulated shift
The label-only baseline ctx_mean_profile is logged alongside B and real.

Run from ICL/:
  .venv/bin/python -m icl.train --name lr3e-4 [--lr 3e-4 --steps 10000 --seed 0 ...]
Outputs runs/<name>/: config.yaml, git_hash.txt, metrics.csv, ckpt_last.pt, ckpt_best.pt
"""
import argparse
import csv
import math
import os
import subprocess
import time
from dataclasses import replace

import numpy as np
import torch
import yaml

import icl.episodes as E
from icl import config as C
from icl.encoder import EEGNetEncoder, FeatureTokenEncoder
from icl.metrics import context_mean_profile, r2_score
from icl.model import ICLDecoder, masked_mse
from icl.shifts import ShiftConfig, TemplateBank, sample_shift

RUNS = os.path.join(C.ROOT, 'runs')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_bool(text):
    if text.lower() in ('1', 'true', 'yes'):
        return True
    if text.lower() in ('0', 'false', 'no'):
        return False
    raise argparse.ArgumentTypeError(f'expected true/false, got {text!r}')


def load_config(argv=None):
    with open(os.path.join(os.path.dirname(__file__), 'configs', 'base.yaml')) as file:
        cfg = yaml.safe_load(file)
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--config', default=None, help='YAML file overriding base.yaml')
    for key, value in cfg.items():
        kind = type(value[0]) if isinstance(value, list) else type(value)
        if kind is bool:
            kind = parse_bool
        parser.add_argument(f'--{key}', type=kind, nargs='+' if isinstance(value, list) else None, default=None)
    args = parser.parse_args(argv)
    if args.config:
        with open(args.config) as file:
            cfg.update(yaml.safe_load(file))
    cfg.update({k: v for k, v in vars(args).items() if v is not None and k not in ('name', 'config')})
    cfg['name'] = args.name
    return cfg


def build_model(cfg):
    if cfg['encoder'] == 'eegnet':
        encoder = EEGNetEncoder(d=cfg['d_model'], dropout=cfg['enc_dropout'])
    else:
        encoder = FeatureTokenEncoder(d=cfg['d_model'])
    return ICLDecoder(encoder, d=cfg['d_model'], n_layer=cfg['n_layer'], n_head=cfg['n_head'], dropout=cfg['dropout'])


def lr_lambda(cfg):
    def factor(step):
        if step < cfg['warmup']:
            return (step + 1) / cfg['warmup']
        progress = (step - cfg['warmup']) / max(1, cfg['steps'] - cfg['warmup'])
        return cfg['min_lr_frac'] + (1 - cfg['min_lr_frac']) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return factor


@torch.no_grad()
def predict(model, batch, chunk=16):
    out = []
    for i in range(0, batch['x'].shape[0], chunk):
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == 'cuda'):
            out.append(model(batch['x'][i:i + chunk], batch['y'][i:i + chunk], batch['valid'][i:i + chunk]).float())
    return torch.cat(out)


class Validator:
    """Fixed validation sets, built once so every evaluation sees exactly the same episodes."""

    def __init__(self, cfg):
        self.cfg, self.splits = cfg, E.load_splits()
        self.ks = cfg['val_ks']
        train_sessions = E.sessions_where('train', 0)
        # A: held-out block of the training sessions (10 trials each -> 9 context + 1 query)
        store_a = E.TrialStore(train_sessions, parts=['val'], device=DEVICE)
        sampler_a = E.EpisodeSampler(store_a, E.EpisodeConfig(n_pool=9, n_extra=1, shift=self._shift_cfg()),
                                     bank=TemplateBank.load(('train',)), seed=0)
        rng = np.random.default_rng(1000)
        self.batches_a = [sampler_a.sample(16, rng=rng) for _ in range(4)]
        # B: validation-day no-stim sessions + validation-session templates
        val_bank = TemplateBank.load(('val',))
        self.sessions_b = E.sessions_where('val', 0)
        self.store_b = E.TrialStore(self.sessions_b, device=DEVICE)
        rng = np.random.default_rng(2000)
        self.shifts_b = {(s, d): sample_shift(rng, ShiftConfig(kind='template'), bank=val_bank)
                         for s in self.sessions_b for d in range(cfg['val_draws'])}
        # real: validation stim sessions, no simulated shift
        self.sessions_real = E.sessions_where('val', 1)
        self.store_real = E.TrialStore(self.sessions_real, device=DEVICE)

    def _shift_cfg(self):
        return replace(ShiftConfig(), kind=self.cfg['shift_kind'])

    def run_a(self, model):
        rows = []
        model.eval()
        pred = torch.cat([predict(model, b) for b in self.batches_a])
        y = torch.cat([b['y'] for b in self.batches_a])
        v = torch.cat([b['valid'] for b in self.batches_a])
        for k in [k for k in self.ks if k < y.shape[1]] + ([9] if 9 not in self.ks else []):
            rows.append(('A', 'icl', k, r2_score(pred[:, k], y[:, k], v[:, k])))
        model.train()
        return rows

    def run_splits(self, model, part):
        """B or real: R2 per (session, k, draw), averaged over sessions and draws."""
        store, sessions = (self.store_b, self.sessions_b) if part == 'B' else (self.store_real, self.sessions_real)
        scores = {}
        model.eval()
        for session in sessions:
            for k in self.ks:
                for draw in range(self.cfg['val_draws'] if k > 0 else 1):
                    shift = self.shifts_b[(session, draw)] if part == 'B' else None
                    batch = E.eval_batch(store, session, k, draw, self.splits, shift=shift,
                                         rng=np.random.default_rng(draw), clip=self.cfg['clip'])
                    query_y, query_v = batch['y'][:, -1], batch['valid'][:, -1]
                    pred = predict(model, batch)[:, -1]
                    scores.setdefault(('icl', k), []).append(r2_score(pred, query_y, query_v))
                    baseline = context_mean_profile(batch['y'], batch['valid'])
                    scores.setdefault(('ctx_mean_profile', k), []).append(r2_score(baseline, query_y, query_v))
        model.train()
        return [(part, method, k, float(np.nanmean(values))) for (method, k), values in scores.items()]


def git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C.ROOT, text=True).strip()
    except Exception:
        return 'unknown'


def main(argv=None):
    cfg = load_config(argv)
    out = os.path.join(RUNS, cfg['name'])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'config.yaml'), 'w') as file:
        yaml.safe_dump(cfg, file)
    with open(os.path.join(out, 'git_hash.txt'), 'w') as file:
        file.write(git_hash() + '\n')

    torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = True
    sessions = E.sessions_where('train', 0) + (E.sessions_where('train', 1) if cfg['train_stim'] else [])
    store = E.TrialStore(sessions, parts=['train'], device=DEVICE)
    episode_cfg = E.EpisodeConfig(max_jitter_bins=cfg['max_jitter_bins'], clip=cfg['clip'],
                                  shift=replace(ShiftConfig(), kind=cfg['shift_kind']))
    sampler = E.EpisodeSampler(store, episode_cfg, bank=TemplateBank.load(('train',)), seed=cfg['seed'])
    validator = Validator(cfg)

    model = build_model(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=tuple(cfg['betas']),
                                  weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(cfg))
    print(f"{cfg['name']}: {sum(p.numel() for p in model.parameters()):,} parameters, {cfg['steps']} steps", flush=True)

    metrics_path = os.path.join(out, 'metrics.csv')
    with open(metrics_path, 'w', newline='') as file:
        csv.writer(file).writerow(['step', 'split', 'method', 'k', 'value'])

    def log(rows):
        with open(metrics_path, 'a', newline='') as file:
            csv.writer(file).writerows(rows)

    best, start, running = -float('inf'), time.time(), []
    for step in range(1, cfg['steps'] + 1):
        batch = sampler.sample(cfg['batch'])
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == 'cuda'):
            pred = model(batch['x'], batch['y'], batch['valid'])
        loss = masked_mse(pred.float(), batch['y'], batch['valid'])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        optimizer.step()
        scheduler.step()
        running.append(loss.item())

        if step % cfg['log_every'] == 0:
            log([(step, 'train', 'icl', -1, float(np.mean(running)))])
            print(f"step {step:6d}  loss {np.mean(running):.4f}  lr {scheduler.get_last_lr()[0]:.2e}  "
                  f"{time.time() - start:.0f} s", flush=True)
            running = []
        if step % cfg['val_every'] == 0 or step == cfg['steps']:
            rows = validator.run_a(model) + validator.run_splits(model, 'B')
            if step % cfg['real_val_every'] == 0 or step == cfg['steps']:
                rows += validator.run_splits(model, 'real')
            log([(step, *row) for row in rows])
            b = {k: r for split, method, k, r in rows if split == 'B' and method == 'icl'}
            selection = float(np.mean([b[5], b[20]]))
            print('  val ' + '  '.join(f'{split}/{method[:3]}/k{k}={r:.3f}' for split, method, k, r in rows
                                        if k in (0, 5, 20, 9)) + f'  select={selection:.3f}', flush=True)
            state = dict(model=model.state_dict(), cfg=cfg, step=step, selection=selection)
            torch.save(state, os.path.join(out, 'ckpt_last.pt'))
            if selection > best:
                best = selection
                torch.save(state, os.path.join(out, 'ckpt_best.pt'))
    print(f"done in {time.time() - start:.0f} s; best selection R2 {best:.3f}", flush=True)


if __name__ == '__main__':
    main()
