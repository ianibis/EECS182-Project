"""In-context linear regression with our backbone (plan 5.3.1 sanity check).

Reproduces the toy setting of Garg et al. (src/conf/toy.yaml: n_dims=5, n_points=11, batch 64,
lr 1e-4) with the backbone size planned for the decoder: GPT-2, 4 layers, width 128, 4 heads.
Prompt format and loss follow Garg's TransformerModel: x and y tokens interleaved, predictions
read at x positions, loss averaged over every prefix.

Pass: after training, the error at k >= 7 context points is close to least squares (which is
exact for k >= 5 here) and far below the zero estimator.

Run from ICL/:  .venv/bin/python scripts/sanity_linreg_icl.py [--steps 20000]
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


class LinRegTransformer(nn.Module):
    def __init__(self, n_dims, n_points, d=128, n_layer=4, n_head=4):
        super().__init__()
        self.read_in = nn.Linear(n_dims, d)
        self.backbone = GPT2Model(GPT2Config(
            vocab_size=1, bos_token_id=0, eos_token_id=0, n_positions=2 * n_points, n_embd=d,
            n_layer=n_layer, n_head=n_head, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False))
        self.read_out = nn.Linear(d, 1)

    def forward(self, xs, ys):
        batch, points, dims = xs.shape
        ys_wide = torch.cat([ys[..., None], xs.new_zeros(batch, points, dims - 1)], dim=2)
        zs = torch.stack([xs, ys_wide], dim=2).reshape(batch, 2 * points, dims)     # x1 y1 x2 y2 ...
        hidden = self.backbone(inputs_embeds=self.read_in(zs)).last_hidden_state
        return self.read_out(hidden)[:, ::2, 0]                                     # predict at x positions


def sample(batch, points, dims, device, generator=None):
    xs = torch.randn(batch, points, dims, device=device, generator=generator)
    w = torch.randn(batch, dims, 1, device=device, generator=generator)
    return xs, (xs @ w)[..., 0]


def least_squares_errors(xs, ys):
    """Squared error of min-norm least squares fitted on the first k points, predicting point k."""
    errors = []
    for k in range(xs.shape[1]):
        if k == 0:
            pred = torch.zeros_like(ys[:, 0])
        else:
            w = torch.linalg.pinv(xs[:, :k]) @ ys[:, :k, None]
            pred = (xs[:, k:k + 1] @ w)[:, 0, 0]
        errors.append((pred - ys[:, k]) ** 2)
    return torch.stack(errors, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', type=int, default=5)
    parser.add_argument('--points', type=int, default=11)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LinRegTransformer(args.dims, args.points).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'params {sum(p.numel() for p in model.parameters()):,}  device {device}')

    start = time.time()
    for step in range(1, args.steps + 1):
        xs, ys = sample(args.batch, args.points, args.dims, device)
        loss = ((model(xs, ys) - ys) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 1000 == 0 or step == 1:
            print(f'step {step:6d}  loss/d {loss.item() / args.dims:.4f}  {time.time() - start:.0f} s', flush=True)

    generator = torch.Generator(device=device).manual_seed(12345)
    xs, ys = sample(4096, args.points, args.dims, device, generator)
    with torch.no_grad():
        model_err = ((model(xs, ys) - ys) ** 2).mean(dim=0).cpu().numpy() / args.dims
    ls_err = least_squares_errors(xs, ys).mean(dim=0).cpu().numpy() / args.dims
    zero_err = (ys ** 2).mean(dim=0).cpu().numpy() / args.dims

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = np.column_stack([np.arange(args.points), model_err, ls_err, zero_err])
    np.savetxt(os.path.join(OUT_DIR, 'sanity_linreg.csv'), rows, delimiter=',', fmt=['%d', '%.5f', '%.5f', '%.5f'],
               header='k,transformer,least_squares,zero', comments='')
    print('\n k  transformer  least_squares  zero')
    for k, m, l, z in rows:
        print(f'{int(k):2d}  {m:11.4f}  {l:13.4f}  {z:6.3f}')
    late = model_err[7:].mean()
    print(f'\nmean error at k >= 7: transformer {late:.4f}, least squares {ls_err[7:].mean():.4f}, '
          f'zero {zero_err[7:].mean():.3f}  ->  {"PASS" if late < 0.1 else "FAIL"}')


if __name__ == '__main__':
    main()
