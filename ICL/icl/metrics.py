"""Metrics and label-only baselines shared by training and evaluation."""
import torch


def r2_score(pred, y, valid):
    """R2 over valid bins, computed per output dimension and averaged (decision: vx, vy then mean).

    pred, y: (..., bins, dims); valid: (..., bins) bool. Returns a float (nan if too few bins).
    """
    p, t = pred[valid].double(), y[valid].double()
    if t.shape[0] < 2:
        return float('nan')
    ss_res = ((p - t) ** 2).sum(0)
    ss_tot = ((t - t.mean(0)) ** 2).sum(0)
    return float((1 - ss_res / ss_tot).mean())


def context_mean_profile(y, valid):
    """Label-only baseline: predict each query bin with the mean of the context trials' labels at that bin.

    y: (Q, k + 1, bins, dims) with the query last; valid likewise. Bins where no context trial is
    valid (and k = 0) get 0, i.e. the training mean in model units. Returns (Q, bins, dims).
    """
    context, mask = y[:, :-1], valid[:, :-1, :, None].to(y.dtype)
    if context.shape[1] == 0:
        return torch.zeros_like(y[:, -1])
    return (context * mask).sum(1) / mask.sum(1).clamp(min=1)
