"""A decoder that has to use the context's brain signal: closed-form session calibration.

The first in-context model could ignore the context signal, and it did. This model removes that option.
Each trial is encoded on its own into causal features, and the context is used in two explicit steps:

    prediction(t) = template(t) + features(t) * (global_readout + session_correction)

    template            average context velocity at each time bin, shrunk toward a learned typical trajectory
    global_readout      a learned decoder shared by all sessions, used alone when k = 0
    session_correction  ridge regression from the context features to what the template and the global
                        readout got wrong on the context trials

The correction is computed in closed form, so no weights are updated at test time. It depends on the pairing
between context signal and context velocity, so it can learn which features stimulation has corrupted in this
session. The ridge strength is learned: if correcting does not help, training can turn it down.

Development uses VALIDATION days only. This script never reads the test days.

Pilot decision rule, fixed before running. A setting passes only if both hold at k = 20:

    A. Uses the context signal   breaking the signal-velocity pairing (each label gets another context trial's
                                 signal) lowers validation R² by at least 0.01, and the 95% session interval
                                 excludes 0, for both inputs
    B. No collapse               best validation R² is within 0.02 of the first model's seed 1

The passing setting with the highest validation R² goes on to 5 seeds. If none passes, no seeds are trained.

Run from the repository root, on one free GPU (replace 0 with its index):
    CUDA_VISIBLE_DEVICES=0 python task5/task5_calibrated.py --pilot
    CUDA_VISIBLE_DEVICES=0 python task5/task5_calibrated.py --final
    CUDA_VISIBLE_DEVICES=0 python task5/task5_calibrated.py --compare
"""

import argparse
import json
import math
import os

import pandas as pd
import torch
from torch import nn

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task5_analyze as analysis
import task5_diagnose_context as diagnose
from task5_mixing import train_with_mixing
from task5_model import TokenEncoder
from task5_run_final import SEEDS

N_FEATURES = 32
ALPHA_MAX_VALUES = (0.0, 0.25, 0.5)     # channel mixing during training; 0 means none
PILOT_SEED = 1
CHECK_K = 20
MAX_R2_LOSS = 0.02
INPUTS = ('recorded', 'amp10')
CHOSEN_PATH = os.path.join(cfg.RESULTS_DIR, 'calibrated_chosen.json')


class CalibratedDecoder(nn.Module):
    """Has the same encode / decode interface as InContextDecoder, so the same evaluation code scores it."""

    def __init__(self, config):
        super().__init__()
        self.encoder = TokenEncoder(config)
        self.time_embedding = nn.Embedding(cfg.N_BINS, config.d_model)
        layer = nn.TransformerEncoderLayer(config.d_model, config.n_heads, config.ff_size, config.dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.n_layers, enable_nested_tensor=False)
        self.to_features = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, N_FEATURES),
                                         nn.LayerNorm(N_FEATURES))
        self.global_readout = nn.Parameter(torch.zeros(N_FEATURES, 2))
        self.typical = nn.Parameter(torch.zeros(cfg.N_BINS, 2))
        self.log_template_prior = nn.Parameter(torch.tensor(math.log(10.0)))
        self.log_ridge = nn.Parameter(torch.tensor(math.log(100.0)))

    def encode(self, lfp, channel_keep):
        """Encode each trial on its own: (B, T, C, S) -> causal features (B, T, 40, N_FEATURES)."""
        n_episodes, n_trials = lfp.shape[:2]
        device = lfp.device
        flat = lfp.reshape(n_episodes * n_trials, *lfp.shape[2:])
        keep = channel_keep[:, None].expand(-1, n_trials, -1).reshape(n_episodes * n_trials, -1)
        tokens = self.encoder(flat, keep) + self.time_embedding(torch.arange(cfg.N_BINS, device=device))
        mask = nn.Transformer.generate_square_subsequent_mask(cfg.N_BINS, device=device, dtype=torch.bool)
        output = self.transformer(tokens, mask=mask, is_causal=True)
        return self.to_features(output).reshape(n_episodes, n_trials, cfg.N_BINS, -1)

    def template(self, velocity, valid):
        """Context mean velocity per bin, shrunk toward the learned typical trajectory: (B, 40, 2)."""
        prior = self.log_template_prior.exp()
        counts = valid.sum(dim=1)[..., None].float()
        totals = (velocity * valid[..., None]).sum(dim=1)
        return (totals + prior * self.typical) / (counts + prior)

    def session_correction(self, features, velocity, valid, template, readout):
        """Ridge regression from context features to the remaining context error: (B, N_FEATURES, 2)."""
        n_episodes = len(features)
        weight = valid[..., None].float()
        error = (velocity - template[:, None] - features @ readout[:, None]) * weight
        x = (features * weight).reshape(n_episodes, -1, N_FEATURES)
        y = error.reshape(n_episodes, -1, 2)
        ridge = self.log_ridge.exp() * torch.eye(N_FEATURES, device=x.device)
        return torch.linalg.solve(x.transpose(1, 2) @ x + ridge, x.transpose(1, 2) @ y)

    def decode(self, context_features, context_velocity, context_valid, query_features):
        """Predict (B, 40, 2) query velocities from the context and the query's own features."""
        with torch.autocast(query_features.device.type, enabled=False):    # the linear solve needs full precision
            context_features, query_features = context_features.float(), query_features.float()
            template = self.template(context_velocity.float(), context_valid)
            readout = self.global_readout[None].expand(len(query_features), -1, -1)
            if context_features.shape[1] > 0:
                readout = readout + self.session_correction(context_features, context_velocity.float(),
                                                             context_valid, template, readout)
            return template + query_features @ readout

    def forward(self, episode):
        keep = episode.channel_keep
        query_features = self.encode(episode.query_lfp[:, None], keep)[:, 0]
        if episode.context_lfp.shape[1] > 0:
            context_features = self.encode(episode.context_lfp, keep)
        else:
            context_features = query_features[:, :0, None].expand(-1, 0, cfg.N_BINS, -1)
        return self.decode(context_features, episode.context_velocity, episode.context_valid, query_features)


def run_name(alpha_max, seed):
    return f'calibrated_a{alpha_max:g}_seed{seed}'


def load_calibrated(name, device):
    # These checkpoints are created by this project's training code.
    saved = torch.load(os.path.join(cfg.RUNS_DIR, name, 'best.pt'), map_location=device, weights_only=False)
    model = CalibratedDecoder(cfg.ModelConfig(**saved['model_config'])).to(device)
    model.load_state_dict(saved['model'])
    return model.eval(), saved['velocity_scale'].to(device), saved['step']


def evaluate_setting(alpha, reference_r2):
    """Apply the two pilot gates to one trained model."""
    name = run_name(alpha, PILOT_SEED)
    with open(os.path.join(cfg.RUNS_DIR, name, 'summary.json')) as file:
        best_r2 = json.load(file)['best_val_r2']
    summary = diagnose.summarize(diagnose.score_all(runs={PILOT_SEED: name}, loader=load_calibrated))
    at_k = summary[summary.k == CHECK_K]
    rotated = at_k[at_k.condition == 'rotated'].set_index('input')
    zero = at_k[at_k.condition == 'zero'].set_index('input')

    row = dict(alpha_max=alpha, run=name, best_val_r2=best_r2, reference_val_r2=reference_r2)
    for input_name in INPUTS:
        row.update({f'r2_k20_{input_name}': rotated.loc[input_name, 'r2_real'],
                    f'labels_only_k20_{input_name}': rotated.loc[input_name, 'r2_labels_only'],
                    f'rotated_drop_{input_name}': rotated.loc[input_name, 'drop'],
                    f'rotated_ci_low_{input_name}': rotated.loc[input_name, 'ci_low'],
                    f'rotated_ci_high_{input_name}': rotated.loc[input_name, 'ci_high'],
                    f'zero_drop_{input_name}': zero.loc[input_name, 'drop']})
    gate_a = bool(rotated.loc[list(INPUTS), 'matters'].all())
    gate_b = bool(best_r2 >= reference_r2 - MAX_R2_LOSS)
    row.update(gate_a_uses_context_signal=gate_a, gate_b_no_collapse=gate_b, passed=gate_a and gate_b)
    return row


def run_pilot():
    for alpha in ALPHA_MAX_VALUES:
        train_with_mixing(run_name(alpha, PILOT_SEED), alpha, PILOT_SEED, model_class=CalibratedDecoder)
    with open(os.path.join(cfg.RUNS_DIR, f'final_in_context_seed{PILOT_SEED}', 'summary.json')) as file:
        reference_r2 = json.load(file)['best_val_r2']
    table = pd.DataFrame([evaluate_setting(alpha, reference_r2) for alpha in ALPHA_MAX_VALUES])
    table.to_csv(os.path.join(cfg.RESULTS_DIR, 'calibrated_pilot.csv'), index=False)
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
        train_with_mixing(run_name(alpha, seed), alpha, seed, model_class=CalibratedDecoder)
    print('ALL CALIBRATED SEEDS DONE', flush=True)


def paired_row(input_name, k, name, difference):
    low, high = analysis.bootstrap_interval(difference.to_numpy())
    day_low, day_high = analysis.bootstrap_interval(difference.to_numpy(),
                                                    [analysis.day_of(s) for s in difference.index])
    return {f'{name}': difference.mean(), f'{name}_ci_low': low, f'{name}_ci_high': high,
            f'{name}_day_ci_low': day_low, f'{name}_day_ci_high': day_high,
            f'{name}_sessions_positive': int((difference > 0).sum())}


def run_compare():
    """Validation comparison of the 5 calibrated seeds with the first model and the labels-only reference."""
    with open(CHOSEN_PATH) as file:
        alpha = json.load(file)['alpha_max']
    table = diagnose.score_all(runs={seed: run_name(alpha, seed) for seed in SEEDS}, loader=load_calibrated)
    diagnose.summarize(table).to_csv(os.path.join(cfg.RESULTS_DIR, 'calibrated_context_signal_val.csv'), index=False)

    real = table[table.condition == 'real']
    new = real.groupby(['session', 'input', 'k', 'seed']).r2.mean().groupby(['session', 'input', 'k']).mean()
    saved = pd.read_csv(os.path.join(cfg.RESULTS_DIR, 'scores_val.csv'), keep_default_na=False)
    saved = saved[saved.draw < diagnose.DRAWS]      # the same context draws as the new model
    first = analysis.session_scores(saved).reset_index().set_index(['session', 'input', 'k']).r2
    labels = analysis.labels_only_scores(saved).reset_index().set_index(['session', 'input', 'k']).r2

    rows = []
    for input_name in INPUTS:
        for k in diagnose.K_VALUES:
            sessions = new.xs((input_name, k), level=('input', 'k'))
            first_part = first.xs((input_name, k), level=('input', 'k'))[sessions.index]
            labels_part = labels.xs((input_name, k), level=('input', 'k'))[sessions.index]
            row = dict(input=input_name, k=k, n_sessions=len(sessions), r2_calibrated=sessions.mean(),
                       r2_first_model=first_part.mean(), r2_labels_only=labels_part.mean())
            row.update(paired_row(input_name, k, 'minus_first_model', sessions - first_part))
            row.update(paired_row(input_name, k, 'minus_labels_only', sessions - labels_part))
            rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(os.path.join(cfg.RESULTS_DIR, 'calibrated_comparison_val.csv'), index=False)
    print(comparison.round(4).T.to_string())
    print('DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--compare', action='store_true')
    arguments = parser.parse_args()
    if arguments.pilot:
        run_pilot()
    if arguments.final:
        run_final()
    if arguments.compare:
        run_compare()
