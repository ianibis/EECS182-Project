"""Improve the calibrated decoder in three fixed steps, using VALIDATION days only.

This script never reads the test days. Real stimulation trials are still not used for training.

    Step 1  Reduce overfitting. Try a smaller model, stronger dropout, random time jitter, and their combinations.
    Step 2  Predict timing and amplitude. Let the query signal shift the movement template in time, and optionally
            scale it, using only the signal up to the current time bin.
    Step 3  Train 5 seeds of the chosen setting and compare one model with the average of the 5 predictions.

Decision rules, fixed before training. The score is the one that already selects checkpoints: mean validation R²
over recorded and 10x inputs at k = 2, 5, 10 and 20.

    A change is adopted only if both hold for seed 1:
    1. Its score is at least 0.01 above the setting it would replace (seed-to-seed spread is about 0.003).
    2. It still reads the context signal: at k = 20, breaking the signal-label pairing lowers R² by at least 0.01,
       with a 95% session interval above 0, for both inputs.
    If several changes qualify, the highest score wins. If none qualifies, the current setting stays.

Run from the repository root, on one free GPU:
    CUDA_VISIBLE_DEVICES=0 python task5/task5_improve.py --step1 --step2 --step3
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
import torch
from torch import nn

import task5_config as cfg  # first: it puts the shared Task 3 code on the path
import task5_analyze as analysis
import task5_data as data
import task5_diagnose_context as diagnose
from task5_baselines import LabelsOnlyBaseline
from task5_calibrated import N_FEATURES, CalibratedDecoder, load_calibrated
from task5_episodes import OffTrialBank, sample_episode
from task5_evaluate import context_positions, load_held_out, n_draws, predict_queries, session_r2
from task5_model import count_parameters, masked_mse
from task5_run_final import FINAL_ITERATIONS, SEEDS
from task5_shift import TemplateLibrary
from task5_train import VALIDATION_K, learning_rate, set_seed, validation_score

PILOT_SEED = 1
CHECK_K = 20
DRAWS = 3
MIN_GAIN = 0.01
MIN_CONTEXT_DROP = 0.01
MAX_SHIFT_BINS = 10     # the template may move by at most 0.5 s
MAX_GAIN = 2.0          # the template may be scaled between 1/2 and 2 times
JITTER_BINS = 2         # time jitter of at most 100 ms per trial
INPUTS = ('recorded', 'amp10')
CHOICE_PATH = os.path.join(cfg.RESULTS_DIR, 'improve_choice.json')


# ------------------------------------------------------------------------------- settings

@dataclass(frozen=True)
class Setting:
    name: str
    small: bool = False             # 2 layers of width 64 instead of 4 layers of width 128
    strong_dropout: bool = False    # dropout 0.3 and encoder dropout 0.5 instead of 0.1 and 0.25
    jitter_bins: int = 0            # random time shift of each training trial, signal and labels together
    timing: str = 'none'            # 'none', 'shift', or 'shift_gain'

    def model_config(self):
        values = {}
        if self.small:
            values.update(d_model=64, n_layers=2, ff_size=256)
        if self.strong_dropout:
            values.update(dropout=0.3, encoder_dropout=0.5)
        return cfg.ModelConfig(**values)


BASELINE = Setting('baseline')      # the 5 calibrated decoders that are already trained
STEP1_SETTINGS = (
    Setting('small', small=True),
    Setting('dropout', strong_dropout=True),
    Setting('jitter', jitter_bins=JITTER_BINS),
    Setting('small_dropout', small=True, strong_dropout=True),
    Setting('small_jitter', small=True, jitter_bins=JITTER_BINS),
    Setting('dropout_jitter', strong_dropout=True, jitter_bins=JITTER_BINS),
    Setting('small_dropout_jitter', small=True, strong_dropout=True, jitter_bins=JITTER_BINS),
)
TIMING_OPTIONS = ('shift', 'shift_gain')


def run_name(setting, seed):
    if setting == BASELINE:
        return f'calibrated_a0_seed{seed}'
    return f'improve_{setting.name}_seed{seed}'


# ------------------------------------------------------------------------------- model

class TimingDecoder(CalibratedDecoder):
    """A calibrated decoder whose movement template can move in time and change size on each trial.

        prediction(t) = gain(t) * template(t - shift(t)) + features(t) * (global_readout + session_correction)

    shift(t) and gain(t) are read from the features at time t, which see only the signal up to t. They start at
    0 and 1, so an untrained TimingDecoder makes the same prediction as a CalibratedDecoder.
    """

    def __init__(self, config, timing):
        super().__init__(config)
        self.use_gain = timing == 'shift_gain'
        self.to_timing = nn.Linear(N_FEATURES, 2)
        nn.init.zeros_(self.to_timing.weight)
        nn.init.zeros_(self.to_timing.bias)

    def warped_template(self, template, features):
        """Read a (B, 40, 2) template at each trial's own times: returns (B, T, 40, 2)."""
        n_trials = features.shape[1]
        raw = self.to_timing(features)
        shift = MAX_SHIFT_BINS * torch.tanh(raw[..., 0])
        times = torch.arange(cfg.N_BINS, device=features.device)[None, None]
        position = (times - shift).clamp(0, cfg.N_BINS - 1)
        lower = position.floor().long().clamp(max=cfg.N_BINS - 2)
        fraction = (position - lower)[..., None]
        expanded = template[:, None].expand(-1, n_trials, -1, -1)
        before = expanded.gather(2, lower[..., None].expand(-1, -1, -1, 2))
        after = expanded.gather(2, (lower + 1)[..., None].expand(-1, -1, -1, 2))
        warped = before + fraction * (after - before)
        if self.use_gain:
            warped = warped * torch.exp(math.log(MAX_GAIN) * torch.tanh(raw[..., 1]))[..., None]
        return warped

    def session_correction(self, features, velocity, valid, fitted, readout):
        """Ridge regression from context features to what the warped template and global readout got wrong."""
        n_episodes = len(features)
        weight = valid[..., None].float()
        error = (velocity - fitted - features @ readout[:, None]) * weight
        x = (features * weight).reshape(n_episodes, -1, N_FEATURES)
        y = error.reshape(n_episodes, -1, 2)
        ridge = self.log_ridge.exp() * torch.eye(N_FEATURES, device=x.device)
        return torch.linalg.solve(x.transpose(1, 2) @ x + ridge, x.transpose(1, 2) @ y)

    def decode(self, context_features, context_velocity, context_valid, query_features):
        with torch.autocast(query_features.device.type, enabled=False):    # the linear solve needs full precision
            context_features, query_features = context_features.float(), query_features.float()
            velocity = context_velocity.float()
            template = self.template(velocity, context_valid)
            readout = self.global_readout[None].expand(len(query_features), -1, -1)
            if context_features.shape[1] > 0:
                fitted = self.warped_template(template, context_features)
                readout = readout + self.session_correction(context_features, velocity, context_valid, fitted,
                                                             readout)
            return self.warped_template(template, query_features[:, None])[:, 0] + query_features @ readout


def build_model(setting):
    config = setting.model_config()
    if setting.timing == 'none':
        return CalibratedDecoder(config)
    return TimingDecoder(config, setting.timing)


def load_run(name, device):
    if name.startswith('calibrated_'):
        return load_calibrated(name, device)
    # These checkpoints are created by this script.
    saved = torch.load(os.path.join(cfg.RUNS_DIR, name, 'best.pt'), map_location=device, weights_only=False)
    model = build_model(Setting(**saved['setting'])).to(device)
    model.load_state_dict(saved['model'])
    return model.eval(), saved['velocity_scale'].to(device), saved['step']


# ------------------------------------------------------------------------------- time jitter

def jitter_trials(lfp, velocity, valid, max_bins, generator):
    """Shift each trial's signal and labels by the same whole number of bins. Bins moved in are invalid."""
    device = lfp.device
    n_trials, n_channels, n_samples = lfp.shape
    lags = torch.randint(-max_bins, max_bins + 1, (n_trials,), generator=generator, device=device)[:, None]

    samples = (torch.arange(n_samples, device=device)[None] - lags * cfg.BIN_SAMPLES).clamp(0, n_samples - 1)
    shifted_lfp = lfp.gather(2, samples[:, None].expand(-1, n_channels, -1))

    bins = torch.arange(cfg.N_BINS, device=device)[None] - lags
    inside = (bins >= 0) & (bins < cfg.N_BINS)
    bins = bins.clamp(0, cfg.N_BINS - 1)
    shifted_velocity = velocity.gather(1, bins[..., None].expand(-1, -1, 2))
    return shifted_lfp, shifted_velocity, valid.gather(1, bins) & inside


def jitter_episode(episode, max_bins, generator):
    if max_bins == 0:
        return episode
    query_lfp, query_velocity, query_valid = jitter_trials(episode.query_lfp, episode.query_velocity,
                                                           episode.query_valid, max_bins, generator)
    episode = replace(episode, query_lfp=query_lfp, query_velocity=query_velocity, query_valid=query_valid)
    n_episodes, k = episode.context_lfp.shape[:2]
    if k == 0:
        return episode
    lfp, velocity, valid = jitter_trials(episode.context_lfp.flatten(0, 1), episode.context_velocity.flatten(0, 1),
                                         episode.context_valid.flatten(0, 1), max_bins, generator)
    return replace(episode, context_lfp=lfp.reshape(n_episodes, k, *lfp.shape[1:]),
                   context_velocity=velocity.reshape(n_episodes, k, *velocity.shape[1:]),
                   context_valid=valid.reshape(n_episodes, k, *valid.shape[1:]))


# ------------------------------------------------------------------------------- training

def train_setting(setting, seed, iterations=FINAL_ITERATIONS, name=None, device='cuda'):
    """The same training as the calibrated decoder, with this setting's model and jitter."""
    name = name or run_name(setting, seed)
    run_dir = os.path.join(cfg.RUNS_DIR, name)
    if os.path.exists(os.path.join(run_dir, 'summary.json')):
        print(f'{name}: already finished', flush=True)
        return
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(cfg.RESULTS_DIR, 'chosen_setting.json')) as file:
        chosen = json.load(file)
    config = cfg.TrainConfig(seed=seed, iterations=iterations, learning_rate=chosen['learning_rate'],
                             warmup_iterations=chosen['warmup'])
    shift_config = cfg.ShiftConfig()
    set_seed(seed)

    bank, library = OffTrialBank(device), TemplateLibrary(device)
    baseline = LabelsOnlyBaseline()
    generator = torch.Generator(device=device).manual_seed(seed)
    model = build_model(setting).to(device)
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
            episode = jitter_episode(episode, setting.jitter_bins, generator)
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
                for index in range(10):     # held-out OFF trials, without jitter
                    held_k = VALIDATION_K[index % len(VALIDATION_K)]
                    batch = sample_episode(bank, library, shift_config, config.batch_size, held_k, generator,
                                           use_holdout_query=True)
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
                                velocity_scale=bank.velocity_scale.cpu(), setting=asdict(setting),
                                model_config=asdict(setting.model_config()), train_config=asdict(config)),
                           os.path.join(run_dir, 'best.pt'))
            print(f'[{name}] step {step:5d}  train {row["train_loss"]:.4f}  holdout {row["holdout_loss"]:.4f}  '
                  f'val R2 {score:+.4f}  vs labels-only {gain:+.4f}  context gain {context_gain:+.4f}  '
                  f'{row["minutes"]:.1f} min{marker}', flush=True)

    with open(os.path.join(run_dir, 'summary.json'), 'w') as file:
        json.dump(dict(name=name, setting=asdict(setting), best_val_r2=float(best_score), **asdict(config)), file,
                  indent=2)


# ------------------------------------------------------------------------------- decisions

def best_score(setting, seed=PILOT_SEED):
    with open(os.path.join(cfg.RUNS_DIR, run_name(setting, seed), 'summary.json')) as file:
        return json.load(file)['best_val_r2']


def check_setting(setting):
    """The score of seed 1, and whether it still reads the context signal."""
    name = run_name(setting, PILOT_SEED)
    summary = diagnose.summarize(diagnose.score_all(runs={PILOT_SEED: name}, loader=load_run))
    rotated = summary[(summary.k == CHECK_K) & (summary.condition == 'rotated')].set_index('input')
    row = dict(setting=setting.name, run=name, score=best_score(setting))
    for input_name in INPUTS:
        row.update({f'r2_k20_{input_name}': rotated.loc[input_name, 'r2_real'],
                    f'rotated_drop_{input_name}': rotated.loc[input_name, 'drop'],
                    f'rotated_ci_low_{input_name}': rotated.loc[input_name, 'ci_low']})
    row['reads_context'] = all(rotated.loc[i, 'drop'] >= MIN_CONTEXT_DROP and rotated.loc[i, 'ci_low'] > 0
                               for i in INPUTS)
    return row


def choose(candidates, current):
    """Apply the decision rule. Returns the adopted setting and the table of all candidates."""
    for setting in candidates:
        train_setting(setting, PILOT_SEED)
    table = pd.DataFrame([check_setting(setting) for setting in (current, *candidates)])
    reference = table.loc[table.setting == current.name, 'score'].iloc[0]
    table['gain_over_current'] = table.score - reference
    table['adopted'] = False
    qualified = table[(table.setting != current.name) & table.reads_context & (table.gain_over_current >= MIN_GAIN)]
    if qualified.empty:
        return current, table
    winner = qualified.loc[qualified.score.idxmax(), 'setting']
    table.loc[table.setting == winner, 'adopted'] = True
    return next(s for s in candidates if s.name == winner), table


def save_choice(key, setting):
    choices = {}
    if os.path.exists(CHOICE_PATH):
        with open(CHOICE_PATH) as file:
            choices = json.load(file)
    choices[key] = asdict(setting)
    with open(CHOICE_PATH, 'w') as file:
        json.dump(choices, file, indent=2)


def load_choice(key):
    with open(CHOICE_PATH) as file:
        return Setting(**json.load(file)[key])


def run_step1():
    chosen, table = choose(STEP1_SETTINGS, BASELINE)
    table.to_csv(os.path.join(cfg.RESULTS_DIR, 'improve_step1.csv'), index=False)
    save_choice('step1', chosen)
    print(table.round(4).T.to_string())
    print(f'STEP 1 RESULT: {chosen.name}', flush=True)


def run_step2():
    current = load_choice('step1')
    prefix = '' if current == BASELINE else f'{current.name}_'
    candidates = tuple(replace(current, name=f'{prefix}{timing}', timing=timing) for timing in TIMING_OPTIONS)
    chosen, table = choose(candidates, current)
    table.to_csv(os.path.join(cfg.RESULTS_DIR, 'improve_step2.csv'), index=False)
    save_choice('step2', chosen)
    print(table.round(4).T.to_string())
    print(f'STEP 2 RESULT: {chosen.name}', flush=True)


# ------------------------------------------------------------------------------- step 3: 5 seeds and averaging

def score_groups(groups, device='cuda'):
    """R² of one model (mean over seeds) and of the 5-seed average, per session, input, k and draw."""
    loaded = {group: [load_run(run_name(setting, seed), device) for seed in SEEDS]
              for group, setting in groups.items()}
    scale = next(iter(loaded.values()))[0][1]
    rows = []
    for session in data.usable_sessions('val', 'stim'):
        for input_name in INPUTS:
            held = load_held_out(session, input_name, scale, device)
            for k in VALIDATION_K:
                for draw in range(n_draws(k, DRAWS)):
                    positions = context_positions(held, k, draw)
                    for group, models in loaded.items():
                        outputs = [(predict_queries(model, held, positions) * scale).cpu().numpy()
                                   for model, _, _ in models]
                        one = np.mean([session_r2(o, held.query_velocity, held.query_valid)[0] for o in outputs])
                        average = session_r2(np.mean(outputs, axis=0), held.query_velocity, held.query_valid)[0]
                        rows.append(dict(session=session, input=input_name, k=k, draw=draw, group=group,
                                         one_model=one, seed_average=average))
    return pd.DataFrame(rows)


def difference_row(name, difference):
    low, high = analysis.bootstrap_interval(difference.to_numpy())
    day_low, day_high = analysis.bootstrap_interval(difference.to_numpy(),
                                                    [analysis.day_of(s) for s in difference.index])
    return {name: difference.mean(), f'{name}_ci_low': low, f'{name}_ci_high': high,
            f'{name}_day_ci_low': day_low, f'{name}_day_ci_high': day_high}


def run_step3():
    final = load_choice('step2')
    for seed in SEEDS:
        if final != BASELINE:
            train_setting(final, seed)

    groups = {'calibrated': BASELINE}
    if final != BASELINE:
        groups['final'] = final
    scores = score_groups(groups)
    scores.to_csv(os.path.join(cfg.RESULTS_DIR, 'improve_scores_val.csv'), index=False)
    per_session = scores.groupby(['session', 'input', 'k', 'group'])[['one_model', 'seed_average']].mean().unstack()
    final_group = 'final' if final != BASELINE else 'calibrated'

    saved = pd.read_csv(os.path.join(cfg.RESULTS_DIR, 'scores_val.csv'), keep_default_na=False)
    saved = saved[saved.draw < DRAWS]
    labels = analysis.labels_only_scores(saved).droplevel('group')
    first = analysis.session_scores(saved).droplevel('group')

    rows = []
    for input_name in INPUTS:
        for k in VALIDATION_K:
            part = per_session.xs((input_name, k), level=('input', 'k'))
            one, average = part[('one_model', final_group)], part[('seed_average', final_group)]
            old_one, old_average = part[('one_model', 'calibrated')], part[('seed_average', 'calibrated')]
            labels_part = labels.xs((input_name, k), level=('input', 'k'))[part.index]
            first_part = first.xs((input_name, k), level=('input', 'k'))[part.index]
            row = dict(input=input_name, k=k, n_sessions=len(part), final_setting=final.name, final_one_model=one.mean(),
                       final_seed_average=average.mean(), calibrated_one_model=old_one.mean(),
                       calibrated_seed_average=old_average.mean(), first_model=first_part.mean(),
                       labels_only=labels_part.mean())
            row.update(difference_row('average_minus_one_model', average - one))
            row.update(difference_row('one_model_minus_calibrated', one - old_one))
            row.update(difference_row('average_minus_calibrated_average', average - old_average))
            row.update(difference_row('average_minus_first_model', average - first_part))
            row.update(difference_row('average_minus_labels_only', average - labels_part))
            rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(os.path.join(cfg.RESULTS_DIR, 'improve_comparison_val.csv'), index=False)

    runs = {seed: run_name(final, seed) for seed in SEEDS}
    context = diagnose.summarize(diagnose.score_all(runs=runs, loader=load_run))
    context.to_csv(os.path.join(cfg.RESULTS_DIR, 'improve_context_signal_val.csv'), index=False)
    print(comparison.round(4).T.to_string())
    print(f'STEP 3 RESULT: final setting {final.name}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--step1', action='store_true')
    parser.add_argument('--step2', action='store_true')
    parser.add_argument('--step3', action='store_true')
    arguments = parser.parse_args()
    if arguments.step1:
        run_step1()
    if arguments.step2:
        run_step2()
    if arguments.step3:
        run_step3()
