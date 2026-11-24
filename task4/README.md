# Task 4

Three notebooks, one per sub-task, sharing a single helper module. Outputs live
beside the code so a task is self-contained.

```text
task4/
  utils.py                     shared code
  task4_1_draws.ipynb          4.1 calibration draws
  task4_2_eegnet.ipynb         4.2 EEGNet regression decoder
  task4_3_parrm_ridge.ipynb    4.3 PARRM then ridge refit
  make_draws.py                CLI wrapper for the 4.1 sampler
  eval_amp10.py                CLI: score the 4.2 checkpoints on the 10x recordings
  results/  checkpoints/  training_logs/  figures/
```

Run everything from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
jupyter lab task4/
```

`utils.py` resolves the repository root from its own location, so the notebooks
work whether Jupyter is started at the root or inside `task4/`.

## Real, semi-synthetic and synthetic input

Four different inputs appear in this task; three of them are recordings on disk.

| Input | What it is | Where it comes from | Who uses it |
|---|---|---|---|
| Recorded | the export as acquired, stimulation ON or OFF | `data/export_outputs/h5_export_outputs/` | 4.2, 4.3, GRU baseline |
| PARRM-cleaned | recorded ON minus the estimated artifact | `data/features/parrm_none/` | 4.3 |
| Semi-synthetic 10x | recorded ON + 9 × estimated artifact | `data/semi_synthetic/parrm_amp10/h5_export_outputs/` | 4.2.10 only |
| Synthetic tones | a pure sine added in memory | `utils.add_stimulation_tones` | 4.2.6 training only |

Only the 10x row is a separate dataset to build; the others are variants of the same
export. The 10x recordings are ON-session evaluation data, never training data.

The semi-synthetic files are built by `semisynthetic/EECS182_make_semisynthetic_h5.py`:
a copy of the export with only `/lfp` changed, on the 35 channels the decoders read, as
`lfp + (scale − 1) × parrm_artifact` with `scale = 10`. The artifact is the measured PARRM
template — periodic at the AM rate, but a real waveform, not a sine. Regenerate with

```bash
.venv/bin/python semisynthetic/EECS182_make_semisynthetic_h5.py --root data --split val-test --h5-only
```

`--root data` is required in this repository, where the recordings live under `data/`
rather than beside `splits.json`. `--h5-only` skips the feature caches, which only the
ridge and GRU paths need. `utils.semisynthetic_paths()` and `utils.amp10_test_datasets()`
read them; `task4/eval_amp10.py` is the CLI.

## 4.1 Calibration draws

For each stimulated test session and each k in {2, 5, 10, 20}, sample 10 sets of k
stimulation-ON trials. The remaining ON trials are that draw's query set, per
`docs/project-spec.md`. Writes `results/draws.json`.

- Draws are nested: draw `i` at k=2 is contained in draw `i` at k=5, 10 and 20.
- Seeded per session, so the file reproduces and adding a session does not shift
  the others.
- Normalisation uses the drawn trials, because the query set is the leftover ON
  trials and would otherwise overlap the statistics.

CLI equivalent: `.venv/bin/python task4/make_draws.py`.

The earlier baseline run used a fixed 30-trial query set instead; that file is
kept as `results/draws_fixed_query_legacy.json` and is not read by these
notebooks.

## 4.2 EEGNet regression decoder (braindecode)

EEGNet on 1 kHz LFP predicting Index fingertip velocity per 50 ms window, trained
on stimulation-OFF trials.

- **4.2.1** braindecode `EEGNet` emits one velocity pair per 50 ms bin (pooling 2
  then 25 over 2000 samples) with `final_conv_length=1`. Its stock convolutions
  pad symmetrically, so each padded convolution is rebuilt as an explicit
  left-only pad plus an unpadded convolution. The notebook asserts causality by
  perturbing the last input sample and checking that only the last bin moves.
- **4.2.2** Early stopping on validation R² with the best-epoch weights restored,
  and a NaN-masked loss (`U.masked_mse`) so bins without kinematics do not
  contribute. `U.train_trial_model` deliberately does not stop early.
- **4.2.3** Five seeds, held-out OFF R² compared against ridge (task 3.1) plus the KNN,
  MLP and GRU baselines. OFF R² is scored on each session's same-day no-stimulation
  recording, which none of the decoders trained on, so the earlier baseline run stays
  comparable. Days are the unit: 3–4 sessions share each OFF reference.

| Decoder | Input | Adapts? |
|---|---|---|
| Ridge | cached features, 9 lags | refit per draw in 4.3 |
| KNN | cached features | per draw (`knn_incontext`) |
| MLP | cached features, dense layers | no |
| GRU | cached features, per-bin sequence | no |
| EEGNet (4.2) | raw 1 kHz LFP, convolutions | no |

The MLP is not a variant of the EEGNet: it is fully connected on cached per-bin
features, while EEGNet convolves the raw 1 kHz LFP. Their OFF scores differ for that
reason, not because of a tuning gap.

The training loop lives in this notebook and the model in `utils.CausalEEGNet`.
`CausalEEGNet.__init__` builds an explicit `nn.Sequential`, so `print(CausalEEGNet())`
lists every layer in forward order, and the single definition means this notebook and
`kaggle/train_task4_2.py` build and reload the same module. Architecture from Lawhern
et al. 2018, <https://arxiv.org/abs/1611.08024>.

### Hyperparameters

Model arguments of `CausalEEGNet`, defined in `task4_2_eegnet.ipynb`:

| Argument | Default | Values worth sweeping | Effect |
|---|---|---|---|
| `temporal_filters` (F1) | 8 | 4, 8, 16 | temporal filters before the first pool |
| `depth_multiplier` (D) | 2 | 1, 2 | depthwise filters per temporal filter; F2 = F1 × D |
| `kernel_samples` | 500 | 125, 250, 500, 1000 | temporal receptive field, 125–1000 ms at 1 kHz |
| `separable_samples` | 62 | 31, 62, 125 | depthwise temporal kernel |
| `pool_first`, `pool_second` | 2, 25 | (2, 25), (5, 10), (10, 5), (25, 2) | output rate: bins = 2000 / (pool_first × pool_second) |
| `dropout` | 0.5 | 0.25, 0.5, 0.75 | regularisation between the two blocks |

`pool_first × pool_second` **must divide 2000 into 40 bins**, because the kinematics
target has one velocity pair per 50 ms bin. Any other product silently misaligns
predictions and targets, and the constructor raises rather than reshaping.

Fixed by the task: `channel_count=35`, `samples=2000`, `outputs=2`, and
`final_conv_length=1` inside the class — the last one is what keeps one prediction per
bin instead of averaging the time axis away.

Training knobs:

| Setting | Default | Where | Values worth sweeping |
|---|---|---|---|
| `U.ADAM_LR` | 1e-3 | `utils.py` | 3e-4, 1e-3, 3e-3 (Adam, biases and batch-norm) |
| `U.MUON_LR` | 0.02 | `utils.py` | 0.01, 0.02, 0.05 (Muon, weight matrices) |
| `U.LR_FLOOR` | 0.01 | `utils.py` | floor of the cosine decay |
| `U.TRIAL_BATCH_SIZE` | 16 | `utils.py` | 8, 16, 32 |
| `MAX_EPOCHS`, `PATIENCE`, `MIN_DELTA` | 200, 20, 1e-4 | notebook | stopping budget and tolerance |
| `SEEDS` | (0, 1, 2, 3, 4) | notebook | more seeds; 5 cannot resolve the ±0.03 seed spread |

There is no weight decay: `U.build_optimizers` uses Muon on the weight matrices and
Adam on vectors, with no `weight_decay`. Adding it means touching that function, so it
affects every decoder, not just EEGNet.

braindecode's `EEGNet` exposes more that the wrapper does not pass through — `F2`,
`pool_mode` (`mean`/`max`), `activation`, batch-norm `momentum`/`eps`/`affine`,
`conv_spatial_max_norm`, `norm_rate`. Widen the constructor signature if you want to
sweep those.

### 4.2.10 Semi-synthetic 10x artifact, and what PARRM does to it

The stimulation artifact is the largest thing in an ON recording, and the real data cannot
say what happens if it were larger: the amplitude was fixed for every session. The 10x files
answer that artificially, as `recorded + 9 × parrm_artifact` on the 35 decoded channels.

Nothing is retrained. The seeds from 4.2.5 and 4.2.6 are reloaded and scored on four input
arms that share the same same-day OFF control:

| Arm | Input | What it isolates |
|---|---|---|
| `recorded` | the export | baseline |
| `amp10` | `recorded + 9 × artifact` | sensitivity to artifact amplitude |
| `parrm_recorded` | PARRM-cleaned export | what cleaning buys a *fixed* decoder |
| `parrm_amp10_reuse` | `10x − 1 × artifact` | cleaning with the wrong template scale |

PARRM's estimate scales with the amplitude of the periodic component, so re-estimating it on
the amplified signal gives `10x − 10 × artifact = clean`. The `parrm_recorded` arm therefore
*is* the re-estimated result; `task4/make_parrm_amp10.py --verify` checks the identity to
3e-8 relative RMS and the scores below reproduce it to four decimals. The fourth arm is what
happens when the existing template library is reused instead, leaving eight copies.

Measured on the 14 test ON sessions (`results/task4_2_amp10_summary.csv`):

| Method | recorded | 10x | PARRM, recorded | PARRM, 10x re-estimated | PARRM, 1x template reused | OFF |
|---|---:|---:|---:|---:|---:|---:|
| `eegnet` | −0.145 | −3.244 | **+0.050** | +0.050 | −2.944 | 0.037 |
| `eegnet_ensemble` | −0.082 | −2.544 | **+0.106** | +0.106 | −2.327 | 0.082 |
| `eegnet_stimulation_tones` | +0.030 | −0.322 | +0.026 | +0.026 | −0.246 | 0.032 |
| `eegnet_stimulation_tones_ensemble` | +0.052 | −0.205 | +0.050 | +0.050 | −0.151 | 0.048 |

Three things follow.

1. **Cleaning rescues a fixed decoder, where labels did not have to.** The plain EEGNet goes
   from −0.145 on the recorded ON input to +0.050 cleaned, and the ensemble from −0.082 to
   +0.106, which is above its own OFF control (0.082). Task 4.3 found PARRM worth ~0.0005
   *after* a refit on labelled ON trials; the difference is that a refit can learn to ignore
   the artifact, while a fixed decoder cannot. Cleaning is the only lever a fixed decoder has.
2. **Re-estimating matters more than cleaning.** Reusing the 1x template leaves eight copies
   and stays catastrophic (−2.944). The rescue needs the template to be scaled to the
   artifact actually present, which is exactly what re-running PARRM on the amplified signal
   does.
3. **The tone-augmented arm is already clean-robust and gains nothing** (+0.030 → +0.026), so
   the two fixes overlap: whichever you do first absorbs most of the benefit.

`loader.normalize` z-scores each channel by its MAD, so a larger artifact also inflates the
scale and shrinks the neural signal in z-units (MAD ratio 1.9 at 3 Hz, 1.3 at 20 Hz). The
arms therefore differ in normalisation as well as artifact amplitude, and PARRM removes
everything phase-locked to the AM cycle, brain included. The variance characterisation in
`semisynthetic/semi_synthetic_README.md` quantifies the added term directly.

## 4.3 PARRM then ridge refit on each draw

Refit a ridge decoder on the k calibration trials of every draw, score that
draw's query trials, for raw and PARRM-cleaned features. The ridge penalty is
chosen once on the ON validation sessions and held fixed across draws, so
differences come from the refit data rather than a per-draw hyperparameter
search.

The notebook imports `task3/task3_common.py` rather than reimplementing ridge:
`C.ridge_path` solves for a new draw, `C.LAMBDAS` is the 13-value penalty grid,
and `C.CLIP_Z` clips z-scored features to ±10 so a single artifact bin cannot
dominate the weights. Task 3 and task 4.3 therefore make the same preprocessing
and solver choices; only the fit data differs (fixed OFF trials vs. the drawn ON
trials).

## Outputs

| File | Contents |
|---|---|
| `results/draws.json` | 4.1 draws plus invariant checks |
| `results/task4_1_draws_summary.csv` | Per session and k: draw counts, query sizes |
| `results/task4_2_eegnet_off.csv` | Per session and seed: query and OFF R² |
| `results/task4_2_seeds.csv` | Best validation R² and stopping epoch per seed |
| `results/task4_2_vs_ridge.csv` | EEGNet minus ridge 3.1, per session |
| `results/task4_2_eegnet_amp10.csv` | Per session, method, seed and variant: recorded ON, 10x ON and OFF R² |
| `results/task4_2_amp10_summary.csv` | Per method: recorded ON, 10x ON, OFF, and the 10x − recorded delta |
| `results/task4_3_parrm_ridge.csv` | Per session, k, draw and feature variant |
| `results/task4_3_summary.csv` | Per variant and k with a session bootstrap CI |
