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

The model and the training loop live in this one notebook. `CausalEEGNet.__init__`
builds an explicit `nn.Sequential`, so `print(CausalEEGNet())` lists every layer in
forward order. Architecture from Lawhern et al. 2018,
<https://arxiv.org/abs/1611.08024>.

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
| `results/task4_3_parrm_ridge.csv` | Per session, k, draw and feature variant |
| `results/task4_3_summary.csv` | Per variant and k with a session bootstrap CI |
