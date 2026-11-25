# Task 4

Self-contained directory containing shared utilities, three sub-task notebooks, and local output folders (`checkpoints/`, `training_logs/`, `figures/`); CSVs land in the
repository-level `task4_results/`.

```text
task4/
  README.md                    documentation
  utils.py                     loaders, metrics, training loop, sampler, statistics
  task4_1_draws.ipynb          4.1 calibration draws
  task4_2_eegnet.ipynb         4.2 EEGNet decoder; MLP/GRU/EEGNet definitions
  task4_3_parrm_ridge.ipynb    4.3 PARRM + ridge refit; ridge/kNN definitions
```

Model classes (`MLPRegressor`, `GRURegressor`, `CausalEEGNet` in 4.2; `RidgeRegressor`, `KNNRegressor` in 4.3) are defined in notebook cells tagged `MODELS:`. `utils.py` executes these cells so downstream scripts (`kaggle/train_task4_2.py`, `kaggle/train_gru.py`, figure scripts) import directly from `utils`.

## Setup & Execution

Run from the repository root (`utils.py` resolves paths relative to the root):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
jupyter lab task4/
```

To regenerate the 10x semi-synthetic evaluation HDF5 files:

```bash
.venv/bin/python semisynthetic/EECS182_make_semisynthetic_h5.py --root data --split val-test --h5-only
```

To run the GRU sweep and 5-seed ensemble (requires `data/features/export_none/`):

```bash
python kaggle/train_gru.py
```

## Data Layout & Inputs

- **Cached features:** `(n_bins, 2800)` per 50 ms bin (`35 channels × 8 features × 10 lags`) or `(n_trials, 40, 2800)` per trial. Used by Ridge, kNN, MLP (bin-level), and GRU (40-step sequence).
- **LFP:** `(n_trials, 35, 2000)` per trial (`35 channels × 2000 samples` at 1 kHz). Used by CausalEEGNet.
- **Targets:** `(n_bins, 2)` or `(n_trials, 40, 2)` representing `(Index_vx, Index_vy)`.
- **Sessions:** `STIMULATION_ON` sessions pair with same-day `STIMULATION_OFF` control sessions via `splits.json` (`off_reference`), scored under `r2_off_*`.

| Input | Source | Usage |
|---|---|---|
| Recorded | `data/export_outputs/h5_export_outputs/` | 4.2, 4.3, GRU baseline |
| PARRM-cleaned | `data/features/parrm_none/` | 4.3, 4.2.10 evaluation |
| Semi-synthetic 10x | `data/semi_synthetic/parrm_amp10/h5_export_outputs/` (`lfp + 9 × parrm_artifact`) | 4.2.10 evaluation only |
| Synthetic frequency augmentation | `utils.add_synthetic_frequencies` (in-memory sine addition) | 4.2.6 training only |

## Sub-tasks

### 4.1 Calibration Draws (`task4_1_draws.ipynb`)
- Calls `utils.build_draws` to generate `task4_results/draws.json` (v3, `query_definition = "fixed"`).
- For each of the 14 test ON sessions, samples non-nested calibration draws of size $k \in \{2, 5, 10\}$ (10 draws each) and $k = 20$ (1 full-pool draw) from the 20-trial pool (blocks 1–2), totaling 434 partitions.
- Evaluates all draws against a fixed 30-trial query (blocks 3–5); normalizes features using calibration trials only. Controls evaluate all 50 OFF trials.

### 4.2 EEGNet Decoder (`task4_2_eegnet.ipynb`)
- **CausalEEGNet:** Rebuilds braindecode `EEGNet` with explicit left-only padding for causality, `channel_count=35`, `samples=2000`, `outputs=2`, and `final_conv_length=1`. `pool_first × pool_second` must equal 50 to map 2000 samples to 40 bins.
- **Training:** Trains on OFF trials for 5 seeds using Muon (weight matrices, `LR=0.02`) and Adam (vectors, `LR=1e-3`) with NaN-masked MSE (`U.masked_mse`) and early stopping on validation $R^2$ (`MAX_EPOCHS=200`, `PATIENCE=20`, `MIN_DELTA=1e-4`).
- **4.2.10 Amp10 Evaluation:** Evaluates saved 4.2.5/4.2.6 checkpoints without retraining across `recorded`, `amp10`, `parrm_recorded`, and `parrm_amp10_reuse` arms. Re-estimating PARRM removes the amplified artifact; reusing the 1x template leaves 8x residual artifact.

### 4.3 PARRM + Ridge Refit (`task4_3_parrm_ridge.ipynb`)
- Uses `task3/task3_common.py` (`C.ridge_path`, `C.LAMBDAS`, `C.CLIP_Z = ±10`).
- Selects a single ridge penalty on ON validation sessions and refits on the $k$ calibration trials of each draw for raw and PARRM-cleaned features.

## Key Outputs (`task4_results/`)

- **Draws:** `draws.json`, `task4_1_draws_summary.csv`
- **EEGNet:** `task4_2_eegnet_off.csv`, `task4_2_seeds.csv`, `task4_2_vs_ridge.csv`, `task4_2_eegnet_amp10.csv`, `task4_2_amp10_summary.csv`, `amp10_noise_variance.csv`, `amp10_noise_summary.csv`
- **GRU & Ridge:** `gru_sweep.csv`, `task4_gru_final.csv`, `task4_gru_final_seeds.csv`, `task4_3_parrm_ridge.csv`, `task4_3_summary.csv`
- **Cross-model & Audits:** `decoder_comparison.csv`, `decoder_paired.csv`, `checkpoint_audit.csv`, `artifact_severity.csv`