# Task 3: movement decoding during stimulation

| Script | Use |
| --- | --- |
| `task3/task3_1_ridge.py` | Linear OFF-to-ON benchmark (Task 3.1) |
| `task3/task3_1_ridge_semisynthetic.py` | Ridge on recorded, PARRM-cleaned and 10x ON features |
| `task3/task3_1_ridge_gru_matched.py` | Ridge with the GRU's 9 features, 19 lags and same-day calibration on 20 OFF trials |
| `task3/task3_make_amp10_features.py` | 10x ON feature caches from the local `data/` layout |
| `task3/task3_2_parrm.py` | Traditional pipeline (3.2): OFF ridge on recorded vs. PARRM-cleaned ON features (k = 0), then PARRM + ridge refit on k = 2, 5, 10, 20 labeled ON trials |
| `task3/task3_3_stif.py` | Stimulation monitor-only control (3.3) |
| `task3/task3_1_xgboost.py` | Optional tree-model comparison |

The ridge scripts select their own linear hyperparameters and do not use the GRU checkpoint. They use LMP + 7 band powers per channel and default to 19 lags (1 s of history, as in the GRU); earlier results used 9 lags (`--lags 9`). These are **not** the same experiment as the later 19-lag residual ridge result reported in the project notes.

**Traditional pipeline (`task3_2_parrm.py`).** The OFF ridge decoder W0 is refit on k labeled, PARRM-cleaned trials of each stimulation-ON session, minimizing `||Y - XW - b||² + mu ||W - W0||²`. The k trials are the 10 draws per session in `splits.json` (`calibration[session].subsets[k]`, from blocks 1-2). mu is chosen for each k on the validation-day ON sessions, and the refit models are scored on the query trials (blocks 3-5). k = 0 is the OFF decoder on PARRM features. The summary reports the mean R² per condition and k, a 95% bootstrap interval over sessions, and the fraction of the ON-OFF loss recovered. Features of every validation and test session are normalized with its 20 calibration-pool trials (features only, no labels), even when k < 20.

The GRU uses 35 channels with nine features per channel, a 48-unit projection,
and a 64-unit GRU. It predicts horizontal and vertical velocity residuals,
with auxiliary speed and direction heads during training. A trial contains
19 warm-up bins and 40 scored 50-ms bins. Each checkpoint has 37,316 parameters.
The ensemble averages five seed predictions.

## Setup and inputs

Use the checkout's existing loader and features modules. From the repository root:

```sh
python -m pip install -r requirements.txt
export TASK3_DATA_ROOT="/path/to/Box/project"
```

The decoder needs NumPy, pandas, h5py, and PyTorch. Use Python 3.10 or later.
The [file map](../EECS182_file_map_261117.md) lists exact Box input paths.
The existing `TASK3_DATA_ROOT` setting is shared by the GRU and MLP. Checkout
results stay local. Calibration and checkpoint lookups prefer generated
checkout artifacts before falling back to the archived Box files.

## Commands

Run the existing ridge entry points with the same data setting:

```sh
python -B task3/task3_1_ridge.py
python -B task3/task3_2_parrm.py
python -B task3/task3_3_stif.py
```

The ridge, XGBoost, and stimulation-monitor scripts write to `task3_results/`.
Their existing feature generator is `task3/task3_make_amp10_features.py`.

For training and calibration, run these in a clean checkout with the Box
data and caches available:

```sh
python -B task3/task3_gru_multitask_v1.py --mode augmented --target velocity --device cpu
python -B task3/task3_gru_full_finetune_v1.py --device cpu
python -B task3/task3_gru_temporal_adapt_v1.py --device cpu
python -B task3/task3_gru_ensemble.py --device cpu
```

The multitask script trains seed 1 with the established loss, dropout, seed,
and early stopping. The next two scripts fit the saved calibration grids
using validation-day pools. The ensemble reuses archived peers with the
archived seed-1 checkpoint. With a locally trained seed-1 checkpoint it
creates any missing local peers from the original training split. Use a
fresh output area for a full regeneration. Keep all generated checkpoints
outside GitHub.

Run the established evaluations on the local or archived checkpoints:

```sh
python -B task3/task3_gru_evaluate.py --device cpu --split val
python -B task3/task3_gru_verify.py --split validation
python -B task3/task3_gru_evaluate.py --device cpu --split test
python -B task3/task3_gru_verify.py --split test
python -B task3/task3_gru_results.py --device cpu --split val --dataset all
```

Each adaptation uses 20 same-day stimulation-OFF trials and an OFF-fitted
time correction, then scores 30 disjoint query trials per session. Query
conditions include OFF, recorded 1× ON, 10× ON, and PARRM-cleaned ON.
Task 4.1 matched ON-context draws are a separate protocol.

`task3_gru_core.py` implements model loading, features, calibration, and
scoring. `task3_paths.py` maps checkout and Box paths. The imported
`task3_common.py`, `task3_diag_levels_v7.py`, `task3_hf_velocity_v1.py`,
`task3_gru_v2_vxfocus.py`, `task3_gru_trial_v2.py`,
`task3_gru_readout_recal_v1.py`, and `task3_gru_partial_finetune_v1.py`
provide shared training, feature, and calibration code.

Training, calibration, and ensemble files are under
`task3_results/gru/{training,calibration,ensemble}/`.
Validation and test outputs contain `sessions.csv` and `aggregate.csv`.
The verifier writes `benchmark.csv` using archived comparators, with a
score tolerance of 0.0001 and exact scored-bin counts.
The older condition-specific runner writes to `task3_results/gru/recording_modes/`.
The archived five-seed OFF R² is 0.204977 on validation and 0.197518 on test.

The ensemble, older results runner, and multitask training script expose
`--self-test`. If a checkout has workflows using an older layout, update
their script paths to `task3/` and output paths to `task3_results/gru/`
before scheduling them.
