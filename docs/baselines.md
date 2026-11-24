# CS182 recalibration baselines under stimulation

Baselines for [the proposal](proposal.md).
Five methods, one decoding target (Index finger velocity, `Index_vx_cm`/`Index_vy_cm`), one
fixed 30-trial query set.

| # | Method | Input | Adapts? |
|---|---|---|---|
| B0 | Ridge / Wiener, 9 causal lags (4 conditions) | cached LMP + 7 band powers | conditions (3),(4) refit on k |
| B1 | KNN | cached features | mode (b) uses only the k trials |
| B2 | MLP (2 hidden layers) | cached features | no |
| B3 | EEGNet (fully convolutional) | raw 1 kHz LFP, 35 channels | no |
| B4 | GRU | cached features, per-bin sequence | no |

Ridge and KNN are evaluated per draw and per k. MLP, EEGNet and GRU are trained once
per seed on the stimulation-off training split and scored on the fixed query set, so
their rows carry `k=0`.

## Setup

Data are not in version control. `task4/` and `task3/` expect this layout:

| Path | Contents |
|---|---|
| `data/export_outputs/h5_export_outputs/` | `<session>.h5`, directly inside this directory |
| `data/export_outputs/trials/` | `<session>_trials.csv` |
| `data/parrm_outputs/parrm_h5/` | PARRM-cleaned `<session>.h5` |
| `data/parrm_outputs/template_library.h5` | PARRM template library |
| `data/features/export_none/`, `data/features/parrm_none/` | Precomputed features |
| `data/kinematics_v3/bins50ms/` | 50 ms kinematics |
| `EECS182_data_manifest.csv`, `splits.json` | Session metadata and fixed splits, at the repository root |
| `data/manifest.csv`, `data/splits.json` | Identical loader fallbacks for a `data/` root |

`Paths.export_dir` must point at the directory that directly contains `<session>.h5`, which is why the notebook sets it to `data/export_outputs/h5_export_outputs`. `task4/utils.py` and `task3/task3_common.py` must agree on these paths; both currently do. Set `TASK3_DATA_ROOT` to point `task3/` at a different data root.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running

Run commands from the repository root.

```bash
jupyter lab task4/
```

Task 4 is split into three notebooks sharing [task4/utils.py](../task4/utils.py):

| Notebook | Task |
|---|---|
| `task4/task4_1_draws.ipynb` | 4.1 k-shot calibration draws, query = remaining ON trials |
| `task4/task4_2_eegnet.ipynb` | 4.2 braindecode EEGNet regression, early stopping, 5 seeds vs ridge |
| `task4/task4_3_parrm_ridge.ipynb` | 4.3 PARRM cleaning then ridge refit on each draw |

Outputs land beside the code: `task4/results/`, `task4/checkpoints/`, `task4/training_logs/`,
`task4/figures/`. `scripts/make_diagnostics.py` rebuilds the report figures from
`results/checkpoints` and `results/results_*.parquet`.

The 4.1 draws have a command-line entry point that calls the same sampler:

```bash
.venv/bin/python task4/make_draws.py                        # writes task4/results/draws.json
.venv/bin/python task4/make_draws.py --n-draws 20 --seed 7
```

Task 3 checks:

```bash
.venv/bin/python task3/task3_1_ridge.py --lags 9
.venv/bin/python task3/task3_1_ridge_within_session.py --lags 9
.venv/bin/python task3/task3_1_xgboost.py --lags 9
.venv/bin/python task3/task3_2_parrm.py --lags 9
.venv/bin/python task3/task3_3_stif.py
```

Outputs go to `results/task3/`. The baseline report is `reports/baselines/index.html`.

## Design decisions worth knowing

**Query set: fixed in the earlier baselines, leftover in task 4.** `docs/project-spec.md` says
"the remaining on trials are the query trials". The earlier baseline run used the fixed 30-trial
set (blocks 3–5) instead, which keeps R² comparable across k, draws, seeds and methods and keeps
`pool ∩ query = ∅`. That file is archived as `task4/results/draws_fixed_query_legacy.json`.
Task 4.1 now follows the spec literally: each draw scores the ON trials it did not use, so the
query set shrinks with k (50 − k trials per session).

**Normalisation never sees query trials.** Feature statistics are median / (1.4826·MAD) per
channel and feature. Training sessions use their training blocks and the earlier baselines use
the calibration pool. Task 4.1's leftover-query draws would overlap that pool, so `task4` normalises
each draw on its own k trials through `U.drawn_trial_windows`. `EECS182_NHP_features.load_features`
normalises on all included trials by default, so this code always passes an explicit
normalisation set through `trials`.

**EEGNet is fully convolutional.** A stock EEGNet averages over time, so it cannot
emit one prediction per 50 ms window. Here the output rate is 20 Hz (pooling strides 2 and
25 at 1 kHz), kernels are specified in milliseconds and converted, and temporal/separable
convolutions use left-only padding so window *j* only sees samples up to the end of bin *j*.
With `padding='same'` the network sees 250 ms of future and beats the causal ridge for free.

**Ridge alpha is tuned once per feature variant** on validation days, not by inner CV inside
a k=2 calibration set (80 windows). All four ridge conditions share it.

**KNN searches the raw window features.** There is no PCA (or any other projection) in front
of the neighbour search: distances are Euclidean on the same 2,800-dimensional window ridge
and MLP consume. Earlier versions projected to 48 principal components first; the projection
only re-weighted dimensions that KNN's inverse-distance weighting already handles, and it
cost a fitted transform that had to be re-fit per calibration draw.

**The torch models are deliberately vanilla.** Weight matrices are updated with Muon
(orthogonalised momentum, `MUON_LR`), biases and BatchNorm parameters with plain Adam, and
both rates follow a cosine decay to `LR_FLOOR` of their initial value over a fixed epoch
budget. There is no dropout, weight decay, gradient clipping, or early stopping. Early
stopping was removed because a decay-to-zero schedule assumes the full budget runs; the
validation R² trace is still logged per epoch (`results/training_logs/`) and the best epoch
is reported in the training-curve figure.

**Confidence intervals resample sessions.** `scripts/make_diagnostics.py` bootstraps per-session mean R² and
recovery fractions, so each held-out session counts once.

## Findings that change how results should be read

- **Absolute R² is low.** A within-session ridge on cached features reaches ~0.12 for a
  held-out block; cross-session transfer to a held-out test session is ~0.03–0.10. Speed
  (`Index_speed_cm`) is roughly 10× more decodable than per-bin velocity, which is
  consistent with the velocity target being derivative-noise dominated (DeepLabCut gives
  ~1 confident frame per 50 ms bin, and `Index_vx_cm` has ±140 cm/s outliers).
- **The OFF ceiling is therefore small**, so the recovery fraction
  `ρ = (R²_method − R²_no_adapt)/(R²_off − R²_no_adapt)` has a small denominator and is
  reported with its CI next to the raw R², never alone.
- **`Butters-250711-101424` has no screen-cm kinematics** (July 11 has no touch data), so
  it contributes zero valid training windows.
- **In the earlier baseline draws, k=20 was the whole pool**, so all 10 draws at that k are the
  same set and averaging over draws changed nothing; five sessions also had a repeated k=2 draw.
  Task 4.1 samples from all 50 stimulation-ON trials instead, so its draws stay distinct at k=20.
