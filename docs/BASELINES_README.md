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

Data are not in version control. The notebook and `task3/` expect this layout:

| Path | Contents |
|---|---|
| `data/exports/export_h5_LFPs/` | `<session>.h5`, directly inside this directory |
| `data/exports/export_trials/` | `<session>_trials.csv` |
| `data/parrm/parrm_h5_LFPs/` | PARRM-cleaned `<session>.h5` |
| `data/parrm/template_library.h5` | PARRM template library |
| `data/features/export_none/`, `data/features/parrm_none/` | Precomputed features |
| `data/kinematics_bins50ms/bins50ms/` | 50 ms kinematics |
| `EECS182_data_manifest.csv`, `splits.json` | Session metadata and fixed splits, read by the notebook and `task3/` |
| `data/manifest.csv`, `data/splits.json` | Identical loader fallbacks for a `data/` root |

`Paths.export_dir` must point at the directory that directly contains `<session>.h5`, which is why the notebook sets it to `data/exports/export_h5_LFPs`. The notebook passes the root manifest and splits paths to the loader. The `data/` copies keep the loader defaults working.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running

Run commands from the repository root.

```bash
jupyter lab cs182_decoder_grouped.ipynb
jupyter nbconvert --to notebook --execute cs182_decoder_grouped.ipynb --output cs182_decoder_grouped.executed.ipynb
```

Set `RUN_PIPELINE = True` in the last cell to retrain every model and overwrite `results/`. The default runs the saved checkpoints and writes figures. `scripts/make_diagnostics.py` rebuilds the report figures from `results/checkpoints` and `results/results_*.parquet`.

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

**Query set is fixed (option A).** `docs/project_specs.md` says "the remaining on trials are
the query trials". Taken literally the scored set would change with k (48 trials at k=2,
30 at k=20) and would contain pool trials, so pool-fitted feature normalisation would see
query data. The fixed 30-trial set (blocks 3–5) keeps R² comparable across k, draws, seeds
and methods, and keeps `pool ∩ query = ∅`.

**Normalisation never sees query trials.** Feature statistics are median / (1.4826·MAD) per
channel and feature, computed from the calibration pool for held-out sessions and from the
training blocks for training sessions. `EECS182_NHP_features.load_features` normalises on all
included trials by default, so this code passes the calibration-pool rows through `trials` for
held-out sessions and the training blocks for training sessions.

**EEGNet is fully convolutional.** A stock EEGNet averages over time, so it cannot
emit one prediction per 50 ms window. Here the output rate is 20 Hz (pooling strides 2 and
25 at 1 kHz), kernels are specified in milliseconds and converted, and temporal/separable
convolutions use left-only padding so window *j* only sees samples up to the end of bin *j*.
With `padding='same'` the network sees 250 ms of future and beats the causal ridge for free.

**Ridge alpha is tuned once per feature variant** on validation days, not by inner CV inside
a k=2 calibration set (80 windows). All four ridge conditions share it.

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
- **k=20 is the whole pool**, so all 10 draws at that k are the same set and averaging over
  draws changes nothing. Five sessions also have a repeated k=2 draw in `splits.json`.
