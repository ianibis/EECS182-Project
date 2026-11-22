# Recalibrating a brain decoder under epidural stimulation

Decoding fingertip velocity from motor-cortex LFP while epidural amplitude-modulated
stimulation is on. Five baselines, one target, one fixed query set.

Target: `Index_vx_cm` and `Index_vy_cm`, in 50 ms bins. Input: 35 electrodes × 8 features
per bin (LMP plus 7 band powers), with 10 bins of history.

## Results

Mean validation R² over the 30-trial query set of each held-out stimulation session.

| Method | 3 Hz AM | 20 Hz AM |
|---|---:|---:|
| Same-day stimulation-off ceiling | 0.048 | 0.028 |
| Ridge, no adaptation | 0.025 | 0.016 |
| Ridge, refit on 20 ON trials | 0.030 | 0.055 |
| KNN, k = 20 ON trials | 0.029 | 0.051 |
| MLP, 5 seeds | 0.039 | −0.032 |
| GRU, 5 seeds | 0.028 | 0.029 |
| EEGNet, 1 seed | −0.052 | −0.027 |

Ridge is weak even without stimulation, so the ceiling is low. Refitting on 20 labelled ON
trials recovers most of the 20 Hz loss; PARRM cleaning alone does not. Full tables are in
[docs/baseline-results.md](docs/baseline-results.md).

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
jupyter lab cs182_decoder_grouped.ipynb
```

The notebook trains nothing on load. Set `RUN_PIPELINE = True` in the last cell to retrain
every model and overwrite `results/`. Task 3 scripts read the data root from
`TASK3_DATA_ROOT`; see [task3/task3_README.md](task3/task3_README.md).

```bash
TASK3_DATA_ROOT="$PWD/data" .venv/bin/python task3/task3_1_ridge.py --lags 9
```

## Layout

Code and metadata only. `data/` (~15 GB) and `results/` stay local and are git-ignored.

```text
cs182_decoder_grouped.ipynb     five baselines, grouped by predictor
EECS182_NHP_data_loader.py      session, LFP and kinematics reader
EECS182_NHP_features.py         LMP and band-power features
EECS182_data_manifest.csv       70 rows, one per session
conditions_eACS_AM-kHz_v2.csv   102 candidate sessions with inclusion decisions
splits.json                     train / val / test days, calibration subsets
task3/                          ridge, XGBoost, PARRM and StiF checks
semisynthetic/                  10x artifact-amplified recordings generator
scripts/                        figures, diagrams and parameter sweeps
reports/                        generated HTML reports and figures
docs/                           proposal, spec, dataset, notes
```

Data layout, all under `data/`:

```text
export_outputs/h5_export_outputs/   <session>.h5, continuous LFP
export_outputs/trials/              <session>_trials.csv
parrm_outputs/parrm_h5/             PARRM-cleaned LFP
parrm_outputs/template_library.h5   PARRM templates
features/<variant>/                 precomputed features per session
kinematics_v3/bins50ms/             50 ms kinematics
```

## Reports

| File | Contents |
|---|---|
| [reports/baselines/index.html](reports/baselines/index.html) | Every baseline on one page, with figures |
| [reports/model_diagram.html](reports/model_diagram.html) | What goes in and what comes out, from a real trial |
| [reports/model_architectures.html](reports/model_architectures.html) | Per-baseline architecture, validation animation, sweeps |

## Docs

| File | Contents |
|---|---|
| [docs/proposal.md](docs/proposal.md) | Project proposal |
| [docs/project-spec.md](docs/project-spec.md) | Scope and method |
| [docs/DATASET.md](docs/DATASET.md) | Dataset handout: sessions, task, file formats |
| [docs/baselines.md](docs/baselines.md) | Baseline design decisions |
| [docs/baseline-results.md](docs/baseline-results.md) | Full result tables |
| [docs/task3-notes.md](docs/task3-notes.md) | Task 3 analysis |
| [docs/file-map.md](docs/file-map.md) | Where every file came from |
