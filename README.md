# Recalibrating a brain decoder under epidural stimulation

Decoding fingertip velocity from motor-cortex LFP while epidural amplitude-modulated
stimulation is on. Five baselines, one target, one fixed query set.

Target: `Index_vx_cm` and `Index_vy_cm`, in 50 ms bins. Input: 35 electrodes × 8 features
per bin (LMP plus 7 band powers), with 10 bins of history.

## Dataset

One rhesus macaque (Butters), one 64-channel array, 70 recording sessions on 26 days
(2025-05-20 to 2025-07-11); 35 of the 64 channels are used for decoding. Every session has
50 rewarded trials in 5 blocks of 10. In stimulation sessions the stimulator runs during
each block and is off during the breaks, so every rewarded trial is entirely
stimulation-on or entirely stimulation-off.

| Condition | Sessions | Rewarded trials |
|---|---:|---:|
| No stimulation | 22 | 1,100 |
| 3 Hz AM (3.5 kHz carrier, 1.75 mA) | 22 | 1,100 |
| 20 Hz AM (3.5 kHz carrier, 1.75 mA) | 26 | 1,300 |
| Total | 70 | 3,500 |

The task is sequential target acquisition: hold a button, wait for the go cue, touch five
targets in a fixed order, collect juice. Two fixed orders are used — `1-9-4-3-7` (36
sessions) and `3-7-6-1-9` (34 sessions); the `sequence` column of `manifest.csv` says which.
Target codes are phone-keypad positions on a 3 × 3 grid given in screen cm.
`docs/DATASET.md` (local only) has the coordinate table, the HDF5 and trial-table layouts,
the channel list, and the PARRM and kinematics details.

`EECS182_data_manifest.csv` has one row per session with counts, quality measures and known
problems; `conditions_eACS_AM-kHz_v2.csv` covers all 102 candidate sessions with inclusion
decisions and exclusion reasons.

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
`docs/baseline-results.md`, which is kept locally only.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
jupyter lab task4/
```

Task 4 runs as three notebooks under [task4/](task4), one per sub-task, sharing
[task4/utils.py](task4/utils.py); see [task4/README.md](task4/README.md). Task 3 scripts read the
data root from `TASK3_DATA_ROOT`; see [task3/task3_README.md](task3/task3_README.md).

The 10× semi-synthetic recordings (`recorded + 9 × measured PARRM artifact`) are built by
[semisynthetic/EECS182_make_semisynthetic_h5.py](semisynthetic/EECS182_make_semisynthetic_h5.py)
and explained in [semisynthetic/semi_synthetic_README.md](semisynthetic/semi_synthetic_README.md).
They are not the synthetic 3/20 Hz frequencies injected to augment task 4.2 training.

```bash
TASK3_DATA_ROOT="$PWD/data" .venv/bin/python task3/task3_1_ridge.py --lags 9
```

## Layout

Results live in one folder per task: `task3_results/`, `task4_results/`,
`task5_results/`. Recordings and cached features live under `data/`; the ~15 GB of raw
exports are not committed, only the small task 5 artifacts and READMEs.

```text
task3/                          ridge, XGBoost, PARRM, StiF and GRU checks
task3_results/                  task 3 result CSVs, GRU training/calibration outputs
task4/                          calibration draws, EEGNet regression, PARRM ridge (task 4)
  utils.py                      shared data, metrics, training, decoders, draws
  task4_1_draws.ipynb           4.1 calibration draws
  task4_2_eegnet.ipynb          4.2 braindecode EEGNet regression decoder
  task4_3_parrm_ridge.ipynb     4.3 PARRM then ridge refit on each draw
  MLP/                          task 4 MLP decoder
task4_results/                  task 4 result CSVs, draws, sweeps
task5/                          task 5 simulation, calibration and baseline scripts
task5_results/                  task 5 result CSVs
EECS182_NHP_data_loader.py      session, LFP and kinematics reader
EECS182_NHP_features.py         LMP and band-power features
EECS182_data_manifest.csv       70 rows, one per session
conditions_eACS_AM-kHz_v2.csv   102 candidate sessions with inclusion decisions
splits.json                     train / val / test days, calibration subsets
semisynthetic/                  10x artifact-amplified recordings generator and notes
scripts/                        figures, diagrams and parameter sweeps
kaggle/                         Kaggle training and kernel bundles
data/                           recordings, features, task 5 checkpoints and predictions
reports/                        local only: writeups, figures, HTML reports
docs/                           local only: proposal, spec, dataset, notes
```

Data layout, all under `data/`:

```text
export_outputs/h5_export_outputs/   <session>.h5, continuous LFP
export_outputs/trials/              <session>_trials.csv
parrm_outputs/parrm_h5/             PARRM-cleaned LFP
parrm_outputs/template_library.h5   PARRM templates
features/<variant>/                 precomputed features per session
kinematics_v3/bins50ms/             50 ms kinematics
task5_runs/                         task 5 checkpoints and run logs (committed)
task5_predictions/                  task 5 per-trial predictions (committed)
```

## Reports and docs (local only)

The writeups and figures are not part of this repository, so the paths below only resolve
on a machine that has them. `scripts/` regenerates the figures from `data/` and the
`task*_results/` folders.

```text
reports/baselines/index.html            every baseline on one page, with figures
reports/model_diagram.html              what goes in and what comes out, from a real trial
reports/model_architectures.html        per-baseline architecture, validation animation, sweeps

docs/proposal.md                        project proposal
docs/project-spec.md                    scope and method
docs/DATASET.md                         dataset handout: sessions, task, file formats
docs/baselines.md                       baseline design decisions
docs/baseline-results.md                full result tables
docs/task3-notes.md                     task 3 analysis
docs/file-map.md                        where every file came from
```
