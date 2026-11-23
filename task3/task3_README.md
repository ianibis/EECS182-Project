# Task 3: movement decoding during stimulation

This package contains the ridge scripts and a shorter GRU evaluation pipeline for the Butters LFP dataset. The main task is to compare movement-velocity decoding on stimulation-OFF and stimulation-ON recordings, with and without PARRM cleaning. The 10x artifact-amplified ON recordings are a separate test condition.

## Files and runs

| Script | Use |
| --- | --- |
| `Task3/task3_1_ridge.py` | Linear OFF-to-ON benchmark (Task 3.1) |
| `Task3/task3_1_ridge_semisynthetic.py` | Ridge on recorded, PARRM-cleaned and 10x ON features |
| `Task3/task3_1_ridge_gru_matched.py` | Ridge with the GRU's 9 features, 19 lags and same-day calibration on 20 OFF trials |
| `Task3/task3_make_amp10_features.py` | 10x ON feature caches from the local `data/` layout |
| `Task3/task3_2_parrm.py` | Ridge on recorded versus PARRM-cleaned ON features (3.2) |
| `Task3/task3_3_stif.py` | Stimulation monitor-only control (3.3) |
| `Task3/task3_1_xgboost.py` | Optional tree-model comparison |
| `Task3/task3_gru_results.py` | GRU OFF/ON, PARRM, and 10x comparisons (3.1–3.2) |
| `Task3/task3_gru_core.py` | Feature loading, saved model, calibration, and scoring |
| `Task3/task3_make_semisynthetic_h5_v1.py` | Generate 10x ON recordings and feature caches |
| `EECS182_NHP_data_loader.py`, `EECS182_NHP_features.py` | Shared recording and feature readers |

The GRU runner loads five pretrained checkpoints. It does not train the models from scratch. Its inputs are 35 electrodes with nine features each (LMP, seven original bands, and 200–400 Hz power). Each trial has 19 warm-up bins followed by 40 scored 50-ms bins. The saved GRUs have a 48-unit projection and 64 hidden units. They are fully fine-tuned on 20 same-day OFF trials, and a time-dependent correction is fitted on those trials. The other 30 OFF trials and all ON query trials are scored without using their labels for adaptation.

From the project root:

```bash
python -m pip install -r requirements.txt
python Task3/task3_gru_results.py --self-test
python Task3/task3_make_semisynthetic_h5_v1.py --self-test
python Task3/task3_make_semisynthetic_h5_v1.py --split val-test
python Task3/task3_gru_results.py --split test --dataset all
```

The GRU script writes `results/task3_gru/task3_gru_test_sessions.csv`, `task3_gru_test_off.csv`, `task3_gru_test_summary.csv`, and `task3_gru_test_parrm.csv`. Use `--split val` to check the validation sessions. The output distinguishes recorded, PARRM-cleaned, and 10x ON inputs and reports matched same-day OFF scores. Missing original or PARRM feature caches are computed from the corresponding recording and saved. The 10x feature caches must first be produced by the generator.

The five GRU checkpoint files and the two original calibration CV CSVs must be present under `results/task3/`. They are deliberately not included here. They were generated earlier on the lab machine; see `docs/file_map.md`. The ridge scripts select their own linear hyperparameters and do not use the GRU checkpoint. Their default 9-lag ridge result is **not** the same experiment as the later 19-lag residual ridge result reported in the project notes.

## Data location

The default layout is the Box project folder:

```text
EECS282_finalProject/
  EECS182_data_manifest.csv
  splits.json
  export_outputs/h5_export_outputs/<session>.h5
  export_outputs/trials/<session>_trials.csv
  parrm_outputs/parrm_h5/<session>.h5
  features/export_none/<session>.h5
  features/parrm_none/<session>.h5
  kinematics_v3/bins50ms/<session>_bins50ms.csv
  results/task3/checkpoints/*.pt
```

If the code is in a separate GitHub checkout, point the GRU runner to the existing Box folder:

```bash
python Task3/task3_gru_results.py --root "/Users/io/Library/CloudStorage/Box-Box/EECS282_finalProject" --split test --dataset all
```

The ridge scripts read the data root from the `TASK3_DATA_ROOT` environment variable. Set it to the Box folder before running ridge or GRU. Results are written to the local repository.

## GitHub

Commit the Python files, READMEs, manifest and splits (if sharing is permitted), and small result CSVs. Do **not** commit the continuous LFP exports, kinematics recordings, PARRM outputs, or checkpoint files without lab approval. The supplied `.gitignore` excludes large binaries by default.

The check workflow runs syntax and synthetic tests on a GitHub-hosted runner. The results workflow requires a **self-hosted macOS runner** that can read the Box folder. Set the repository variable `TASK3_DATA_ROOT` to that folder. From the GitHub **Actions** tab, select **Run Task 3 GRU**, choose the split and dataset, and run it. The workflow uploads the result CSVs and commits those CSVs to the branch. It cannot run the real data analysis on a standard GitHub runner unless the data and checkpoints are made accessible there.

The new code was checked against the original GRU model on synthetic arrays: forward outputs, weight updates during full fine-tuning, and the temporal correction were numerically identical. A synthetic two-session data test also produced the expected result CSVs. The complete pipeline has **not** been run against the actual lab dataset in this environment, so compare its first real run against the previous 0.1975 OFF test R² and the saved 3/20 Hz ON results.

## Local setup: code in GitHub, data in Box

From the local GitHub checkout, set the Box path (replace this example if yours differs):

```bash
export TASK3_DATA_ROOT="/Users/io/Library/CloudStorage/Box-Box/EECS282_finalProject"
python Task3/task3_gru_results.py --root "$TASK3_DATA_ROOT" --split test --dataset all
python Task3/task3_1_ridge.py
python Task3/task3_2_parrm.py
python Task3/task3_3_stif.py
```

The GRU runner writes CSVs to `results/task3_gru/` in the local GitHub checkout. The ridge, XGBoost and StiF scripts write CSVs to `task3_results/`, which is committed. Without `TASK3_DATA_ROOT` they read a local copy of the data from `data/` in the repository. No HDF5 recordings, feature caches, or checkpoints belong in Git.

The generator was already run on the Box data; it need not be run again. To document the generated dataset, copy `semi_synthetic_README.md` from this repository to the Box folder as `semi_synthetic/parrm_amp10/README.md`. The generated `metadata.csv` is different from `semi_synthetic_metadata_schema.csv`, which is an explanatory field list.

The runner does not retrain the five base GRUs and does not perform a k-labeled-ON-trial in-context comparison. It adapts on 20 same-day OFF trials and scores the 30 held-out query trials. `task3/task3_1_ridge_semisynthetic.py` scores the ridge decoder on the recorded, PARRM-cleaned and 10x ON recordings. It reads 10x features from `features/parrm_amp10/`, made by `task3/task3_make_amp10_features.py`.
