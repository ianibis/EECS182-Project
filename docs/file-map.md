# Project files: original and semi-synthetic data

## Original and semi-synthetic file correspondence

| Original file | Semi-synthetic equivalent | Relationship |
|---|---|---|
| `export_outputs/h5_export_outputs/<session>.h5` | `semi_synthetic/parrm_amp10/h5_export_outputs/<session>.h5` | Same stimulation-ON recording, with the estimated PARRM artifact amplified to 10× on the 35 selected channels. |
| `parrm_outputs/parrm_h5/<session>.h5` | No separate equivalent; input to the generator | Supplies the estimated artifact used to modify the original recording. |
| `features/export_none/<session>.h5` | `features/parrm_amp10/<session>.h5` | Same LMP and seven band-power features, recomputed from the modified LFP. |
| Original 200–400 Hz feature for the corresponding recording | `features/parrm_amp10_hf/<session>.h5` | 200–400 Hz feature recomputed from the modified LFP. The existing `features/hf200_400_off/` cache contains OFF-session features, so it is **not** a direct ON-session equivalent. |

Trial files, kinematics, movement labels, and session splits remain unchanged.

## GitHub

| Path | Purpose |
|---|---|
| `task3/task3_common.py` | Shared paths and configuration for the existing ridge scripts. |
| `task3/task3_gru_core.py` | GRU model, feature loading, calibration, and scoring. |
| `task3/task3_gru_results.py` | Runs GRU comparisons and saves results. |
| `task3/task3_README.md` | Instructions for Task 3. |
| `semisynthetic/EECS182_make_semisynthetic_h5.py` | Generates 10× artifact-amplified HDF5 recordings and features. |
| `semisynthetic/semi_synthetic_README.md` | Describes the semi-synthetic dataset. |
| `semisynthetic/semi_synthetic_metadata_schema.csv` | Defines the generated metadata columns. |
| `task4/eval_amp10.py` | Scores the task 4.2 EEGNet checkpoints on the 10× recordings. |

## Which code reads which variant

| Variant | Path under the data root | Read by |
|---|---|---|
| Recorded export | `export_outputs/h5_export_outputs/<session>.h5` | `EECS182_NHP_data_loader`, task 4.2 LFP, generator input |
| PARRM-cleaned | `parrm_outputs/parrm_h5/<session>.h5` | `task3_common`, `task3_gru_core`, generator input |
| Raw features | `features/export_none/<session>.h5` | task 3.1–3.3, task 4.3, task 4 GRU baseline |
| PARRM features | `features/parrm_none/<session>.h5` | task 3.2, task 4.3 |
| 10× recordings | `semi_synthetic/parrm_amp10/h5_export_outputs/<session>.h5` | `task4/utils.semisynthetic_paths`, `task4/eval_amp10.py` |
| 10× features | `features/parrm_amp10/`, `features/parrm_amp10_hf/` | `task3_gru_core` (`--dataset amp10`) |

The 10× recordings are `recorded + 9 × PARRM_artifact` on the 35 decoded channels; see
`semisynthetic/semi_synthetic_README.md`. They are not the synthetic 3/20 Hz tones used to
augment task 4.2 training (`task4/utils.add_stimulation_tones`).

Keep the existing ridge scripts, shared `EECS182_NHP_data_loader.py` and `EECS182_NHP_features.py`, and project README. Commit small result CSVs after checking them; do not commit recordings, feature caches, or checkpoints.

## Box

| Path | Purpose |
|---|---|
| `semi_synthetic/parrm_amp10/h5_export_outputs/` | 48 artifact-amplified stimulation-ON HDF5 recordings. |
| `semi_synthetic/parrm_amp10/metadata.csv` | Source session, split, scale, and output paths. |
| `semi_synthetic/parrm_amp10/README.md` | Dataset documentation stored beside the recordings. |
| `features/parrm_amp10/` | 48 feature files with LMP and seven power bands. |
| `features/parrm_amp10_hf/` | 48 files with the additional 200–400 Hz feature. |
| `results/task3_gru/` | Recorded, PARRM and 10× GRU scores (`--dataset all`). |

Generate the 10× files with

```bash
python semisynthetic/EECS182_make_semisynthetic_h5.py --root <data root> --split val-test --h5-only
```

`--root` points at the folder holding `export_outputs/` and `parrm_outputs/`; `splits.json`
and the manifest are found in that folder or its parent.



## Task 4

Code and outputs live together under `task4/`.

| Path | Purpose |
|---|---|
| `task4/utils.py` | Shared data loaders, metrics, training loop, decoders, draws sampler and reporting helpers. |
| `task4/task4_1_draws.ipynb` | Task 4.1: sample k-shot ON draws with a fixed 30-trial query (blocks 3-5) and write the draws file. |
| `task4/task4_2_eegnet.ipynb` | Task 4.2: braindecode EEGNet adapted to regression, early stopping, 5 seeds vs ridge 3.1. |
| `task4/task4_3_parrm_ridge.ipynb` | Task 4.3: PARRM cleaning then ridge refit on each draw. |
| `task4/make_draws.py` | CLI wrapper over `utils.build_draws` for the 4.1 draws. |
| `task4/compare_decoders.py` | Rebuilds the cross-model tables, session CIs, paired differences and the checkpoint audit. |
| `task4/sweep_gru.py` | Twelve-setting GRU sweep (width, depth, dropout) ranked by clean validation R2. |
| `task4/train_gru_final.py` | Retrains five seeds at the sweep winner and scores every stimulated test session. |
| `kaggle/train_gru.py` | The same sweep and five-seed run, packaged for a GPU kernel. |
| `kaggle/build_gru_kernel.py` | Writes `kaggle/kernel_gru/` (notebook + metadata) from `train_gru.py`. |
| `task4/results/gru_sweep.csv` | Sweep output: 12 settings with validation R2 and parameter counts. |
| `task4/results/task4_gru_final.csv` | Per session and seed plus the ensemble, for the retrained GRU. |
| `task4/results/task4_gru_final_seeds.csv` | Validation R2 and stopping epoch per seed. |
| `task4/eval_amp10.py` | Scores the 4.2 checkpoints on the recorded, 10×, PARRM-re-estimated and PARRM-reuse inputs. |
| `task4/make_parrm_amp10.py` | Builds the reused-template 10× files and verifies `10x - 10*artifact = parrm(recorded)`. |
| `task4/characterise_amp10_noise.py` | Variance of the added 10× term: DC, AC, phase split, harmonics, bin-averaged RMS. |
| `docs/decoder-comparison.md` | GRU versus EEGNet: paired CIs, the 3 Hz-specific collapse, and the checkpoint audit. |
| `reports/task4_writeup.tex` | One-page paper section (model, loss, sweep, results) plus appendices A--H; compiles standalone, and is written so the section can be `\input` into the main paper. |
| `task4/results/draws.json` | 4.1 output (v3, fixed query): 14 test ON sessions × k∈{2,5,10,20}; 10 independent subsets for k<20, one full-pool draw for k=20; 434 partitions. |
| `task4/results/draws_fixed_query_legacy.json` | Archived v1 fixed-query draws from the earlier baseline run (10 draws at k=20); not read. |
| `task4/results/` | Task 4 metric tables (`task4_1_draws_summary.csv`, `task4_2_*.csv`, `task4_3_*.csv`). |
| `task4/checkpoints/` | Task 4 model weights. |
| `task4/training_logs/` | Per-epoch training logs. |
| `task4/figures/` | Task 4 figures. |
