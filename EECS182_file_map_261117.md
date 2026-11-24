# Project files: original and semi-synthetic data

## Original and semi-synthetic file correspondence

| Original file | Semi-synthetic equivalent | Relationship |
|---|---|---|
| `export_outputs/h5_export_outputs/<session>.h5` | `semisynthetic/parrm_amp10/h5_export_outputs/<session>.h5` | Same stimulation-ON recording, with the estimated PARRM artifact amplified to 10× on the 35 selected channels. |
| `parrm_outputs/parrm_h5/<session>.h5` | No separate equivalent; input to the generator | Supplies the estimated artifact used to modify the original recording. |
| `features/export_none/<session>.h5` | `features/task3_parrm_amp_s0010p000/<session>.h5` | Same LMP and seven band-power features, recomputed from the modified LFP. |
| Original 200–400 Hz feature for the corresponding recording | `features/task3_parrm_amp_hf_s0010p000/<session>.h5` | 200–400 Hz feature recomputed from the modified LFP. The existing `features/hf200_400_off/` cache contains OFF-session features, so it is **not** a direct ON-session equivalent. |

Trial files, kinematics, movement labels, and session splits remain unchanged.

## GitHub

| Path | Purpose |
|---|---|
| `task3/task3_common.py` | Shared paths and configuration for the existing ridge scripts. |
| `task3/task3_gru_core.py` | GRU model, feature loading, calibration, and scoring. |
| `task3/task3_gru_results.py` | Runs GRU comparisons and saves results. |
| `task3/task3_README.md` | Instructions for Task 3. |
| `semisynthetic/EECS182_make_semisynthetic_h5_v1.py` | Generates 10× artifact-amplified HDF5 recordings and features. |
| `semisynthetic/semi_synthetic_README.md` | Describes the semi-synthetic dataset. |
| `semisynthetic/semi_synthetic_metadata_schema.csv` | Defines the generated metadata columns. |

Keep the existing ridge scripts, shared `EECS182_NHP_data_loader.py` and `EECS182_NHP_features.py`, and project README. Commit small result CSVs after checking them; do not commit recordings, feature caches, or checkpoints.

## Box

| Path | Purpose |
|---|---|
| `semisynthetic/parrm_amp10/h5_export_outputs/` | 48 artifact-amplified stimulation-ON HDF5 recordings. |
| `semisynthetic/parrm_amp10/metadata.csv` | Source session, split, scale, and output paths. |
| `semisynthetic/parrm_amp10/README.md` | Dataset documentation stored beside the recordings. |
| `features/task3_parrm_amp_s0010p000/` | 48 feature files with LMP and seven power bands. |
| `features/task3_parrm_amp_hf_s0010p000/` | 48 files with the additional 200–400 Hz feature. |

## Decoder inputs and output paths

`TASK3_DATA_ROOT` points to the Box project root. Box paths below retain their
original capitalization. The checkout can also use the repository-style data
tree `data/exports/{export_h5_LFPs,export_trials}/`,
`data/parrm/parrm_h5_LFPs/`, `data/kinematics_bins50ms/bins50ms/`, and
`data/features/` when populated externally.

| Box path | Required input |
| --- | --- |
| `EECS182_data_manifest.csv`, `splits.json` | Sessions, days, trial split, and same-day OFF mapping |
| `export_outputs/trials/`, `kinematics_v3/bins50ms/` | Trial events and velocity labels |
| `features/export_none/`, `features/parrm_none/` | Base features for recorded and cleaned recordings |
| `features/hf200_400_off/`, `features/task3_parrm_amp_hf_s0001p000/` | OFF and recorded ON high-band features |
| `features/task3_parrm_amp_s0010p000/`, `features/task3_parrm_amp_hf_s0010p000/` | Existing 10× features |
| `semisynthetic/parrm_amp10/metadata.csv`, `semisynthetic/parrm_amp10/h5_export_outputs/` | 10× metadata and recording-presence checks |
| `Task4/results/feature_cache/` | Optional archived cleaned high-band features. GRU evaluation can build missing high-band caches locally from `parrm_outputs/parrm_h5/`. |
| `Task4/results/mlp/recorded_high_band/` | Recorded ON training high-band cache when absent from `features/` |
| `results/task3/checkpoints/` | Seed-1 multitask GRU and seed-2 through seed-5 epoch-7 ensemble checkpoints |
| `results/task3/task3_gru_full_finetune_v1_pool_cv.csv`, `results/task3/task3_gru_temporal_adapt_v1_pool_cv.csv` | Saved calibration choices, also supplied as small release CSVs |
| `Task4/results/mlp/{off,on_1x,on_10x}/checkpoint.pt` | Archived MLPs. The OFF-only entry point uses the first checkpoint. |

New GRU files are written to `task3_results/gru/`. New MLP files are written to
`task4_results/mlp/`. GitHub receives the code, guides, and selected small
CSV summaries. Keep raw recordings, trial/kinematics datasets, derived binary
caches, model checkpoints, and temporary files in external storage. The
existing artifact generator is outside this release. This release reads the
established artifact data and does not regenerate it.
