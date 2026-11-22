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
| `Task3/task3_common.py` | Shared paths and configuration for the existing ridge scripts. |
| `Task3/task3_gru_core.py` | GRU model, feature loading, calibration, and scoring. |
| `Task3/task3_gru_results.py` | Runs GRU comparisons and saves results. |
| `task3/task3_README.md` | Instructions for Task 3. |
| `semisynthetic/EECS182_make_semisynthetic_h5_v1.py` | Generates 10× artifact-amplified HDF5 recordings and features. |
| `semisynthetic/semi_synthetic_README.md` | Describes the semi-synthetic dataset. |
| `semisynthetic/semi_synthetic_metadata_schema.csv` | Defines the generated metadata columns. |

Keep the existing ridge scripts, shared `EECS182_NHP_data_loader.py` and `EECS182_NHP_features.py`, and project README. Commit small result CSVs after checking them; do not commit recordings, feature caches, or checkpoints.

## Box

| Path | Purpose |
|---|---|
| `semi_synthetic/parrm_amp10/h5_export_outputs/` | 48 artifact-amplified stimulation-ON HDF5 recordings. |
| `semi_synthetic/parrm_amp10/metadata.csv` | Source session, split, scale, and output paths. |
| `semi_synthetic/parrm_amp10/README.md` | Dataset documentation stored beside the recordings. |
| `features/parrm_amp10/` | 48 feature files with LMP and seven power bands. |
| `features/parrm_amp10_hf/` | 48 files with the additional 200–400 Hz feature. |


