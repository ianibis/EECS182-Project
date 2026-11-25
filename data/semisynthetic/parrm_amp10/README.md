# PARRM artifact amplification: 10×

These are modified versions of the stimulation-ON recordings from the Butters dataset. The 3 Hz and 20 Hz AM conditions use the same transformation. No movement labels or trial timings were changed.

For the 35 fixed channels used by the decoder, each sample is computed as:

    modified LFP = recorded LFP + 9 × PARRM-estimated artifact

The 10× figure refers to the estimated artifact component, not the stimulation current or the full LFP. The other channels in each 64-channel export are left as recorded. The files have the same datasets and metadata as the original exports, plus attributes documenting the modification.

The scale was chosen after a validation sweep. Relative to the recorded stimulation-ON signals (1×), 10× lowered mean test velocity R² by about 0.028 for both 3 Hz and 20 Hz AM with the fixed GRU. This is a semi-synthetic challenge condition, not a recording from stronger stimulation.

## Files

- `h5_export_outputs/<session>.h5`: modified stimulation-ON recordings; unchanged HDF5 layout.
- `metadata.csv`: source files, split, stimulation frequency, scale, and reconstruction check for each session.
- `features/task3_parrm_amp_s0010p000/`: LMP and seven 1–200 Hz power bands from the modified recordings.
- `features/task3_parrm_amp_hf_s0010p000/`: additional 200–400 Hz power for the GRU.

The last two paths are relative to the project root, not this README. The CSV contains the actual project-relative file paths.

## Using the files

Keep `splits.json`, `EECS182_data_manifest.csv`, trial CSVs, and `kinematics_v3/bins50ms` unchanged. Use the synthetic HDF5 only for stimulation-ON sessions. Stimulation-OFF sessions continue to use `export_outputs/h5_export_outputs`.

The current data loader does not automatically search both directories. Resolve each session path before loading it:

```python
from pathlib import Path

def recording_path(root, session):
    root = Path(root)
    modified = root / "semi_synthetic/parrm_amp10/h5_export_outputs" / f"{session}.h5"
    if modified.is_file():
        return modified
    return root / "export_outputs/h5_export_outputs" / f"{session}.h5"
```

For comparisons with the original GRU, use the fixed 35-channel mask and the existing calibration/query trial split. The feature caches contain unnormalized values; normalization must be fitted from the calibration pool, not the query labels.

Do not treat the original `parrm_outputs` files as the result of applying PARRM anew to these modified recordings. Re-run PARRM on the modified input if that specific baseline is needed.

This dataset is separate from the planned shift generator that adds stimulation templates to OFF trials for in-context model training. It provides modified real stimulation-ON sessions for evaluation, not new OFF-derived training episodes.
