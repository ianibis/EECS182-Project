# 10x PARRM artifact dataset

These are altered versions of the recorded 3 Hz and 20 Hz AM sessions. Only the 35 channels kept for decoding are changed. For each sample:

    LFP_10x = LFP_recorded + 9 * PARRM_artifact

The original exports, trial times, and velocity labels are unchanged. The 10x refers to the estimated artifact, not the stimulation current. This is an artificial stress condition, not a stronger-stimulation experiment.

The validation sweep selected 10x as the largest tested scale. On the five OFF test days, the GRU ensemble gave R² = 0.1975. On test stimulation sessions, the recorded and 10x results were 0.0895 and 0.0615 at 3 Hz, and 0.1086 and 0.0805 at 20 Hz. These are session means, not pooled-bin scores.

The generator writes `semi_synthetic/parrm_amp10/h5_export_outputs/<session>.h5`, `metadata.csv`, and two feature caches under `features/task3_parrm_amp_s0010p000/` and `features/task3_parrm_amp_hf_s0010p000/`.

For each ON session, `metadata.csv` records the original export, PARRM file, generated recording, feature files, artifact scale, and reconstruction check. File paths are relative to the data root.

Do not replace the original recordings. Use `splits.json` to select training, calibration, and query trials. Normalize features on the first 20 calibration trials for validation and test sessions; the remaining 30 trials are held out for scoring. The 10x files are ON-session evaluation data. They are not the template-shifted OFF training episodes proposed for the in-context model.

The current PARRM-cleaned recordings were computed from the original ON data. To evaluate PARRM against the amplified 10x recording itself, run PARRM again on the amplified signal.
