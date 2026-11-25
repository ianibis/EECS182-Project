# Semi-synthetic 10x artifact dataset

## What this dataset is

Altered copies of the recorded 3 Hz and 20 Hz AM stimulation-ON sessions. Only the 35
channels kept for decoding are changed, and only the `/lfp` dataset is rewritten:

    LFP_10x = LFP_recorded + (scale - 1) * PARRM_artifact,   scale = 10

The original exports, trial times, kinematics and velocity labels are unchanged, and every
other HDF5 dataset and attribute is copied verbatim. The "10x" refers to the estimated
artifact, not to the stimulation current: this is an artificial stress condition, not a
stronger-stimulation experiment.

Because `recorded = cleaned + artifact` holds to within 3 % relative RMS, the result is
close to `clean + 10 * artifact` — the same session and the same labels, with ten times the
estimated artifact. The generator checks that identity on every session and stores the
relative RMS error in `metadata.csv`.

The validation sweep selected 10x as the largest tested scale. On the five OFF test days,
the GRU ensemble gave R2 = 0.1975. On test stimulation sessions, the recorded and 10x
results were 0.0895 and 0.0615 at 3 Hz, and 0.1086 and 0.0805 at 20 Hz. These are session
means, not pooled-bin scores.

## This is not the sinusoid

Two different things in this repository are called "synthetic". They are unrelated:

| | Semi-synthetic 10x (this folder) | Synthetic frequencies |
|---|---|---|
| Added signal | `(scale - 1) x PARRM_artifact` | `strength x gain x sin(2 pi f t + phase)` |
| Waveform | measured, stimulation-locked artifact estimate | pure sine |
| Applied to | `/lfp` of stimulation-ON **recordings**, written to disk | stimulation-OFF **training** trials, in memory |
| Purpose | harder artifact at test time | augmentation so a model learns to ignore a frequency |
| Code | `semisynthetic/EECS182_make_semisynthetic_h5.py` | `task4/utils.py`, `add_synthetic_frequencies` |
| Consumers | `task3/task3_gru_results.py --dataset amp10`, `task4_2_eegnet.ipynb` (arms cell) | `task4_2_eegnet.ipynb` section 4.2.6 only |

The artifact is periodic at the AM rate — one period per 50 ms at 20 Hz, per ~333 ms at
3 Hz — but it is the *measured* residual: harmonics, amplitude drift and the 35-channel
spatial pattern all come from the recording. A sine has none of that, which is why the 10x
condition is the more faithful stress test.

## Where the code is

| Piece | Location |
|---|---|
| Recording generator | `semisynthetic/EECS182_make_semisynthetic_h5.py`, `make_recording` |
| Feature caches from the same LFP | `make_features` in the same file |
| Session selection | `select_sessions` (reads `splits.json` and the manifest) |
| Project-file lookup | `project_file` (data root, then its parent) |
| Round-trip check | `--self-test`, which rebuilds two toy sessions in a temp folder |
| GRU consumer | `task3/task3_gru_core.py` (`feature_variant`, `high_band_path`, `prepare_features`) |
| EEGNet consumer | `task4/utils.py` (`semisynthetic_paths`, `amp10_test_datasets`), called from the 4.2 notebook |

### Generating

```bash
python semisynthetic/EECS182_make_semisynthetic_h5.py --root data --split val-test --h5-only
```

`--root` is the data root, the folder that holds `export_outputs/` and `parrm_outputs/`.
In the Box layout that is the project folder itself; in this repository the recordings live
under `data/`, so `--root data` is required. `splits.json` and the manifest are looked up in
the data root and then in its parent, which covers both layouts.

Useful flags: `--split val-test|train|val|test|all`, `--h5-only` (skip the feature caches),
`--overwrite`, `--tolerance` (relative RMS error allowed in `raw = clean + artifact`),
`--self-test`. Only `--scale 10` is accepted; this release packages the validated condition.

### Outputs

`<root>/semi_synthetic/parrm_amp10/`:

- `h5_export_outputs/<session>.h5` — the modified ON recordings, unchanged HDF5 layout,
  stamped with `semi_synthetic_method`, `semi_synthetic_artifact_scale`,
  `semi_synthetic_formula`, `semi_synthetic_channels` and
  `semi_synthetic_reconstruction_relative_rms_error`;
- `metadata.csv` — one row per session: split, condition, AM frequency, matched OFF
  reference, scale, modified/total channels, reconstruction error, and the source and
  output paths relative to the data root;
- `README.md` — this documentation, copied next to the recordings.

With the feature caches enabled (no `--h5-only`), also
`<root>/features/parrm_amp10/<session>.h5` (LMP plus seven band powers) and
`<root>/features/parrm_amp10_hf/<session>.h5` (200-400 Hz power).

## Using the files

Keep `splits.json`, `EECS182_data_manifest.csv`, the trial CSVs and
`kinematics_v3/bins50ms` unchanged. Stimulation-OFF sessions always keep using
`export_outputs/h5_export_outputs`. Resolve each session path before loading:

```python
from pathlib import Path

def recording_path(root, session):
    root = Path(root)
    modified = root / "semi_synthetic/parrm_amp10/h5_export_outputs" / f"{session}.h5"
    if modified.is_file():
        return modified
    return root / "export_outputs/h5_export_outputs" / f"{session}.h5"
```

Do not replace the original recordings. Use `splits.json` to select training, calibration
and query trials. The 10x files are ON-session evaluation data. They are not the
template-shifted OFF training episodes proposed for the in-context model.

## Characterising the added term as 0-centred noise

The variance characterisation behind `task4_results/amp10_noise_variance.csv` treats the added term

    n_c(t) = (scale - 1) * artifact_c(t) = 9 * artifact_c(t)

as a noise term, removes each channel's mean, and reports what is left. It runs on the
stimulation-on samples only, since the artifact estimate is zero elsewhere. Outputs are
`task4_results/amp10_noise_variance.csv` (per session), `amp10_noise_summary.csv` (median
per AM frequency) and `results/fig_amp10_noise.png`; both CSVs are under `task4_results/`.

Medians over the seven test sessions at each frequency:

| Statistic | 3 Hz | 20 Hz | What it means |
|---|---:|---:|---|
| DC RMS, before centring | 3.0 | 2.0 | small; any intercept absorbs it |
| AC RMS, after centring | 43.4 | 20.2 | the part that cannot be absorbed |
| fraction of AC variance explained by AM phase | 0.12 | 0.42 | the estimate is not a clean periodic template |
| fraction of AC variance at k·f0, k ≤ 10 | 0.07 | 0.06 | narrowband content is a small share |
| share of AC variance in the five worst channels | 0.47 | 0.47 | concentrated in about a seventh of the array |
| **AC variance retained by 50 ms binning** | **0.56** | **0.21** | what survives into the decoder's input |
| lag-1 autocorrelation of the binned term | 0.35 | 0.98 | oscillatory at 3 Hz, a slow drift at 20 Hz |
| **RMS of the binned term** | **32.7** | **9.7** | comparable to the neural signal at 3 Hz only |
| SNR of the binned term vs the cleaned LFP | −0.1 dB | −10.2 dB | 10 dB apart |
| MAD ratio, 10x over recorded | 1.92 | 1.29 | how much normalisation inflates the scale |

Read together, the term is **not** white and **not** Gaussian, so a single sigma would be
right about the magnitude and wrong about the structure:

- After centring, only 12 % (3 Hz) to 42 % (20 Hz) of the variance is explained by the AM
  phase. The rest is cycle-to-cycle amplitude change: PARRM's estimate is a per-window gain
  times a template, and the gain drifts.
- Almost none of it sits in the first ten harmonics (6–7 %). It is broadband on the scale of
  the recording, which is why a "narrowband interference" picture of the artifact is wrong.
- The half that matters is what survives the 50 ms binning the decoders see. At 20 Hz one
  artifact period equals one bin, so binning removes 79 % of the variance and leaves a
  lag-1 correlation of 0.98 — a slow drift. At 3 Hz a bin covers 1/6.7 of a cycle, so 56 %
  survives and stays oscillatory (lag-1 0.35).
- That statistic tracks the failure. Per session, the RMS of the binned term correlates with
  EEGNet's ON R² at Spearman −0.94 across all 14 test sessions, and with the GRU's at +0.10.
  The GRU reads band-power features with 9 bins of history, which is insensitive to a slow
  drift; EEGNet reads raw 1 kHz LFP per bin, so the surviving variance is signal-shaped
  interference it cannot average away.

A note on the 20 Hz outlier: session 250616-102124 has the largest raw AC RMS of the set
(91) yet a binned fraction of only 0.018, so almost none of it reaches the decoder and its
EEGNet score stays near zero. Raw amplitude alone would have predicted the wrong answer;
the binned variance is the quantity to use.

## PARRM applied to the amplified signal

`semi_synthetic/parrm_amp10_reuse/parrm_h5/` holds the 10x recording cleaned with the
*existing* 1x template (`10x - artifact = recorded + 8 * artifact`), so the normal
`source='parrm'` loader reads it. Those files were built and verified against the
complementary identity `10x - 10 * artifact = parrm(recorded)`; the worst relative RMS error
is 3.7e-8, so re-estimating PARRM on the amplified signal returns the recorded cleaned LFP and
needs no new file. Task 4.2.10 scores both arms: re-estimating rescues the fixed EEGNet,
reusing the 1x template does not.

## Limits worth repeating

- **Normalisation moves with the artifact.** Both `EECS182_NHP_data_loader.normalize` and
  `EECS182_NHP_features.load_features` z-score by a median/MAD over the task interval, so a
  larger artifact also inflates the scale and shrinks the neural signal in z-units. A 10x
  score mixes "harder artifact" with "lower effective SNR". Task 3's GRU used the same
  convention, so its 10x numbers and task 4.2.10's are comparable.
- **Broadband RMS does not grow tenfold.** The artifact is narrowband and already present in
  the recording, so on the kept channels the 10x LFP's standard deviation is roughly
  1.1-1.5x the recorded one even though the artifact term itself is 10x. Judge the
  condition at the artifact frequencies, not by total variance.
- **Two PARRM arms, and only one needs building.** The `parrm_none` feature caches were
  computed from the original ON data, so they are not 10x inputs. For the amplified
  recording, the reused-template arm lives in
  `semi_synthetic/parrm_amp10_reuse/parrm_h5/`; the re-estimated arm needs no new file
  because `10x - 10 * artifact` returns the recorded cleaned LFP. The 200-400 Hz companion
  cache is only built for the feature pipeline, so a power-band decoder on the 10x
  recording also needs `--h5-only` dropped from the generation command.
- **Coverage.** `--split val-test` produces the 20 ON sessions in the validation and test
  splits; `--split all` produces all 48.
