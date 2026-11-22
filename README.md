# EECS 182 final project data: NHP (Butters), 64-channel LFP under AM epidural stimulation

This dataset contains motor cortex LFP recordings for decoding hand movement during epidural amplitude-modulated stimulation (eACS) and without stimulation. It covers 70 sessions from one rhesus macaque (Butters), recorded with a 64-channel array over 26 days (2025-05-20 to 2025-07-11). Each session includes LFP, stimulation signals, a trial table, and hand kinematics from DeepLabCut. Stimulation sessions also have a PARRM-cleaned copy of the LFP.

| Condition | Sessions | Rewarded trials used |
|---|---|---|
| No stimulation | 22 | 1,100 |
| 3 Hz AM (3.5 kHz carrier, 1.75 mA) | 22 | 1,100 (all stim on) |
| 20 Hz AM (3.5 kHz carrier, 1.75 mA) | 26 | 1,300 (all stim on) |
| Total | 70 | 3,500 |

Every session has 50 rewarded trials, with 10 trials in each of five task blocks. During stimulation sessions, the stimulator is on for the task blocks and usually off for the ~30 s breaks. Two sessions have stimulation continuing through one break. All rewarded trials are therefore entirely ON in stimulation sessions and entirely OFF in no-stim sessions. The session counts, quality measures, and known problems are in `manifest.csv`.

## Repository layout

Data stays local and git ignores it. Run notebooks and scripts from the repository root so `data/` and `results/` resolve.

| Path | Contents |
|---|---|
| `data/exports/export_h5_LFPs/` | Continuous LFP export per session |
| `data/exports/export_trials/` | One trial table per session |
| `data/parrm/parrm_h5_LFPs/` | PARRM-cleaned LFP for stimulation sessions |
| `data/features/export_none/`, `data/features/parrm_none/` | Precomputed LMP and band-power features |
| `data/kinematics_bins50ms/bins50ms/` | Kinematics resampled to 50 ms bins |
| `EECS182_data_manifest.csv`, `splits.json` | Session metadata and fixed splits, used by the notebook and `task3/` |
| `data/manifest.csv`, `data/splits.json` | Identical loader fallbacks for code that uses `data/` as its root |
| `cs182_decoder_grouped.ipynb` | Ridge, KNN, MLP, GRU and EEGNet baselines |
| `task3/` | Task 3 ridge, XGBoost, PARRM and StiF checks |
| `scripts/notebook_api.py` | Loads the notebook's shared cells for the scripts |
| `scripts/make_diagnostics.py` | Rebuilds the figures in the report |
| `scripts/make_model_diagram.py` | Rebuilds the input and model diagrams from one real trial |
| `scripts/make_model_architectures.py` | Rebuilds the per-baseline architecture graphs |
| `scripts/make_validation_animation.py` | Rebuilds the validation vector animation |
| `scripts/make_parameter_sweeps.py` | Rebuilds the R² and RMSE parameter sweeps |
| `results/` | Generated metrics, checkpoints, figures and training logs; ignored by git |
| `reports/baselines/index.html` | Generated baseline report with figures |
| `reports/model_diagram.html` | Input, decoder, and output diagrams |
| `reports/model_architectures.html` | Architecture graphs, validation animation, and parameter sweeps |
| `docs/` | Proposal, project spec and baseline notes |

The notebook and `task3/task3_common.py` read `EECS182_data_manifest.csv` and `splits.json` at the root. The `data/` copies keep loader defaults working when a script points its root at `data/`.

## Task

The task is a sequence of five touchscreen targets. Butters holds a button until the go cue, then touches the targets in order for a juice reward. Two sequences were used: `1-9-4-3-7` (2025-05-20 to 2025-06-06 and 2025-07-07 to 2025-07-11; 36 sessions) and `3-7-6-1-9` (2025-06-09 to 2025-07-03; 34 sessions). The sequence for each recording is in `manifest.csv`. Target codes follow a phone keypad arrangement on a 3 × 3 grid. Positions are in screen cm:

| Code | x | y |
|---|---|---|
| 1 | -6.55 | 7.3 |
| 3 | 6.55 | 7.3 |
| 4 | -6.55 | 0.75 |
| 6 | 6.55 | 0.75 |
| 7 | -6.55 | -5.8 |
| 9 | 6.55 | -5.8 |

## Files

| Path | Contents |
|---|---|
| `manifest.csv` | One row per session: condition, counts, timing, quality, known problems |
| `conditions_eACS_AM-kHz_v2.csv` | All 102 candidate sessions with inclusion decisions, exclusion reasons and `qc_flag` |
| `splits.json` | Train, validation and test days and sessions; held-out trials; calibration subsets (see Splits) |
| `EECS182_NHP_data_loader.py` | Python loader (see Loader) |
| `session_summary.csv` | DeepLabCut export summary per session (confidence per keypoint, Start and Target positions, rest positions, reach endpoints) |
| `export_outputs/<session>.h5` | Continuous LFP, stimulation signals, task samples and video frame times for one session |
| `export_outputs/trials/<session>_trials.csv` | One row per trial |
| `export_outputs/qc_figures/<session>_qc.png` | Trial PSD vs same-day no-stim trials and stim-off rest, plus per-channel spectral-shape correlation |
| `export_outputs/export_summary.csv` | Per-session export log |
| `export_outputs/qc_all_sessions.mat`, `task_timing_all.mat` | Collected QC spectra and task timing (used for checks) |
| `export_outputs/trial_reconstruction_*.csv` | Validation of the July 11 trial rebuild |
| `export_outputs/onoff_spectra/<session>_onoff.png` | Trial spectra of the session and of the other sessions that day (made by the loader) |
| `export_outputs/kinematics_calibration.csv` | Per-session map from Index pixels to screen cm (see Kinematics) |
| `parrm_outputs/<session>.h5` | PARRM-cleaned LFP and artifact estimate, 48 stimulation sessions (see PARRM) |
| `parrm_outputs/template_library.h5` | One group per stimulation session: mean stimulation-locked waveform per channel and its gains |
| `parrm_outputs/parrm_summary.csv`, `figures/` | Per-session PARRM results and figures |
| `features/<source>_<reference>/<session>.h5` | LMP and band power per channel and 50 ms bin (see Features); `export_none`, `parrm_none` and `export_car` computed |
| `EECS182_NHP_features.py` | Computes and loads the features |

The kinematics exports are under `kinematics_v3/`, in `bins50ms/`, `frames/`, and `reaches/`.

## Clock

Use the TDT recording clock for alignment. Times are in seconds from the start of the recording. The task program clock (`/task/t_file`) differs from TDT time by -72 to +49 s across sessions and should not be used to align the data.

## HDF5 layout (`<session>.h5`)

Shapes below are in h5py order. `N` is the number of LFP samples, at roughly 1,000 samples/s.

| Path | Shape, type | Contents |
|---|---|---|
| `/lfp` | (64, N) float32 | LFP in microvolts, all 64 channels in channel order 1-64 |
| `/t` | (N,) float64 | Sample times (s) |
| `/channel_ids` | (64,) | Physical channel number of each row (1-64) |
| `/channel_keep_fixed` | (64,) uint8 | **Channel mask to use** (1 = keep, 35 channels). Same in every session. See Channels. |
| `/channel_keep` | (64,) uint8 | Original per-session mask from the export (29-41 channels). Superseded. |
| `/bad_sheet`, `/bad_auto` | (k,) | Per-session bad channels from the lab sheet and from an automatic power check (record only) |
| `/stim/on` | (N,) uint8 | 1 while the stimulator is on (from `PC1_` events; gaps over 1 s separate segments) |
| `/stim/segments` | (n, 2) | Start and end time (s) of each stimulation segment. Absent in no-stim sessions. |
| `/stim/envelope` | (N,) float32 | AM envelope of the stimulus monitor (StiF), uncalibrated monitor units |
| `/stim/phase` | (N,) float32 | AM phase (rad), 0 at the envelope peak; NaN when stimulation is off |
| `/task/t` | (M,) | Time (s) of each task-program sample (`PC2_` events, one per sample, ~30 Hz) |
| `/task/t_file` | (M,) | Same samples on the task program's clock (do not use for alignment) |
| `/task/touch_t` | (M,) | `/task/t` minus the 70 ms touchscreen lag: when each touch sample actually happened |
| `/task/n_contacts` | (M,) uint8 | Number of fingers on the screen |
| `/task/touch_first_xy`, `/task/touch_centroid_xy` | (M, 2) float32 | First contact and centroid of all contacts, screen cm; NaN with no contact |
| `/task/target_xy` | (M, 2) float32 | Current target position, screen cm |
| `/task/state_code` | (M,) uint8 | Task state code. **The code-to-name mapping differs between sessions**: always decode with the `state_names` attribute of this dataset (1-based, comma-separated). |
| `/task/task_block` | (M,) uint8 | Task block (1-5) |
| `/video/frame_t` | (F,) | Time (s) of each side-camera frame (50 fps, recorded in bursts during task blocks). Frame n of the video is element n. |
| `/qc/f`, `/qc/psd_task`, `/qc/psd_rest`, `/qc/psd_ref_task`, `/qc/shape_r_task` | | Spectra used for the QC figures (rows are channels) |

The three July 11 sessions have no `/task` group because the task files are missing. Their trial tables were reconstructed from TDT events.

Root attributes: `session`, `date`, `stim`, `am_freq_sheet_hz`, `am_freq_measured_hz` (from the StiF envelope), `mod_depth`, `envelope_valid`, `amplitude_ma`, `sequence`, `lfp_units`, `lfp_fs`, `lfp_fs_original`, `lfp_filter`, `clock`, `task_time_source`, `touch_lag_s`, `frame_time_source`, `layout`, `ref_block`, `created_by`, `created`.

## LFP processing

The raw signal comes from the TDT `LFP4` store at 3,051.76 Hz. Each channel was filtered in the FFT domain with zero phase: a 0.1 Hz high-pass, notches at 60, 120, and 180 Hz (Q = 10), and a low-pass response flat through 250 Hz and tapered to zero at 300 Hz. The signal was then resampled in the FFT domain to about 1,000 Hz. The exact sampling rate is in `lfp_fs`; use `/t` for timing.

There is residual line noise around 55-65 and 115-125 Hz in many sessions, including recordings without stimulation. Avoid those ranges when choosing feature bands.

## Trial table (`trials/<session>_trials.csv`)

The trial CSV includes rewarded and unrewarded trials. Use `include == 1` to select the rewarded ON or OFF trials.

| Column | Meaning |
|---|---|
| `session`, `trial`, `task_block` | Session, trial number (1..), task block (1-5) |
| `t_start` | Trial start (video trigger state) |
| `t_go` | Go cue: first sample of the first target |
| `t_first_touch` | First screen contact at or after go, corrected for the 70 ms touch lag. **Not a movement-onset measure**: in many sessions the hand is already resting on the screen at go (`touch_at_go_frac` in the manifest, up to 0.9 in June), so it equals `t_go` minus 0.07 s. |
| `t_acq1` ... `t_acq5` | Acquisition of targets 2-5 (target position changes) and the final target hold |
| `t_reward` | Reward onset |
| `t_end` | First task-break sample after go |
| `outcome` | `success`, `timeout_error` (task files), or `unrewarded` (July 11 rebuild) |
| `n_targets_reached` | Targets reached |
| `tgt1_x` ... `tgt5_y` | Target positions in order, screen cm |
| `stim_on_frac` | Fraction of [`t_go`, `t_reward`] with stimulation on |
| `stim_label` | `ON` (>= 0.99), `OFF` (<= 0.01) or `MIXED` |
| `am_freq_hz`, `sequence` | Condition and target order |
| `include`, `exclude_reason` | 1 for rewarded trials that are entirely ON or OFF |

Go-to-reward time is 0.83-2.2 s across the 1st-99th percentiles.

The July 11 trial tables were reconstructed from intervals between task-program events, which change with task state. The reconstruction was checked against the 67 sessions with task files: 99.9% of reward times matched exactly, 99.5% of go times were within 80 ms (85% exact), and task block assignments matched throughout. Each July 11 session has 50 rewarded trials, 10 per block. `t_first_touch` and `t_acq1`-`t_acq4` are empty, `t_acq5` marks the final hold, and unrewarded trials have outcome `unrewarded`.

## Channels

For consistent decoding across sessions, use `/channel_keep_fixed`. It selects the same 35 channels in all recordings:

`21 22 26 27 28 29 30 31 32 33 34 35 36 37 38 41 42 43 44 45 47 48 49 51 52 53 54 55 57 58 59 60 61 62 64`

The excluded channels are recorded in the HDF5 attributes:

| Group | Channels | Reason |
|---|---|---|
| `excluded_canonical` | 1-10, 12-15, 17-20, 24, 25, 40, 50, 63 | Lab's list of bad channels for this array |
| `excluded_degrading` | 16, 39, 46, 56 | Normal early, abnormally high power in later no-stim sessions (46 from June 9, 39 from June 17, 56 from July 7, 16 in several sessions and in every no-stim session from July 1) |
| `excluded_stim_pickup` | 11, 23 | Healthy without stimulation, strong 20 Hz harmonic pickup in 7-8 of the 26 20 Hz sessions |

The `/lfp` dataset still contains all 64 channels. You can use another mask, for example to test whether including the stimulation-pickup channels changes decoding.

## PARRM (`parrm_outputs/`)

`run_PARRM_LFP_v5.m` was applied to all 48 stimulation sessions, processing each stimulation segment separately with a ±6 s window for both AM frequencies. The AM period was estimated by fitting a straight line to the StiF envelope phase across segments. The fitted frequencies were 2.9971 and 19.8911 Hz.

| Path | Shape | Contents |
|---|---|---|
| `/lfp` | (64, N) float32 | Export LFP with the PARRM estimate subtracted inside stimulation segments; identical to the export elsewhere (microvolts) |
| `/artifact` | (64, N) float32 | The subtracted estimate (zero outside stimulation segments) |
| `/stim/phase_model` | (N,) float32 | Fitted AM phase (rad), 0 at the envelope peak |
| `/template/mean`, `/template/std` | (64, phase bins) | Mean and SD of the stimulation-locked waveform over one AM cycle (each cycle demeaned first) |
| `/template/gain`, `/template/window_rms` | (64, windows) | Least-squares gain against the mean template, and RMS, in 1 s windows (`gain_t_start`, `gain_t_end`, `gain_segment`) |
| `/sham/...` | | The same filter applied to a no-stim session within this session's stimulation timing: template, spectra, segments |
| `/qc/...` | | Task spectra before and after cleaning and of the same-day reference |

`template_library.h5` contains a template group for every stimulation session, along with `sham_mean`. Each group records the AM frequency, period, and `template_snr`.

PARRM subtracts the component phase-locked to the AM signal. That includes the stimulation artifact and potentially neural activity entrained to the stimulus; the filter cannot distinguish between them. For comparison, the sham applies the same filter at the same period to no-stim data:

| | 3 Hz (22 sessions) | 20 Hz (26 sessions) |
|---|---|---|
| Power left in the AM-frequency bin after cleaning (median) | 0.955 | 0.831 |
| Same, sham on no-stim data | 1.018 | 0.991 |
| Power elsewhere, after / before | 1.039 | 1.008 |
| Same, sham | 1.041 | 1.009 |
| Template signal-to-noise vs sham (median) | 1.9 | 5.7 |

At 20 Hz, PARRM removes a median 17% of the power in the AM-frequency bin, while the sham shows almost no reduction. At 3 Hz, the reduction is about 5%, with about a 4% increase in broadband power. The sham has a similar broadband increase, suggesting that this effect comes from the filtering rather than stimulation. Session values are in the manifest under `parrm_*`.

There are a few limitations to keep in mind:

- No-stim sessions have no PARRM file. With `source='parrm'`, the loader uses their export LFP, so ON trials are filtered but OFF trials are not. At 3 Hz, this produces about a 4% broadband power mismatch, mostly removed by per-session normalization. The sham-cleaned LFP was not saved.
- Templates for 250606-104043 (SNR 0.97), 250609-111203 (1.16), and 250529-102830 (1.46) are only slightly above sham and are poor candidates for semi-synthetic artifact generation.
- Nine of the 48 shams use a no-stim recording from the nearest other day because the same-day no-stim session was excluded (`parrm_sham_same_day = 0`).

## Kinematics (`kinematics_v3/`)

DeepLabCut (ResNet-50, shuffle 2, `snapshot_best-100`) tracked Start, Target, Wrist, Thumb, and Index in the side-camera video. The camera runs at 50 fps, with one TDT trigger per frame, so frame times are known on the TDT clock. Points below 0.6 confidence are set to NaN. Confident points are accurate to about 4 px. Index confidence is 81% at the median session, but falls to 49% in 250620-094859.

`bins50ms/<session>_bins50ms.csv` has one row per 50 ms bin, aligned to multiples of 0.05 s on the TDT clock. The loader uses this same grid. For each of `Wrist`, `Thumb`, and `Index`, the columns are:

| Columns | Meaning |
|---|---|
| `_x`, `_y` | Position in image pixels relative to the Start block |
| `_dx_rest`, `_dy_rest` | Displacement from the most recent resting position on the IR button |
| `_vx`, `_vy`, `_speed` | Velocity (px/s), central difference of the bin means |
| `_n_valid` | Confident frames in the bin |
| `_progress` | Projection on the Start-to-Target line (used by the loader's screen-plane mask; keep it) |

Other columns are `n_frames`, `in_hold` (fraction of the bin with the button held), `reach_id`, and the reach endpoint (`end_t`, `Index_end_x`, `Index_end_y`, `Index_end_dist`). Image positions are relative to Start because the camera moved between days. The pixel-to-cm scale changed by about 4% in July.

The loader converts Index positions to screen cm (x right, y up, matching the target coordinates) using a separate affine fit for each session. It uses touchscreen contacts where the Index point is confident and moving slower than 300 px/s. To avoid fitting mostly to resting positions, the samples are balanced across the screen, with at most 15 points per 2 cm cell, followed by two rounds of outlier rejection.

Across the 67 sessions with touch data, R² ranges from 0.885-0.967 for screen x and 0.925-0.977 for y. Median residuals are about 1.4 cm and 1.1 cm, and the scale is within 7% of the median for sessions with the same camera placement. Results are in `kinematics_calibration.csv` and the manifest's `kin_*` columns.

The mapping applies only near the screen plane. The cm columns are NaN when Index progress is outside the range observed during contact; this leaves about 80-90% of bins in the trial windows checked. July 11 has no touch data, so those sessions have no cm values. Both screen axes can be estimated from the side view during contact, but target identity still comes from the task files.

## Known problems

The manifest records session-specific issues in `qc_flag`, `known_problems`, and `sheet_notes`. The ones most relevant to decoding are:

- **250602-102701 and 250605-102648 (3 Hz):** power above ~25 Hz is 3-10x the same-day no-stim session in both stim-on trials and stim-off rest, equally on all channels. Kept for training, out of validation and test (`test_eligible = 0`).
- **250603-142553 (no-stim):** during trials, power above ~40 Hz is 2-3x that of the same-day stim sessions, while its between-trial power is not elevated, consistent with the sheet note on grimacing/chewing. As the day's OFF reference it would bias ON-vs-OFF comparisons, so it has `qc_flag = emg_in_trials` and `test_eligible = 0`, which keeps the 250603 day in training.
- **250530-105932:** a large artifact appears near the end of the recording, outside the trials. The trials themselves are clean. Use the first-to-last-trial window (`first_go_s` to `last_reward_s` in the manifest) for normalization, as with the other sessions.
- **250616-102124 (20 Hz):** leftover peaks at 20, 40 and ~100 Hz during stim-on trials; PARRM removes 69% of the 20 Hz bin there.
- **Between-session power:** broadband power varies substantially (`broadband_ratio` 0.43-3.68). Normalize features within each session.
- **3 Hz AM:** the stimulation-locked component at 3 Hz cannot be cleanly separated from possible neural entrainment by PARRM or SASS. The local motor potential (below 4 Hz) is the feature most exposed to it.
- **250620-094859:** distracted subject (73 trials, 23 unrewarded) and low DeepLabCut confidence (Index 49% of frames).
- **250710-101340 (no-stim):** LMP transients over 300 µV on channels 21, 27, 52, 54, 60 and 62 at about 221-222 s and 310 s. One overlaps included trial 44, which is kept.
- **250711-103614 (3 Hz):** LMP transients up to 2 mV on channels 21 and 27 between task blocks (about 10, 62-78 and 188 s). None fall inside a trial window, and the robust normalization is not affected.
- **Similar spectra across kept channels:** the 25-75th percentile spread is about 0.007 log10 units, suggesting a substantial common component. The loader can subtract a common-average reference (CAR). Compare both references when decoding.

## Splits (`splits.json`)

The splits support a comparison of in-context adaptation, fine-tuning, LoRA, and PARRM + ridge. On held-out days, each method receives `k` ON calibration trials from a stimulation session and is evaluated on separate ON trials from that session. The OFF reference is the same-day no-stim session.

- **Split by day:** all sessions from one day stay in the same partition. This prevents test and training sessions from sharing that day's electrode state and behavior. The OFF reference for each stimulation session is its same-day no-stim session (`off_reference`).
- **Validation and test days:** each has a no-stim reference, at least one stimulation session, touch data, and no session with `test_eligible = 0`. July 11 can therefore appear only in training. The test split has 5 days, at least 5 sessions per AM frequency, no more than a 2-session difference between frequencies, and at least 2 days of each target sequence. Validation has 3 days, at least 2 sessions per AM frequency, and both target sequences. Days were sampled randomly from combinations satisfying these constraints; the seed is in `splits.json`.
- **Training sessions:** hold out one task block (10 trials) per session using `train_val_trials`. The other 40 trials are available for fitting.
- **Calibration and query trials:** in validation and test sessions, blocks 1-2 provide 20 calibration trials and blocks 3-5 provide 30 query trials. `calibration[session].subsets[k]` stores 10 random draws for `k = 2, 5, 10, 20`. The draws are nested (each 5-trial draw contains its 2-trial draw); `k = 20` uses the full pool. Calibration trials precede query trials in time.

| Split | Days | No-stim | 3 Hz | 20 Hz |
|---|---|---|---|---|
| Train | 18 | 14 | 13 | 15 |
| Validation | 3 | 3 | 2 | 4 |
| Test | 5 | 5 | 7 | 7 |

`make_splits_v1.py` regenerates the splits and checks that no day or session appears in more than one partition.

## Loader (`EECS182_NHP_data_loader.py`)

Dependencies: numpy, pandas, h5py, and matplotlib.

```python
import EECS182_NHP_data_loader as L

P = L.Paths(root='.../EECS282_finalProject', kin_dir='.../kinematics_v3/bins50ms')
M, S = L.load_manifest(P), L.load_splits(P)
L.check_splits(S, M)                                    # fails if a test session is in training

s = L.load_session('Butters-250606-104043', P)          # fixed 35 channels, export LFP
s = L.load_session('Butters-250606-104043', P, source='parrm', reference='car')
L.normalize(s)                                          # per-channel robust z-score, first-to-last-trial window
K = L.load_kinematics(s.name, P)                        # bins in px and screen cm; K.attrs['calibration']
q = L.split_trials(S, s, 'query')                       # or 'pool', 'train', 'val', 'all'
ex = L.trial_examples(s, K, align='t_go', pre=0.5, post=1.5, trials=q, with_lmp=True)
ex[0]['lfp'].shape                                      # (35 channels, 40 bins, 50 samples)
ex[0]['lmp'].shape                                      # (35, 40): local motor potential at each bin end
ex[0]['kin'], ex[0]['kin_valid']                        # (40, 2) Index vx, vy in cm/s; (40,) True where defined

T = L.load_templates('Butters-250606-104043', P)        # one session of the PARRM template library
L.calibrate_all(P)                                      # refit the cm calibration for all sessions
L.plot_all_on_off_spectra(P, overwrite=True)            # one figure per session into onoff_spectra/
```

- **Decoding target:** the default is Index velocity in screen cm/s (`kin_cols=('Index_vx_cm', 'Index_vy_cm')`), consistent with common LFP hand-velocity decoding. For pixel velocity, including July 11, use `('Index_vx', 'Index_vy')`. Exclude bins with `kin_valid = False` from the loss and metric.
- **`source`:** `'export'` (default) or `'parrm'`. The latter uses cleaned LFP and fitted AM phase for stimulation sessions; no-stim sessions use the original export.
- **`reference`:** `'none'` keeps the recording reference (default); `'car'` subtracts the common average of the selected channels after PARRM cleaning. CAR has been reported to help LMP decoding slightly, but it may also remove the spatially common 3 Hz stimulation-locked component. Keep the reference consistent across methods.
- **LMP:** a causal 256 ms moving average of the LFP, sampled at each bin end, following Flint et al. (2012) and later decoding work. Run `normalize()` first because it clears the cached LMP.
- **Bins:** all signals use the same TDT grid. Bin `j` covers `[0.05*j, 0.05*(j+1))` seconds. A fixed window has `round((pre + post) / 0.05)` bins; `until='t_reward'` instead ends the window at reward plus `post`.
- **Spectra:** each figure shows median channel power during trials (go - 0.5 s through reward + 0.5 s). The current session has a thick line; other sessions from the same day are dashed (black OFF, red 3 Hz, blue 20 Hz). The grey line is between-trial, stimulation-off power from the current session. Welch PSD uses 1 s Hann windows with 50% overlap, restricted to complete windows within each period.

## Features (`features/`, `EECS182_NHP_features.py`)

Features are saved once per session and preprocessing variant, named `<source>_<reference>` as in the loader. `export_none` (default), `parrm_none`, and `export_car` are computed for all 70 sessions. The script can also generate `parrm_car`. In no-stim sessions, `parrm_none` is the same as `export_none` (attribute `source = 'export'`).

Features are computed for each kept channel at 50 ms intervals, using only samples available by the end of each bin:

| Feature | Definition | Window |
|---|---|---|
| `lmp` | Local motor potential: mean LFP over the window (µV) | 256 ms |
| `pow_1_4` | log10 mean PSD (µV²/Hz) in 1-4 Hz, Hann-windowed FFT | 1 s |
| `pow_4_8`, `pow_8_13` | same, 4-8 and 8-13 Hz | 0.5 s |
| `pow_13_30`, `pow_30_50`, `pow_70_110`, `pow_130_200` | same | 0.256 s |

Frequency bands follow Flint et al. (2012) and later LFP decoding work, with finer divisions at low frequencies. We omit 50-70, 110-130, and 170-190 Hz because of residual line noise. Feature values are NaN if the analysis window begins before the recording (roughly the first second).

Each feature HDF5 file has `bin_id` (B), `t_bin` (B, start time in seconds), `lmp` (C, B), `logpow` (C, 7, B), `channel_ids` (C), and attributes describing the bands, windows, source, reference, and feature definitions.

```python
import EECS182_NHP_features as F
f = F.load_features('Butters-250606-104043', P, variant='export_none')   # X: (bins, 35, 8), robust z per channel and feature
rows = lambda part: s.trials[s.trials.trial.isin(L.split_trials(S, s, part))]
f = F.load_features(s.name, P, normalize='pool', trials=rows('pool'))   # statistics from the calibration pool only
X, Y, valid, bins = F.trial_arrays(f, rows('query'), kin=K, align='t_go', pre=0.5, post=1.5, n_lags=9)
# X: (trials, 40, 10 * 35 * 8) with 9 causal lags as in Wiener/ridge decoders; Y: (trials, 40, 2) Index vx, vy in cm/s
```

- **Normalization:** median and 1.4826 × MAD per channel and feature, estimated from the first-go to last-reward interval (± 1 s). For held-out sessions, `normalize='pool'` computes statistics only from the calibration pool, not the query trials.
- **`valid`:** False wherever a feature or target is NaN. Mask these bins in both the loss and the metric.
- **Recording reference vs CAR:** this choice is still open. The recording reference is the default, with CAR as an ablation. Ahmadi et al. (2021) reported the best LMP decoding with CAR in macaque M1, while Young et al. (2018) found linear regression reference better for stimulation artifacts. Shirhatti et al. (2016) also showed that referencing changes PSD slopes and phase. Because the kept channels share much of their signal, CAR could remove movement information here. The first baseline compares `export_none` and `export_car` on validation days; the result will replace this note.
- **Compute:** run `python EECS182_NHP_features.py --variants export_none parrm_none`. Each variant takes about 6.5 s per session. Existing files are skipped unless `--overwrite` is supplied.
