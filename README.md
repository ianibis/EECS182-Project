# EECS 182/282A project data: Butters, 64-channel LFP under AM epidural stimulation

Data for decoding hand movement from motor cortex LFP while epidural AM stimulation (eACS) is on or off. One rhesus macaque (Butters), one 64-channel array, 70 recording sessions on 26 days (2025-05-20 to 2025-07-11). Each session has the LFP, the stimulation signals, a trial table, hand kinematics from DeepLabCut, and (stimulation sessions) a PARRM-cleaned copy of the LFP.

| Condition | Sessions | Rewarded trials used |
|---|---|---|
| No stimulation | 22 | 1,100 |
| 3 Hz AM (3.5 kHz carrier, 1.75 mA) | 22 | 1,100 (all stim on) |
| 20 Hz AM (3.5 kHz carrier, 1.75 mA) | 26 | 1,300 (all stim on) |
| Total | 70 | 3,500 |

Every session has 50 rewarded trials (5 task blocks of 10). In stimulation sessions the stimulator is on during each task block and off during the ~30 s breaks between blocks, so every rewarded trial is entirely stim on (stim sessions) or entirely stim off (no-stim sessions). Two stim sessions had the stimulator left on through one break. `manifest.csv` has one row per session with counts, quality measures and known problems.

## Task

Sequential target task on a touchscreen. The monkey holds a button, a go cue appears, and he touches 5 targets in a fixed order, then gets juice. Two fixed target orders are used: `1-9-4-3-7` (2025-05-20 to 2025-06-06 and 2025-07-07 to 2025-07-11; 36 sessions) and `3-7-6-1-9` (2025-06-09 to 2025-07-03; 34 sessions). The `sequence` column gives it per session. Target codes follow a phone keypad on a 3 x 3 grid, in screen cm:

| Code | x | y |
|---|---|---|
| 1 | -6.55 | 7.3 |
| 3 | 6.55 | 7.3 |
| 4 | -6.55 | 0.75 |
| 6 | 6.55 | 0.75 |
| 7 | -6.55 | -5.8 |
| 9 | 6.55 | -5.8 |

## GRU and MLP decoders

The [Task 3 GRU guide](task3/task3_README.md) and
[Task 4 MLP guide](task4/README.md) describe training, same-day OFF calibration,
and evaluation. Set `TASK3_DATA_ROOT` to the Box project directory before
running commands from this checkout. See the
[file map](EECS182_file_map_261117.md) for external inputs and archived results.

## Files

Under `EECS282_finalProject/`:

| Path | Contents |
|---|---|
| `manifest.csv` | One row per session: condition, counts, timing, quality, known problems |
| `conditions_eACS_AM-kHz_v2.csv` | All 102 candidate sessions with inclusion decisions, exclusion reasons and `qc_flag` |
| `splits.json` | Train, validation and test days and sessions; held-out trials; calibration subsets (see Splits) |
| `butters_loader_v2.py` | Python loader (see Loader) |
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
| `butters_features_v1.py` | Computes and loads the features |

Kinematics are in `kinematics_v3/` (`bins50ms/`, `frames/`, `reaches/`), currently on io's computer at `/Users/io/Documents/Documents/BME/PhD/Berkeley/Deep_Learning/5_Final_Project/TC_Videos/kinematics_v3`, not in the Box folder. Copy it next to the other folders to share it, and point the loader at it with `Paths(kin_dir=...)`.

## Clock

All times in every file are seconds on the TDT recording clock, with 0 at the start of the recording. The task program's own clock (`/task/t_file`) is offset from it by -72 to +49 s depending on the session; do not use it for alignment.

## HDF5 layout (`<session>.h5`)

Shapes are as h5py reads them. N is the number of LFP samples (about 1,000 per second).

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

July 11 sessions have no `/task` group (their task files are missing); their trial tables were rebuilt from TDT events instead.

Root attributes: `session`, `date`, `stim`, `am_freq_sheet_hz`, `am_freq_measured_hz` (from the StiF envelope), `mod_depth`, `envelope_valid`, `amplitude_ma`, `sequence`, `lfp_units`, `lfp_fs`, `lfp_fs_original`, `lfp_filter`, `clock`, `task_time_source`, `touch_lag_s`, `frame_time_source`, `layout`, `ref_block`, `created_by`, `created`.

## LFP processing

From the TDT store `LFP4` (3,051.76 Hz), each channel was filtered in the FFT domain (zero phase): high-pass 0.1 Hz, notches at 60, 120 and 180 Hz (Q 10), low-pass flat to 250 Hz and tapering to zero at 300 Hz, then resampled in the FFT domain to about 1,000 Hz (the exact rate is in `lfp_fs`; use `/t` for times). The low-pass is lower than the planned 400 Hz on purpose: the 3.5 kHz carrier aliases to about 448 Hz at the recording rate, too close to 500 Hz for a 400 Hz cutoff to remove it.

Line noise leaks just outside the notches (about 55-65 and 115-125 Hz) in many sessions, with or without stimulation. Feature bands should avoid those ranges.

## Trial table (`trials/<session>_trials.csv`)

One row per trial, including unrewarded ones. Use rows with `include == 1`.

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

Go to reward takes 0.83-2.2 s (1st-99th percentile across sessions).

**July 11 (3 sessions):** task files are missing, so trials were rebuilt from the timing of the task-program events, whose interval pattern changes with task state. Validated on the 67 sessions that have task files: 99.9% of rewards found at the exact time, 99.5% of go times within 80 ms (85% exact), task blocks always correct. July 11 has 50 rewarded trials, 10 per block, in all three sessions. `t_first_touch` and `t_acq1`-`t_acq4` are empty; `t_acq5` is the final hold; unrewarded trials are labelled `unrewarded`.

## Channels

Use `/channel_keep_fixed`. It keeps the same 35 channels in every session:

`21 22 26 27 28 29 30 31 32 33 34 35 36 37 38 41 42 43 44 45 47 48 49 51 52 53 54 55 57 58 59 60 61 62 64`

Excluded, recorded as attributes of the dataset:

| Group | Channels | Reason |
|---|---|---|
| `excluded_canonical` | 1-10, 12-15, 17-20, 24, 25, 40, 50, 63 | Lab's list of bad channels for this array |
| `excluded_degrading` | 16, 39, 46, 56 | Normal early, abnormally high power in later no-stim sessions (46 from June 9, 39 from June 17, 56 from July 7, 16 in several sessions and in every no-stim session from July 1) |
| `excluded_stim_pickup` | 11, 23 | Healthy without stimulation, strong 20 Hz harmonic pickup in 7-8 of the 26 20 Hz sessions |

All 64 channels stay in `/lfp`, so a different set can be used (for example, adding the pickup channels back as a robustness test).

## PARRM (`parrm_outputs/`)

`run_PARRM_LFP_v5.m` removed the stimulation-locked component from each of the 48 stimulation sessions, separately within each stimulation segment, with a +/-6 s window at both AM frequencies. The AM period comes from a straight-line fit to the StiF envelope phase across segments (fitted frequencies 2.9971 and 19.8911 Hz).

| Path | Shape | Contents |
|---|---|---|
| `/lfp` | (64, N) float32 | Export LFP with the PARRM estimate subtracted inside stimulation segments; identical to the export elsewhere (microvolts) |
| `/artifact` | (64, N) float32 | The subtracted estimate (zero outside stimulation segments) |
| `/stim/phase_model` | (N,) float32 | Fitted AM phase (rad), 0 at the envelope peak |
| `/template/mean`, `/template/std` | (64, phase bins) | Mean and SD of the stimulation-locked waveform over one AM cycle (each cycle demeaned first) |
| `/template/gain`, `/template/window_rms` | (64, windows) | Least-squares gain against the mean template, and RMS, in 1 s windows (`gain_t_start`, `gain_t_end`, `gain_segment`) |
| `/sham/...` | | The same filter applied to a no-stim session within this session's stimulation timing: template, spectra, segments |
| `/qc/...` | | Task spectra before and after cleaning and of the same-day reference |

`template_library.h5` holds the template group of every session (plus `sham_mean`), with the AM frequency, period and `template_snr` as attributes.

**How to read it.** PARRM removes whatever is phase-locked to the AM, so the removed part is the stimulation-locked component: artifact plus any neural activity entrained to the AM. It cannot tell the two apart. The sham control runs the same filter on no-stim data at the same period:

| | 3 Hz (22 sessions) | 20 Hz (26 sessions) |
|---|---|---|
| Power left in the AM-frequency bin after cleaning (median) | 0.955 | 0.831 |
| Same, sham on no-stim data | 1.018 | 0.991 |
| Power elsewhere, after / before | 1.039 | 1.008 |
| Same, sham | 1.041 | 1.009 |
| Template signal-to-noise vs sham (median) | 1.9 | 5.7 |

At 20 Hz PARRM removes a median 17% of the AM-frequency bin and the sham removes none, so the removed component is specific to stimulation. At 3 Hz it removes about 5% and adds about 4% broadband power; the sham shows the same increase, so that is a cost of the filter itself. Per session: `parrm_*` columns in the manifest.

**Limits:**
- No-stim sessions have no PARRM file; with `source='parrm'` the loader returns their export LFP. ON data is filtered and OFF data is not, a mismatch of about 4% broadband power (3 Hz only, mostly removed by per-session normalization). The sham-cleaned LFP was not saved.
- Templates in 250606-104043 (SNR 0.97), 250609-111203 (1.16) and 250529-102830 (1.46) are barely above their sham, so they are weak material for semi-synthetic artifacts.
- 9 of the 48 shams come from the nearest other day, because the same-day no-stim session was excluded (`parrm_sham_same_day = 0`).

## Kinematics (`kinematics_v3/`)

DeepLabCut (ResNet-50, shuffle 2, `snapshot_best-100`) tracked Start, Target, Wrist, Thumb and Index in the side camera (50 fps, one TDT trigger per frame, so every frame has an exact TDT time). Points below 0.6 confidence are NaN. Confident points are accurate to about 4 px; about 80% of Index frames are confident (median 81%; 49% in 250620-094859).

`bins50ms/<session>_bins50ms.csv` has one row per 50 ms bin on the TDT clock (edges at multiples of 0.05 s, the same grid the loader uses). Per keypoint (`Wrist`, `Thumb`, `Index`):

| Columns | Meaning |
|---|---|
| `_x`, `_y` | Position in image pixels relative to the Start block |
| `_dx_rest`, `_dy_rest` | Displacement from the most recent resting position on the IR button |
| `_vx`, `_vy`, `_speed` | Velocity (px/s), central difference of the bin means |
| `_n_valid` | Confident frames in the bin |
| `_progress` | Projection on the Start-to-Target line (used by the loader's screen-plane mask; keep it) |

Also `n_frames`, `in_hold` (fraction of the bin holding the button), `reach_id`, and the reach endpoint (`end_t`, `Index_end_x`, `Index_end_y`, `Index_end_dist`). Positions are relative to Start because the camera moved between days; the px-to-cm scale changed by about 4% (July).

**Screen cm.** The loader converts Index to screen cm (x right, y up, the target axes) with an affine map fitted per session against the touchscreen: contact samples with a confident, slow (< 300 px/s) Index, balanced over the screen (at most 15 per 2 cm cell), two rounds of outlier rejection. Across the 67 sessions with touch data: R^2 0.885-0.967 for screen x and 0.925-0.977 for y, residual about 1.4 cm and 1.1 cm (median), and a scale within 7% of the median of sessions with the same camera placement (`kinematics_calibration.csv`, `kin_*` columns in the manifest). The map holds in the screen plane only, so cm values are NaN when the hand is off the screen (Index progress outside the range seen at contacts); in trial windows that kept about 80-90% of bins in the sessions checked. July 11 has no touch data, so its cm columns are empty. The fit shows that both screen axes are recoverable from the side view while the finger is on the screen. Target identity still comes from the task files.

## Known problems

Per-session details are in `manifest.csv` (`qc_flag`, `known_problems`, `sheet_notes`). The main ones:

- **250602-102701 and 250605-102648 (3 Hz):** power above ~25 Hz is 3-10x the same-day no-stim session in both stim-on trials and stim-off rest, equally on all channels. Kept for training, out of validation and test (`test_eligible = 0`).
- **250603-142553 (no-stim):** during trials, power above ~40 Hz is 2-3x that of the same-day stim sessions, while its between-trial power is not elevated, consistent with the sheet note on grimacing/chewing. As the day's OFF reference it would bias ON-vs-OFF comparisons, so it has `qc_flag = emg_in_trials` and `test_eligible = 0`, which keeps the 250603 day in training.
- **250530-105932:** very large artifact outside the trials near the end of the recording. Trials are clean. Compute any normalization from the first-to-last-trial window (`first_go_s` to `last_reward_s` in the manifest), which is good practice for every session.
- **250616-102124 (20 Hz):** leftover peaks at 20, 40 and ~100 Hz during stim-on trials; PARRM removes 69% of the 20 Hz bin there.
- **Day-to-day power differences:** broadband power varies between days and sessions (`broadband_ratio` 0.43-3.68), so normalize features within each session.
- **3 Hz AM:** the stimulation-locked component at 3 Hz cannot be cleanly separated from possible neural entrainment by PARRM or SASS. The local motor potential (below 4 Hz) is the feature most exposed to it.
- **250620-094859:** distracted subject (73 trials, 23 unrewarded) and low DeepLabCut confidence (Index 49% of frames).
- **250710-101340 (no-stim):** LMP transients over 300 µV on channels 21, 27, 52, 54, 60 and 62 at about 221-222 s and 310 s. One overlaps included trial 44, which is kept.
- **250711-103614 (3 Hz):** LMP transients up to 2 mV on channels 21 and 27 between task blocks (about 10, 62-78 and 188 s). None fall inside a trial window, and the robust normalization is not affected.
- **The kept channels have nearly identical spectra** (25-75th percentile spread across channels about 0.007 log10 units), which suggests a large component common to all channels. The loader's common-average reference option removes it; compare decoders with and without it.

## Splits (`splits.json`)

The comparison the splits are built for: on held-out days, decoding accuracy on stimulation-ON trials of each held-out stim session after adapting with k ON trials from that session (in-context vs fine-tuning, LoRA and PARRM + ridge), relative to the OFF trials of the same-day no-stim session.

- **The unit is a day.** All sessions of a day go to the same split, so a test session never shares a day (electrode state, behavior) with training data. Because every trial in a stim session is ON, the OFF reference for a stim session is the same-day no-stim session (`off_reference`).
- **Test and validation days** must have a same-day no-stim session, at least one stim session, no session with `test_eligible = 0`, and touch data (so July 11 is training only). Test: 5 days with at least 5 sessions of each AM frequency, at most 2 sessions difference between them, and at least 2 days of each target sequence. Validation: 3 days with at least 2 sessions of each AM frequency and both sequences. Days were drawn at random (seed in the file) from all combinations meeting these rules.
- **Training sessions:** one task block per session (10 trials) is held out for validation (`train_val_trials`); use the other 40 for fitting.
- **Validation and test sessions:** blocks 1-2 are the calibration pool (20 trials), blocks 3-5 the query trials (30). `calibration[session].subsets[k]` holds 10 random draws of k = 2, 5, 10 and 20 calibration trials, nested within each draw (the k = 5 draw contains the k = 2 draw). k = 20 always uses the whole pool. Calibration comes before the queries in time, as it would in use.

| Split | Days | No-stim | 3 Hz | 20 Hz |
|---|---|---|---|---|
| Train | 18 | 14 | 13 | 15 |
| Validation | 3 | 3 | 2 | 4 |
| Test | 5 | 5 | 7 | 7 |

Regenerate with `make_splits_v1.py`; it asserts that no session or day is in two splits.

## Loader (`butters_loader_v2.py`)

Needs numpy, pandas, h5py and matplotlib.

```python
import butters_loader_v2 as L

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

- **Default decoding target: Index velocity in screen cm/s** (`kin_cols=('Index_vx_cm', 'Index_vy_cm')`), the field's usual target (hand velocity; position second). Use `('Index_vx', 'Index_vy')` for px/s, which also covers July 11. Bins where a target is undefined have `kin_valid = False`; mask them in the loss and the metric.
- **`source`:** `'export'` (default) or `'parrm'`. With `'parrm'`, stimulation sessions use the cleaned LFP and PARRM's fitted AM phase, and no-stim sessions fall back to the export.
- **`reference`:** `'none'` (recording reference, default) or `'car'` (common average of the selected channels, after PARRM cleaning). In the one direct comparison of referencing schemes for LFP decoding of hand kinematics, CAR helped the LMP feature slightly. Because the 3 Hz stimulation-locked component is mostly common to all channels, CAR also removes part of it, so apply the same reference to every method being compared.
- **LMP:** causal 256 ms moving average of the LFP (as in Flint et al. 2012 and later LFP decoding work), sampled at each bin end. `normalize()` clears it, so normalize first.
- **Bins** are on one global grid: bin j covers [0.05 j, 0.05 (j + 1)) s on the TDT clock, the same grid as the kinematics. Fixed windows have `round((pre + post) / 0.05)` bins; `until='t_reward'` gives variable-length windows ending at reward plus `post`.
- **Spectrum figures:** for each session, the median over channels of the trial spectrum (go - 0.5 s to reward + 0.5 s) for that session (thick) and for the other sessions that day (dashed: black OFF, red 3 Hz, blue 20 Hz), plus the session's between-trial, stim-off spectrum (grey). Each curve is the Welch average (1 s Hann windows, 50% overlap) over all whole windows inside the masked periods.

## Features (`features/`, `butters_features_v1.py`)

One file per session and variant. The variant is `<source>_<reference>` with the loader's meanings: `export_none` (default), `parrm_none` and `export_car` are computed for all 70 sessions; `parrm_car` can be added with the same script. For no-stim sessions `parrm_none` equals `export_none` (attribute `source = 'export'`).

For each kept channel and each 50 ms bin of the global grid, using only samples up to the end of the bin:

| Feature | Definition | Window |
|---|---|---|
| `lmp` | Local motor potential: mean LFP over the window (µV) | 256 ms |
| `pow_1_4` | log10 mean PSD (µV²/Hz) in 1-4 Hz, Hann-windowed FFT | 1 s |
| `pow_4_8`, `pow_8_13` | same, 4-8 and 8-13 Hz | 0.5 s |
| `pow_13_30`, `pow_30_50`, `pow_70_110`, `pow_130_200` | same | 0.256 s |

The bands follow Flint et al. (2012) and later LFP decoding work, split finer at low frequencies. 50-70, 110-130 and 170-190 Hz are left out because of the line-noise shoulders. Bins whose window starts before the recording are NaN (about the first second).

HDF5 layout: `bin_id` (B), `t_bin` (B, bin start in s), `lmp` (C, B), `logpow` (C, 7, B), `channel_ids` (C), and attributes for the bands, windows, source, reference and definition.

```python
import butters_features_v1 as F
f = F.load_features('Butters-250606-104043', P, variant='export_none')   # X: (bins, 35, 8), robust z per channel and feature
rows = lambda part: s.trials[s.trials.trial.isin(L.split_trials(S, s, part))]
f = F.load_features(s.name, P, normalize='pool', trials=rows('pool'))   # statistics from the calibration pool only
X, Y, valid, bins = F.trial_arrays(f, rows('query'), kin=K, align='t_go', pre=0.5, post=1.5, n_lags=9)
# X: (trials, 40, 10 * 35 * 8) with 9 causal lags as in Wiener/ridge decoders; Y: (trials, 40, 2) Index vx, vy in cm/s
```

- **Normalization:** median and 1.4826 x MAD per channel and feature over the first-go to last-reward window (± 1 s). For held-out sessions, `normalize='pool'` keeps the query trials out of the statistics.
- **`valid`** is False where any feature or target is NaN; mask those bins in the loss and the metric.
- **Reference (recording vs CAR):** not settled yet. The recording reference is the default and CAR an ablation. In the closest published comparison (Ahmadi et al. 2021, Utah array in macaque M1, hand kinematics), CAR gave the best decoding with the LMP feature; for stimulation artifacts, linear regression reference outperformed CAR (Young et al. 2018); and the reference changes PSD slopes and phase (Shirhatti et al. 2016). Because the kept channels share most of their signal, CAR may remove movement information here. The first baseline decoder (section 3) compares `export_none` and `export_car` on validation days, and its result will replace this note.
- **Compute:** `python butters_features_v1.py --variants export_none parrm_none` (about 6.5 s per session per variant; skips existing files unless `--overwrite`).

## Provenance

| Step | Script |
|---|---|
| Session list and inclusion | `conditions_eACS_AM-kHz_v2.csv` (from the lab sheet, task files and video checks) |
| Export | `export_session_v2.m` |
| Fixed channel set | `patch_channel_keep_v1.m` |
| QC collection | `collect_qc_v1.m` |
| July 11 trials | `dump_task_timing_v1.m`, then `reconstruct_trials_tdt_v2.py` |
| Splits | `make_splits_v1.py` |
| PARRM | `run_PARRM_LFP_v5.m` |
| Kinematics | DeepLabCut, then `export_kinematics_v3.py` |
| Screen-cm calibration | `butters_loader_v2.calibrate_all()` |
| Features | `butters_features_v1.py` |
| Session manifest | built from `export_summary.csv`, `conditions_eACS_AM-kHz_v2.csv`, the trial tables, `parrm_summary.csv`, `session_summary.csv` and `kinematics_calibration.csv` |

Excluded from the 102 candidate sessions (reasons in `conditions_eACS_AM-kHz_v2.csv`): random target order (21), video frame count mismatch or no video (8), a stimulation waveform different from the others (1), days whose stimulation had no AM (2).
