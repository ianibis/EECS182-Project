# GRU vs EEGNet: what the trained checkpoints actually say

Every number here was rebuilt from the saved artifacts; the tables are committed under
`task4/results/` (`decoder_comparison.csv`, `decoder_paired.csv`,
`checkpoint_audit.csv`).

It reads `results/results_{ridge_knn,mlp,gru,eegnet}.parquet`, `task4/results/task4_2_*.csv`
and the checkpoints themselves, then writes:

| Output | Contents |
|---|---|
| `task4/results/decoder_comparison.csv` | Every method × input × frequency, with session CIs |
| `task4/results/decoder_paired.csv` | Paired differences with CIs and win/loss counts |
| `task4/results/checkpoint_audit.csv` | Each saved weight file rescored against its cached row |
| `task4/results/artifact_severity.csv` | Per session, artifact amplitude relative to the ON spread |
| `results/fig_grueegnet.png` | ON R² by method and frequency |

## The comparison is valid

All four baseline parquets and the 4.2 table cover the same 14 held-out stimulation-ON
sessions, score the same 30 query trials per session, and report the same per-session bin
counts (937–1,070). Ridge and KNN are evaluated per draw and per k; only the largest
k = 20 rows are used so the "refit k=20" label is accurate. Seeds and draws are averaged
inside a session before anything else, because the sessions are the independent unit;
confidence intervals resample the 14 sessions (7 per AM frequency), not the 70 seed-rows.

## Query R² on stimulated trials

Fixed models never see an ON label; adapted models spend 20 labelled ON trials.

| Model | Adapts | All (14) | 3 Hz (7) | 20 Hz (7) | OFF |
|---|---|---:|---:|---:|---:|
| Ridge, raw, refit k=20 | 20 ON trials | **+0.042** | +0.030 | **+0.055** | 0.031 |
| Ridge, PARRM, refit k=20 | 20 ON trials | +0.041 | +0.029 | +0.053 | 0.030 |
| KNN, in-context k=20 | 20 ON trials | +0.040 | +0.029 | +0.051 | 0.027 |
| EEGNet + tones (5 seeds) | — | +0.030 | +0.031 | +0.030 | 0.032 |
| GRU (5 seeds) | — | +0.029 | +0.028 | +0.029 | 0.033 |
| KNN, OFF bank | — | +0.026 | +0.026 | +0.027 | 0.028 |
| Ridge, raw, no adapt | — | +0.008 | +0.025 | −0.010 | 0.034 |
| MLP (5 seeds) | — | +0.004 | +0.039 | −0.032 | 0.045 |
| Ridge, PARRM, no adapt | — | +0.002 | +0.025 | −0.020 | 0.034 |
| EEGNet, 5-seed ensemble | — | −0.082 | −0.280 | **+0.117** | 0.082 |
| EEGNet, single seed | — | −0.145 | −0.337 | +0.047 | 0.037 |

Session-bootstrap 95 % CIs for the rows that matter: GRU [0.018, 0.040], EEGNet
[−0.258, −0.034], EEGNet ensemble [−0.193, 0.025], EEGNet + tones [0.001, 0.058], ridge refit
[0.032, 0.054]. The OFF column is the same decoder on the same day's no-stimulation
recording, which is the ceiling for that session.

## Paired differences

Same sessions on both sides, so these are paired; the CI resamples sessions.

| Comparison | All (14) | 3 Hz (7) | 20 Hz (7) |
|---|---:|---:|---:|
| GRU − EEGNet | **+0.174** [+0.063, +0.287] | **+0.365** [+0.282, +0.439], 7/7 wins | −0.017 [−0.082, +0.041] |
| GRU − EEGNet ensemble | **+0.110** [+0.003, +0.223] | +0.308 [+0.236, +0.378] | **−0.087** [−0.132, −0.045], 1/6 |
| EEGNet + tones − EEGNet | **+0.175** [+0.074, +0.283] | +0.367 [+0.300, +0.433] | −0.017 [−0.042, +0.011] |
| EEGNet ensemble − EEGNet | +0.064 [+0.047, +0.082] | +0.057 [+0.037, +0.087] | +0.070 [+0.045, +0.093] |
| Ridge refit k=20 − GRU | +0.014 [−0.001, +0.028] | +0.002 [−0.013, +0.015] | +0.026 [+0.002, +0.047] |
| KNN in-context − ridge refit | −0.002 [−0.011, +0.006] | −0.000 [−0.011, +0.013] | −0.004 [−0.017, +0.008] |

## What this says

**1. EEGNet's failure is specific to 3 Hz, not to deep models.** Overall the GRU beats
EEGNet by 0.174 with a CI well clear of zero, but that number is entirely the 3 Hz
sessions: +0.365 there, 7 of 7 sessions, and −0.017 at 20 Hz with no separation. Reporting
a single pooled "GRU beats EEGNet" would be wrong about both halves.

**2. At 20 Hz the EEGNet ensemble is better than the GRU.** Ensemble − GRU at 20 Hz is
+0.087 [+0.045, +0.132], 6 of 7 sessions. The GRU is the *stable* baseline, not the best
one; the ensemble is the better 20 Hz model. A single seed hides this (−0.017 against the
GRU at 20 Hz, no separation), which is the argument for reporting ensembles of small models
rather than single runs.

**3. The three deep models fail at different frequencies.** EEGNet collapses at 3 Hz
(−0.337) and is fine at 20 Hz (+0.047). The MLP does the opposite: fine at 3 Hz (+0.039),
negative at 20 Hz (−0.032). The GRU is flat at both (0.028 / 0.029). Three architectures,
three different signatures, so this is not seed noise and not a property of "deep models".

**4. Tone augmentation removes the asymmetry, and only buys 3 Hz.** Adding a synthetic
3/20 Hz sine to the OFF training trials moves EEGNet from −0.337 to +0.031 at 3 Hz, a gain
of 0.367 with 7 of 7 sessions, and changes nothing at 20 Hz (−0.017). The result lands
almost exactly on the GRU's flat profile, which is what a model that has learned to ignore
the interfering frequency should look like.

**5. Twenty labelled ON trials beat the GRU, but not decisively, and not at 3 Hz.** Ridge
refit k=20 − GRU is +0.014 overall with a CI that just touches zero, +0.026 at 20 Hz, and
+0.002 at 3 Hz. KNN with the same 20 trials is indistinguishable from ridge (−0.002). The
project's claim — that labels, not cleaning, close the gap — holds, but the margin over a
fixed deep baseline is 0.01–0.03 R², not the 0.05 the older tables implied.

## Why 3 Hz is the one that breaks

`task4/results/amp10_noise_variance.csv` and the 10× arms in `task4/README.md` §4.2.10 isolate the
mechanism; `semisynthetic/semi_synthetic_README.md` has the full table.

- The artifact is **not** zero-mean white noise. After removing each channel's mean, only
  12 % (3 Hz) to 42 % (20 Hz) of the variance is explained by the AM phase, and only 6–7 %
  sits in the first ten harmonics — it is a drifting gain times a template, not a tone.
- What reaches the decoder is the 50 ms bin average. At 20 Hz one artifact period equals one
  bin, so binning removes 79 % of the variance and leaves a lag-1 correlation of 0.98: a slow
  drift that the 9-bin feature history and the ridge intercept absorb. At 3 Hz a bin covers
  1/6.7 of a cycle, so 56 % survives and stays oscillatory (lag-1 0.35).
- That statistic tracks the failure: per session, the RMS of the binned term correlates with
  EEGNet's ON R² at Spearman **−0.94** across all 14 sessions, and with the GRU's at +0.10.
  The GRU reads band-power features, which have already averaged the artifact away; EEGNet
  reads raw 1 kHz LFP and cannot.
- Cleaning the input rescues the fixed decoder: EEGNet goes from −0.145 (recorded ON) to
  **+0.050** on the PARRM-cleaned ON recording, and the ensemble from −0.082 to **+0.106**,
  above its own OFF control. Reusing the 1× template on the amplified signal does not help
  (−2.944), so the rescue needs the template scaled to the artifact actually present.

The one-line version: the three deep models differ in how they average the artifact, not in
how much capacity they have. Band-power features and 50 ms pooling hide it at 20 Hz; only
raw-LFP, single-bin EEGNet is exposed to the 3 Hz oscillation.

## The GRU was retrained at its own sweep winner

Everything above uses the archived GRU (`hidden=128`, one layer), whose weights do not
reproduce their cached rows (see the audit below). A twelve-setting sweep over hidden width,
depth and dropout, ranked by clean validation R2 on the same three held-out sessions, selects
**hidden 256, one layer, no dropout** at +0.0951 against +0.0751 for the 128-unit setting, and
that model was retrained for five seeds:

| GRU | clean | stimulated | 3 Hz | 20 Hz |
|---|---:|---:|---:|---:|
| single-seed mean (5 seeds) | +0.036 [+0.017,+0.056] | +0.035 [+0.020,+0.049] | +0.034 | +0.037 |
| 5-seed ensemble | **+0.049** [+0.031,+0.068] | **+0.046** [+0.030,+0.061] | +0.045 | +0.048 |

Day-level bootstrap, five held-out days. Both rows come from `utils.day_level_summary` applied to
`task4_gru_final.csv`: the point estimate and interval for the ensemble use the `seed == -1` rows,
and the single-seed row uses the `seed >= 0` rows, which averages the five seeds inside each day
before the day bootstrap. Per-frequency columns are the stimulated scores.

Two consequences for the claims above. First, the retrained GRU beats the archived 128-unit one
on the stimulated input (+0.046 against +0.029) and on the ensemble-versus-single comparison,
though not on clean input (+0.049 against +0.039); the "stable baseline" reading therefore holds
more strongly, not less. Second, it is *tied* with
spending 20 labelled stimulated trials on a ridge refit: paired over 14 sessions, ridge refit
minus the GRU ensemble is +0.0004 [-0.017,+0.017], and every per-frequency interval spans zero.
The paper's claim is therefore that labels buy a few hundredths of R2, not a decisive margin.

Artifacts: `task4/results/gru_sweep.csv`, `task4_gru_final.csv`, `task4_gru_final_seeds.csv`,
`task4_gru_final_config.json`; the sweep and the five-seed run both live in
`kaggle/train_gru.py`.

## Checkpoint audit

`task4/results/checkpoint_audit.csv` reloads every saved weight file and rescoring it:

| Artifact | Reproduces its cached row? |
|---|---|
| `task4/checkpoints/task4_2_*.pt` (10 files) | **Yes**, to 0.0000 per session |
| `results/checkpoints/ridge_{raw,parrm}.npz` | **Yes**, to 0.0000 per session |
| `results/checkpoints/gru_seed0.pt` | **No**: 0.0314 vs 0.0309 ON, 0.0389 vs 0.0357 OFF, worst session gap 0.032 |
| `results/checkpoints/mlp_seed0.pt` | **No**: 0.0260 vs −0.0123 ON, 0.0532 vs 0.0420 OFF, worst session gap 0.438 |

The 4.2 weights and the ridge coefficients reproduce the published tables exactly. The older
GRU and MLP weights do not, and only seed 0 of each was kept, so four of the five seeds in
those parquets cannot be audited at all. Read the GRU and MLP rows as "the cached run", not
as "the saved model". The MLP row in particular should not be cited without a rerun.

## Caveats

- **Absolute R² is at the noise floor.** Fitting and testing on different trials of the same
  no-stimulation session gives R² ≈ 0.067 (0.039–0.126 across days), so the best number in
  this table (+0.055) is at that ceiling. Differences under ~0.01 are not resolvable with 14
  sessions; that is why the paired CIs, not the ranks, carry the claims.
- **Task 3's GRU ensemble is a different experiment.** Its OFF R² is 0.198 because it
  fine-tunes on 20 same-day OFF trials and adds a time-dependent correction; its ON scores
  (0.090 at 3 Hz, 0.109 at 20 Hz) are not comparable to anything here and are reported
  separately in `task3/task3_README.md`.
- **Seven sessions per frequency.** All 3 Hz sessions fall on three days and all 20 Hz
  sessions on two, so frequency is confounded with day. The 3 Hz/20 Hz split is a real
  signal in the artifact structure, but the size of the gap may partly be "those days were
  easier".
- **The PARRM arm changes normalisation too.** `loader.normalize` z-scores by a MAD over the
  task interval, so cleaning also changes the scale (MAD ratio 1.9 at 3 Hz, 1.3 at 20 Hz).
  The cleaned-arm comparison is not a pure artifact-amplitude manipulation.
