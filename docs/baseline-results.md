# Results: five baselines, 14 held-out stimulation sessions

Target: Index finger velocity (`Index_vx_cm`, `Index_vy_cm`), R² on the fixed 30-trial query
set (blocks 3–5), 50 ms bins, NaN targets masked. Ridge and KNN are evaluated per draw and
per k (10 draws each); MLP, EEGNet and GRU are trained once per seed on the stimulation-off
training split and never see the calibration trials.

`r2_off` is the same decoder applied to that day's no-stimulation session, which sets the ceiling.

## Main table (mean over sessions × draws × seeds)

| Method | k | 3 Hz R² | 20 Hz R² | off 3 Hz | off 20 Hz |
|---|---|---|---|---|---|
| Ridge, raw, no adapt (B0 cond. 1) | 0 | 0.025 | **−0.010** | 0.039 | 0.029 |
| Ridge, PARRM, no adapt (cond. 2) | 0 | 0.025 | **−0.020** | 0.039 | 0.029 |
| Ridge, raw, refit (cond. 3) | 20 | 0.030 | 0.055 | 0.023 | 0.039 |
| Ridge, PARRM, refit (cond. 4) | 20 | 0.029 | 0.053 | 0.021 | 0.039 |
| KNN, stim-off bank | 0 | 0.026 | 0.027 | 0.036 | 0.020 |
| KNN, k calibration trials | 2 | −0.002 | −0.001 | −0.003 | −0.003 |
| | 5 | 0.005 | 0.010 | 0.004 | 0.008 |
| | 10 | 0.017 | 0.030 | 0.011 | 0.021 |
| | 20 | 0.029 | 0.051 | 0.021 | 0.032 |
| MLP (2 hidden, 90 k params) | 0 | 0.037 | −0.023 | 0.047 | 0.041 |
| EEGNet (5.9 k params) | 0 | −0.077 | 0.068 | −0.072* | 0.066* |
| GRU | 0 | 0.026 | 0.030 | 0.036 | 0.029 |

\* Stale. These EEGNet off scores came from the stimulated session instead of the no-stimulation
reference. The code is fixed, so rerun the EEGNet section of `cs182_decoder_grouped.ipynb` or set `RUN_PIPELINE = True` to replace them.

## What the numbers say

1. **20 Hz AM destroys the decoder, 3 Hz does not.** The no-adapt ridge is *negative* at
   20 Hz (−0.010 raw, −0.020 PARRM): worse than predicting the mean. At 3 Hz it survives at
   0.025. Refitting on k=20 ON trials lifts 20 Hz to 0.053–0.055, so the loss is
   recoverable. This is the effect the project is about.
2. **PARRM cleaning without labels does not help** (20 Hz: −0.020 vs −0.010 raw). Cleaning
   the input while keeping a decoder fitted on other days changes nothing; the gain only
   appears once labelled ON trials are used. That is the cleaning-vs-labels decomposition
   the four ridge conditions were designed to separate.
3. **k=2 recovers nothing; the gain is nearly all between k=10 and k=20.** Monotone in both
   frequencies and for both ridge and KNN. With 30 query trials and 7 sessions per frequency
   the CIs are wide, but the ordering is consistent across all 10 draws.
4. **Nonparametric adaptation is competitive.** KNN with the k calibration trials as its
   entire model reaches 0.029 (3 Hz) and 0.051 (20 Hz) at k=20. That is within 0.004 of ridge
   refit, with zero weight updates. This is the cheapest version of the in-context claim.
5. **EEGNet is not a reliable baseline here.** It has the best *validation* R² (0.10–0.11 on
   held-out blocks of training sessions) and the worst test behaviour at 3 Hz (−0.077), while
   being the best baseline at 20 Hz (0.068). Trainability does not translate across days.
6. **The GRU is the most stable deep baseline** (0.026 / 0.030, off 0.036 / 0.029) and needs
   no tuning to be sane. The MLP is better than the GRU at 3 Hz (0.037) and worse at 20 Hz
   (−0.023), so the MLP-vs-GRU ordering depends on the artifact frequency.

## Two problems with the evaluation design that the project should know about

**1. Absolute decoding accuracy is at the noise floor.** Within one session, fitting and
testing on different trials of the same no-stim session gives R² ≈ 0.067
(`results/diagnostic_within_day_off.csv`, range 0.039–0.126), with no stimulation and no
cross-day transfer involved. Speed is ~10× more decodable than per-bin velocity, which
points at the velocity target rather than the features: DeepLabCut yields ~1.2 confident
frames per 50 ms bin and `Index_vx_cm` has ±140 cm/s outliers, so a central-difference
velocity is noise-dominated. Recommendation: report speed or a 3-bin smoothed velocity
alongside per-bin velocity, and expect all R² values to be small.

**2. Consequently the recovery fraction is not usable.** ρ =
(R²_method − R²_no_adapt)/(R²_off − R²_no_adapt) has a denominator of ~0.01–0.04, so it
swings between −19 and +8 with CIs spanning an order of magnitude
(`results/table_recovery.csv`, `results/recovery_vs_k.png`, kept only as evidence of the
instability). Report raw R² and paired
differences. For a usable ρ the experiment needs either more query trials, a stronger
decoder, or a less noisy target.

## Figures

| File | What it shows |
|---|---|
| `fig1_r2_vs_k.png` | R² vs k for every method, 3 Hz and 20 Hz, 95 % session-bootstrap bands, no-adaptation baselines as dashed lines |
| `fig2_method_comparison.png` | Grouped bars per method: single hatched bar when the method cannot adapt, k=0 vs k=20 pairs otherwise |
| `fig3_ridge_factorial.png` | The attribution figure: cleaning × adaptation 2×2. At 20 Hz the whole gain sits in the label column |
| `fig4_session_heatmap.png` | Per-session R² for all methods; EEGNet flips between −0.15 and +0.15 across sessions |
| `fig5_training_curves.png` | Validation R² per epoch with the early-stopping checkpoint marked |
| `fig6_noise_floor.png` | Same-day fit vs cross-day transfer on no-stimulation sessions, the ceiling that makes ρ unstable |
| `predictions/*_actions.png` | One session decoded: velocity, speed, scatter and per-method R² |
| `predictions/*_trials_speed.png` | Speed envelope per trial, measured vs decoded |
| `predictions/*_predictions.npz` | per-bin measured and decoded velocity for every query trial |
| `results/training_logs/*.json` | per-epoch training/validation curves behind fig5 |

The prediction overlays show what the R² numbers mean physically: the decoders recover the
*movement onset* (speed rises ~0.25 s after the go cue) but compress the amplitude by roughly
5×, so the measured 85–120 cm/s peaks appear as 10–25 cm/s predictions. Nothing is smoothed
in those panels.

## Files

| Path | Contents |
|---|---|
| `results/draws.json` | 14 test stim sessions × k∈{2,5,10,20} × 10 nested draws, fixed query set |
| `results/table_main.csv` | per method × frequency × k: mean R² with 95 % CI, off R², SEM |
| `results/table_recovery.csv` | ρ with 95 % bootstrap CI (unstable, see above) |
| `results/fig*.png` | the figure suite above |
| `results/table_main.{csv,md,tex}` | main table in three formats |
| `results/table_ridge_factorial.csv` | the four ridge conditions |
| `results/table_per_session.csv` | per-session R² for every method |
| `results/diagnostic_within_day_off.csv` | within-day OFF R² ceiling per test day |
| `results/results_*.parquet` | raw rows: method × session × k × draw × seed |
