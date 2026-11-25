# Task 5: in-context recalibration of a movement decoder

This is the experiment plan for Task 5. It was written **before any Task 5 model was trained**, so the
success criteria below cannot be bent to fit the results. Any later change is recorded in the
[change log](#8-change-log) with the reason for it.

## 1. The question

Given `k` labeled stimulation-ON trials from a new session, can a transformer recalibrate **without any
weight update**, simply by reading those trials as context?

| ID | Hypothesis (from the proposal) | How we test it |
|---|---|---|
| H1 | In-context adaptation improves with `k` | R² at k = 0, 2, 5, 10, 20 |
| H2 | At k ≤ 5 it matches or beats fine-tuning and LoRA; by k = 20 fine-tuning catches up | Same draws as Task 4. **Needs the Task 4 results, which do not exist yet** |
| H3 | Shuffling the context labels removes the in-context gain | Control C1 |

## 2. The main risk: the model could score well without reading the brain

Task 3 found that the average velocity at each moment after the go cue reaches R² ≈ 0.089, while ridge on
brain features reaches about 0.05. The monkey repeats the same sequence, so **context labels alone reveal
the average trajectory**. A transformer that copies this average would appear to "recalibrate" while
ignoring the neural signal.

So every result is reported next to two **labels-only references**, neither of which sees the brain signal:
a shrinkage average of the context trajectories, and a blind transformer with the same architecture and
training. The headline number is the **neural gain** against whichever reference is stronger:

```
neural gain(k) = R²(model, k) − R²(stronger labels-only reference, k)
```

The stronger reference is chosen per input and `k` on validation days, then frozen for the test days.
Control C3 tests the same thing from the model side, by hiding the query trial's brain signal.

## 3. Data rules

- **Splits:** `splits.json`, unchanged. 18 training days, 3 validation days, 5 test days.
- **Target:** Index fingertip velocity (vx, vy) in cm/s, 50 ms bins, from 0.5 s before go to 1.5 s after.
  Invalid bins are masked from both loss and metric. July 11 is excluded (no cm/s velocity).
- **Input:** 35 fixed channels of 1 kHz LFP, recording reference. Each token sees the past 500 ms.
- **Normalization:** per-channel robust z-score from the 20-trial calibration pool (labels not used), then
  clipped to ±10, as agreed in Task 3 because of the channel 64 artifact.
- **Training uses only stimulation-OFF trials from training days**, with simulated artifacts. Artifact
  templates come only from **training-day** stimulation sessions, excluding the sessions flagged as poor
  templates in the data README.
- **The test days are scored once**, after every choice has been fixed on validation days.

## 4. Method

**Shift generator (5.1).** One shift is sampled per episode and applied to every trial in it:
AM frequency (3 or 20 Hz), a real PARRM template, per-channel gains, amplitude, phase, period jitter, slow
amplitude modulation, mild nonlinearity, a shared component across channels, a non-periodic 1/f residual,
channel gain drift, and channel dropout. Each realism option can be switched off.

**Episode (5.2).** As at test time: draw a 20-trial pool and a separate query trial from one session,
apply one shift, normalize from the pool, and use the first `k` pool trials as context.

**Model (5.3).** EEGNet-style encoder → one token per 50 ms → 4-layer causal transformer (width 128,
4 heads, about 820k parameters). Context is laid out as alternating signal and movement tokens, followed by
the query's signal tokens. The loss is MSE on valid query bins only.

**Training (5.4).** Overfit check, then a learning-rate and warmup sweep on validation days, then 5 seeds.

## 5. Evaluation

All scores use the Task 3 definition: pool the valid bins of a session's 30 query trials, compute R² for
vx and vy, and average them. Each session, `k`, draw (10 from `splits.json`) and seed is saved.

| Condition | Data |
|---|---|
| Real | Recorded ON trials of held-out sessions |
| Semi-synthetic | The same trials with the PARRM artifact amplified 10× (the main result, per Task 3) |
| OFF reference | Same-day no-stim session |

| Control | Change | Expected if the model truly uses context |
|---|---|---|
| C1 Shuffled labels | Context velocities are shuffled across trials and time | The gain from `k` disappears |
| C2 OFF context | Context comes from the same-day OFF session | Worse than ON context under a strong artifact |
| C3 No query signal | The query's brain signal is zeroed | Falls to the labels-only baseline |
| C4 Stimulation monitor only | A GRU reads only the StiF envelope and phase, from the trial's own monitor or another trial's | Equal R², so stimulation timing leaks no movement information |

Uncertainty: 95% bootstrap intervals over sessions, and over seeds where stated. Task 3 showed that a
single session can move a mean, so paired per-session differences are preferred.

## 6. Success criteria (fixed in advance)

| Check | Pass condition |
|---|---|
| Overfit test | Loss on one small batch falls below 5% of its starting value |
| Simulated vs real | Simulated spectra show peaks at the AM frequency and harmonics, as real ON trials do |
| H1 | R² at k = 20 exceeds k = 0, and the paired 95% interval excludes 0 |
| **Real neural gain (primary)** | Neural gain against the stronger labels-only reference at k = 20 is above 0, and the paired 95% interval over sessions excludes 0 |
| H3 | Shuffling labels loses at least half of the k = 20 gain |
| C2 | ON context beats OFF context at k = 20, and the paired 95% interval excludes 0 |
| C3 | Hiding the query signal lowers R², and the paired 95% interval excludes 0 |
| C4 | R²(own monitor) − R²(another trial's monitor) is at most 0.01 (revised, see change log) |

Neural gain at other `k` values is descriptive and does not change the primary verdict. Each paired check
also reports a stricter interval that resamples whole days.

Two measures describe recovery from stimulation. `fraction_recovered` is the proposal's metric, the gain
from context divided by the OFF-minus-ON loss at k = 0. It can exceed 100% because context also helps OFF
sessions. `gap_closed` compares ON and OFF at the same `k`, so it counts only stimulation-specific recovery.

If the model does not beat the labels-only baseline, we report that as the finding rather than tuning
against the test days.

## 7. Scope

Task 5 does not implement the Task 4 baselines (EEGNet, fine-tuning, LoRA, oracle). The result files use the
per-session, k, draw and seed format so that those methods can be joined later. A ridge refit on the same
`k` trials is included because Task 5.7 needs a least-squares reference.

## 8. Change log

Changes made after this plan was written are listed here, with the evidence that prompted them.

| Time | Change | Reason |
|---|---|---|
| Nov 23, before training | Encoder normalization changed from GroupNorm to per-time-step LayerNorm | The causality unit test failed: GroupNorm averages over the trial, so early tokens could see future samples |
| Nov 23, after a 100-step smoke test | **Neural gain is now measured against a blind transformer** (same architecture and training, brain signal hidden). A shrinkage labels-only baseline is reported beside it. This replaces the raw k-trial average in Section 2 | After only 50 steps the old "gain" was +0.15. The raw average of 2 trials is so noisy that learning the typical trajectory beats it, so the old baseline overstated neural decoding |
| Nov 23, after check 5.1.3 | Added a non-periodic 1/f residual to the simulated artifact, 10-55% of its amplitude and 90% shared across channels | On training-day sessions the real artifact has broadband power between harmonics (median amplitude share 0.16 at 3 Hz, 0.24 at 20 Hz). The first generator was almost perfectly periodic |
| Nov 23, validation control run | **C4 pass condition changed** from "monitor-only R² ≤ 0.01" to "matched minus mismatched monitor R² ≤ 0.01". The old wording cannot be met by any recurrent model | A GRU reaches R² ≈ 0.05 even when given another trial's monitor signal, because it can count time since the trial start. Only the difference between the two isolates a real stimulation-movement link |
| Nov 23, sweep | Grid extended to a learning rate of 2e-3, and finalists compared at the full 6000 steps. **Final setting: learning rate 3e-4, warmup 300, 6000 steps** | 1e-3 won at the grid edge after 3000 steps (validation R² 0.139) but overfit by 6000 steps (best 0.131, final 0.103). 3e-4 reached 0.153 at 6000 steps |
| Nov 23, after the first final seed | **Primary neural gain now uses the stronger of the two labels-only references** at each input and k, chosen on validation days and frozen before the test days are opened | The blind transformer learned poorly (validation R² about 0.05, below the labels-only average at about 0.08), so using it alone would overstate the neural gain |
| Nov 23, after the first final seed | Added `gap_closed` beside `fraction_recovered` | Context also helps OFF sessions, so `fraction_recovered` exceeded 100% without showing that the stimulation-specific gap had shrunk |
| Nov 23, before opening the test days | **All choices frozen after the 5-seed validation run.** k = 20 is the primary neural-gain test, C2 is listed as a pass/fail check with the paired test already used in the analysis, and checksums are in `task5_results/frozen_before_test.sha256` | Validation showed significant gains at some smaller `k` but not at k = 20. Keeping the planned k = 20 test prevents picking the most favorable `k` after seeing results |
