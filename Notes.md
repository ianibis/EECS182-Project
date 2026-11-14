# Notes

These notes summarize what we found in Task 3, the "problem size check". The question behind it: before we build anything fancy, how well does a simple decoder predict hand movement from the brain signal, and how much does stimulation hurt it?

## Reproducing the numbers

The `results/` folder isn't on GitHub, so you'll need to run the scripts yourself. First, one-time setup:

1. Unzip the shared data into a `data/` folder in the repo. The expected folder names are listed in `PATHS` at the top of `task3_common.py`.
2. Create the virtual environment and install the packages:
   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```

On Mac or Linux, use `.venv/bin/pip` and `.venv/bin/python` instead.

Then run the scripts. Each one takes about 1 to 2 minutes and saves CSV files to `results/task3/`:

```
.venv\Scripts\python task3_1_ridge.py                  # Task 3.1 with the default settings
.venv\Scripts\python task3_1_ridge.py --lags 19        # longer history (we also tried 14 and 29)
.venv\Scripts\python task3_1_ridge.py --features bands # band power only, no LMP
.venv\Scripts\python task3_2_parrm.py                  # Task 3.2 (same options as 3.1)
.venv\Scripts\python task3_3_stif.py                   # Task 3.3 (accepts --lags)
```

## Terms used below

- **ON and OFF trials.** ON trials are recorded while stimulation is running; OFF trials have no stimulation. In a stimulation session all 50 trials are ON, so OFF trials only come from the no-stim sessions.
- **Calibration pool and query trials.** Each validation and test session has 50 trials in 5 blocks of 10. The first 20 trials (blocks 1 and 2) are the calibration pool. The last 30 (blocks 3 to 5) are the query trials. Every R² in these notes is computed on the query trials, the same trials the later methods (fine-tuning, LoRA, in-context learning) are scored on, so our numbers line up with theirs.
- **R².** How close the predicted velocity is to the real one. 1 means perfect, 0 means no better than always guessing the average velocity, and below 0 means worse than that. We compute it separately for sideways (vx) and up-down (vy) velocity and report the average.
- **LMP and band power.** The two kinds of features the decoder reads from each channel. LMP (local motor potential) is the slow average voltage over the last quarter second. Band power is how strong the signal is in each of 7 frequency ranges, such as 13 to 30 Hz.
- **Lags (history).** How many earlier 50 ms time steps the decoder sees in addition to the current one. 9 lags means half a second of signal in total.
- **λ (lambda).** The strength of the ridge penalty. A larger λ keeps the decoder's weights small and pushes its guesses toward the average.
- **Recording reference vs. CAR.** Two versions of the brain signal. The recording reference is the signal as recorded. CAR (common average reference) subtracts the average of all channels at each moment.

## Task 3.1: How much does stimulation hurt a standard decoder?

### What we did

We trained a ridge decoder that predicts fingertip velocity from the brain signal, one prediction every 50 ms. It only ever saw OFF trials during training: 40 trials from each of the 13 no-stim sessions on training days, about 17,600 time steps in total. The three July 11 sessions were left out because they have no velocity in cm/s.

For each trial, the decoder looks at the window from half a second before the go cue to 1.5 seconds after it. Features are normalized within each session, so sessions with louder or quieter recordings become comparable. For validation and test sessions, the normalization is computed from the first 20 trials only, so the graded trials stay untouched. After normalizing, we cap every value at ±10 to protect against artifacts (see the channel 64 outlier below).

We tried 13 values of λ, from 1 to 1,000,000, with both the recording reference and CAR. We kept whichever combination scored best on the 3 validation-day OFF sessions. With those settings fixed, we scored the decoder on the test days: on each of the 14 stimulation sessions (7 at 3 Hz, 7 at 20 Hz), and on the no-stim session recorded the same day. The difference between the two (ON minus OFF) is how much accuracy stimulation costs.

### What we found

Each row of the table is one setting. "OFF R²" is the average over the 5 test-day no-stim sessions. The last two columns show the ON score, the same-day OFF score, and the difference in brackets (negative means stimulation made it worse).

| Features | History | Validation R² | Chosen λ | OFF R² (test days) | 3 Hz: ON vs. OFF | 20 Hz: ON vs. OFF |
|---|---|---|---|---|---|---|
| Band power only | 9 lags (0.5 s) | 0.021 | 316,228 | 0.015 | 0.012 vs. 0.010 (+0.001) | 0.017 vs. 0.016 (+0.001) |
| LMP + band power | 9 lags (0.5 s) | 0.035 | 31,623 | 0.045 | 0.025 vs. 0.048 (−0.023) | 0.016 vs. 0.028 (−0.012) |
| LMP + band power | 14 lags (0.75 s) | 0.052 | 31,623 | 0.051 | 0.025 vs. 0.051 (−0.025) | 0.021 vs. 0.032 (−0.011) |
| LMP + band power | 19 lags (1 s) | **0.053** | 316,228 | 0.053 | 0.031 vs. 0.046 (−0.016) | 0.029 vs. 0.041 (−0.011) |
| LMP + band power | 29 lags (1.5 s) | 0.052 | 316,228 | 0.052 | 0.032 vs. 0.044 (−0.012) | 0.029 vs. 0.043 (−0.014) |

The main takeaway is that **the decoder is weak even without stimulation**. The best version reaches an R² of about 0.05, far below the 0.2 that slide 24 uses as the bar for "poor accuracy even without stimulation".

A few other things stood out:

- **LMP helps.** With half a second of history, adding LMP to band power raised validation R² from 0.021 to 0.035, and test-day OFF R² from 0.015 to 0.045.
- **More history helps a little, then stops helping.** Going from 0.5 to 0.75 seconds raised validation R² from 0.035 to 0.052. Beyond that it stays flat. One second (19 lags) scored highest on validation, but 0.75 to 1.5 seconds are practically tied.
- **The recording reference beat CAR in every setting.** For example, 0.035 vs. 0.021 for LMP + band power with 9 lags.
- **The decoder needs a strong penalty.** The best λ was always 31,623 or 316,228. With weaker penalties it memorizes the training trials and does worse on new ones.
- **Stimulation costs a little accuracy, mostly at 3 Hz.** With LMP included, ON trials score about 0.01 to 0.025 lower than the same-day OFF session. At 3 Hz, ON scored lower than OFF in 7 of 7 sessions with 9 or 14 lags, 6 of 7 with 19 lags, and 5 of 7 with 29 lags. That fits the README, which says LMP is the feature most affected by 3 Hz stimulation. With band power alone there's no gap, but there's also almost nothing to lose.

### Why is accuracy so low?

To make sure the low numbers weren't caused by a bug, we ran a few extra checks. These used throwaway scripts that aren't in the repo, were run before we added the ±10 cap, and used 9 lags.

| Check | Band power | LMP + band power | What it tells us |
|---|---|---|---|
| Score on its own training data (λ = 100) | 0.18 | 0.26 | The decoder can partly memorize the training data, but that doesn't carry over to new trials |
| Train on 40 trials, test on the held-out block from the same sessions | 0.022 | 0.042 | About as poor as on new days, so day-to-day drift isn't the problem |
| A separate decoder for each session (best case) | typical 0.03, best 0.07 | typical 0.08, best 0.16 | Even the most favorable setup stays below 0.2 |
| No brain data at all: predict the average velocity at each moment after the go cue | 0.089 | 0.089 | This beats the brain decoder |

The last row is the most telling. The monkey reaches for the same targets in the same order on every trial, so just knowing the time since the go cue predicts the movement better than the brain features do. That means the velocity data and the trial timing are fine. The weak link is how little velocity information these features carry.

### Keep in mind: the OFF session was always recorded first

On all 5 test days, the no-stim session was recorded before the stimulation sessions. So when ON scores lower than OFF, part of that could come from being later in the day (the monkey getting tired, attention drifting, electrodes changing) rather than from the stimulation itself. Comparing against the same day's OFF session rules out differences between days, but not this ordering within a day.

### Outlier: a huge artifact on channel 64 in session 250616-102853

This 20 Hz test-day session has a very large voltage spike on **channel 64** during its last three graded trials (trials 45, 46 and 47 in block 5, roughly 392 to 400 seconds into the recording).

It's far too big to be brain activity. The LMP on that channel reaches about 78,700 µV in trial 45, then −4,670 µV in trial 46 and −1,500 µV in trial 47. Normal LMP values are tens to hundreds of µV. After normalization, the peak is about 5,300, when clean values stay within about ±10. The manifest doesn't mention this artifact: its `known_problems` entry for this session only notes a disagreement about the stimulation frequency.

Before we handled it, this one artifact dragged the session's R² down to −0.71 (LMP + band power, 9 lags), enough to pull the whole 20 Hz average below zero. Dropping just the 86 affected time steps brought it back to 0.043, in line with the other sessions. The band-power-only decoder barely noticed (R² 0.030), because the artifact shows up mainly in the LMP.

**How we handle it:** after normalization, every feature value is capped at ±10 (`CLIP_Z` in `task3_common.py`), in every session, for both training and evaluation. With the cap, this session scores 0.040.

**If your task uses LMP from this session** (fine-tuning, LoRA, the in-context model, PARRM + ridge), you'll hit the same artifact. Please use the same ±10 cap, or let's agree on a different fix as a team, so all methods are compared fairly.

## Task 3.2: Does PARRM cleaning help?

### What we did

PARRM is a filter that removes the part of the signal that repeats in step with the stimulation. We reused the exact decoder from Task 3.1: same training trials, same λ, same reference. No retraining was needed, because PARRM only touches stimulation sessions. We checked that the features of all 22 no-stim sessions are identical with and without PARRM.

Each test-day stimulation session was then scored twice: once on the original features (as in 3.1) and once on the PARRM-cleaned features.

One note for later: if CAR ever wins the reference comparison, there's no cleaned version of the CAR features yet (`parrm_car`). In that case `task3_2_parrm.py` stops and prints the command that computes them, which takes about 5 minutes.

### What we found

For each frequency, the table shows the same-day OFF score, the ON score without cleaning, and the ON score with cleaning. "Cleaning effect" is how much PARRM changed the ON score, and in how many of the 7 sessions it improved things.

| Features | History | 20 Hz: OFF / ON / ON cleaned | Cleaning effect, 20 Hz | 3 Hz: OFF / ON / ON cleaned | Cleaning effect, 3 Hz |
|---|---|---|---|---|---|
| LMP + band power | 9 lags | 0.028 / 0.016 / 0.017 | +0.001 (better in 3 of 7) | 0.048 / 0.025 / 0.025 | 0.000 (better in 4 of 7) |
| LMP + band power | 14 lags | 0.032 / 0.021 / 0.022 | +0.001 (better in 4 of 7) | 0.051 / 0.025 / 0.024 | −0.001 (better in 3 of 7) |
| LMP + band power | 19 lags | 0.041 / 0.029 / 0.031 | +0.002 (better in 4 of 7) | 0.046 / 0.031 / 0.030 | 0.000 (better in 4 of 7) |
| LMP + band power | 29 lags | 0.043 / 0.029 / 0.030 | +0.001 (better in 4 of 7) | 0.044 / 0.032 / 0.032 | 0.000 (better in 4 of 7) |
| Band power only | 9 lags | 0.016 / 0.017 / 0.018 | +0.001 (better in 2 of 7) | 0.010 / 0.012 / 0.012 | +0.001 (better in 3 of 7) |

**PARRM cleaning recovers almost none of the lost accuracy.** At 20 Hz it closes only about 6 to 16% of the gap between ON and OFF. At 3 Hz it closes none of it. Session by session, cleaning helps about half the time and hurts the other half, which is what you'd expect from random noise.

That's consistent with the README. PARRM removes only about 17% of the power at the stimulation frequency for 20 Hz, and about 5% for 3 Hz. At 3 Hz it can't tell the stimulation apart from brain activity that happens to follow the same rhythm.

The biggest single improvement (+0.007 with 9 lags) is in session 250616-102124. That's the 20 Hz session where PARRM removes the most, 69% of the 20 Hz power, and it's listed as a known problem in the manifest. Since it's just one session, treat it as a hint rather than a finding.

The recording-order caution from Task 3.1 applies here too.

## Task 3.3: Can the stimulation signal alone predict movement?

### What we did

This is a sanity check. The stimulator runs on its own clock, independently of the task, so the stimulation signal shouldn't tell us anything about how the hand moves. If it did, a brain decoder could score well on ON trials by picking up the stimulation rather than real brain activity.

We built five simple features from the stimulation monitor signal (StiF) for every 50 ms time step:

- how strong the stimulation is on average during the step, and how much it varies within the step (both normalized within each session, like the brain features);
- where in the stimulation cycle we are at the end of the step, encoded as a sine and a cosine (both set to 0 while the stimulator is off);
- what fraction of the step the stimulator is switched on.

We trained a ridge decoder on these features alone, using the 26 stimulation sessions on training days (July 11 excluded), about 35,000 time steps. λ was chosen on the 6 validation-day stimulation sessions, and the decoder was scored on the 14 test-day stimulation sessions. The time window, history lengths, ±10 cap and λ values are the same as in Task 3.1.

### What we found

| History | Best validation R² (λ) | Test R², 20 Hz | Test R², 3 Hz |
|---|---|---|---|
| 9 lags | −0.002 (λ = 1,000,000) | −0.001 | 0.000 |
| 14 lags | −0.002 (λ = 1,000,000) | −0.001 | 0.000 |
| 19 lags | −0.002 (λ = 1,000,000) | −0.001 | 0.000 |
| 29 lags | −0.002 (λ = 1,000,000) | −0.001 | 0.000 |

**The check passes.** Every one of the 14 test sessions scores between −0.005 and 0.000, at every history length. The stimulation signal carries no information about the hand's movement. That includes the moments when the stimulator switches on at the start of a block.

Every λ scored below zero on validation, from −0.0034 at λ = 1 up to −0.0020 at λ = 1,000,000. The best λ sits at the very top of the range we tried, which would normally mean extending the range. Here it makes sense: the strongest penalty forces the decoder to just predict the average velocity, so the "best" stimulation-signal decoder is one that ignores the stimulation signal entirely.

This also means the small accuracy the Task 3.1 decoder reaches on ON trials (R² of 0.016 to 0.032) really comes from the brain signal, not from the stimulation.

One limit: this was a linear check with simple features. It's strong evidence, but it doesn't completely rule out a far more flexible model, like the in-context transformer, finding a pattern that a linear model can't.

## Bottom line for Task 3

Our results match **outcome 3 on slide 24: poor accuracy even without stimulation.**

- The best decoder without stimulation only reaches an R² of about 0.05. That's after the two improvements slide 24 suggests: we added LMP and longer history, and per-session normalization was already in place.
- Stimulation costs only about 0.01 to 0.025 in R². Some of that may be due to recording order rather than the stimulation itself, and PARRM cleaning recovers almost none of it.
- The stimulation signal on its own predicts nothing.

Following slide 24, the project should make the **semi-synthetic results the main result**. On the real data there's very little accuracy to lose, and so very little for any recalibration method to win back.
