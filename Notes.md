# Notes

These notes summarize what we found in Task 3, the "problem size check". The question behind it: before we build anything fancy, how well does a simple decoder predict hand movement from the brain signal, and how much does stimulation hurt it?

## Reproducing the numbers

The `results/` folder isn't on GitHub, so you'll need to run the scripts yourself. First, one-time setup:

1. Unzip the shared data into a `data/` folder in the repo. The expected folder names are listed in `PATHS` at the top of `task3/task3_common.py`.
2. Create the virtual environment and install the packages:
   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```

On Mac or Linux, use `.venv/bin/pip` and `.venv/bin/python` instead.

Then run the scripts from the repository root. All Task 3 code is in the `task3/` folder, and each script saves CSV files to `results/task3/`:

```
.venv\Scripts\python task3/task3_1_ridge.py                  # Task 3.1 with the default settings (9 lags)
.venv\Scripts\python task3/task3_1_ridge.py --lags 19        # longer history (we also tried 14 and 29)
.venv\Scripts\python task3/task3_1_ridge_within_session.py   # Task 3.1 same-day check (accepts --lags)
.venv\Scripts\python task3/task3_1_xgboost.py --device cuda  # Task 3.1 with XGBoost (drop --device cuda without an NVIDIA GPU)
.venv\Scripts\python task3/task3_2_parrm.py                  # Task 3.2 (same options as 3.1)
.venv\Scripts\python task3/task3_3_stif.py                   # Task 3.3 (accepts --lags)
```

## Terms used below

- **ON and OFF trials.** ON trials are recorded while stimulation is running; OFF trials have no stimulation. In a stimulation session all 50 trials are ON, so OFF trials only come from the no-stim sessions.
- **Calibration pool and query trials.** Each validation and test session has 50 trials in 5 blocks of 10. The first 20 trials (blocks 1 and 2) are the calibration pool. The last 30 (blocks 3 to 5) are the query trials. Every validation and test R² in these notes is computed on the 30 query trials, the same trials the later methods (fine-tuning, LoRA, in-context learning) are scored on, so our numbers line up with theirs. The only exception is the same-day check in Task 3.1, which scores held-out blocks of training sessions.
- **Validation.** λ (and, for XGBoost, the tree depth and number of trees) is always chosen on the 3 validation days. Each training session also has one block of 10 trials set aside in `splits.json` (`train_val_trials`). We don't train on it, and we don't use it for validation either.
- **R².** How close the predicted velocity is to the real one. 1 means perfect, 0 means no better than always guessing the average velocity, and below 0 means worse than that. For each session, we compute it in three steps:
  1. Pool all valid 50 ms time steps from the session's 30 query trials. A time step is valid when its features and its velocity are both available.
  2. Compute one R² for sideways velocity (vx) and one R² for up-down velocity (vy), each over all of the pooled time steps.
  3. The session's R² is the average of the R² for vx and the R² for vy.

  We never compute R² for a single trial. When a table gives one R² for several sessions (for example, the 3 validation sessions), it's the average of the sessions' R² values.
- **LMP and band power.** The two kinds of features the decoder reads from each channel. LMP (local motor potential) is the slow average voltage over the last quarter second. Band power is how strong the signal is in each of 7 frequency ranges, such as 13 to 30 Hz.
- **Lags (history).** How many earlier 50 ms time steps the decoder sees in addition to the current one. 9 lags means half a second of signal in total.
- **λ (lambda).** The strength of the ridge penalty. A larger λ keeps the decoder's weights small and pushes its guesses toward the average.
- **Recording reference vs. CAR.** Two versions of the brain signal. The recording reference is the signal as recorded. CAR (common average reference) subtracts the average of all channels at each moment.

## Task 3.1: How much does stimulation hurt a standard decoder?

### What we did

We trained a ridge decoder that predicts fingertip velocity from the brain signal, one prediction every 50 ms. It reads LMP and the 7 band powers from every channel. It only ever saw OFF trials during training: 40 trials from each of the 13 no-stim sessions on training days, about 17,600 time steps in total. The three July 11 sessions were left out because they have no velocity in cm/s.

For each trial, the decoder looks at the window from half a second before the go cue to 1.5 seconds after it. Features are normalized within each session, so sessions with louder or quieter recordings become comparable. For validation and test sessions, the normalization is computed from the first 20 trials only, so the graded trials stay untouched. After normalizing, we cap every value at ±10 to protect against artifacts (see the channel 64 outlier below).

We tried 13 values of λ, from 1 to 1,000,000, with both the recording reference and CAR. We kept whichever combination scored best on the 3 validation-day OFF sessions. With those settings fixed, we scored the decoder on the test days: on each of the 14 stimulation sessions (7 at 3 Hz, 7 at 20 Hz), and on the no-stim session recorded the same day. The difference between the two (ON minus OFF) is how much accuracy stimulation costs.

### What we found

Each row of the table is one setting. "OFF R²" is the average over the 5 test-day no-stim sessions. The last two columns show the ON score, the same-day OFF score, and the difference in brackets (negative means stimulation made it worse).

| History | Validation R² | Chosen λ | OFF R² (test days) | 3 Hz: ON vs. OFF | 20 Hz: ON vs. OFF |
|---|---|---|---|---|---|
| 9 lags (0.5 s) | 0.035 | 31,623 | 0.045 | 0.025 vs. 0.048 (−0.023) | 0.016 vs. 0.028 (−0.012) |
| 14 lags (0.75 s) | 0.052 | 31,623 | 0.051 | 0.025 vs. 0.051 (−0.025) | 0.021 vs. 0.032 (−0.011) |
| 19 lags (1 s) | **0.053** | 316,228 | 0.053 | 0.031 vs. 0.046 (−0.016) | 0.029 vs. 0.041 (−0.011) |
| 29 lags (1.5 s) | 0.052 | 316,228 | 0.052 | 0.032 vs. 0.044 (−0.012) | 0.029 vs. 0.043 (−0.014) |

The main takeaway is that **the decoder is weak even without stimulation**. The best version reaches an R² of about 0.05, far below the 0.2 that slide 24 uses as the bar for "poor accuracy even without stimulation".

A few other things stood out:

- **We use LMP + band power.** An earlier version also tried band power alone. It did worse (validation R² 0.021 vs. 0.035 with half a second of history), so we dropped it.
- **More history helps a little, then stops helping.** Going from 0.5 to 0.75 seconds raised validation R² from 0.035 to 0.052. Beyond that it stays flat. One second (19 lags) scored highest on validation, but 0.75 to 1.5 seconds are practically tied.
- **The recording reference beat CAR in every setting.** For example, 0.035 vs. 0.021 for LMP + band power with 9 lags.
- **The decoder needs a strong penalty.** The best λ was always 31,623 or 316,228. With weaker penalties it memorizes the training trials and does worse on new ones.
- **Stimulation costs a little accuracy, mostly at 3 Hz.** ON trials score about 0.01 to 0.025 lower than the same-day OFF session. At 3 Hz, ON scored lower than OFF in 7 of 7 sessions with 9 or 14 lags, 6 of 7 with 19 lags, and 5 of 7 with 29 lags. That fits the README, which says LMP is the feature most affected by 3 Hz stimulation.

### Why is accuracy so low?

To make sure the low numbers weren't caused by a bug, we ran a few extra checks. These used throwaway scripts that aren't in the repo, were run before we added the ±10 cap, and used 9 lags.

| Check | R² | What it tells us |
|---|---|---|
| Score on its own training data (λ = 100) | 0.26 | The decoder can partly memorize the training data, but that doesn't carry over to new trials |
| Train on 40 trials, test on the held-out block from the same sessions | 0.042 | About as poor as on new days, so day-to-day drift isn't the problem |
| No brain data at all: predict the average velocity at each moment after the go cue | 0.089 | This beats the brain decoder |

The last row is the most telling. The monkey reaches for the same targets in the same order on every trial, so just knowing the time since the go cue predicts the movement better than the brain features do. That means the velocity data and the trial timing are fine. The weak link is how little velocity information these features carry.

### Same-day check: does a decoder trained on the same session do better?

The main decoder is trained on some days and scored on other days. Brain recordings change from day to day, so that is a hard test. To see how much the change of day costs, we also trained and scored ridge within single sessions (`task3/task3_1_ridge_within_session.py`).

**How it works:**

- We used each of the 14 no-stim sessions on the training days. Each session has 5 blocks of 10 trials.
- For each session, we trained ridge on 4 blocks and scored it on the 5th. We repeated this 5 times, so every block is scored once ("leave one block out").
- Everything else matches the main decoder: LMP + band power, the same time window, history lengths, ±10 cap, recording reference and λ values.
- Normalization uses only the 4 training blocks, so the scored block isn't used for anything except R².
- R² is computed as described under "Terms used below", except on the 10 trials of the scored block instead of 30 query trials. The session's R² is the average over its 5 scored blocks.
- For each session, λ is the value that works best on the other sessions, so no session gets to pick its own λ.
- July 11 has no velocity in cm/s, so we decoded it in pixels/s. Pixel directions follow the camera, not the screen, so July 11 is left out of the averages below. On its own, it scores 0.024 to 0.032.

**Results.** "Same session" is the average of the 13 sessions with velocity in cm/s. "Across days" is the main Task 3.1 decoder from the table above.

| History | Same session | Across days: validation | Across days: test-day OFF |
|---|---|---|---|
| 9 lags | 0.054 | 0.035 | 0.045 |
| 14 lags | 0.071 | 0.052 | 0.051 |
| 19 lags | **0.076** | 0.053 | 0.053 |
| 29 lags | 0.074 | 0.052 | 0.052 |

**Training on the same day helps, but only a little.** The same-session decoder scores 0.01 to 0.025 higher than the across-days decoder. The best single session reaches 0.133, and no session comes close to 0.2. So the change of day is not what keeps accuracy low.

**The improvement comes entirely from sideways movement.** Here is R² split by direction, for 19 lags (the other history lengths look the same):

| Decoder (19 lags) | Sideways (vx) | Up-down (vy) |
|---|---|---|
| Same session | 0.054 | 0.097 |
| Across days, test-day OFF sessions | 0.012 | 0.094 |

Up-down velocity is predicted equally well either way. Sideways velocity can be predicted within a session, but almost not at all across days. So sideways movement is in the brain features, but the way it shows up changes from day to day. We don't know yet whether that change is in the brain signal or in the camera calibration, which is fitted separately for each session.

**The best λ is about the same as for the across-days decoder:** 31,623 for 9 and 14 lags, and 100,000 for 19 and 29 lags. Letting each session pick its own best λ would add only 0.001 to 0.003.

**Even within a session, the brain decoder doesn't beat the timing-only baseline** (0.089 on the validation days), which uses no brain data at all.

An earlier throwaway version of this check estimated "typical 0.08, best 0.16". It was too optimistic, and these numbers replace it.

### XGBoost (nonlinear model): does a more flexible decoder do better?

The diagnostics above left one question open: is the low accuracy the features' fault, or the linear decoder's? To test that, we swapped ridge for XGBoost, a gradient-boosted trees model that can pick up curved relationships and interactions between features. Everything else stayed the same as for ridge: training trials, time window, normalization, the ±10 cap, the validation days and the test design.

**How it's set up** (`task3/task3_1_xgboost.py`):

- One model predicts sideways velocity (vx) and a second predicts up-down velocity (vy).
- Each tree sees a random 80% of the training time steps and a random 30% of the inputs, and every leaf must cover at least 20 time steps. Both choices help against overfitting.
- Trees are added one at a time, up to 2,000, and training stops once the score on the validation-day OFF sessions stops improving ("early stopping"). We deliberately don't use XGBoost's usual random hold-out of training time steps: neighboring 50 ms steps are near-copies of each other, so a random hold-out would leak and make the model look better than it is.
- We tried tree depths 2, 3, 4, 6, 8 and 10 and kept the one with the best validation R². Depth 10 never won, so the script now tries depths 2 to 8 by default. To include it again, run with `--depths 2 3 4 6 8 10`.
- The script runs on the CPU by default, or on an NVIDIA GPU with `--device cuda`. GPU and CPU runs don't build exactly the same trees, and the random sampling of time steps and inputs also varies, so scores can differ slightly from run to run. Treat differences of a few thousandths as noise.
- Deep trees with long histories need a lot of memory. If a depth doesn't fit on the GPU, the script retrains it on the CPU, and if it doesn't fit there either, it records that depth as skipped and moves on. In our runs, depth 10 at 29 lags didn't fit, so the 29-lag row below covers depths 2 to 8 only. Depth 10 never won in any other setting, so this shouldn't change the conclusion.

**Results.** Each cell shows ridge first, then XGBoost. "All 19 test sessions" averages the 14 ON sessions and the 5 OFF sessions together.

| History | Best depth | Validation R² | Test OFF R² | 3 Hz ON R² | 20 Hz ON R² | All 19 test sessions |
|---|---|---|---|---|---|---|
| 9 lags | 6 | 0.035 → 0.070 | 0.045 → 0.045 | 0.025 → 0.046 | 0.016 → 0.041 | 0.027 → 0.044 |
| 14 lags | 6 | 0.052 → 0.082 | 0.051 → 0.056 | 0.025 → 0.059 | 0.021 → 0.057 | 0.030 → 0.058 |
| 19 lags | 6 | 0.053 → **0.091** | 0.053 → **0.062** | 0.031 → 0.065 | 0.029 → 0.063 | 0.036 → **0.064** |
| 29 lags | 6 | 0.052 → 0.092 | 0.052 → 0.061 | 0.032 → 0.065 | 0.029 → 0.055 | 0.036 → 0.061 |

**XGBoost beats ridge in every setting, but only by a little.** It gains about 0.03 to 0.04 on the validation days and about 0.02 to 0.03 across the 19 test sessions. The best setting (19 lags, depth 6) reaches 0.091 on validation and about 0.06 on the test days, still far below the 0.2 bar.

The validation numbers flatter XGBoost a bit, because the validation days also chose its depth and when to stop adding trees. The test days are the fair comparison, and there the gain on the OFF sessions is small (0.000 to 0.009). Most of the test-day gain comes from the ON sessions.

Even the best XGBoost decoder only matches the timing-only baseline from the diagnostics (0.089 on the validation days), which uses no brain data at all.

**Deeper trees memorize instead of generalizing.** Here is the depth search for the best setting (19 lags):

| Tree depth | Trees kept (vx / vy) | R² on its own training data | Validation R² |
|---|---|---|---|
| 2 | 132 / 392 | 0.165 | 0.074 |
| 3 | 96 / 206 | 0.203 | 0.080 |
| 4 | 9 / 200 | 0.219 | 0.082 |
| **6** | 10 / 133 | 0.337 | **0.091** |
| 8 | 27 / 88 | 0.518 | 0.089 |
| 10 | 28 / 57 | 0.606 | 0.083 |

Deeper trees keep fitting the training data better (up to 0.61), but validation R² peaks at depth 6 and then falls. Depth 6 won at every history length.

**Across days, almost all of the signal is in up-down movement.** Splitting R² by direction shows that neither model can predict sideways velocity on new days:

| Model (19 lags) | Sideways (vx), OFF / ON | Up-down (vy), OFF / ON |
|---|---|---|
| Ridge | 0.012 / −0.015 | 0.094 / 0.074 |
| XGBoost | 0.003 / −0.002 | 0.121 / 0.130 |

This is also why the vx models stop after only a handful of trees: there's almost nothing to learn that carries over to new days. Even up-down velocity on its own stays below 0.2. Sideways movement isn't missing from the brain signal, though: the same-day check above predicts it with an R² of about 0.05. It just doesn't carry over from one day to the next.

**ON vs. OFF looks different at the two frequencies.**

- **At 3 Hz the pattern matches ridge.** ON scores 0.009 to 0.018 below the same-day OFF session, lower in 5 or 6 of the 7 sessions.
- **At 20 Hz, ON appears to score higher than OFF** (by 0.026 to 0.037). That's not stimulation helping. Four of the seven 20 Hz sessions share one OFF reference, 250616-101326, which XGBoost happens to decode poorly (R² between −0.015 and 0.028 depending on the setting), so the comparison is dragged by one weak session.
- **XGBoost improved the ON sessions more than the OFF sessions,** compared with ridge. It may cope with stimulation-induced changes in the features a little better, but with this few sessions we can't separate that from noise.

**Takeaway:** a nonlinear decoder helps a little, but not nearly enough to reach the 0.2 bar. The main limit is how little velocity information these features carry, not the linear model. This doesn't change the Task 3 conclusion.

### Keep in mind: the OFF session was always recorded first

On all 5 test days, the no-stim session was recorded before the stimulation sessions. So when ON scores lower than OFF, part of that could come from being later in the day (the monkey getting tired, attention drifting, electrodes changing) rather than from the stimulation itself. Comparing against the same day's OFF session rules out differences between days, but not this ordering within a day.

### Outlier: a huge artifact on channel 64 in session 250616-102853

This 20 Hz test-day session has a very large voltage spike on **channel 64** during its last three graded trials (trials 45, 46 and 47 in block 5, roughly 392 to 400 seconds into the recording).

It's far too big to be brain activity. The LMP on that channel reaches about 78,700 µV in trial 45, then −4,670 µV in trial 46 and −1,500 µV in trial 47. Normal LMP values are tens to hundreds of µV. After normalization, the peak is about 5,300, when clean values stay within about ±10. The manifest doesn't mention this artifact: its `known_problems` entry for this session only notes a disagreement about the stimulation frequency.

Before we handled it, this one artifact dragged the session's R² down to −0.71 (LMP + band power, 9 lags), enough to pull the whole 20 Hz average below zero. Dropping just the 86 affected time steps brought it back to 0.043, in line with the other sessions. The band-power-only decoder we tried earlier barely noticed (R² 0.030), because the artifact shows up mainly in the LMP.

**How we handle it:** after normalization, every feature value is capped at ±10 (`CLIP_Z` in `task3/task3_common.py`), in every session, for both training and evaluation. With the cap, this session scores 0.040.

**If your task uses LMP from this session** (fine-tuning, LoRA, the in-context model, PARRM + ridge), you'll hit the same artifact. Please use the same ±10 cap, or let's agree on a different fix as a team, so all methods are compared fairly.

## Task 3.2: Does PARRM cleaning help?

### What we did

PARRM is a filter that removes the part of the signal that repeats in step with the stimulation. We reused the exact decoder from Task 3.1: same training trials, same λ, same reference. No retraining was needed, because PARRM only touches stimulation sessions. We checked that the features of all 22 no-stim sessions are identical with and without PARRM.

Each test-day stimulation session was then scored twice: once on the original features (as in 3.1) and once on the PARRM-cleaned features.

One note for later: if CAR ever wins the reference comparison, there's no cleaned version of the CAR features yet (`parrm_car`). In that case `task3/task3_2_parrm.py` stops and prints the command that computes them.

### What we found

For each frequency, the table shows the same-day OFF score, the ON score without cleaning, and the ON score with cleaning. "Cleaning effect" is how much PARRM changed the ON score, and in how many of the 7 sessions it improved things.

| History | 20 Hz: OFF / ON / ON cleaned | Cleaning effect, 20 Hz | 3 Hz: OFF / ON / ON cleaned | Cleaning effect, 3 Hz |
|---|---|---|---|---|
| 9 lags | 0.028 / 0.016 / 0.017 | +0.001 (better in 3 of 7) | 0.048 / 0.025 / 0.025 | 0.000 (better in 4 of 7) |
| 14 lags | 0.032 / 0.021 / 0.022 | +0.001 (better in 4 of 7) | 0.051 / 0.025 / 0.024 | −0.001 (better in 3 of 7) |
| 19 lags | 0.041 / 0.029 / 0.031 | +0.002 (better in 4 of 7) | 0.046 / 0.031 / 0.030 | 0.000 (better in 4 of 7) |
| 29 lags | 0.043 / 0.029 / 0.030 | +0.001 (better in 4 of 7) | 0.044 / 0.032 / 0.032 | 0.000 (better in 4 of 7) |

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

- The best decoder without stimulation only reaches an R² of about 0.05 with ridge. That's after the two improvements slide 24 suggests: we added LMP and longer history, and per-session normalization was already in place. A nonlinear decoder (XGBoost) raises this only to about 0.06 on the test days (0.09 on the validation days).
- A ridge decoder trained and scored within the same session only reaches about 0.05 to 0.08, so the change of day isn't what keeps accuracy low. The only thing that improves within a session is sideways velocity.
- Stimulation costs only about 0.01 to 0.025 in R². Some of that may be due to recording order rather than the stimulation itself, and PARRM cleaning recovers almost none of it.
- The stimulation signal on its own predicts nothing.

Following slide 24, the project should make the **semi-synthetic results the main result**. On the real data there's very little accuracy to lose, and so very little for any recalibration method to win back.
