# Notes

## Task 3.1

### Outlier: channel 64 artifact in Butters-250616-102853 (20 Hz, test day)

- **What:** a very large slow-voltage transient on **channel 64** during the last three query trials of block 5 (**trials 45, 46, 47**, around **392–400 s**).
- **Size:** the LMP feature peaks at about **78,700 µV** at 392.65 s (trial 45), then −4,670 µV (trial 46) and −1,500 µV (trial 47). Normal LMP values are tens to hundreds of µV. After per-session normalization this is z ≈ 5,300; clean features stay within about ±10.
- **Not documented:** the manifest's `known_problems` for this session only mentions a sheet-note conflict about the AM frequency. This artifact is not listed.
- **Effect on decoding (ridge, LMP + band power, 9 lags, before any fix):** the session's query R² was **−0.71**, which pulled the 20 Hz average below zero. Dropping just the 86 affected 50 ms bins gave R² = 0.043, in line with the other sessions. The band-power-only decoder was barely affected (R² 0.030) because the artifact is mainly in the LMP.
- **How it is handled:** every normalized feature is clipped to **±10** (`CLIP_Z` in `task3_common.py`). This applies to all sessions, training and evaluation alike. After clipping, this session's query R² is **0.040** (LMP + band power, 9 lags).
- **For other tasks:** any method that uses LMP from this session (fine-tuning, LoRA, in-context model, PARRM + ridge) will see the same artifact. To keep comparisons fair, use the same handling (clip at ±10) or agree on another one as a team.
