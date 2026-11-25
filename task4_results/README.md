# Task 4 results

Committed CSV artifacts for task 4. The notebooks and scripts write here through
`utils.TASK4_RESULTS`, so regenerating a notebook overwrites its rows in place.

| Path | Written by |
|---|---|
| `draws.json` | `task4/task4_1_draws.ipynb` — calibration partitions (v3, fixed query set) |
| `eegnet_sweep.csv` | 4.2 hyperparameter sweep |
| `task4_2_seeds.csv` | 4.2.5 five-seed baseline, validation R² per seed |
| `task4_2_baseline_off.csv`, `task4_2_eegnet_off.csv` | 4.2.5/4.2.6 per-session scores |
| `task4_2_vs_ridge.csv` | 4.2 comparison against the task 3.1 ridge baseline |
| `gru_sweep.csv` | GRU sweep (`kaggle/train_gru.py`) |
| `task4_3_parrm_ridge.csv`, `task4_3_summary.csv` | `task4/task4_3_parrm_ridge.ipynb` |
| `MLP/` | Task 4 MLP runs, one folder per evaluation protocol (see `MLP/README.md`) |

Larger intermediate artifacts stay out of git: checkpoints in `task4/checkpoints/` are
tracked only for the five-seed runs, and the parquet score tables live in the local
`results/` folder.
