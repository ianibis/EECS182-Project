# Task 5 checkpoints

Every Task 5 training run. Each folder holds `best.pt` (the checkpoint with the best validation score),
`history.csv` (validation scores during training) and `summary.json` (settings and best score). The `*.log` files
are the console output of each step. Results are explained in `task5/RESULTS.md`.

## Which checkpoint to use

| Folder | What it is | Section of RESULTS.md |
|---|---|---|
| `final_in_context_seed1` to `5` | **First model**, the pre-specified in-context transformer | 2, 3 |
| `final_blind_seed1` to `5` | Blind control: the same model without the brain signal | 3 |
| `calibrated_a0_seed1` to `5` | **Calibrated decoder**: closed-form session correction from the context | 8, 8.1 |
| `improve_jitter_seed1` to `5` | **Jitter decoder**: the calibrated decoder trained with time jitter; the best model | 8.2, 8.3 |

Use all 5 seeds. Reported numbers average R² over seeds; Section 8.3 also scores the mean of the 5 predictions.

## Other runs

| Folder | What it is |
|---|---|
| `smoke`, `pilot` | Early code checks (100 and 1500 steps); not used for results |
| `sweep_lr*_warmup*` | Learning-rate sweep, 3000 steps each (`task5_results/sweep.csv`) |
| `diag_6000`, `full_lr0.001_warmup1000` | Full-length sweep candidates; `diag_6000` was chosen (`task5_results/chosen_setting.json`) |
| `diversity_n*_seed1` | Task 5.7: training with 4 to 4096 distinct simulated shifts (Section 5) |
| `mixing_a*_seed1` | Channel-mixing pilot for the first model, not adopted (Section 7) |
| `calibrated_a0.25_seed1`, `calibrated_a0.5_seed1` | Channel-mixing pilot for the calibrated decoder, not adopted (Section 8) |
| `improve_*_seed1` other than jitter | Step 1 and Step 2 candidates of Section 8.2, not adopted |

## Loading

Run from the repository root:

```python
import sys
sys.path.insert(0, 'task5')
from task5_score import load_model              # final_*, sweep_*, diag_6000, full_*, smoke, pilot, diversity_*, mixing_*
from task5_calibrated import load_calibrated    # calibrated_*
from task5_improve import load_run              # improve_* (and calibrated_*)

model, velocity_scale, step = load_run('improve_jitter_seed1', 'cuda')
```

`task5/task5_evaluate.py` shows how a session is loaded, normalized and scored. The SHA-256 files in
`task5_results/frozen_*.sha256` record the checkpoints that were frozen before each test-day read.
