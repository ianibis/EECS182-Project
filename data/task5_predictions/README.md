# Per-trial predictions of the first Task 5 in-context model

`first_model_predictions_val.npz` and `first_model_predictions_test.npz` hold the 5 seeds' predicted velocity
for every query trial, with the same context draws as `task5_results/scores_{val,test}.csv` (control `none`).
They were made by `task5/task5_export_predictions.py`, which checked that these predictions give back the saved
R² before writing the files.

Each array has one entry per row, and a row is one (session, input, k, draw):

| array | shape | meaning |
|---|---|---|
| `session` | (R,) | session name |
| `input` | (R,) | `recorded` (real data) or `amp10` (recorded + 9x PARRM artifact; stim sessions only) |
| `k` | (R,) | number of context trials: 0, 2, 5, 10, 20 |
| `draw` | (R,) | index into `splits.json["calibration"][session]["subsets"][str(k)]`: 0-9 for k = 2, 5, 10; 0 for k = 0 and 20 |
| `context_trials` | (R, 20) | trial numbers given as context, in the order given (ascending); -1 pads to 20 |
| `query_trials` | (R, 30) | the 30 query trials (blocks 3-5), ascending, same as `splits.json` |
| `query_first_bin` | (R, 30) | each query trial's first bin on the session's 50 ms grid |
| `prediction` | (R, 5, 30, 40, 2) | predicted (vx, vy) in cm/s; axis 1 is the seed (`seeds`) |
| `velocity` | (R, 30, 40, 2) | recorded (vx, vy) in cm/s; NaN where not valid |
| `valid` | (R, 30, 40) | the bins that are scored |
| `seeds` | (5,) | 1, 2, 3, 4, 5 |

Bins are 50 ms, from 0.5 s before go to 1.5 s after. The LFP of each session was normalized with its 20 pool trials only.

## Reproduce a saved score

```python
import numpy as np

d = np.load('first_model_predictions_test.npz')

def session_r2(truth, prediction):
    """Task 3 score: R² per axis over the pooled valid bins, then the mean of vx and vy."""
    ss_res = ((truth - prediction) ** 2).sum(axis=0)
    ss_tot = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)
    return (1 - ss_res / ss_tot).mean()

row = np.flatnonzero((d['input'] == 'recorded') & (d['k'] == 5) & (d['draw'] == 0))[0]
valid = d['valid'][row]
print(d['session'][row], [session_r2(d['velocity'][row][valid], d['prediction'][row, s][valid]) for s in range(5)])
```

The reported numbers average this R² over draws, then over seeds, then over sessions. They average R², not the
predictions. The main Task 5 results use the stimulation sessions; the files also hold the same-day OFF sessions
(`recorded` only).
