# Task 4.2 on Kaggle

| File | Role |
|---|---|
| `train_task4_2.py` | the whole of task 4.2 training: model, five seeds, plain and synthetic-frequency-augmented, scoring |
| `build_dataset.py` | assembles `dataset/` with the 36 sessions the decoders touch and the code they import |
| `build_kernel.py` | writes `kernel/` (notebook + metadata) with the training script embedded |

`dataset/` is hardlinked to the repository, so building it costs no extra disk.

## Commands

```bash
python kaggle/build_dataset.py
kaggle datasets create -p kaggle/dataset            # first time
kaggle datasets version -p kaggle/dataset -m "..."  # after rebuilding

python kaggle/build_kernel.py
kaggle kernels push -p kaggle/kernel
kaggle kernels status kevinseriousying/task4-2-eegnet-lfp
kaggle kernels output kevinseriousying/task4-2-eegnet-lfp -p kaggle/out
```

## Mount path

Kaggle mounts an attached dataset **read-only** at
`/kaggle/input/datasets/<owner>/<slug>`, not `/kaggle/input/<slug>`. The script therefore
resolves its data root by searching for `task4/utils.py` under `/kaggle/input`, so it works
whatever the mount layout is. Outputs always go to `/kaggle/working/task4_2`.

## What the kernel writes

`/kaggle/working/task4_2/`

- `checkpoints/task4_2_{eegnet,eegnet_synthetic_frequencies}_seed{0..4}.pt`
- `training_logs/{eegnet,eegnet_synthetic_frequencies}_seed{0..4}.json`
- `task4_2_eegnet_off.csv` — one row per method, session and seed
- `task4_2_summary.csv` — mean clean and stimulated R² per method

and `task4_2_run.zip` beside it.

## Why the data subset

36 sessions: 14 stimulation-off training, 3 stimulation-off validation, 14 stimulation-on test
and the 5 same-day stimulation-off recordings the test sessions are scored against. Nothing else
is read, so the upload is 3.4 GB instead of the full 14 GB.

## Running it locally first

```bash
SMOKE=1 python kaggle/train_task4_2.py     # one epoch, one seed, one test session
python kaggle/train_task4_2.py             # full run on CPU or Apple GPU
```
