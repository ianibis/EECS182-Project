# Task 4: MLP decoder

The MLP uses the same 315 features per bin and fingertip velocity targets
as the GRU. A shared 48-unit projection feeds a causal 20-bin window into
a 64-unit hidden layer and a two-axis velocity-residual readout. The
training objective retains the speed and direction heads. There are
76,932 parameters per checkpoint.

## Run from the repository root

```sh
python -m pip install -r requirements.txt
export TASK3_DATA_ROOT="/path/to/Box/project"

python -B task4/task4_mlp_train.py --run all --device cpu
python -B task4/task4_mlp_evaluate.py --split val
python -B task4/task4_mlp_evaluate.py --split test
python -B task4/task4_mlp_evaluate.py --split val --ablation
python -B task4/task4_mlp_evaluate.py --split test --ablation
python -B task4/task4_mlp_evaluate_off.py --split val
python -B task4/task4_mlp_evaluate_off.py --split test
```

`task4_mlp_train.py` trains the OFF model and paired real 1× and semi-synthetic
10× ON models. Use `--run off` or `--run on` to select a subset. Training uses
the saved splits, AdamW, the archived multitask objective, seed 1 and
validation-gain early stopping. It computes the sequence-specific behavioral
profile from training trials by default. To reproduce the archived OFF
profile exactly, add `--gru-checkpoint`, or pass an explicit checkpoint path
after that flag. MLP weights initialize independently.

`task4_mlp_evaluate.py` reads three MLP checkpoints. Run the Task 3 GRU
evaluator first to generate the local GRU comparator CSVs, or supply the
existing Box GRU result tree. `task4_mlp_evaluate_off.py` needs only the OFF
checkpoint. It uses fixed calibration settings and includes the GRU CSV
comparison only with `--compare-gru`. Both scripts use 20 same-day OFF
calibration trials and 30 disjoint query trials per session. Task 4.1's ON
calibration draws are separate from these commands.

The standard run reports full fine-tuning with OFF-fitted time correction.
`--ablation` scores frozen, ridge-readout, and full fine-tuned models with
and without that correction. Inputs are the held-out OFF, recorded 1× ON,
semi-synthetic 10× ON, and PARRM-cleaned original ON features. When cleaned
high-band features are missing, run the GRU evaluator first to generate
a local cache. See the [file map](../EECS182_file_map_261117.md).

Training CSVs and checkpoints go to `task4_results/mlp/{off,on_1x,on_10x}/`.
Evaluation writes `sessions.csv` and `aggregate.csv` within validation and
test subdirectories. Ablation and OFF-only results have separate
subdirectories. The archived OFF-trained MLP scores 0.186438 on validation
OFF queries and 0.206771 on test OFF queries. Compare new scores with the
archived CSVs before replacing any published results.
