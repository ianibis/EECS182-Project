"""Score the task 4.2 EEGNet checkpoints across the 10x input arms.

Trains nothing. Loads the saved CausalEEGNet seeds and scores each stimulated test
session on four inputs:

    recorded            the recorded stimulation-ON query trials
    amp10               the same trials, recorded + 9 * PARRM artifact
    parrm_recorded      PARRM re-estimated on the amplified signal, which returns the
                        recorded cleaned LFP (see make_parrm_amp10.py --verify)
    parrm_amp10_reuse   the 10x signal cleaned with the 1x template, leaving eight copies

Every arm shares the same same-day OFF control, so the only thing that changes is the
input recording.

    python task4/eval_amp10.py
    python task4/eval_amp10.py --split val
    python task4/eval_amp10.py --checkpoints kaggle/out/task4_2/checkpoints
    python task4/eval_amp10.py --arms recorded amp10

The 10x recordings must exist first:

    python semisynthetic/EECS182_make_semisynthetic_h5.py --root data --split val-test --h5-only
    python task4/make_parrm_amp10.py --split val-test

Writes task4/results/task4_2_eegnet_amp10.csv (one row per session, method, seed, arm)
and task4/results/task4_2_amp10_summary.csv (one row per method, one column per arm).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils as U  # noqa: E402

METHODS = (U.EEGNET_METHOD, U.EEGNET_TONE_METHOD)


def score_arm(models, datasets, method, feature_label):
    rows = []
    for seed, model in enumerate(models):
        rows += [U.score_trial_model(model, session, method, feature_label, seed, query, off)
                 for session, (query, off) in datasets.items()]
    rows += [U.score_ensemble(models, session, f'{method}_ensemble', feature_label, query, off)
             for session, (query, off) in datasets.items()]
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints', default=U.CHECKPOINT_DIR,
                        help='folder holding task4_2_<method>_seed<0-4>.pt')
    parser.add_argument('--split', choices=('test', 'val'), default='test')
    parser.add_argument('--methods', nargs='+', default=list(METHODS), choices=METHODS)
    parser.add_argument('--arms', nargs='+', default=list(U.ARM_VARIANTS),
                        choices=list(U.ARM_VARIANTS))
    args = parser.parse_args()

    sessions = U.sessions_in_split(args.split, U.STIMULATED_ON)
    datasets = U.amp10_arms(sessions, args.arms)
    print(f'{len(sessions)} {args.split} ON sessions | arms {list(datasets)} '
          f'| checkpoints {args.checkpoints}')

    rows = []
    for method in args.methods:
        models = [U.load_eegnet_checkpoint(f'task4_2_{method}_seed{seed}', args.checkpoints)
                  for seed in range(5)]
        for arm, arm_datasets in datasets.items():
            rows += score_arm(models, arm_datasets, method, U.ARM_VARIANTS[arm])
        print('scored', method, flush=True)

    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(U.TASK4_RESULTS, 'task4_2_eegnet_amp10.csv'), index=False)

    per_session = (scores.groupby(['method', 'feature_variant', 'session'])
                   [['r2_ave', 'r2_off_ave']].mean().groupby(['method', 'feature_variant'])
                   .mean().unstack())
    summary = pd.DataFrame({f'r2_{arm}': per_session[('r2_ave', U.ARM_VARIANTS[arm])]
                            for arm in datasets if ('r2_ave', U.ARM_VARIANTS[arm]) in per_session})
    summary['r2_off'] = per_session[('r2_off_ave', U.ARM_VARIANTS['recorded'])]
    if 'r2_amp10' in summary and 'r2_recorded' in summary:
        summary['amp10_minus_recorded'] = summary.r2_amp10 - summary.r2_recorded
    if 'r2_parrm_amp10_reuse' in summary and 'r2_parrm_recorded' in summary:
        summary['reuse_minus_reestimated'] = (summary.r2_parrm_amp10_reuse
                                              - summary.r2_parrm_recorded)
    summary = summary.round(4).reset_index()
    summary.to_csv(os.path.join(U.TASK4_RESULTS, 'task4_2_amp10_summary.csv'), index=False)

    print(f'\n{len(scores)} rows -> task4/results/task4_2_eegnet_amp10.csv')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
