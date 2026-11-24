"""Rebuild the GRU vs EEGNet comparison from the saved artifacts.

Reads only cached results, so it is fast and re-runnable:

    results/results_{ridge_knn,mlp,gru,eegnet}.parquet   task 4 baselines
    task4/results/task4_2_eegnet_off.csv                 the 4.2 run
    task4/results/task4_2_eegnet_amp10.csv               the 10x run
    task4/checkpoints/*.pt, results/checkpoints/*        the weights themselves

Writes
    task4/results/decoder_comparison.csv   per method x input x frequency, with session CIs
    task4/results/decoder_paired.csv       paired differences between methods
    task4/results/checkpoint_audit.csv     reload each checkpoint, compare to its cached row
    task4/results/artifact_severity.csv    per-session artifact amplitude, for the diagnosis
    results/fig_grueegnet.png              ON R2 by method and frequency

    python task4/compare_decoders.py
    python task4/compare_decoders.py --skip-checkpoints     # tables only, no scoring
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils as U  # noqa: E402

PARQUETS = {
    'results_ridge_knn.parquet': None,
    'results_mlp.parquet': None,
    'results_gru.parquet': None,
    'results_eegnet.parquet': None,
}

LABELS = {
    'ridge_raw_off': 'Ridge raw, no adapt',
    'ridge_parrm_off': 'Ridge PARRM, no adapt',
    'ridge_raw_refit': 'Ridge raw, refit k=20',
    'ridge_parrm_refit': 'Ridge PARRM, refit k=20',
    'knn_off': 'KNN OFF bank',
    'knn_incontext': 'KNN in-context k=20',
    'mlp': 'MLP',
    'gru': 'GRU',
    'eegnet': 'EEGNet',
    'eegnet_ensemble': 'EEGNet ensemble',
    'eegnet_stimulation_tones': 'EEGNet + tones',
    'eegnet_stimulation_tones_ensemble': 'EEGNet + tones ensemble',
}

# Fixed models never see ON labels; the rest spend k labelled ON trials from the draws.
FIXED = ('mlp', 'gru', 'eegnet', 'eegnet_ensemble',
         'eegnet_stimulation_tones', 'eegnet_stimulation_tones_ensemble',
         'ridge_raw_off', 'ridge_parrm_off', 'knn_off')
ADAPTED = ('ridge_raw_refit', 'ridge_parrm_refit', 'knn_incontext')

# Paired comparisons worth a CI. Each pair is (a, b) and the report is mean(a - b).
PAIRS = (
    ('gru', 'eegnet'),
    ('gru', 'eegnet_ensemble'),
    ('eegnet_stimulation_tones', 'eegnet'),
    ('eegnet_stimulation_tones_ensemble', 'eegnet_ensemble'),
    ('eegnet_ensemble', 'eegnet'),
    ('ridge_raw_refit', 'gru'),
    ('ridge_raw_refit', 'eegnet_stimulation_tones_ensemble'),
    ('knn_incontext', 'ridge_raw_refit'),
)


def load_scores():
    frames = [pd.read_parquet(os.path.join(U.RESULTS_DIR, name)) for name in PARQUETS]
    scores = pd.concat(frames, ignore_index=True)

    # results_eegnet.parquet and task4_2_eegnet_off.csv are both missing the
    # tone-augmented ensemble's 14 rows, although the run produced them. Recover them from
    # the four-arm file, whose recorded arm agrees with the cached rows exactly for every
    # method that is present (verified: max |difference| = 0.0).
    ninex = pd.read_csv(os.path.join(U.TASK4_RESULTS, 'task4_2_eegnet_amp10.csv'))
    recorded_arm = ninex[ninex.feature_variant == 'raw']
    absent = set(recorded_arm.method.unique()) - set(scores.method.unique())
    if absent:
        recovered = recorded_arm[recorded_arm.method.isin(absent)]
        scores = pd.concat([scores, recovered.drop(columns='feature_variant')],
                           ignore_index=True)

    # Ridge and KNN are evaluated per draw and per k. Keep the largest k so the
    # "refit k=20" label matches the rows; the k=0 rows are the no-adaptation arms.
    largest = scores[scores.k > 0].k.max()
    adapted = scores.method.isin(ADAPTED)
    scores = scores[~adapted | (scores.k == largest)]
    # One row per session, method, seed; the 10 draws are nested, so collapse them.
    per_session = (scores.groupby(['method', 'session', 'am_freq', 'seed'], as_index=False)
                   [['r2_ave', 'r2_off_ave']].mean())
    return per_session


def summarise(per_session):
    """Mean, session-bootstrap CI and n for every method at each frequency and overall."""
    rows = []
    for method, part in per_session.groupby('method'):
        # Average seeds inside a session: seeds are nested, sessions are the unit.
        by_session = part.groupby(['session', 'am_freq'])[['r2_ave', 'r2_off_ave']].mean().reset_index()
        groups = [('all', by_session)] + [(f'{freq:g} Hz', by_session[by_session.am_freq == freq])
                                          for freq in sorted(by_session.am_freq.unique())]
        for frequency, group in groups:
            low, high = U.bootstrap_confidence_interval(group.r2_ave)
            rows.append(dict(
                method=method, label=LABELS.get(method, method), frequency=frequency,
                adaptation='fixed' if method in FIXED else 'adapted',
                sessions=len(group), r2_on=group.r2_ave.mean(), ci_low=low, ci_high=high,
                r2_off=group.r2_off_ave.mean()))
    return pd.DataFrame(rows)


def paired(per_session, pairs=PAIRS):
    """Paired differences on aligned sessions, with a bootstrap CI over sessions."""
    by_session = (per_session.groupby(['method', 'session', 'am_freq'])[['r2_ave']]
                  .mean().reset_index())
    wide = by_session.pivot(index=['session', 'am_freq'], columns='method', values='r2_ave')
    rows = []
    for a, b in pairs:
        if a not in wide or b not in wide:
            continue
        for frequency, part in [('all', wide)] + [
                (f'{freq:g} Hz', wide[wide.index.get_level_values('am_freq') == freq])
                for freq in sorted(by_session.am_freq.unique())]:
            aligned = part[[a, b]].dropna()
            if len(aligned) < 3:
                continue
            difference = (aligned[a] - aligned[b]).to_numpy()
            low, high = U.bootstrap_confidence_interval(difference)
            rows.append(dict(
                method_a=LABELS.get(a, a), method_b=LABELS.get(b, b), frequency=frequency,
                sessions=len(aligned), mean_difference=difference.mean(),
                ci_low=low, ci_high=high,
                wins=int((difference > 0).sum()), losses=int((difference < 0).sum())))
    return pd.DataFrame(rows)


def artifact_severity(sessions):
    """Per-session artifact amplitude, as a fraction of the ON recording's own spread.

    Reads the PARRM artifact estimate for the 35 decoded channels and decimates by 10 to
    keep the whole comparison to a few seconds.
    """
    import h5py

    rows = []
    for session in sessions:
        with h5py.File(os.path.join(U.DATA_DIR, 'export_outputs', 'h5_export_outputs',
                                    f'{session}.h5')) as file:
            keep = file['channel_keep_fixed'][:].astype(bool)
            on = file['lfp'][keep][:, ::10]
        with h5py.File(os.path.join(U.PARRM_DIR, f'{session}.h5')) as file:
            artifact = file['artifact'][keep][:, ::10]

        def mad(values):
            centre = np.median(values, axis=1, keepdims=True)
            return 1.4826 * np.median(np.abs(values - centre), axis=1)

        rows.append(dict(session=session,
                         am_freq=float(U.SESSION_TABLE.loc[session, 'am_freq_hz']),
                         artifact_fraction=float(np.median(mad(artifact) / mad(on)))))
    return pd.DataFrame(rows)


def audit_checkpoints(test, amplified, cleaned, skip):
    """Reload every checkpoint and score it against the row already on disk."""
    rows = []
    sessions = sorted(test) if test else []
    cached = pd.read_csv(os.path.join(U.TASK4_RESULTS, 'task4_2_eegnet_off.csv'))
    for method in (U.EEGNET_METHOD, U.EEGNET_TONE_METHOD):
        for seed in range(5):
            name = f'task4_2_{method}_seed{seed}'
            path = os.path.join(U.CHECKPOINT_DIR, f'{name}.pt')
            expected = cached[(cached.method == method) & (cached.seed == seed)]
            row = dict(checkpoint=f'task4/checkpoints/{name}.pt', available=os.path.isfile(path),
                       expected_on=expected.r2_ave.mean(), expected_off=expected.r2_off_ave.mean(),
                       reloaded_on=np.nan, reloaded_off=np.nan, max_abs_session_difference=np.nan)
            if row['available'] and not skip:
                model = U.load_eegnet_checkpoint(name)
                scored = pd.DataFrame([U.score_trial_model(model, s, method, 'raw', seed, q, off)
                                       for s, (q, off) in test.items()])
                row['reloaded_on'] = scored.r2_ave.mean()
                row['reloaded_off'] = scored.r2_off_ave.mean()
                aligned = (scored.set_index('session').r2_ave
                           - expected.set_index('session').r2_ave).abs().max()
                row['max_abs_session_difference'] = float(aligned)
            rows.append(row)

    # The older baseline checkpoints: only seed 0 was kept, so compare that one. They read
    # different tensors, so each needs its own loader and scorer: the GRU is a trial model on
    # 40 bins x 280 features, the MLP a window model on 280 features x 10 lags.
    legacy = {
        'gru': dict(model=U.GRURegressor(280), path='results/checkpoints/gru_seed0.pt',
                    parquet='results_gru.parquet',
                    datasets=(lambda: U.load_gru_datasets(U.RAW_VARIANT, sessions)[2]),
                    score=lambda model, data: U.evaluate_gru(model, 0, data, 'raw')),
        'mlp': dict(model=U.MLPRegressor(280 * (U.LAG_COUNT + 1)),
                    path='results/checkpoints/mlp_seed0.pt', parquet='results_mlp.parquet',
                    datasets=(lambda: {s: U.load_query_and_off_windows(s, U.RAW_VARIANT,
                                                                       U.LAG_COUNT)
                                       for s in sessions}),
                    score=lambda model, data: U.evaluate_mlp(model, 0, data, 'raw')),
    }
    import torch
    for method, entry in legacy.items():
        path = entry['path']
        full = os.path.join(U.PROJECT_ROOT, path)
        reference = pd.read_parquet(os.path.join(U.RESULTS_DIR, entry['parquet']))
        expected = reference[reference.seed == 0]
        row = dict(checkpoint=path, available=os.path.isfile(full),
                   expected_on=expected.r2_ave.mean(), expected_off=expected.r2_off_ave.mean(),
                   reloaded_on=np.nan, reloaded_off=np.nan, max_abs_session_difference=np.nan)
        if row['available'] and not skip:
            model = entry['model']
            model.load_state_dict(torch.load(full, map_location='cpu', weights_only=True))
            model.eval()
            scored = pd.DataFrame(entry['score'](model, entry['datasets']()))
            row['reloaded_on'] = scored.r2_ave.mean()
            row['reloaded_off'] = scored.r2_off_ave.mean()
            row['max_abs_session_difference'] = float(
                (scored.set_index('session').r2_ave
                 - expected.set_index('session').r2_ave).abs().max())
        rows.append(row)

    # Ridge: the npz holds the OFF decoder's coefficients, so replay evaluate_ridge and
    # compare its no-adaptation rows (k=0, draw=0) against the cached table.
    if not skip:
        draws = U.load_json(U.DRAWS_PATH)
        reference = pd.read_parquet(os.path.join(U.RESULTS_DIR, 'results_ridge_knn.parquet'))
        for variant, label, path in (
                (U.RAW_VARIANT, 'ridge_raw', 'results/checkpoints/ridge_raw.npz'),
                (U.PARRM_VARIANT, 'ridge_parrm', 'results/checkpoints/ridge_parrm.npz')):
            full = os.path.join(U.PROJECT_ROOT, path)
            method = f'ridge_{label.split("_")[-1]}_off'
            expected = reference[reference.method == method]
            row = dict(checkpoint=path, available=os.path.isfile(full),
                       expected_on=expected.r2_ave.mean(),
                       expected_off=expected.r2_off_ave.mean(),
                       reloaded_on=np.nan, reloaded_off=np.nan, max_abs_session_difference=np.nan)
            if row['available']:
                landscape = np.load(full)
                model = U.RidgeRegressor(float(landscape['alpha']))
                model.coefficients = landscape['coefficients']
                model.intercept = landscape['intercept']
                replayed = pd.DataFrame(U.evaluate_ridge(
                    draws, model, float(landscape['alpha']), variant,
                    label.split('_')[-1], U.LAG_COUNT, sorted(sessions)))
                replayed = replayed[replayed.method == method]
                row['reloaded_on'] = replayed.r2_ave.mean()
                row['reloaded_off'] = replayed.r2_off_ave.mean()
                row['max_abs_session_difference'] = float(
                    (replayed.set_index('session').r2_ave
                     - expected.set_index('session').r2_ave).abs().max())
            rows.append(row)

    # The new arms: recorded, PARRM-cleaned and 10x, for the two ensemble rows.
    if not skip:
        for method in (U.EEGNET_METHOD, U.EEGNET_TONE_METHOD):
            models = [U.load_eegnet_checkpoint(f'task4_2_{method}_seed{seed}')
                      for seed in range(5)]
            for arm, datasets, variant in (('recorded', test, 'raw'),
                                           ('parrm_cleaned', cleaned, U.PARRM_VARIANT),
                                           ('amp10', amplified, U.AMP10_VARIANT)):
                scored = pd.DataFrame([U.score_ensemble(models, s, f'{method}_ensemble',
                                                        variant, q, off)
                                       for s, (q, off) in datasets.items()])
                rows.append(dict(checkpoint=f'{method}_ensemble on {arm}', available=True,
                                 expected_on=np.nan, expected_off=np.nan,
                                 reloaded_on=scored.r2_ave.mean(),
                                 reloaded_off=scored.r2_off_ave.mean(),
                                 max_abs_session_difference=np.nan))
    return pd.DataFrame(rows)


def figure(comparison, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    order = ['eegnet', 'eegnet_ensemble', 'mlp', 'gru', 'eegnet_stimulation_tones',
             'eegnet_stimulation_tones_ensemble', 'ridge_raw_off', 'knn_off',
             'ridge_raw_refit', 'ridge_parrm_refit', 'knn_incontext']
    rows = [m for m in order if m in set(comparison.method)]
    overall = comparison[comparison.frequency == 'all'].set_index('method').loc[rows]
    three = comparison[comparison.frequency == '3 Hz'].set_index('method').loc[rows]
    twenty = comparison[comparison.frequency == '20 Hz'].set_index('method').loc[rows]

    figure_axes, axis = plt.subplots(figsize=(9, 4.6))
    positions = np.arange(len(rows))
    width = 0.27
    for offset, part, colour, name in ((-width, three, '#c44e52', '3 Hz'),
                                        (0.0, overall, '#4c72b0', 'all'),
                                        (width, twenty, '#55a868', '20 Hz')):
        axis.bar(positions + offset, part.r2_on, width, label=name, color=colour,
                 yerr=[part.r2_on - part.ci_low, part.ci_high - part.r2_on],
                 capsize=2, error_kw=dict(lw=0.8))
    axis.axhline(0, color='black', lw=0.8)
    axis.set_xticks(positions)
    axis.set_xticklabels([LABELS.get(m, m) for m in rows], rotation=35, ha='right', fontsize=8)
    axis.set_ylabel('query $R^2$ on stimulated trials')
    axis.set_title('Fixed models (left) versus models that spend 20 labelled ON trials (right)',
                   fontsize=9)
    axis.legend(title='sessions', fontsize=8)
    figure_axes.tight_layout()
    figure_axes.savefig(path, dpi=150)
    plt.close(figure_axes)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-checkpoints', action='store_true',
                        help='tables only; do not reload and score the weights')
    args = parser.parse_args()

    per_session = load_scores()
    comparison = summarise(per_session)
    differences = paired(per_session)

    comparison.to_csv(os.path.join(U.TASK4_RESULTS, 'decoder_comparison.csv'), index=False)
    differences.to_csv(os.path.join(U.TASK4_RESULTS, 'decoder_paired.csv'), index=False)

    sessions = sorted(per_session.session.unique())
    severity = artifact_severity(sessions)
    severity.to_csv(os.path.join(U.TASK4_RESULTS, 'artifact_severity.csv'), index=False)

    if args.skip_checkpoints:
        audit = audit_checkpoints(None, None, None, True)
    else:
        test = U.stimulated_test_datasets()
        amplified = U.amp10_test_datasets(sessions)
        cleaned = U.parrm_test_datasets(sessions)
        audit = audit_checkpoints(test, amplified, cleaned, False)
    audit.to_csv(os.path.join(U.TASK4_RESULTS, 'checkpoint_audit.csv'), index=False)

    figure(comparison, os.path.join(U.RESULTS_DIR, 'fig_grueegnet.png'))

    print(comparison[comparison.frequency == 'all']
          .sort_values('r2_on', ascending=False)
          [['label', 'adaptation', 'sessions', 'r2_on', 'ci_low', 'ci_high', 'r2_off']]
          .round(4).to_string(index=False))
    print('\npaired differences:')
    print(differences[differences.frequency == 'all']
          [['method_a', 'method_b', 'sessions', 'mean_difference', 'ci_low', 'ci_high', 'wins', 'losses']]
          .round(4).to_string(index=False))
    print('\ncheckpoint audit:')
    print(audit.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
