"""Plot the ridge alpha and KNN neighbour sweeps with R2 and RMSE."""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

sys.path.insert(0, os.path.join(ROOT, "task4"))
import utils as U

COLORS = {'raw': '#2f5d8a', 'parrm': '#d9772a', 'bank': '#2f7d5c', 'cal': '#7b4fa3'}
LAGS = 9


def metrics(y, yhat, valid):
    """Return mean R2 and RMSE over valid bins."""
    err = (y - yhat)[valid]
    return float(U.mean_r2(y, yhat, valid)), float(np.sqrt(np.mean(err ** 2)))


SESSIONS_OFF = U.sessions_in_split('val', 0)
SESSIONS_ON = U.sessions_in_split('val', 1)


def validation_sets(variant):
    return {
        'off': [U.valid_bin_rows(U.load_split_dataset(s, 'query', variant, LAGS))
                for s in SESSIONS_OFF],
        'on': [U.valid_bin_rows(U.load_split_dataset(s, 'query', variant, LAGS))
               for s in SESSIONS_ON],
    }


def score_model(model, sets):
    scores = {'off': [], 'on': []}
    for group, pairs in sets.items():
        for X, y in pairs:
            r2, rmse = metrics(y, model.predict(X), np.ones(len(y), bool))
            scores[group].append((r2, rmse))
    return {group: (float(np.mean([v[0] for v in values])),
                    float(np.mean([v[1] for v in values])))
            for group, values in scores.items()}


print('Ridge alpha sweep')
ridge = {}
for variant, label in [('export_none', 'raw'), ('parrm_none', 'parrm')]:
    X_train, y_train = U.stack_session_bin_rows(U.STIMULATION_OFF_TRAINING_SESSIONS, 'train', variant, LAGS)
    sets = validation_sets(variant)
    rows = []
    for alpha in U.RIDGE_PENALTIES:
        model = U.RidgeRegressor(alpha).fit(X_train, y_train)
        rows.append((alpha, score_model(model, sets)))
        print(label, alpha, rows[-1][1])
    ridge[label] = rows


print('KNN neighbour sweep')
X_bank, y_bank = U.stack_session_bin_rows(U.STIMULATION_OFF_TRAINING_SESSIONS, 'train', 'export_none', LAGS)
X_val_off, y_val_off = U.stack_session_bin_rows(SESSIONS_OFF, 'query', 'export_none', LAGS)
X_val_on, y_val_on = U.stack_session_bin_rows(SESSIONS_ON, 'query', 'export_none', LAGS)
calibration = U.SPLITS['calibration']
cal = [U.load_calibration_bin_rows(s, calibration[s]['subsets'][str(U.CALIBRATION_TUNE_TRIALS)][0],
                                      calibration[s]['pool'], 'export_none', LAGS)
       for s in SESSIONS_ON]
X_cal, y_cal = [np.concatenate(part) for part in zip(*cal)]

knn = {'bank_off': [], 'bank_on': [], 'cal_on': []}
for k in U.KNN_NEIGHBOR_GRID:
    for name, X_bank_part, y_bank_part, X_val, y_val in [
            ('bank_off', X_bank, y_bank, X_val_off, y_val_off),
            ('bank_on', X_bank, y_bank, X_val_on, y_val_on),
            ('cal_on', X_cal, y_cal, X_val_on, y_val_on)]:
        yhat = U.KNNRegressor(k).fit(X_bank_part, y_bank_part).predict(X_val)
        r2, rmse = metrics(y_val, yhat, np.ones(len(y_val), bool))
        knn[name].append((k, r2, rmse))
        print(name, k, r2, rmse)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='white')
fig.suptitle('Parameter sweeps: validation R² and RMSE', fontsize=16, weight='bold', y=0.98)
fig.text(0.5, 0.935, 'Selection uses mean validation R². RMSE is shown for comparison.',
         ha='center', fontsize=9.5, color='#5c6670')

for column, metric_index in [(0, 0), (1, 1)]:
    ax = axes[0, column]
    for group, style in [('off', '-'), ('on', '--')]:
        xs = [row[0] for row in ridge['raw']]
        ys = [row[1][group][metric_index] for row in ridge['raw']]
        ax.plot(xs, ys, style, marker='o', color=COLORS['raw'],
                label=f'raw, {group.upper()}')
    ax.set_xscale('log')
    ax.axvline(1e5, color='k', linestyle=':', linewidth=0.8)
    ax.set_xlabel('ridge alpha')
    ax.set_ylabel('R²' if metric_index == 0 else 'RMSE (cm/s)')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

for column, metric_index in [(0, 1), (1, 2)]:
    ax = axes[1, column]
    for name, label, style in [('bank_off', 'OFF bank', '-'), ('bank_on', 'ON bank', '--'),
                               ('cal_on', 'ON calibration', '-.')]:
        xs = [row[0] for row in knn[name]]
        ys = [row[metric_index] for row in knn[name]]
        ax.plot(xs, ys, style, marker='o', color=COLORS['bank' if 'bank' in name else 'cal'],
                label=label)
    ax.set_xscale('log')
    ax.set_xlabel('KNN neighbours k')
    ax.set_ylabel('R²' if metric_index == 1 else 'RMSE (cm/s)')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

fig.text(0.5, 0.015,
         'Current code tunes on ON validation sessions. OFF curves show the no-stimulation split. Ridge curves are raw; PARRM overlaps.',
         ha='center', fontsize=8.5, color='#5c6670')
fig.tight_layout(rect=[0, 0.04, 1, 0.92])
fig.savefig('reports/fig_parameter_sweeps.png', dpi=180, facecolor='white')
plt.close(fig)
print('wrote reports/fig_parameter_sweeps.png')
