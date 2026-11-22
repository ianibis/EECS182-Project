import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import notebook_api

OUT = os.path.join('reports', 'baselines', 'diagnostics')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 160, 'savefig.bbox': 'tight'})

ns = notebook_api.load(extra=notebook_api.KNN + notebook_api.NEURAL)

session = 'Butters-250616-102124'
methods = ['Ridge', 'KNN', 'MLP', 'GRU', 'EEGNet']
colors = {'Ridge': '#1f77b4', 'KNN': '#2ca02c', 'MLP': '#d62728',
          'GRU': '#9467bd', 'EEGNet': '#8c564b'}

# ---------------------------------------------------------------- predictions
q9 = ns['load_split_dataset'](session, 'query', 'export_none', 9)
X9, y9 = ns['valid_windows'](q9)
v_tab = q9['valid_bins']

ridge = np.load('results/checkpoints/ridge_raw.npz')
p_ridge = X9 @ ridge['coefficients'] + ridge['intercept']

X_bank, y_bank = ns['collect_windows'](ns['OFF_TRAIN_SESSIONS'], 'train', 'export_none', 9)
knn = ns['KNNRegressor'](100).fit(X_bank, y_bank)
p_knn = knn.predict(X9)

mlp = ns['MLPRegressor'](X9.shape[1])
mlp.load_state_dict(torch.load('results/checkpoints/mlp_seed0.pt', map_location='cpu'))
mlp.eval()
p_mlp = ns['predict'](mlp, X9)

q0 = ns['load_split_dataset'](session, 'query', 'export_none', 0)
v_gru = q0['valid_bins']
gru = ns['GRURegressor'](q0['features'].shape[-1])
gru.load_state_dict(torch.load('results/checkpoints/gru_seed0.pt', map_location='cpu'))
gru.eval()
p_gru = ns['predict_trials'](gru, q0)[v_gru]

eeg_q = ns['load_lfp_split'](session, 'query')
v_eeg = eeg_q['valid_bins']
eeg = ns['EEGNetRegressor'](eeg_q['features'].shape[1])
eeg.load_state_dict(torch.load('results/checkpoints/eegnet_seed0.pt', map_location='cpu'))
eeg.eval()
p_eeg = ns['predict_trials'](eeg, eeg_q)[v_eeg]

preds = {
    'Ridge': (y9, p_ridge, v_tab),
    'KNN': (y9, p_knn, v_tab),
    'MLP': (y9, p_mlp, v_tab),
    'GRU': (q0['targets'][v_gru], p_gru, v_gru),
    'EEGNet': (eeg_q['targets'][v_eeg], p_eeg, v_eeg),
}

def fill_trials(pred, valid):
    out = np.full((*valid.shape, 2), np.nan)
    out[valid] = pred
    return out

# ---------------------------------------------------------------- 01 method bars
results = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob('results/results*.parquet'))],
                    ignore_index=True)
rows = []
for method, label in [('ridge_raw_off', 'Ridge raw'), ('ridge_parrm_off', 'Ridge PARRM'),
                      ('ridge_raw_refit', 'Ridge raw refit'), ('ridge_parrm_refit', 'Ridge PARRM refit'),
                      ('knn_off', 'KNN off'), ('knn_incontext', 'KNN k=20'),
                      ('mlp', 'MLP'), ('gru', 'GRU'), ('eegnet', 'EEGNet')]:
    d = results[results.method == method]
    if method in ('ridge_raw_refit', 'ridge_parrm_refit', 'knn_incontext'):
        d = d[d.k == 20]
    for freq in (3, 20):
        rows.append((label, freq, d[d.am_freq == freq].r2_ave.mean()))
bars = pd.DataFrame(rows, columns=['method', 'freq', 'r2'])
labels = bars.method.unique()
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(7.2, 2.8))
for j, freq in enumerate((3, 20)):
    v = [bars[(bars.method == l) & (bars.freq == freq)].r2.iloc[0] for l in labels]
    ax.bar(x + (j - 0.5) * 0.38, v, 0.38, label=f'{freq} Hz')
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha='right')
ax.set_ylabel('R2'); ax.legend(); ax.set_title('baseline comparison')
fig.savefig(os.path.join(OUT, '01_method_comparison.png')); plt.close(fig)

# ---------------------------------------------------------------- 02 training curves
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
for ax, method in zip(axes, ('mlp', 'gru', 'eegnet')):
    with open(f'results/training_logs/{method}_seed0.json') as file:
        h = json.load(file)
    e = [r['epoch'] for r in h['training_log']]; v = [r['val_r2'] for r in h['training_log']]
    ax.plot(e, v, color=colors[method.upper() if method != 'eegnet' else 'EEGNet'])
    ax.axvline(h['best_epoch'], color='k', ls='--', lw=0.8)
    ax.set_title(method.upper()); ax.set_xlabel('epoch')
axes[0].set_ylabel('validation R2')
fig.savefig(os.path.join(OUT, '02_training_curves.png')); plt.close(fig)

# ---------------------------------------------------------------- 03 ridge weights
coef = ridge['coefficients'].reshape(10, 35, 8, 2)
weight = np.abs(coef).sum(axis=1)
bands = ['LMP', '1-4', '4-8', '8-13', '13-30', '30-50', '70-110', '130-200']
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
for j, ax in enumerate(axes):
    im = ax.imshow(weight[:, :, j].T, aspect='auto', cmap='magma')
    ax.set_yticks(range(8)); ax.set_yticklabels(bands)
    ax.set_xticks(range(10)); ax.set_xticklabels([str(i) for i in range(10)])
    ax.set_xlabel('lag bin'); ax.set_title(('vx', 'vy')[j])
fig.colorbar(im, ax=axes, shrink=0.8)
fig.savefig(os.path.join(OUT, '03_ridge_weights.png')); plt.close(fig)

# ---------------------------------------------------------------- 04 knn neighbors
val_sessions = ns['split_sessions']('val', 1)
X_val, y_val = ns['collect_windows'](val_sessions, 'query', 'export_none', 9)
calibration = ns['SPLITS']['calibration']
cal = [ns['load_calibration_windows'](s, calibration[s]['subsets']['10'][0],
                                      calibration[s]['pool'], 'export_none', 9)
       for s in val_sessions]
X_cal, y_cal = [np.concatenate(part) for part in zip(*cal)]
scores_bank, scores_cal = [], []
for n in ns['NEIGHBORS']:
    scores_bank.append(ns['mean_r2'](y_val, ns['KNNRegressor'](n).fit(X_bank, y_bank).predict(X_val)))
    scores_cal.append(ns['mean_r2'](y_val, ns['KNNRegressor'](n).fit(X_cal, y_cal).predict(X_val)))
fig, ax = plt.subplots(figsize=(4.2, 2.4))
ax.plot(ns['NEIGHBORS'], scores_bank, 'o-', label='stim-off bank')
ax.plot(ns['NEIGHBORS'], scores_cal, 'o-', label='k=10 calibration')
ax.set_xscale('log'); ax.set_xlabel('neighbors'); ax.set_ylabel('validation R2')
ax.legend(); ax.set_title('KNN neighbor sweep')
fig.savefig(os.path.join(OUT, '04_knn_neighbors.png')); plt.close(fig)

# ---------------------------------------------------------------- 05 prediction scatter
fig, axes = plt.subplots(len(methods), 2, figsize=(7.2, 9), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.35)
for i, name in enumerate(methods):
    y, p, valid = preds[name]
    for j, ax in enumerate(axes[i]):
        ax.scatter(y[:, j], p[:, j], s=4, alpha=0.25, color=colors[name])
        lim = [np.nanmin(y[:, j]), np.nanmax(y[:, j])]
        ax.plot(lim, lim, 'k--', lw=0.7)
        ax.set_title(f'{name} {"vx" if j == 0 else "vy"} R2={ns["r2_per_output"](y, p)[j]:.3f}')
        if j == 0: ax.set_ylabel('predicted')
        if i == len(methods) - 1: ax.set_xlabel('measured')
fig.suptitle('prediction scatter, held-out query bins')
fig.savefig(os.path.join(OUT, '05_prediction_scatter.png')); plt.close(fig)

# ---------------------------------------------------------------- 06 trial speed
common = v_tab & v_gru & v_eeg
trial = int(np.flatnonzero(common.all(axis=1))[0]) if common.all(axis=1).any() else 0
times = np.arange(40) * 0.05 - 0.5
fig, axes = plt.subplots(1, 5, figsize=(10, 2.2), sharey=True)
for ax, name in zip(axes, methods):
    y, p, valid = preds[name]
    yt = fill_trials(y, valid)[trial]
    pt = fill_trials(p, valid)[trial]
    ax.plot(times, np.hypot(yt[:, 0], yt[:, 1]), 'k', lw=2, label='measured')
    ax.plot(times, np.hypot(pt[:, 0], pt[:, 1]), color=colors[name], lw=1.2, label='decoded')
    ax.set_title(name); ax.set_xlabel('s from go')
axes[0].set_ylabel('speed (cm/s)'); axes[0].legend(fontsize=6)
fig.suptitle(f'speed, session {session}, trial {trial}', y=1.02)
fig.subplots_adjust(top=0.78)
fig.savefig(os.path.join(OUT, '06_trial_speed.png')); plt.close(fig)

# ---------------------------------------------------------------- 07 error vs speed
fig, ax = plt.subplots(figsize=(4.6, 2.6))
for name in methods:
    y, p, valid = preds[name]
    sm = np.hypot(y[:, 0], y[:, 1]); sp = np.hypot(p[:, 0], p[:, 1])
    edges = np.nanpercentile(sm, [0, 25, 50, 75, 100])
    centers, errs = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (sm >= a) & (sm <= b)
        if m.any():
            centers.append(np.nanmean(sm[m])); errs.append(np.nanmean(sp[m] - sm[m]))
    ax.plot(centers, errs, 'o-', color=colors[name], label=name)
ax.axhline(0, color='k', lw=0.7)
ax.set_xlabel('measured speed (cm/s)'); ax.set_ylabel('mean decoded - measured')
ax.legend(fontsize=6); ax.set_title('speed compression')
fig.savefig(os.path.join(OUT, '07_error_vs_speed.png')); plt.close(fig)

# ---------------------------------------------------------------- 08 eegnet filters
sd = torch.load('results/checkpoints/eegnet_seed0.pt', map_location='cpu')
fig, axes = plt.subplots(1, 3, figsize=(10, 2.5))
k = sd['layers.1.weight'].squeeze().numpy()
axes[0].plot(k.T, lw=0.7); axes[0].set_title('temporal kernels (8 x 500)')
axes[0].set_xlabel('ms')
s = sd['layers.3.weight'].squeeze().numpy()
im = axes[1].imshow(s, aspect='auto', cmap='RdBu_r', vmin=-np.abs(s).max(), vmax=np.abs(s).max())
axes[1].set_title('spatial filters (16 x 35)'); axes[1].set_xlabel('channel')
out = sd['layers.15.weight'].squeeze().numpy()
axes[2].imshow(out, aspect='auto', cmap='RdBu_r', vmin=-np.abs(out).max(), vmax=np.abs(out).max())
axes[2].set_title('output weights (2 x 16)'); axes[2].set_yticks([0, 1]); axes[2].set_yticklabels(['vx', 'vy'])
fig.colorbar(im, ax=axes[1], shrink=0.8)
fig.savefig(os.path.join(OUT, '08_eegnet_filters.png')); plt.close(fig)

# ---------------------------------------------------------------- copy summary figures
for src, dst in [('results/fig1_r2_vs_k.png', '09_r2_vs_k.png'),
                 ('results/fig4_session_heatmap.png', '10_session_heatmap.png'),
                 ('results/fig6_noise_floor.png', '11_noise_floor.png')]:
    if os.path.exists(src):
        import shutil
        shutil.copyfile(src, os.path.join(OUT, dst))

print('wrote', OUT)
