"""Smoke test for the data loader, features and PARRM templates (plan section 2.3).

Run from ICL/:  .venv/bin/python scripts/smoke_loader.py
Data root defaults to ~/data/EECS182 (override with EECS182_DATA).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import icl  # noqa: E402,F401  (makes the team loader importable)
import EECS182_NHP_data_loader as L
import EECS182_NHP_features as F

DATA = os.environ.get('EECS182_DATA', os.path.expanduser('~/data/EECS182'))
P = L.Paths(root=DATA, kin_dir=f'{DATA}/kinematics_v3/bins50ms')


def section(title):
    print(f'\n=== {title}')


section('manifest and splits')
M, S = L.load_manifest(P), L.load_splits(P)
assert L.check_splits(S, M)
print(len(M), 'sessions;', {part: len(S['sessions'][part]) for part in ('train', 'val', 'test')})

section('val 3 Hz session, export source (Butters-250606-104043)')
s = L.load_session('Butters-250606-104043', P)
print('fs', s.fs, '| lfp', s.lfp.shape, s.lfp.dtype, '| channels', s.channels.tolist())
print('stim on fraction', round(float(s.stim_on.mean()), 3), '| attrs am_freq_measured_hz',
      s.attrs.get('am_freq_measured_hz'), '| units', s.attrs.get('lfp_units'))
K = L.load_kinematics(s.name, P)
cal = K.attrs['calibration']
print('kinematics rows', len(K), '| calibration r2', round(cal['r2_x'], 3), round(cal['r2_y'], 3))
pool = L.split_trials(S, s, 'pool')
ex = L.trial_examples(s, K, align='t_go', pre=2.0, post=1.5, trials=pool)
e = ex[0]
print('pool trials', len(ex), '| lfp', e['lfp'].shape, '| kin', e['kin'].shape,
      '| valid bins in target window', int(e['kin_valid'][-40:].sum()), '/ 40')
print('lfp NaN in any pool trial:', any(np.isnan(x['lfp']).any() for x in ex))
lfp_uv = np.concatenate([x['lfp'].reshape(35, -1) for x in ex], axis=1)
print('raw LFP (uV) per-channel robust sd: median', round(float(np.median(1.4826 * np.median(
    np.abs(lfp_uv - np.median(lfp_uv, axis=1, keepdims=True)), axis=1))), 1))
vel = np.concatenate([x['kin'][-40:][x['kin_valid'][-40:]] for x in ex])
print('Index velocity cm/s: sd per axis', np.round(vel.std(axis=0), 1), '| max speed', round(float(np.hypot(*vel.T).max()), 1))

section('same session, PARRM source + artifact')
sp = L.load_session(s.name, P, source='parrm', with_artifact=True)
print('source', sp.source, '| artifact rms during stim (uV)', round(float(np.sqrt(np.mean(sp.artifact[:, sp.stim_on] ** 2))), 2))

section('no-stim training session (Butters-250602-100529)')
s0 = L.load_session('Butters-250602-100529', P)
K0 = L.load_kinematics(s0.name, P)
tr = L.split_trials(S, s0, 'train')
va = L.split_trials(S, s0, 'val')
print('train trials', len(tr), '| val trials', len(va), '| stim on fraction', float(s0.stim_on.mean()))

section('July 11 session has no cm calibration (Butters-250711-101424)')
K11 = L.load_kinematics('Butters-250711-101424', P)
print('calibration', K11.attrs['calibration'], '| Index_vx_cm all NaN:', bool(K11['Index_vx_cm'].isna().all()))

section('features (export_none, pool normalisation)')
rows = lambda sess, part: sess.trials[sess.trials.trial.isin(L.split_trials(S, sess, part))]
f = F.load_features(s.name, P, variant='export_none', normalize='pool', trials=rows(s, 'pool'))
X, Y, valid, bins = F.trial_arrays(f, rows(s, 'query'), kin=K, align='t_go', pre=0.5, post=1.5, n_lags=9)
print('X', X.shape, '| Y', Y.shape, '| valid fraction', round(float(valid.mean()), 3), '| features', f['feature_names'])

section('PARRM template library')
T = L.load_templates(s.name, P)
for key, value in T.items():
    print(f'  {key:20s}', getattr(value, 'shape', value), getattr(value, 'dtype', ''))
print('\nsmoke test passed')
