"""Simulated vs real stimulation on training days (plan 5.1.3), and calibration of amp = 1.

For every training-split stim session that has a same-day no-stim session (splits.json
off_reference):

1. Spectra. Real: ON trials vs the OFF session's trials. Simulated: the OFF trials plus the ON
   session's own PARRM template scaled by kappa, vs the same OFF trials. The AM excess E is the
   ON/OFF power ratio at the AM frequency bin divided by the median ratio over flanking bins
   (f0 +- 2.5 Hz, excluding +- 0.75 Hz), so broadband differences between sessions cancel.
   kappa_fit is the scale at which the simulated excess equals the real one; the real artifact
   RMS is then kappa_fit times the template RMS.
2. Decoding. A ridge decoder (LMP + 7 bands, 9 lags, per-condition robust normalisation) is
   trained on the OFF session's training trials and tested on its held-out block (OFF), the same
   block with simulated artifact at kappa = 1 and at kappa_fit, and all trials of the real ON session.

Run from ICL/:  .venv/bin/python scripts/check_shift_vs_real.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from sklearn.linear_model import Ridge

from icl import config as C
from icl.cache import load_cached, load_index, target_bins
from icl.features_torch import add_lags, trial_features
from icl.shifts import N_PHASE, Shift, TemplateBank, apply_shift

OUT_CSV = os.path.join(C.RESULTS, 'sim_vs_real.csv')
OUT_JSON = os.path.join(C.RESULTS, 'amp_calibration.json')
OUT_FIG = os.path.join(C.RESULTS, 'figs', 'sim_vs_real_psd.png')
KAPPAS = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32])
N_LAGS = 9
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def psd(x):
    """Trial-averaged Welch PSD per channel: x (trials, C, T) -> freqs, (C, F)."""
    freqs, power = welch(x, fs=C.FS, nperseg=2000, noverlap=1000, axis=-1)
    return freqs, power.mean(axis=0)


def ratio_spectrum(on, off):
    return np.median(on / off, axis=0)                 # median over channels


def am_excess(freqs, ratio, f0):
    peak = ratio[np.argmin(np.abs(freqs - f0))]
    flank = (np.abs(freqs - f0) <= 2.5) & (np.abs(freqs - f0) > 0.75) & (freqs >= 0.5)
    return float(peak / np.median(ratio[flank]))


def template_shift(wave, f_am, kappa):
    return Shift(f_am=f_am, wave=torch.from_numpy(wave), amp=float(kappa), neural_gain=torch.ones(C.N_CHANNELS),
                 drop=torch.zeros(C.N_CHANNELS, dtype=torch.bool), trial_jitter=0.0, kind='template')


def features(x_uv, fs):
    """(trials, C, T) microvolts -> lagged features for the 40 decoded bins, normalised within the set."""
    first = C.TARGET_START_BIN - N_LAGS
    f = trial_features(torch.from_numpy(x_uv).to(DEVICE), fs, first, C.N_TARGET_BINS + N_LAGS)
    flat = f.reshape(-1, *f.shape[-2:])
    median = flat.median(dim=0).values
    scale = 1.4826 * (flat - median).abs().median(dim=0).values
    scale[~(scale > 0)] = 1.0
    # float64: with 2800 features and ~1300 bins sklearn solves the dual, where float32 loses accuracy
    return add_lags((f - median) / scale, N_LAGS).cpu().numpy().astype(np.float64)   # (trials, 40, 2800)


def r2(model, X, Y, valid):
    true = Y[valid]
    pred = model.predict(X[valid]).reshape(true.shape)     # single-target predict returns (n,), not (n, 1)
    per_axis = 1 - ((pred - true) ** 2).sum(axis=0) / ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    return float(per_axis.mean())


def fit_ridge(X, Y, valid, alphas=np.logspace(2, 6, 9), n_folds=5):
    """Ridge with alpha chosen by cross-validation over whole trials.

    (Leave-one-bin-out GCV, as in RidgeCV, picks far too small an alpha because neighbouring
    bins of a trial are strongly correlated.)
    """
    folds = np.array_split(np.random.default_rng(0).permutation(len(X)), n_folds)
    scores = []
    for alpha in alphas:
        fold_scores = []
        for fold in folds:
            rest = np.setdiff1d(np.arange(len(X)), fold)
            model = Ridge(alpha=alpha).fit(X[rest][valid[rest]], Y[rest][valid[rest]])
            fold_scores.append(r2(model, X[fold], Y[fold], valid[fold]))
        scores.append(np.mean(fold_scores))
    best = float(alphas[int(np.argmax(scores))])
    return Ridge(alpha=best).fit(X[valid], Y[valid]), best


def main():
    index = load_index()
    with open(C.SPLITS) as file:
        splits = json.load(file)
    bank = TemplateBank.load(('train',))
    pairs = [(s, splits['off_reference'][s]) for s in splits['sessions']['train']
             if index.get(s, {}).get('stim') == 1 and splits['off_reference'].get(s) in index]
    manifest = pd.read_csv(os.path.join(C.DATA, 'manifest.csv')).set_index('session')
    rng = np.random.default_rng(0)

    rows, spectra = [], {}
    for on_name, off_name in pairs:
        on, off = load_cached(on_name), load_cached(off_name)
        for data in (on, off):                     # decode the standard window
            data['kin'], data['valid'] = data['kin'][:, target_bins()], data['valid'][:, target_bins()]
        key = on['meta']['am_freq_hz']
        f0 = C.AM_FREQS_HZ[key]
        if on_name not in bank.sessions[key]:
            print('skip (template not in training bank):', on_name)
            continue
        i = bank.sessions[key].index(on_name)
        wave, template_rms = bank.waves[key][i], bank.rms_uv[key][i]

        freqs, p_off = psd(off['lfp'])
        _, p_on = psd(on['lfp'])
        real_ratio = ratio_spectrum(p_on, p_off)
        clean = torch.from_numpy(off['lfp'])
        sim_ratio = {}
        for kappa in KAPPAS:
            shifted = apply_shift(clean, template_shift(wave, f0, kappa), rng).numpy()
            sim_ratio[kappa] = ratio_spectrum(psd(shifted)[1], p_off)
        e_real = am_excess(freqs, real_ratio, f0)
        e_sim = np.array([am_excess(freqs, sim_ratio[k], f0) for k in KAPPAS])
        kappa_fit = float(np.interp(e_real, e_sim, KAPPAS)) if e_real > e_sim[0] else 0.0
        spectra[on_name] = (freqs, real_ratio, sim_ratio[1.0],
                            ratio_spectrum(psd(apply_shift(clean, template_shift(wave, f0, kappa_fit), rng).numpy())[1], p_off))

        # decoding: ridge trained on the OFF session's training trials
        fs_off, fs_on = off['meta']['fs'], on['meta']['fs']
        train, val = off['part'] == 'train', off['part'] == 'val'
        X_off = features(off['lfp'], fs_off)
        model, alpha = fit_ridge(X_off[train], off['kin'][train], off['valid'][train])
        result = dict(session=on_name, off_reference=off_name, am_freq_hz=key, template_rms_uv=template_rms,
                      template_snr=bank.snr[key][i], manifest_am_power_ratio=manifest.loc[on_name, 'am_power_ratio'],
                      excess_real=e_real, excess_sim_k1=float(e_sim[KAPPAS == 1][0]), kappa_fit=kappa_fit,
                      kappa_capped=bool(e_real >= e_sim[-1]), real_artifact_rms_uv=kappa_fit * template_rms,
                      ridge_alpha=alpha, r2_off_val=r2(model, X_off[val], off['kin'][val], off['valid'][val]))
        for label, kappa in (('sim_k1', 1.0), ('sim_kfit', kappa_fit)):
            shifted = apply_shift(clean, template_shift(wave, f0, kappa), rng).numpy()
            X = features(shifted, fs_off)
            result[f'r2_{label}_val'] = r2(model, X[val], off['kin'][val], off['valid'][val])
        result['r2_real_on'] = r2(model, features(on['lfp'], fs_on), on['kin'], on['valid'])
        rows.append(result)
        print(f"{on_name} {key:2d} Hz  E real {e_real:5.2f}  sim(k=1) {result['excess_sim_k1']:5.2f}  kappa {kappa_fit:5.2f}"
              f"  R2 off {result['r2_off_val']:.3f}  sim1 {result['r2_sim_k1_val']:.3f}"
              f"  simfit {result['r2_sim_kfit_val']:.3f}  real {result['r2_real_on']:.3f}", flush=True)

    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    table.to_csv(OUT_CSV, index=False)

    calibration = {}
    for key, group in table.groupby('am_freq_hz'):
        calibration[int(key)] = dict(
            n_sessions=int(len(group)), n_kappa_zero=int((group.kappa_fit == 0).sum()),
            n_kappa_capped=int(group.kappa_capped.sum()),
            kappa_fit_median=float(group.kappa_fit.median()),
            real_artifact_rms_uv_median=float(group.real_artifact_rms_uv.median()),
            real_artifact_rms_uv_iqr=[float(group.real_artifact_rms_uv.quantile(q)) for q in (0.25, 0.75)],
            excess_real_median=float(group.excess_real.median()),
            r2_median={c: float(group[c].median()) for c in ('r2_off_val', 'r2_sim_k1_val', 'r2_sim_kfit_val', 'r2_real_on')})
    with open(OUT_JSON, 'w') as file:
        json.dump(calibration, file, indent=1)
    print(json.dumps(calibration, indent=1))
    plot(table, spectra)


def plot(table, spectra):
    INK, INK_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    REAL, SIM1, SIMFIT = INK_2, '#86b6ef', '#1c5cab'
    FREQ_COLORS = {3: '#2a78d6', 20: '#eb6834'}
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK_2, 'axes.labelcolor': INK, 'xtick.color': INK_2,
                         'ytick.color': INK_2, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6), facecolor=SURFACE)

    for column, key in enumerate((3, 20)):
        group = table[table.am_freq_hz == key]
        example = group.iloc[(group.excess_real - group.excess_real.median()).abs().argmin()]
        freqs, real, sim1, simfit = spectra[example.session]
        keep = (freqs >= 1) & (freqs <= 50)
        ax = axes[0, column]
        ax.axhline(1, color=GRID, lw=0.8)
        ax.semilogx(freqs[keep], real[keep], color=REAL, lw=1.2, label='real ON / OFF')
        ax.semilogx(freqs[keep], sim1[keep], color=SIM1, lw=1.6, label='sim: OFF + own template (κ = 1) / OFF')
        ax.semilogx(freqs[keep], simfit[keep], color=SIMFIT, lw=1.6, label=f'sim, κ = {example.kappa_fit:.1f} (fit)')
        ax.set_title(f'{key} Hz AM: {example.session[8:]} vs {example.off_reference[8:]}', loc='left', color=INK,
                     fontsize=10)
        ax.set_xlabel('frequency (Hz)')
        ax.set_ylabel('PSD ratio, median over channels')
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.legend(frameon=False, fontsize=8, loc='upper right')

    ax = axes[1, 0]
    top = max(table.excess_real.max(), table.excess_sim_k1.max()) * 1.05
    ax.plot([0.8, top], [0.8, top], color=GRID, lw=1)
    for key, group in table.groupby('am_freq_hz'):
        ax.scatter(group.excess_sim_k1, group.excess_real, s=36, color=FREQ_COLORS[key], edgecolor=SURFACE,
                   lw=1.5, label=f'{key} Hz ({len(group)} sessions)', zorder=3)
    ax.set_xlabel('AM excess, simulated with own template at κ = 1')
    ax.set_ylabel('AM excess, real')
    ax.set_title('Real vs template-predicted AM excess (line: equal)', loc='left', color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='upper left')

    ax = axes[1, 1]
    columns = ['r2_off_val', 'r2_sim_k1_val', 'r2_sim_kfit_val', 'r2_real_on']
    labels = ['OFF\n(held-out block)', 'sim ON\nκ = 1', 'sim ON\nκ fit', 'real ON\n(same day)']
    for key, group in table.groupby('am_freq_hz'):
        offset = -0.08 if key == 3 else 0.08
        for _, row in group.iterrows():
            ax.plot(np.arange(4) + offset, row[columns].to_numpy(float), color=FREQ_COLORS[key], lw=0.6, alpha=0.35)
        ax.plot(np.arange(4) + offset, group[columns].median().to_numpy(), color=FREQ_COLORS[key], lw=2.2,
                marker='o', ms=6, label=f'{key} Hz median')
    ax.set_xticks(range(4), labels)
    ax.set_ylabel('ridge R² (mean of vx, vy)')
    ax.set_title('OFF-trained ridge decoder', loc='left', color=INK, fontsize=10)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc='lower left')

    fig.suptitle('Simulated vs real stimulation, training days', x=0.01, ha='left', color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, facecolor=SURFACE)
    print('saved', OUT_FIG)


if __name__ == '__main__':
    main()
