"""Load Butters eACS sessions, trial-aligned LFP, and DeepLabCut kinematics.

Times use the TDT clock. Trial and kinematics bins are 50 ms.
"""

import json
import os
from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd

BIN_S = 0.05
DEFAULT_ROOT = '/path/to/root'
DEFAULT_KIN_DIR = '/path/to/kinematics_v3/bins50ms'
LMP_WIN_S = 0.256
CAL_KEYPOINT = 'Index'


@dataclass
class Paths:
    root: str = DEFAULT_ROOT
    export_dir: str = None
    trial_dir: str = None
    parrm_dir: str = None
    kin_dir: str = None
    spectra_dir: str = None

    def __post_init__(self):
        self.export_dir = self.export_dir or os.path.join(self.root, 'export_outputs')
        self.trial_dir = self.trial_dir or os.path.join(self.export_dir, 'trials')
        self.parrm_dir = self.parrm_dir or os.path.join(self.root, 'parrm_outputs')
        self.kin_dir = self.kin_dir or DEFAULT_KIN_DIR
        self.spectra_dir = self.spectra_dir or os.path.join(self.export_dir, 'onoff_spectra')


@dataclass
class Session:
    name: str
    fs: float
    t: np.ndarray
    lfp: np.ndarray
    channels: np.ndarray
    stim_on: np.ndarray
    stim_phase: np.ndarray
    stim_envelope: np.ndarray
    trials: pd.DataFrame
    attrs: dict
    info: pd.Series
    source: str
    reference: str = 'none'
    artifact: np.ndarray = None
    norm: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)

    @property
    def included(self):
        return self.trials[self.trials.include == 1]


def load_manifest(paths=None):
    paths = paths or Paths()
    return pd.read_csv(os.path.join(paths.root, 'manifest.csv'))


def load_splits(paths=None):
    paths = paths or Paths()
    with open(os.path.join(paths.root, 'splits.json')) as file:
        return json.load(file)


def check_splits(splits, manifest):
    train, val, test = (set(splits['sessions'][part]) for part in ('train', 'val', 'test'))
    assert not train & test, 'a test session is in training'
    assert not train & val, 'a validation session is in training'
    assert not val & test, 'a test session is in validation'
    assert train | val | test == set(manifest.session), 'some sessions are in no split'
    for session, calibration in splits['calibration'].items():
        assert session in val | test, f'{session} has calibration trials but is not in val or test'
        assert not set(calibration['pool']) & set(calibration['query']), (
            f'{session}: calibration and query trials overlap'
        )
    return True


def _attr(value):
    if isinstance(value, bytes):
        return value.decode()
    values = np.asarray(value)
    if values.dtype.kind in 'SO':
        values = values.astype(str)
    return values.item() if values.size == 1 else values


def channel_index(file, channels):
    ids = file['channel_ids'][:].astype(int)
    if isinstance(channels, str):
        if channels == 'fixed':
            keep = file['channel_keep_fixed'][:].astype(bool)
        elif channels == 'session':
            keep = file['channel_keep'][:].astype(bool)
        elif channels == 'all':
            keep = np.ones(len(ids), bool)
        else:
            raise ValueError(f'unknown channel set {channels}')
        return np.flatnonzero(keep), ids
    requested = np.asarray(channels, int)
    return np.flatnonzero(np.isin(ids, requested)), ids


def load_session(session, paths=None, channels='fixed', source='export', reference='none',
                 with_lfp=True, with_artifact=False, manifest=None):
    """Load exported or PARRM-cleaned LFP and its session metadata.

    No-stimulation sessions always use the export. CAR is applied to the selected
    channels after loading. PARRM phase uses the fitted envelope peak as zero.
    """
    paths = paths or Paths()
    manifest = load_manifest(paths) if manifest is None else manifest
    info = manifest.set_index('session').loc[session]

    filename = os.path.join(paths.export_dir, f'{session}.h5')
    with h5py.File(filename, 'r') as file:
        indices, channel_ids = channel_index(file, channels)
        attrs = {name: _attr(value) for name, value in file.attrs.items()}
        time = file['t'][:]
        stim_on = file['stim/on'][:].astype(bool)
        stim_phase = file['stim/phase'][:]
        stim_envelope = file['stim/envelope'][:]
        lfp = file['lfp'][indices, :] if with_lfp else None

    used_source = 'export'
    artifact = None
    if source == 'parrm' and int(info.stim) == 1:
        filename = os.path.join(paths.parrm_dir, f'{session}.h5')
        with h5py.File(filename, 'r') as file:
            if with_lfp:
                lfp = file['lfp'][indices, :]
            if with_artifact:
                artifact = file['artifact'][indices, :]
            stim_phase = file['stim/phase_model'][:]
        used_source = 'parrm'
    elif source not in ('export', 'parrm'):
        raise ValueError(f'unknown source {source}')

    if reference == 'car' and lfp is not None:
        lfp = lfp - lfp.mean(axis=0, keepdims=True)
    elif reference != 'none':
        raise ValueError(f'unknown reference {reference}')

    trials = pd.read_csv(os.path.join(paths.trial_dir, f'{session}_trials.csv'))
    return Session(
        name=session, fs=float(attrs['lfp_fs']), t=time, lfp=lfp,
        channels=channel_ids[indices], stim_on=stim_on, stim_phase=stim_phase,
        stim_envelope=stim_envelope, trials=trials, attrs=attrs, info=info,
        source=used_source, reference=reference, artifact=artifact
    )


def trial_window(sess, pad=1.0):
    trials = sess.included
    return float(trials.t_go.min() - pad), float(trials.t_reward.max() + pad)


def normalize(sess, pad=1.0, robust=True):
    """Normalize channels using the task interval; clears cached LMP values.

    The default scale is 1.4826 times the median absolute deviation (MAD).
    """
    sess.cache.clear()
    start, end = trial_window(sess, pad)
    i0, i1 = np.searchsorted(sess.t, [start, end])
    data = sess.lfp[:, i0:i1].astype(np.float64)

    if robust:
        center = np.median(data, axis=1)
        scale = 1.4826 * np.median(np.abs(data - center[:, None]), axis=1)
    else:
        center = data.mean(axis=1)
        scale = data.std(axis=1)

    scale[scale == 0] = 1.0
    sess.lfp = ((sess.lfp - center[:, None]) / scale[:, None]).astype(np.float32)
    sess.norm = dict(mean=center, std=scale, window=(start, end), robust=robust)
    return sess


def _robust_affine(image_xy, screen_xy, rounds=2, k=3.0):
    keep = np.ones(len(image_xy), bool)
    for _ in range(rounds + 1):
        fit = np.linalg.lstsq(image_xy[keep], screen_xy[keep], rcond=None)[0]
        error = np.hypot(*(screen_xy - image_xy @ fit).T)
        median_error = np.median(error[keep])
        mad = np.median(np.abs(error[keep] - median_error))
        keep = error <= median_error + k * 1.4826 * mad

    fit = np.linalg.lstsq(image_xy[keep], screen_xy[keep], rcond=None)[0]
    residual = screen_xy[keep] - image_xy[keep] @ fit
    r2 = [1 - residual[:, axis].var() / screen_xy[keep, axis].var() for axis in range(2)]
    rmse = np.sqrt((residual ** 2).mean(axis=0))
    return fit, keep, r2, rmse


def fit_touch_calibration(session, kin, paths=None, keypoint=CAL_KEYPOINT, cell_cm=2.0, per_cell=15,
                          max_speed_px=300.0, min_span_cm=6.0, min_cells=5, single_min_span_cm=10.0,
                          single_min_r2=0.92, seed=0, method='any_contact'):
    """Fit an affine transform from DeepLabCut pixels to screen centimeters.

    Contacts are limited to per_cell samples per screen grid cell. The fit needs
    at least min_cells cells and min_span_cm on both axes; outliers are rejected
    using 3-MAD residuals. 'auto' prefers a sufficiently broad, accurate
    single-contact fit. The default uses all contacts for cross-session scale
    consistency. Returns None when calibration data are insufficient.
    """
    paths = paths or Paths()
    filename = os.path.join(paths.export_dir, f'{session}.h5')
    with h5py.File(filename, 'r') as file:
        if 'task' not in file or 'touch_t' not in file['task']:
            return None
        touch_time = file['task/touch_t'][:]
        screen_xy = file['task/touch_first_xy'][:]
        n_contacts = file['task/n_contacts'][:]

    # Exported touch times already include the 70 ms lag correction.
    touch_bins = np.floor(touch_time / BIN_S + 1e-9).astype(np.int64)
    samples = kin.reindex(touch_bins)
    image_xy = samples[[f'{keypoint}_x', f'{keypoint}_y']].to_numpy()
    progress = samples[f'{keypoint}_progress'].to_numpy()
    speed = np.nan_to_num(samples[f'{keypoint}_speed'].to_numpy(), nan=0.0)
    usable = (
        np.isfinite(screen_xy).all(axis=1)
        & np.isfinite(image_xy).all(axis=1)
        & (speed < max_speed_px)
    )
    contact_progress = progress[usable & (n_contacts >= 1)]
    contact_progress = contact_progress[np.isfinite(contact_progress)]
    if len(contact_progress) < 20:
        return None

    rng = np.random.default_rng(seed)
    fits = {}
    contact_sets = (
        ('single_contact', usable & (n_contacts == 1)),
        ('any_contact', usable & (n_contacts >= 1)),
    )
    for fit_mode, mask in contact_sets:
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue

        grid = np.floor(screen_xy[indices] / cell_cm).astype(np.int64)
        cell_ids = grid[:, 0] * 100000 + grid[:, 1]
        balanced = np.concatenate([
            rng.permutation(indices[cell_ids == cell])[:per_cell]
            for cell in np.unique(cell_ids)
        ])
        span = np.percentile(screen_xy[balanced], 95, axis=0) - np.percentile(screen_xy[balanced], 5, axis=0)
        n_cells = len(np.unique(cell_ids))
        if n_cells < min_cells or (span < min_span_cm).any():
            continue

        xy_with_offset = np.c_[image_xy[balanced], np.ones(len(balanced))]
        fit, keep, r2, rmse = _robust_affine(xy_with_offset, screen_xy[balanced])
        fits[fit_mode] = dict(
            fit=fit, keep=keep, r2=r2, rmse=rmse,
            n=len(balanced), n_cells=n_cells, span=span
        )

    if not fits:
        return None
    if method == 'auto':
        single = fits.get('single_contact')
        if single is not None and (single['span'] >= single_min_span_cm).all() and min(single['r2']) >= single_min_r2:
            method = 'single_contact'
        else:
            method = 'any_contact'
    if method not in fits:
        return None

    chosen = fits[method]
    affine = chosen['fit']
    progress_lo = float(np.percentile(contact_progress, 1) - 0.02)
    progress_hi = float(np.percentile(contact_progress, 99) + 0.02)
    return dict(
        session=session, keypoint=keypoint, A=affine[:2].T, b=affine[2], method=method,
        n_points=int(chosen['n']), n_cells=int(chosen['n_cells']),
        span_x_cm=float(chosen['span'][0]), span_y_cm=float(chosen['span'][1]),
        progress_lo=progress_lo, progress_hi=progress_hi,
        frac_kept=float(chosen['keep'].mean()),
        r2_x=float(chosen['r2'][0]), r2_y=float(chosen['r2'][1]),
        rmse_x_cm=float(chosen['rmse'][0]), rmse_y_cm=float(chosen['rmse'][1]),
        cm_per_px=float(np.sqrt(abs(np.linalg.det(affine[:2].T))))
    )


def load_kinematics(session, paths=None, calibrate=True, keypoint=CAL_KEYPOINT):
    """Read 50 ms kinematics bins, optionally adding screen-cm position and velocity.

    Screen coordinates are set to NaN outside the contact-derived progress
    range. Calibration results are stored in the DataFrame's attrs.
    """
    paths = paths or Paths()
    filename = os.path.join(paths.kin_dir, f'{session}_bins50ms.csv')
    if not os.path.exists(filename):
        return None

    kin = pd.read_csv(filename)
    kin.index = np.round(kin.t_start / BIN_S).astype(np.int64)
    kin.index.name = 'bin_id'
    if not calibrate:
        return kin

    cal = fit_touch_calibration(session, kin, paths, keypoint)
    kin.attrs['calibration'] = cal
    if cal is None:
        for suffix in ('x_cm', 'y_cm', 'vx_cm', 'vy_cm', 'speed_cm'):
            kin[f'{keypoint}_{suffix}'] = np.nan
        return kin

    in_plane = kin[f'{keypoint}_progress'].between(cal['progress_lo'], cal['progress_hi']).to_numpy()
    position_px = kin[[f'{keypoint}_x', f'{keypoint}_y']].to_numpy()
    velocity_px = kin[[f'{keypoint}_vx', f'{keypoint}_vy']].to_numpy()
    position_cm = position_px @ cal['A'].T + cal['b']
    velocity_cm = velocity_px @ cal['A'].T
    position_cm[~in_plane] = np.nan
    velocity_cm[~in_plane] = np.nan

    kin[f'{keypoint}_x_cm'] = position_cm[:, 0]
    kin[f'{keypoint}_y_cm'] = position_cm[:, 1]
    kin[f'{keypoint}_vx_cm'] = velocity_cm[:, 0]
    kin[f'{keypoint}_vy_cm'] = velocity_cm[:, 1]
    kin[f'{keypoint}_speed_cm'] = np.hypot(velocity_cm[:, 0], velocity_cm[:, 1])
    return kin


def calibrate_all(paths=None, out_csv=None, keypoint=CAL_KEYPOINT, sessions=None):
    """Save a touch-calibration summary for the selected sessions."""
    paths = paths or Paths()
    out_csv = out_csv or os.path.join(paths.export_dir, 'kinematics_calibration.csv')
    if sessions is None:
        if os.path.exists(os.path.join(paths.root, 'manifest.csv')):
            sessions = load_manifest(paths).session.tolist()
        else:
            conditions = pd.read_csv(os.path.join(paths.root, 'conditions_eACS_AM-kHz_v2.csv'))
            sessions = conditions.block[conditions.include == 1].tolist()

    rows = []
    for session in sessions:
        kin = load_kinematics(session, paths, calibrate=False)
        cal = None if kin is None else fit_touch_calibration(session, kin, paths, keypoint)
        row = dict(session=session, has_kinematics=kin is not None, calibrated=cal is not None)
        if cal:
            fields = (
                'method', 'n_points', 'n_cells', 'span_x_cm', 'span_y_cm',
                'frac_kept', 'r2_x', 'r2_y', 'rmse_x_cm', 'rmse_y_cm',
                'progress_lo', 'progress_hi', 'cm_per_px'
            )
            row.update({name: cal[name] for name in fields})
            row.update(
                a11=cal['A'][0, 0], a12=cal['A'][0, 1],
                a21=cal['A'][1, 0], a22=cal['A'][1, 1],
                b1=cal['b'][0], b2=cal['b'][1]
            )
        rows.append(row)
        print(session, 'calibrated' if cal else 'not calibrated')

    summary = pd.DataFrame(rows)
    summary.to_csv(out_csv, index=False)
    return summary


def load_templates(session, paths=None):
    """Read one session's PARRM templates and attributes."""
    paths = paths or Paths()
    filename = os.path.join(paths.parrm_dir, 'template_library.h5')
    with h5py.File(filename, 'r') as file:
        group = file[session]
        templates = {name: group[name][:] for name in group.keys()}
        templates.update({name: _attr(value) for name, value in group.attrs.items()})
    return templates


def lmp(sess, win_s=LMP_WIN_S):
    """Causal LFP moving average; the first win_s seconds are NaN."""
    key = ('lmp', win_s)
    if key not in sess.cache:
        n_samples = int(round(win_s * sess.fs))
        cumulative = np.cumsum(np.pad(sess.lfp.astype(np.float64), ((0, 0), (1, 0))), axis=1)
        result = np.full(sess.lfp.shape, np.nan, np.float32)
        result[:, n_samples - 1:] = (cumulative[:, n_samples:] - cumulative[:, :-n_samples]) / n_samples
        sess.cache[key] = result
    return sess.cache[key]


def bin_lmp(sess, ids, win_s=LMP_WIN_S, bin_s=BIN_S):
    """Take the LMP sample immediately before each bin end."""
    values = lmp(sess, win_s)
    indices = np.searchsorted(sess.t, (ids + 1) * bin_s - 0.5 / sess.fs) - 1
    result = np.full((values.shape[0], len(ids)), np.nan, np.float32)
    valid = (indices >= 0) & (indices < values.shape[1])
    result[:, valid] = values[:, indices[valid]]
    return result


def bin_ids(t0, t1, bin_s=BIN_S):
    """Global IDs for bins spanning the half-open interval [t0, t1)."""
    return np.arange(int(np.floor(t0 / bin_s + 1e-9)), int(np.ceil(t1 / bin_s - 1e-9)))


def cut_bins(sess, ids, bin_s=BIN_S):
    """Return LFP as (channels, bins, samples per bin)."""
    samples_per_bin = int(round(bin_s * sess.fs))
    starts = np.searchsorted(sess.t, ids * bin_s - 0.5 / sess.fs)
    result = np.full((sess.lfp.shape[0], len(ids), samples_per_bin), np.nan, np.float32)
    valid = (starts >= 0) & (starts + samples_per_bin <= sess.lfp.shape[1])
    for bin_index in np.flatnonzero(valid):
        start = starts[bin_index]
        result[:, bin_index, :] = sess.lfp[:, start:start + samples_per_bin]
    return result


def trial_examples(sess, kin=None, align='t_go', pre=0.5, post=1.5, until=None, trials=None,
                   kin_cols=('Index_vx_cm', 'Index_vy_cm'), with_lmp=False, lmp_win_s=LMP_WIN_S,
                   bin_s=BIN_S):
    """Cut trials on the global 50 ms grid, optionally including LMP and kinematics.

    If until is supplied, trials have variable length; otherwise each has
    round((pre + post) / bin_s) bins.
    """
    selected = sess.included if trials is None else sess.trials[sess.trials.trial.isin(trials)]
    examples = []
    for _, trial in selected.iterrows():
        start = trial[align] - pre
        if until:
            ids = bin_ids(start, trial[until] + post, bin_s)
        else:
            first_bin = int(np.floor(start / bin_s + 1e-9))
            ids = np.arange(first_bin, first_bin + int(round((pre + post) / bin_s)))

        velocity = None
        valid_kin = None
        if kin is not None:
            velocity = kin.reindex(ids)[list(kin_cols)].to_numpy(np.float32)
            valid_kin = np.isfinite(velocity).all(axis=1)

        examples.append(dict(
            session=sess.name, trial=int(trial.trial), bin_id=ids, t_bin=ids * bin_s,
            lfp=cut_bins(sess, ids, bin_s),
            lmp=bin_lmp(sess, ids, lmp_win_s, bin_s) if with_lmp else None,
            kin=velocity, kin_valid=valid_kin, row=trial
        ))
    return examples


def split_trials(splits, sess, part):
    """Select included train/val trials or calibration pool/query trials."""
    included = sess.included.trial.astype(int).tolist()
    if part == 'all':
        return included
    if part in ('train', 'val'):
        held_out = set(splits['train_val_trials'][sess.name])
        return [trial for trial in included if (trial in held_out) == (part == 'val')]
    if part in ('pool', 'query'):
        return list(splits['calibration'][sess.name][part])
    raise ValueError(f'unknown part {part}')


def _welch(data, fs, win_s=1.0):
    n_samples = int(round(win_s * fs))
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_samples) / (n_samples - 1)))
    starts = np.arange(0, data.shape[1] - n_samples + 1, n_samples // 2)
    if not len(starts):
        return None, 0

    power = np.zeros((data.shape[0], n_samples // 2 + 1))
    for start in starts:
        spectrum = np.fft.rfft(data[:, start:start + n_samples] * window, axis=1)
        power += np.abs(spectrum) ** 2
    power[:, 1:-1] *= 2
    power /= len(starts) * fs * np.sum(window ** 2)
    return power, len(starts)


def masked_psd(sess, mask, win_s=1.0):
    """Welch PSD using complete windows within each contiguous True region."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    n_samples = int(round(win_s * sess.fs))
    total_power = None
    n_windows = 0

    for start, end in zip(starts, ends):
        if end - start < n_samples:
            continue
        power, count = _welch(sess.lfp[:, start:end], sess.fs, win_s)
        total_power = power * count if total_power is None else total_power + power * count
        n_windows += count

    frequencies = np.fft.rfftfreq(n_samples, 1 / sess.fs)
    return frequencies, (total_power / n_windows if n_windows else None), n_windows


def trial_mask(sess, pad=0.5):
    mask = np.zeros(len(sess.t), bool)
    for _, trial in sess.included.iterrows():
        i0, i1 = np.searchsorted(sess.t, [trial.t_go - pad, trial.t_reward + pad])
        mask[i0:i1] = True
    return mask


def session_spectra(sess, pad=0.5, win_s=1.0):
    """Compare trial spectra with stimulation-off intervals between trials."""
    in_trials = trial_mask(sess, pad)
    start, end = trial_window(sess, 0.0)
    in_session = (sess.t >= start) & (sess.t <= end)
    frequencies, trial_psd, n_trials = masked_psd(sess, in_trials, win_s)
    _, rest_psd, n_rest = masked_psd(sess, in_session & ~in_trials & ~sess.stim_on, win_s)
    return dict(
        f=frequencies, trial=trial_psd, rest=rest_psd,
        n_trial_windows=n_trials, n_rest_windows=n_rest
    )


def plot_on_off_spectra(day_sessions, paths=None, channels='fixed', source='export', fmax=150,
                        out_dir=None, cache=None, reference='none'):
    """Plot each session's trial PSD against other sessions from the same day."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    paths = paths or Paths()
    out_dir = out_dir or paths.spectra_dir
    os.makedirs(out_dir, exist_ok=True)
    manifest = load_manifest(paths)
    cache = {} if cache is None else cache

    for session in day_sessions:
        if session not in cache:
            data = load_session(session, paths, channels, source, reference=reference, manifest=manifest)
            cache[session] = dict(spec=session_spectra(data), info=data.info)

    colors = {0: 'k', 3: 'tab:red', 20: 'tab:blue'}
    names = {0: 'OFF', 3: 'ON 3 Hz', 20: 'ON 20 Hz'}
    saved = []

    for session in day_sessions:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        handles = []
        limits = []
        order = [session] + [other for other in day_sessions if other != session]

        for other in order:
            spectra = cache[other]['spec']
            info = cache[other]['info']
            if spectra['trial'] is None:
                continue

            frequencies = spectra['f']
            selected = (frequencies >= 0.99) & (frequencies <= fmax)
            frequency = frequencies[selected]
            log_power = np.log10(spectra['trial'][:, selected])
            median_log = np.median(log_power, axis=0)
            keep = np.ones(len(frequency), bool)
            for notch_hz in (60, 120, 180):
                keep &= np.abs(frequency - notch_hz) >= 4
            limits.append(median_log[keep])

            color = colors[int(info.am_freq_hz)]
            is_current = other == session
            width = 2.0 if is_current else 1.0
            line_style = '-' if is_current else '--'
            ax.plot(frequency, 10 ** median_log, color=color, lw=width, ls=line_style)
            if is_current:
                low, high = np.percentile(log_power, [25, 75], axis=0)
                ax.fill_between(frequency, 10 ** low, 10 ** high, color=color, alpha=0.25, lw=0)

            label = names[int(info.am_freq_hz)]
            label += ' trials, this session' if is_current else f' trials, {other[-6:]}'
            handles.append(Line2D([], [], color=color, lw=width, ls=line_style, label=label))

        spectra = cache[session]['spec']
        if spectra['rest'] is not None:
            selected = (spectra['f'] >= 0.99) & (spectra['f'] <= fmax)
            frequency = spectra['f'][selected]
            rest_log = np.median(np.log10(spectra['rest'][:, selected]), axis=0)
            keep = np.ones(len(frequency), bool)
            for notch_hz in (60, 120, 180):
                keep &= np.abs(frequency - notch_hz) >= 4
            limits.append(rest_log[keep])
            ax.plot(frequency, 10 ** rest_log, color='0.6', lw=1.2)
            handles.append(Line2D(
                [], [], color='0.6', lw=1.2,
                label='between trials, stim off, this session'
            ))

        if limits and any(len(values) for values in limits):
            all_values = np.concatenate(limits)
            ax.set_ylim(10 ** (all_values.min() - 0.3), 10 ** (all_values.max() + 0.5))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, fmax)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (µV²/Hz)' if source == 'export' else 'PSD')
        ax.legend(handles=handles, fontsize=7, frameon=False, loc='lower left')

        info = cache[session]['info']
        ax.text(0.99, 0.98, f'{session}  {info.condition}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8)
        if int(info.stim) == 1 and not any(int(cache[s]['info'].stim) == 0 for s in day_sessions):
            ax.text(0.99, 0.92, 'no same-day no-stim session',
                    transform=ax.transAxes, ha='right', va='top', fontsize=7)
        fig.tight_layout()
        filename = os.path.join(out_dir, f'{session}_onoff.png')
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        saved.append(filename)

    return saved


def plot_all_on_off_spectra(paths=None, channels='fixed', source='export', sessions=None, reference='none',
                            overwrite=False):
    """Plot daily sessions, skipping days whose figures already exist."""
    paths = paths or Paths()
    manifest = load_manifest(paths)
    if sessions is not None:
        manifest = manifest[manifest.date.isin(manifest.date[manifest.session.isin(sessions)])]

    saved = []
    for date, group in manifest.groupby('date'):
        missing = [
            session for session in group.session
            if overwrite or not os.path.exists(os.path.join(paths.spectra_dir, f'{session}_onoff.png'))
        ]
        if not missing:
            continue
        try:
            files = plot_on_off_spectra(list(group.session), paths, channels, source, reference=reference)
        except Exception as error:
            print(date, 'ERROR', error)
            continue
        saved.extend(files)
        print(date, len(files), 'figures')
    return saved
