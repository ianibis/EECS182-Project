"""Draw the decoder input, model, and output using one real trial."""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import EECS182_NHP_data_loader as L
import EECS182_NHP_features as F

SESSION = 'Butters-250522-095740'
TRIAL = 5
BIN_S = L.BIN_S
PRE, POST = 0.5, 1.5

P = L.Paths(
    root=os.path.join(ROOT, 'data'),
    export_dir=os.path.join(ROOT, 'data', 'exports', 'export_h5_LFPs'),
    trial_dir=os.path.join(ROOT, 'data', 'exports', 'export_trials'),
    parrm_dir=os.path.join(ROOT, 'data', 'parrm', 'parrm_h5_LFPs'),
    kin_dir=os.path.join(ROOT, 'data', 'kinematics_bins50ms', 'bins50ms'),
)
MANIFEST = L.load_manifest(P, os.path.join(ROOT, 'EECS182_data_manifest.csv'))
SESSION_DATA = L.load_session(SESSION, P, with_lfp=True, manifest=MANIFEST)
KIN = L.load_kinematics(SESSION, P)
EXAMPLE = L.trial_examples(SESSION_DATA, KIN, trials=[TRIAL], pre=PRE, post=POST)[0]
RAW = EXAMPLE['lfp']  # (channels, bins, samples per bin)
Y = EXAMPLE['kin'].astype(float)  # (bins, 2)
BIN_IDS = EXAMPLE['bin_id']

FEATURES = F.load_features(SESSION, P, 'export_none', normalize=False)
TRIAL_ROW = SESSION_DATA.trials[SESSION_DATA.trials.trial == TRIAL]
X0, _, VALID0, _ = F.trial_arrays(FEATURES, TRIAL_ROW, KIN, n_lags=0)
X0 = X0[0].reshape(len(BIN_IDS), len(FEATURES['channel_ids']), len(F.FEATURE_NAMES))
SPEED = np.linalg.norm(Y, axis=1)
CANDIDATES = np.flatnonzero(VALID0[0])
BIN_INDEX = int(CANDIDATES[np.argmax(SPEED[CANDIDATES])])

TIME = np.arange(len(BIN_IDS)) * BIN_S - PRE
SAMPLE_TIME = np.linspace(-PRE, POST, RAW.shape[1] * RAW.shape[2], endpoint=False)
KEEP_IDS = set(int(v) for v in FEATURES['channel_ids'])
COLORS = {'blue': '#2f5d8a', 'orange': '#d9772a', 'green': '#2f7d5c', 'grey': '#5c6670'}


def add_box(ax, x, y, w, h, title, body, color='#eef3f7'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.02',
                                linewidth=1.1, edgecolor=COLORS['blue'], facecolor=color))
    ax.text(x + 0.02, y + h - 0.035, title, fontsize=10, weight='bold', va='top',
            color=COLORS['blue'])
    ax.text(x + 0.02, y + h - 0.095, body, fontsize=8.2, va='top', linespacing=1.35,
            color='#1c1c1c')


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14,
                                 linewidth=1.4, color=COLORS['grey']))


def electrode_grid(ax, active_ids):
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(Ellipse((3.5, 3.5), 8.6, 8.6, fill=False, linewidth=1.2,
                         edgecolor=COLORS['grey']))
    ax.add_patch(Rectangle((0.45, 0.45), 6.1, 6.1, fill=False, linewidth=0.9,
                           edgecolor=COLORS['grey'], linestyle=':'))
    for channel in range(1, 65):
        row, col = divmod(channel - 1, 8)
        on = channel in active_ids
        ax.add_patch(Circle((0.8 + col * 0.75, 6.2 - row * 0.75), 0.19,
                            facecolor=COLORS['orange'] if on else '#e8e8e8',
                            edgecolor=COLORS['grey'] if on else '#c9c9c9', linewidth=0.7))
    ax.set_xlim(0, 7.1)
    ax.set_ylim(0, 7.1)
    ax.set_title('64-pad epidural array\n35 fixed channels shown in orange', fontsize=9)


def pipeline():
    fig = plt.figure(figsize=(15, 4.9), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.text(0.5, 0.93, 'One trial through the decoder', ha='center', fontsize=17, weight='bold')
    ax.text(0.5, 0.875, 'A trial is 2.0 s around the go cue: 40 bins of 50 ms. '
                        'The model predicts one 2D velocity per bin.',
            ha='center', fontsize=9.5, color=COLORS['grey'])

    boxes = [
        (0.01, '1. Record',
         '64-pad epidural array\n'
         '35 fixed channels\n'
         '1 kHz sampling\n'
         '~500,000 samples per session\n'
         'channels x time, not a 3D cube', '#f4f7f9'),
        (0.205, '2. Cut one trial',
         '-0.5 s to +1.5 s\n'
         'around the go cue\n'
         '40 bins x 50 ms\n'
         'raw LFP: 35 x 2,000', '#f4f7f9'),
        (0.40, '3. Features per bin',
         'per channel, 8 numbers:\n'
         'LMP + 7 log band powers\n'
         '35 x 8 = 280 per bin\n'
         '10-bin history = 2,800', '#f4f7f9'),
        (0.595, '4. Decoder box',
         'Ridge: 2,800 -> 2\n'
         'KNN: 2,800 -> 2\n'
         'MLP: 2,800 -> 32 -> 16 -> 2\n'
         'GRU: 40 x 280 -> 128 -> 2\n'
         'EEGNet: 35 x 2,000 -> 2', '#fbeee5'),
        (0.79, '5. Output per trial',
         '40 bins x 2 numbers\n'
         'vx: sideways (cm/s)\n'
         'vy: up-down (cm/s)\n'
         'target shape: (40, 2)', '#edf5ef'),
    ]
    width = 0.17
    for x, title, body, color in boxes:
        add_box(ax, x, 0.24, width, 0.40, title, body, color=color)

    for left, right in zip(boxes[:-1], boxes[1:]):
        x0 = left[0] + width + 0.004
        x1 = right[0] - 0.004
        arrow(ax, (x0, 0.44), (x1, 0.44))

    ax.text(0.5, 0.075,
            'Input shapes: Ridge / KNN / MLP (40, 2800).  GRU (40, 280).  '
            'EEGNet (35, 2000).  Target (40, 2).',
            ha='center', fontsize=9, color=COLORS['grey'])

    fig.savefig('reports/model_pipeline.png', dpi=200, facecolor='white')
    plt.close(fig)


def FEATURES_BLOCK():
    block = FEATURES_BIN()
    mean = np.nanmean(block, axis=0)
    scale = np.nanstd(block, axis=0)
    scale = np.where(scale > 0, scale, 1)
    return (block - mean) / scale


def FEATURES_BIN():
    return X0[BIN_INDEX]


def input_figure():
    fig = plt.figure(figsize=(13, 9.4), facecolor='white')
    grid = fig.add_gridspec(3, 3, height_ratios=[1.1, 1, 1], hspace=0.62, wspace=0.30,
                            top=0.88, bottom=0.06, left=0.06, right=0.97)

    ax = fig.add_subplot(grid[0, 0])
    electrode_grid(ax, KEEP_IDS)
    ax.set_title('A. Where the voltages come from\n(8 x 8 schematic, 35 active pads)', fontsize=9)

    ax = fig.add_subplot(grid[0, 1:])
    raw = RAW.reshape(len(FEATURES['channel_ids']), -1)
    clip = np.nanpercentile(np.abs(raw), 98)
    image = ax.imshow(raw, aspect='auto', cmap='RdBu_r', vmin=-clip, vmax=clip,
                      extent=[-PRE, POST, len(FEATURES['channel_ids']), 0])
    ax.axvline(0, color='k', linewidth=0.9)
    ax.axvspan(TIME[BIN_INDEX], TIME[BIN_INDEX] + BIN_S, color='#f2c14e', alpha=0.35)
    ax.set_xlabel('time from go cue (s)')
    ax.set_ylabel('electrode')
    ax.set_title('B. One trial of raw voltage: 35 channels x 2,000 samples\n(stack trials -> a 3D tensor)', fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.85, label='microvolts')

    ax = fig.add_subplot(grid[1, 0])
    for offset, channel in enumerate(np.linspace(0, len(FEATURES['channel_ids']) - 1, 4).astype(int)):
        trace = RAW[channel].reshape(-1)
        trace = trace - np.nanmedian(trace)
        trace = trace / (np.nanstd(trace) + 1e-9)
        ax.plot(SAMPLE_TIME, trace + offset * 2.6, linewidth=0.65, color=COLORS['blue'])
    ax.axvline(0, color='k', linewidth=0.8)
    ax.axvline(TIME[BIN_INDEX], color='#f2c14e', linewidth=2)
    ax.set_xlabel('time from go cue (s)')
    ax.set_yticks([])
    ax.set_title('C. Four example voltage traces\n(offset for visibility)', fontsize=9)

    ax = fig.add_subplot(grid[1, 1])
    block = FEATURES_BLOCK()
    image = ax.imshow(block, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_xticks(range(8))
    ax.set_xticklabels(['LMP', '1-4', '4-8', '8-13', '13-30', '30-50', '70-110', '130-200'],
                       rotation=45, ha='right', fontsize=7)
    ax.set_yticks([])
    ax.set_ylabel('35 channels')
    ax.set_title('D. Features for one 50 ms bin\n(column-normalized)', fontsize=9)

    ax = fig.add_subplot(grid[1, 2])
    image = ax.imshow(block.reshape(1, -1), aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_yticks([])
    ax.set_xlabel('280 numbers')
    ax.set_title('E. The same 280 numbers as a vector\n(column-normalized)', fontsize=9)

    ax = fig.add_subplot(grid[2, 0:2])
    ax.plot(TIME, Y[:, 0], color=COLORS['blue'], linewidth=1.4, label='vx, sideways')
    ax.plot(TIME, Y[:, 1], color=COLORS['orange'], linewidth=1.4, label='vy, up-down')
    ax.axhline(0, color='k', linewidth=0.6)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.axvline(TIME[BIN_INDEX], color='#f2c14e', linewidth=2)
    ax.set_xlabel('time from go cue (s)')
    ax.set_ylabel('velocity (cm/s)')
    ax.set_title('F. Target: one velocity pair per 50 ms bin, shape (40, 2)', fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(grid[2, 2])
    position = np.cumsum(Y * BIN_S, axis=0)
    position -= position[0]
    ax.plot(position[:, 0], position[:, 1], color=COLORS['green'], linewidth=1.2)
    for index in range(0, len(position) - 1, 5):
        ax.annotate('', xy=position[index + 1], xytext=position[index],
                    arrowprops=dict(arrowstyle='-|>', color=COLORS['green'], lw=0.8))
    ax.set_aspect('equal')
    ax.set_xlabel('relative x (cm)')
    ax.set_ylabel('relative y (cm)')
    ax.set_title('G. Velocity integrated into a path', fontsize=9)

    fig.suptitle(f'Inputs and outputs for one trial, session {SESSION}', fontsize=13, weight='bold', y=0.965)
    fig.savefig('reports/input_visualisation.png', dpi=200, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs('reports', exist_ok=True)
    pipeline()
    input_figure()
    print('wrote reports/model_pipeline.png and reports/input_visualisation.png')
