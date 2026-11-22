"""Animate predicted and measured velocity vectors on a held-out validation trial."""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import notebook_api

SESSION = 'Butters-250523-095602'
TRIAL = 43
BIN_S = 0.05
PRE = 0.5
COLORS = {'truth': '#2f5d8a', 'pred': '#d9772a', 'error': '#c0392b', 'grey': '#5c6670', 'gold': '#c99a2e'}


def notebook_namespace():
    """Load the notebook's shared data and model cells."""
    return notebook_api.load()


def load_data():
    ns = notebook_namespace()
    dataset = ns['load_split_dataset'](SESSION, 'query', 'export_none', 9)
    check = np.load('results/checkpoints/ridge_raw.npz')
    W, intercept = check['coefficients'], check['intercept']
    X = dataset['features'].astype(np.float64)
    Y = dataset['targets'].astype(np.float64)
    valid = dataset['valid_bins']
    prediction = X @ W + intercept
    trial_index = int(np.flatnonzero(dataset['trials'] == TRIAL)[0])
    session_r2 = ns['mean_r2'](Y, prediction, valid)
    trial_r2 = ns['mean_r2'](Y[trial_index], prediction[trial_index], valid[trial_index])
    return Y[trial_index], prediction[trial_index], valid[trial_index], session_r2, trial_r2


def animate():
    truth, predicted, valid, session_r2, trial_r2 = load_data()
    frames = np.flatnonzero(valid)
    full_limit = max(1.12 * float(np.max(np.abs(truth[frames]))), 20.0)
    zoom_limit = max(1.35 * float(np.max(np.abs(predicted[frames]))), 5.0)

    fig, (full_ax, zoom_ax) = plt.subplots(1, 2, figsize=(12, 6.4), facecolor='white',
                                           gridspec_kw={'width_ratios': [1.25, 1]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.16, wspace=0.20)

    fig.suptitle('Validation: measured velocity y and predicted velocity yhat',
                 fontsize=16, weight='bold', y=0.96)
    fig.text(0.5, 0.905,
             f'{SESSION}, trial {TRIAL}   |   session R2 = {session_r2:.3f}   |   '
             f'trial R2 = {trial_r2:.3f}',
             ha='center', fontsize=10.5, color=COLORS['grey'])

    for ax, limit, title in [(full_ax, full_limit, 'full screen plane'),
                             (zoom_ax, zoom_limit, f'zoom near origin (+/-{zoom_limit:.0f} cm/s)')]:
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.axhline(0, color='k', linewidth=0.6)
        ax.axvline(0, color='k', linewidth=0.6)
        ax.set_xlabel('vx: right (cm/s)')
        ax.set_ylabel('vy: up (cm/s)')
        ax.set_title(title, fontsize=10.5, weight='bold')
        ax.grid(alpha=0.18)
    full_ax.add_patch(Circle((0, 0), zoom_limit, fill=False, linestyle='--',
                             linewidth=0.9, color=COLORS['gold']))

    truth_arrow = full_ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1,
                                 color=COLORS['truth'], width=0.010, zorder=4)
    pred_arrow = full_ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1,
                                color=COLORS['pred'], width=0.010, zorder=5)
    truth_zoom = zoom_ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1,
                                color=COLORS['truth'], width=0.016, zorder=4)
    pred_zoom = zoom_ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1,
                               color=COLORS['pred'], width=0.016, zorder=5)
    error_line, = full_ax.plot([], [], color=COLORS['error'], linewidth=2.2, zorder=6)
    error_zoom, = zoom_ax.plot([], [], color=COLORS['error'], linewidth=2.2, zorder=6)
    truth_trail, = full_ax.plot([], [], color=COLORS['truth'], alpha=0.30, linewidth=1.2)
    pred_trail, = full_ax.plot([], [], color=COLORS['pred'], alpha=0.45, linewidth=1.2)
    truth_dot, = full_ax.plot([], [], 'o', color=COLORS['truth'], markersize=4)
    pred_dot, = full_ax.plot([], [], 'o', color=COLORS['pred'], markersize=4)

    readout = fig.text(0.5, 0.055, '', ha='center', fontsize=10, color='#1c1c1c')
    fig.text(0.5, 0.018,
             'Same units and same scale. The ridge prediction stays close to the mean velocity.',
             ha='center', fontsize=8.5, color=COLORS['grey'])
    full_ax.text(0.02, 0.98, 'blue: measured y\norange: predicted yhat\nred: tip-to-tip error',
                 transform=full_ax.transAxes, va='top', fontsize=8, color=COLORS['grey'])

    def update(frame_number):
        index = int(frames[frame_number])
        y = truth[index]
        yhat = predicted[index]
        truth_arrow.set_offsets([0, 0])
        truth_arrow.set_UVC(y[0], y[1])
        pred_arrow.set_offsets([0, 0])
        pred_arrow.set_UVC(yhat[0], yhat[1])
        truth_zoom.set_offsets([0, 0])
        truth_zoom.set_UVC(y[0], y[1])
        pred_zoom.set_offsets([0, 0])
        pred_zoom.set_UVC(yhat[0], yhat[1])
        seen = frames[:frame_number + 1]
        truth_trail.set_data(truth[seen, 0], truth[seen, 1])
        pred_trail.set_data(predicted[seen, 0], predicted[seen, 1])
        truth_dot.set_data([y[0]], [y[1]])
        pred_dot.set_data([yhat[0]], [yhat[1]])
        error_line.set_data([y[0], yhat[0]], [y[1], yhat[1]])
        error_zoom.set_data([y[0], yhat[0]], [y[1], yhat[1]])
        time = index * BIN_S - PRE
        readout.set_text(
            f'bin {index:02d}, t = {time:+.2f} s   |   '
            f'y = ({y[0]:+.1f}, {y[1]:+.1f}) cm/s   |   '
            f'yhat = ({yhat[0]:+.1f}, {yhat[1]:+.1f}) cm/s   |   '
            f'|y| = {np.linalg.norm(y):.1f}, |yhat| = {np.linalg.norm(yhat):.1f}, '
            f'error = {np.linalg.norm(y - yhat):.1f}'
        )
        return (truth_arrow, pred_arrow, truth_zoom, pred_zoom, truth_trail, pred_trail,
                truth_dot, pred_dot, error_line, error_zoom, readout)

    animation = FuncAnimation(fig, update, frames=len(frames), interval=120, blit=False)
    update(20)
    fig.savefig('reports/fig1_linear_regression_validation.png', dpi=180, facecolor='white')
    animation.save('reports/fig1_linear_regression_validation.gif', writer=PillowWriter(fps=8),
                   dpi=110)
    plt.close(fig)
    print('wrote reports/fig1_linear_regression_validation.gif')
    print('wrote reports/fig1_linear_regression_validation.png')


if __name__ == '__main__':
    os.makedirs('reports', exist_ok=True)
    animate()
