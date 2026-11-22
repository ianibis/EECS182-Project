"""Draw one architecture infographic per baseline. Start with linear regression."""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import notebook_api

SESSION = 'Butters-250522-095740'
TRIAL = 5
BIN_S = 0.05
PRE = 0.5
COLORS = {'blue': '#2f5d8a', 'orange': '#d9772a', 'green': '#2f7d5c', 'grey': '#5c6670'}


def notebook_namespace():
    """Load the notebook's data and model cells."""
    return notebook_api.load()


def load_example():
    ns = notebook_namespace()
    dataset = ns['load_split_dataset'](SESSION, 'train', 'export_none', 9)
    index = int(np.flatnonzero(dataset['trials'] == TRIAL)[0])
    X = dataset['features'][index].astype(np.float64)
    Y = dataset['targets'][index].astype(np.float64)
    valid = dataset['valid_bins'][index]
    speed = np.linalg.norm(Y, axis=1)
    candidates = np.flatnonzero(valid)
    bin_index = int(candidates[np.argmax(speed[candidates])])
    checkpoint = np.load('results/checkpoints/ridge_raw.npz')
    return X, Y, valid, bin_index, checkpoint['coefficients'], checkpoint['intercept'], float(checkpoint['alpha'])


def box(ax, x, y, w, h, facecolor='#f4f7f9'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.02',
                                linewidth=1.0, edgecolor=COLORS['blue'], facecolor=facecolor))


def ridge():
    X, Y, valid, bin_index, W, b, alpha = load_example()
    x = X[bin_index]
    yhat = x @ W + b
    Yhat = X @ W + b
    time = np.arange(len(X)) * BIN_S - PRE

    fig = plt.figure(figsize=(16, 10), facecolor='white')

    fig.text(0.5, 0.965, 'Baseline 1: linear regression (ridge)', ha='center',
             fontsize=20, weight='bold')
    fig.text(0.5, 0.925, 'One 50 ms bin: a weighted sum of 2,800 input numbers, then two velocity outputs.',
             ha='center', fontsize=10.5, color=COLORS['grey'])

    fig.text(0.5, 0.865, r'$\hat{y} = xW + b$', ha='center', fontsize=28, color=COLORS['blue'])
    fig.text(0.5, 0.825, r'$(1 \times 2800)\;(2800 \times 2) + (1 \times 2) = (1 \times 2)$',
             ha='center', fontsize=11, color=COLORS['grey'])
    fig.text(0.5, 0.792, r'row-vector form of  $y = W^T x + b$', ha='center',
             fontsize=9.5, color=COLORS['grey'])

    # Input vector.
    ax = fig.add_axes([0.045, 0.50, 0.23, 0.25])
    block = x.reshape(10, 280)
    scale = np.nanpercentile(np.abs(block), 98)
    image = ax.imshow(block, aspect='auto', cmap='RdBu_r',
                      norm=TwoSlopeNorm(vcenter=0, vmin=-scale, vmax=scale))
    ax.set_xlabel('280 features per bin (35 channels x 8)')
    ax.set_yticks([0, 9])
    ax.set_yticklabels(['oldest', 'current'])
    ax.set_xticks([0, 279])
    ax.set_title('input x: 2,800 numbers\n(10 bins x 280)', fontsize=11, weight='bold')
    fig.colorbar(image, ax=ax, shrink=0.8, pad=0.02)

    # Weight matrix.
    ax = fig.add_axes([0.345, 0.535, 0.35, 0.19])
    limit = max(float(np.percentile(np.abs(W), 99)), 0.01)
    image = ax.imshow(W.T, aspect='auto', cmap='RdBu_r', interpolation='nearest',
                      norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit))
    for edge in range(280, 2800, 280):
        ax.axvline(edge - 0.5, color='k', linewidth=0.35, alpha=0.25)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['vx weights', 'vy weights'])
    ax.set_xticks([0, 2799])
    ax.set_xticklabels(['input 0', 'input 2799'], fontsize=7)
    ax.set_xlabel('2,800 input weights, 10 lag blocks of 280')
    ax.set_title('W: 2,800 x 2 = 5,600 weights', fontsize=11, weight='bold')
    fig.colorbar(image, ax=ax, shrink=0.8, pad=0.02)

    # Output for one bin as a screen-plane arrow.
    ax = fig.add_axes([0.765, 0.505, 0.18, 0.25])
    limit = max(1.35 * float(np.max(np.abs(yhat))), 2.0)
    ax.quiver(0, 0, yhat[0], yhat[1], angles='xy', scale_units='xy', scale=1,
              color=COLORS['orange'], width=0.022, zorder=3)
    ax.plot([0, yhat[0]], [0, 0], color=COLORS['grey'], linewidth=0.7, linestyle=':')
    ax.plot([0, 0], [0, yhat[1]], color=COLORS['grey'], linewidth=0.7, linestyle=':')
    ax.axhline(0, color='k', linewidth=0.6)
    ax.axvline(0, color='k', linewidth=0.6)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.set_xlabel('vx (cm/s)')
    ax.set_ylabel('vy (cm/s)')
    ax.set_title('output vector yhat', fontsize=11, weight='bold')
    ax.annotate(f'({yhat[0]:.1f}, {yhat[1]:.1f})', xy=(yhat[0], yhat[1]),
                xytext=(6, 6), textcoords='offset points', fontsize=8, color=COLORS['orange'])

    # Whole trial.
    ax = fig.add_axes([0.045, 0.12, 0.46, 0.29])
    ax.plot(time, Yhat[:, 0], color=COLORS['blue'], linewidth=1.5, label='predicted vx')
    ax.plot(time, Yhat[:, 1], color=COLORS['orange'], linewidth=1.5, label='predicted vy')
    ax.axhline(0, color='k', linewidth=0.6)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.axvline(time[bin_index], color='#f2c14e', linewidth=3)
    ax.set_xlabel('time from go cue (s)')
    ax.set_ylabel('velocity (cm/s)')
    ax.set_title('the same W is applied to all 40 bins: X (40 x 2800) -> Yhat (40 x 2)',
                 fontsize=11, weight='bold')
    ax.legend(frameon=False, fontsize=8)

    # Training box.
    ax = fig.add_axes([0.54, 0.08, 0.42, 0.34])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    box(ax, 0.01, 0.02, 0.98, 0.96, '#fbeee5')
    ax.text(0.05, 0.90, 'How W and b are fit', fontsize=12, weight='bold', color=COLORS['blue'])
    ax.text(0.05, 0.82,
            'Training data: 14 stimulation-off sessions\n'
            'X: 17,568 bins x 2,800 inputs\n\n'
            'Ridge loss:\n'
            r'$\|Y - XW - b\|^2 + \alpha\|W\|^2$' '\n\n'
            'Closed form:\n'
            r'$(X^T X + \alpha I)W = X^T Y$' '\n\n'
            f'alpha = {alpha:,.0f}\n\n'
            'No hidden layer. No nonlinearity.\n'
            'Each output is one weighted sum.',
            fontsize=9.0, va='top', linespacing=1.35, color='#1c1c1c')

    fig.text(0.5, 0.035,
             'The 2,800 inputs are 10 bins of history, each with 35 channels x 8 features. '
             'The target y is measured fingertip velocity.',
             ha='center', fontsize=9, color=COLORS['grey'])

    fig.savefig('reports/fig1_linear_regression.png', dpi=200, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs('reports', exist_ok=True)
    ridge()
    print('wrote reports/fig1_linear_regression.png')
