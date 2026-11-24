"""Shared figure style for Task 5, so every figure uses the same colors and chrome."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

# Fixed series colors: a method keeps its color in every figure.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'


def apply_style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'font.family': 'sans-serif', 'font.size': 10, 'text.color': INK,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK_SECONDARY, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'axes.axisbelow': True,
        'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK_SECONDARY,
        'ytick.labelcolor': INK_SECONDARY, 'lines.linewidth': 2.0, 'legend.frameon': False,
        'legend.labelcolor': INK_SECONDARY, 'savefig.dpi': 180, 'savefig.bbox': 'tight',
    })
