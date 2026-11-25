"""Plot the GRU width/depth sweep: hidden units against validation R2.

Reads task4/results/gru_sweep.csv, the twelve settings behind Table 1 of the report
(hidden 64/128/256 x layers 1/2 x dropout 0/0.2, one seed, three validation days), and
writes reports/figures/fig_gru_sweep.png. One line per (layers, dropout) pair so the
width effect is visible without collapsing the other two knobs.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, 'task4', 'results', 'gru_sweep.csv')
OUT = os.path.join(ROOT, 'reports', 'figures', 'fig_gru_sweep.png')

SERIES = [
    # (layers, dropout, colour, linestyle, marker)
    (1, 0.0, '#2f6fdb', '-', 'o'),
    (1, 0.2, '#2f6fdb', '--', 's'),
    (2, 0.0, '#c62828', '-', '^'),
    (2, 0.2, '#c62828', '--', 'v'),
]

plt.rcParams.update({
    'font.size': 9.5,
    'axes.titlesize': 10.5,
    'axes.labelsize': 9.5,
    'legend.fontsize': 8.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})


def main():
    sweep = pd.read_csv(SOURCE)
    widths = sorted(sweep.hidden.unique())

    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor='white')

    for layers, dropout, colour, style, marker in SERIES:
        part = (sweep[(sweep.layers == layers) & (sweep.dropout == dropout)]
                .sort_values('hidden'))
        ax.plot(part.hidden, part.val_r2, style, marker=marker, color=colour,
                markersize=7, linewidth=1.8, markerfacecolor='white',
                markeredgewidth=1.8,
                label=f'{layers} layer{"s" if layers > 1 else ""}, dropout {dropout:.1f}')

    winner = sweep.loc[sweep.val_r2.idxmax()]
    ax.axhspan(winner.val_r2 - 0.011, winner.val_r2 + 0.011, color='#c62828', alpha=0.07,
               zorder=0)
    ax.plot(winner.hidden, winner.val_r2, marker='*', markersize=17, color='#c62828',
            markeredgecolor='white', markeredgewidth=0.8, zorder=5)
    ax.annotate(f'selected: {int(winner.hidden)} units,\n{int(winner.layers)} layer, '
                f'dropout {winner.dropout:.1f}\n$R^2 = {winner.val_r2:+.4f}$',
                xy=(winner.hidden, winner.val_r2), xytext=(-30, -78),
                textcoords='offset points', ha='right', va='top', fontsize=8.4,
                color='#8b1a1a',
                arrowprops=dict(arrowstyle='-', color='#c62828', linewidth=0.9,
                                shrinkA=2, shrinkB=7))
    ax.text(widths[0] * 1.06, winner.val_r2 - 0.011, 'shaded: $\\pm0.011$ seed spread',
            fontsize=7.4, color='#8b1a1a', va='top')

    ax.set_xscale('log', base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.minorticks_off()
    ax.set_xlabel('GRU hidden units')
    ax.set_ylabel('validation $R^2$ (axis-averaged)')
    ax.axhline(0, color='#8b949e', linewidth=0.9, linestyle=':')
    ax.grid(alpha=0.22)
    ax.legend(loc='lower right', frameon=False)
    ax.set_title('GRU sweep: clean validation $R^2$ against hidden width',
                 loc='left', pad=10)

    fig.text(0.01, 0.012,
             'One seed, three held-off stimulation-off validation days, 12 settings.\n'
             'Winner beats 256 units + dropout 0.2 by $+0.003$, inside the $\\pm0.011$ '
             'seed spread: the top two are not separable.',
             fontsize=7.0, color='#5c6670', va='bottom', linespacing=1.45)

    fig.tight_layout(rect=[0, 0.062, 1, 1])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=180, facecolor='white')
    plt.close(fig)
    print(OUT)


if __name__ == '__main__':
    main()
