"""Generate the compact 4.2.3 sampling/training-run diagram.

Emits reports/task4_2_sampling.svg: a two-column flow of one training run (stages
1-7), the seed fan-out, and what each run stores. Sized to fit beside its prose on
one screen rather than dominating it.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "task4_2_sampling.svg")

W = 660
COL = 310
GAP = 16
PAD = 12
BOX_H = 46
ROW = BOX_H + 16

COLORS = {
    "data": ("#f3f7ff", "#2f6fdb"),
    "rand": ("#f1e9ff", "#6b46c1"),
    "step": ("#fff4e0", "#c77c02"),
    "val": ("#e7f6e9", "#2e7d32"),
    "store": ("#f2f3f5", "#6b7280"),
    "score": ("#ffe9e9", "#c62828"),
}

# (title, note, tag) — first column gets the first ceil(n/2)
FLOW = [
    ("14 stimulation-off sessions", "560 trials, z-scored 1 kHz LFP · (560, 35, 2000)", "data"),
    ("one trial = 40 bins", "velocity targets + missing-kinematics mask · (40, 2)", "data"),
    ("epoch start: new permutation of 560", "default_rng(seed).permutation(560)", "rand"),
    ("35 batches of 16 trials", "slices of the permutation, not of the file · (16, 35, 2000)", "step"),
    ("forward, backward, update", "Muon on matrices, Adam on vectors · 35 steps/epoch", "step"),
    ("score 150 held-out trials", "fixed order, R2 on valid bins", "val"),
    ("keep best epoch", "stop after 20 epochs without improvement", "val"),
]

STYLE = """<style>
  svg .name { font-size: 11px; font-weight: 600; fill: #1f2328; }
  svg .sub  { font-size: 8.5px; fill: #57606a; }
  svg .chip { font-size: 10px; font-weight: 600; fill: #1f2328; text-anchor: middle; }
  svg .note { font-size: 9px; fill: #57606a; }
  svg .arrow { stroke: #6e7781; stroke-width: 1.2; }
  svg .head { font-size: 13px; font-weight: 700; fill: #1f2328; }
</style>
<defs>
  <marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5"
          orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e7781"/>
  </marker>
</defs>"""


def box(x, y, title, note, tag, w=COL):
    fill, stroke = COLORS[tag]
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{BOX_H}" rx="7" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.3"/>'
            f'<text x="{x + 10}" y="{y + 18}" class="name">{title}</text>'
            f'<text x="{x + 10}" y="{y + 33}" class="sub">{note}</text>')


def arrow_v(x, y1, y2):
    return (f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" class="arrow" '
            f'marker-end="url(#head)"/>')


def build():
    split = 4  # stages in left column
    parts = []
    top0 = 34
    for i, (title, note, tag) in enumerate(FLOW):
        col = 0 if i < split else 1
        row = i if col == 0 else i - split
        x = PAD + col * (COL + GAP)
        y = top0 + row * ROW
        parts.append(box(x, y, title, note, tag))
        if i:
            if col == 0 or i != split:
                px = x + 14
                parts.append(arrow_v(px, y - ROW + BOX_H + 2, y - 3))
            else:
                # serpentine wrap: out the bottom of the left column, across, up the
                # right edge of the right column, then an arrow into its top box
                lx = PAD + 14
                bottom = top0 + split * ROW - 12
                rx = PAD + 2 * COL + GAP + 8
                parts.append(f'<path d="M {lx} {top0 + (split - 1) * ROW + BOX_H + 2} '
                             f'V {bottom} H {rx} V {top0 - 9}" class="arrow" fill="none"/>')
                parts.append(arrow_v(rx, top0 - 9, y))

    ys = top0 + max(split, len(FLOW) - split) * ROW - 16 + BOX_H + 14
    # seed fan: one line -> five seeds
    parts.append(f'<text x="{PAD}" y="{ys - 6}" class="note">steps 1 and 2 are fixed; steps '
                 f'3 to 7 repeat once per seed, five runs in total, identical data and code</text>')
    chips = []
    cx = PAD
    for k in range(5):
        chips.append(f'<rect x="{cx}" y="{ys + 4}" width="62" height="24" rx="12" '
                     f'fill="#f1e9ff" stroke="#6b46c1" stroke-width="1.1"/>'
                     f'<text x="{cx + 31}" y="{ys + 20}" class="chip">seed {k}</text>')
        cx += 70
    parts.extend(chips)
    fan_y = ys + 28
    parts.append(f'<path d="M {PAD + 31} {fan_y + 4} H {cx - 39}" class="arrow" fill="none"/>')
    parts.append(arrow_v(W / 2, fan_y + 4, fan_y + 14))

    oy = fan_y + 18
    out_w = (W - PAD * 2 - 2 * 10) / 3
    outs = [
        ("checkpoint", "task4_2_eegnet_seedN.pt", "store"),
        ("training log", "eegnet_seedN.json, R2 per epoch", "store"),
        ("70 score rows", "14 sessions x 5 runs, both inputs", "score"),
    ]
    for j, (t, n, tag) in enumerate(outs):
        x = PAD + j * (out_w + 10)
        parts.append(box(x, oy, t, n, tag, w=out_w))

    total = oy + BOX_H + 12
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {total:.0f}" '
        f'width="{W}" height="{total:.0f}" '
        f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
        STYLE,
        f'<text x="{PAD}" y="18" class="head">The CausalEEGNet training pipeline: steps 1 to 7 make one run</text>',
        *parts, "</svg>",
    ]
    return "\n".join(svg)


if __name__ == "__main__":
    with open(OUT, "w") as fh:
        fh.write(build())
    print(OUT)
