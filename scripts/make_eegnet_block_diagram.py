"""Generate the compact CausalEEGNet block diagram.

Emits reports/eegnet_block_diagram.svg: eleven stages in two columns read top-to-bottom
then left-to-right. Each stage is a tensor graphic (batch = stacked offset sheets, sheet
size log-scaled to the C x T extent of that stage) with its name and symbolic shape beside
it, plus a variable legend under the flow. The operations themselves are described in the
report text, not in the figure, so the blocks stay large enough to read.

Hyperparameters are read from task4/utils.py so the diagram cannot silently drift from
the model. Dropout shown is the sweep winner (0.25), not braindecode's default.
"""

import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "eegnet_block_diagram.svg")
UTILS = os.path.join(ROOT, "task4", "utils.py")

DEFAULTS = dict(temporal_filters=8, depth_multiplier=2, kernel_samples=500,
                separable_samples=62)


def read_config():
    cfg = dict(DEFAULTS)
    if os.path.exists(UTILS):
        text = open(UTILS).read()
        m = re.search(r"EEGNET_DEFAULT_SETTINGS = dict\(([^)]*)\)", text)
        if m:
            for key, val in re.findall(r"(\w+)=([\d.]+)", m.group(1)):
                if key in cfg:
                    cfg[key] = float(val)
    return cfg


CFG = read_config()
C, T, N = 35, 2000, 40
F = int(CFG["temporal_filters"])
D = int(CFG["depth_multiplier"])
K1 = int(CFG["kernel_samples"])
K2 = int(CFG["separable_samples"])
F2 = F * D
P, Q, DO = 2, 25, 0.25

COL_W = 232
GAP = 18
PAD = 12
WIDTH = 2 * COL_W + GAP + 2 * PAD
SHEET_DX, SHEET_DY = 4, -3
BATCH_SHEETS = 3
GRAPH_W = 96.0
FACE_H = 62.0
TEXT_W = 118          # name/shape block beside the sheet

COLORS = {
    "io": ("#e7f0ff", "#2f6fdb"),
    "conv": ("#f3f7ff", "#2f6fdb"),
    "norm": ("#fff4e0", "#c77c02"),
    "act": ("#e7f6e9", "#2e7d32"),
    "pool": ("#f1e9ff", "#6b46c1"),
    "head": ("#ffe9e9", "#c62828"),
}

# name, symbolic shape, one-line description, (channels, samples), kind
STAGES = [
    ("input", "(B, C, T)", "LFP at 1 kHz", C, T, "io"),
    ("temporal conv", "(B, F, C, T)", f"{F} filters, kernel 1 x {K1}, causal", C, T, "conv"),
    ("batch norm", "(B, F, C, T)", f"{F} channels", C, T, "norm"),
    ("spatial conv", "(B, FD, 1, T)", f"depthwise {C} x 1, L2 renormalised", 1, T, "conv"),
    ("batch norm + ELU", "(B, FD, 1, T)", f"{F2} maps", 1, T, "act"),
    ("pool + dropout", "(B, FD, 1, T/2)", f"1 x {P} pool, p = {DO}", 1, T // P, "pool"),
    ("separable conv", "(B, FD, 1, T/2)", f"1 x {K2} depthwise + 1 x 1 pointwise", 1, T // P,
     "conv"),
    ("batch norm + ELU", "(B, FD, 1, T/2)", f"{F2} maps", 1, T // P, "act"),
    ("pool + dropout", "(B, FD, 1, N)", f"1 x {Q} to {N} bins, p = {DO}", 1, N, "pool"),
    ("regression head", "(B, N, 2)", f"1 x 1 conv, {F2} to 2", 2, N, "head"),
    ("output", "(B, N, 2)", "vx, vy per 50 ms bin", 2, N, "io"),
]
SPLIT = 6


def face_size(channels, samples):
    # Compressed log range: the true extent ratio is 2000:40 by 35:1, which would leave
    # the late stages as slivers. Keeping every sheet within 55-100 % of the frame makes
    # each stage legible at full page width without pretending the sizes are to scale.
    w = GRAPH_W * (0.55 + 0.45 * math.log10(max(samples, 10)) / math.log10(T))
    h = FACE_H * (0.40 + 0.60 * math.log10(max(channels, 1)) / math.log10(C))
    return max(w, 14.0), max(h, 11.0)


def row_h(ch, smp):
    _, h = face_size(ch, smp)
    return max(h, 34.0) + 12


def col_x(col):
    return PAD + col * (COL_W + GAP)


STYLE = """<style>
  svg .name { font-size: 10.5px; font-weight: 600; fill: #1f2328; }
  svg .shape { font-size: 9px; fill: #8b949e;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  svg .head { font-size: 12.5px; font-weight: 700; fill: #1f2328; }
  svg .note { font-size: 8.5px; fill: #57606a; }
  svg .arrow { stroke: #6e7781; stroke-width: 1.15; }
  svg .var  { font-size: 8.5px; fill: #1f2328;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
</style>
<defs>
  <marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.5" markerHeight="4.5"
          orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e7781"/>
  </marker>
</defs>"""


def build():
    ys, bottoms = {}, {}
    top0 = 40
    for col, group in ((0, range(SPLIT)), (1, range(SPLIT, len(STAGES)))):
        y = top0
        for idx in group:
            ys[idx] = y
            y += row_h(*STAGES[idx][3:5])
        bottoms[col] = y - 12

    body = []
    for idx, (name, shape, sub, ch, smp, kind) in enumerate(STAGES):
        col = 0 if idx < SPLIT else 1
        gx = col_x(col)
        y = ys[idx]
        w, h = face_size(ch, smp)
        fill, stroke = COLORS[kind]
        for i in range(BATCH_SHEETS - 1, 0, -1):
            body.append(f'<rect x="{gx + i * SHEET_DX:.1f}" y="{y + i * SHEET_DY:.1f}" '
                        f'width="{w:.1f}" height="{h:.1f}" rx="2" fill="{fill}" '
                        f'stroke="{stroke}" stroke-width="1" '
                        f'opacity="{0.30 + 0.22 * i:.2f}"/>')
        body.append(f'<rect x="{gx}" y="{y}" width="{w:.1f}" height="{h:.1f}" rx="2" '
                    f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
        tx = gx + GRAPH_W + SHEET_DX * 2 + 10
        mid = y + max(h, 34.0) / 2
        body.append(f'<text x="{tx:.1f}" y="{mid - 6:.1f}" class="name">{name}</text>')
        body.append(f'<text x="{tx:.1f}" y="{mid + 8:.1f}" class="shape">{shape}</text>')
        if idx and not (idx == SPLIT):
            prev = ys[idx - 1]
            _, ph = face_size(*STAGES[idx - 1][3:5])
            px = gx + 3
            body.append(f'<line x1="{px}" y1="{prev + max(ph, 34.0) + 2:.1f}" '
                        f'x2="{px}" y2="{y - 3:.1f}" class="arrow" marker-end="url(#head)"/>')

    # serpentine wrap between columns: out the bottom of the last left box, across the
    # gutter, then down into the top of the first right-column box at its left edge
    lpx = col_x(0) + 3
    py = ys[SPLIT - 1]
    _, ph = face_size(*STAGES[SPLIT - 1][3:5])
    seg_bottom = bottoms[0] + 8
    rpx = col_x(1) + 3
    top_right = ys[SPLIT]
    body.append(f'<path d="M {lpx} {py + max(ph, 34.0) + 2:.1f} V {seg_bottom:.1f} '
                f'H {rpx:.1f} V {top_right - 11:.1f}" class="arrow" fill="none"/>')
    body.append(f'<line x1="{rpx:.1f}" y1="{top_right - 11:.1f}" x2="{rpx:.1f}" '
                f'y2="{top_right - 3:.1f}" class="arrow" marker-end="url(#head)"/>')

    top = bottoms[0] + 22
    footer = [f'<line x1="{PAD}" y1="{top:.1f}" x2="{WIDTH - PAD}" y2="{top:.1f}" '
              'stroke="#eaeef2" stroke-width="1"/>']
    oy = top + 16
    footer.append(f'<text x="{PAD}" y="{oy:.1f}" class="var">B batch     C {C} channels'
                  f'     T {T} samples     N {N} bins     F {F}     D {D}     '
                  f'K1 {K1}     K2 {K2}     p {DO}</text>')
    total = oy + 16

    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {total:.0f}" '
        f'width="{WIDTH}" height="{total:.0f}" '
        f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
        STYLE,
        f'<text x="{PAD}" y="14" class="head">CausalEEGNet: one forward pass</text>',
        f'<text x="{PAD}" y="27" class="note">two columns, left to right; sheets are the B '
        f'samples of one batch, sized to the C x T extent of that stage</text>',
        *body, *footer, "</svg>",
    ])


if __name__ == "__main__":
    with open(OUT, "w") as fh:
        fh.write(build())
    print(OUT)
