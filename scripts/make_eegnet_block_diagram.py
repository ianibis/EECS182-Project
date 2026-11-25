"""Generate the compact CausalEEGNet block diagram.

Emits reports/eegnet_block_diagram.svg: eleven stages in two columns read top-to-bottom
then left-to-right. Each stage is a tensor graphic (batch = stacked offset sheets, sheet
size log-scaled to the C x T extent of that stage) with its name and symbolic shape beside
it. The variable legend, normalization, optimizer and causality notes live in the report
caption, not the figure, so the blocks stay large enough to read.

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
                key = key.removesuffix("_n")  # utils uses temporal_filters_n
                if key in cfg:
                    cfg[key] = float(val)
    return cfg


CFG = read_config()
C, T, N, V = 35, 2000, 40, 2
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
STACK_LIFT = (BATCH_SHEETS - 1) * -SHEET_DY   # back sheets sit this far above the front one
ROW_GAP = 18                                  # clear space between two rows, all of it arrow
GRAPH_W = 96.0
FACE_H = 62.0

COLORS = {
    "io": ("#e7f0ff", "#2f6fdb"),
    "conv": ("#f3f7ff", "#2f6fdb"),
    "norm": ("#fff4e0", "#c77c02"),
    "act": ("#e7f6e9", "#2e7d32"),
    "pool": ("#f1e9ff", "#6b46c1"),
    "head": ("#ffe9e9", "#c62828"),
}

# name, symbolic shape, (channels, samples), kind. What each stage computes is in the
# report text, not the figure.
STAGES = [
    ("input", "(B, C, T)", C, T, "io"),
    ("temporal conv", "(B, F1, C, T)", C, T, "conv"),
    ("BatchNorm2d", "(B, F1, C, T)", C, T, "norm"),
    ("spatial conv", "(B, F2, 1, T)", 1, T, "conv"),
    ("BatchNorm2d + ELU", "(B, F2, 1, T)", 1, T, "act"),
    (f"avg pool, p={DO:.2f}", "(B, F2, 1, T/2)", 1, T // P, "pool"),
    ("separable conv", "(B, F2, 1, T/2)", 1, T // P, "conv"),
    ("BatchNorm2d + ELU", "(B, F2, 1, T/2)", 1, T // P, "act"),
    (f"avg pool, p={DO:.2f}", "(B, F2, 1, N)", 1, N, "pool"),
    ("regression head", "(B, N, V)", V, N, "head"),
    ("output", "(B, N, V)", V, N, "io"),
]
SPLIT = 5


def face_size(channels, samples):
    # Sheet size tracks the real extent of the stage: the log keeps 2000:40 samples and
    # 35:1 channels on one scale, so an input sheet is full height and a 1 x 40 map is a
    # thin strip. Late stages are floored only enough to stay visible.
    w = GRAPH_W * math.log10(max(samples, 10)) / math.log10(T)
    h = FACE_H * math.log10(max(channels, 1)) / math.log10(C)
    return max(w, 14.0), max(h, 6.0)


def row_h(ch, smp):
    _, h = face_size(ch, smp)
    return max(h, 34.0) + ROW_GAP


def col_x(col):
    return PAD + col * (COL_W + GAP)


STYLE = """<style>
  svg .name { font-size: 10.5px; font-weight: 600; fill: #1f2328; }
  svg .shape { font-size: 9px; fill: #8b949e;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  svg .arrow { stroke: #6e7781; stroke-width: 1.15; }
</style>
<defs>
  <marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.5" markerHeight="4.5"
          orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e7781"/>
  </marker>
</defs>"""


def build():
    ys, bottoms = {}, {}
    top0 = 8
    for col, group in ((0, range(SPLIT)), (1, range(SPLIT, len(STAGES)))):
        y = top0
        for idx in group:
            ys[idx] = y
            y += row_h(*STAGES[idx][2:4])
        bottoms[col] = y - ROW_GAP

    body, arrows = [], []
    for idx, (name, shape, ch, smp, kind) in enumerate(STAGES):
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
            _, ph = face_size(*STAGES[idx - 1][2:4])
            px = gx + max(w, 34.0) / 2
            # touch the bottom of the previous sheet and the top of this one
            arrows.append(f'<line x1="{px}" y1="{prev + ph:.1f}" '
                          f'x2="{px}" y2="{y:.1f}" class="arrow" '
                          f'marker-end="url(#head)"/>')

    # serpentine wrap between columns: out the bottom of the last left box, down and
    # across the gutter, up a lane clear of both columns, then into the left edge of the
    # first right-column sheet
    lpx = col_x(0) + face_size(*STAGES[SPLIT - 1][2:4])[0] / 2
    py = ys[SPLIT - 1]
    _, ph = face_size(*STAGES[SPLIT - 1][2:4])
    seg_bottom = bottoms[0] + 8
    lane = col_x(1) - 14          # inside the gutter, clear of both columns
    _, entry_h = face_size(*STAGES[SPLIT][2:4])
    entry = ys[SPLIT] + entry_h / 2   # the sheet's own middle, not the row box's
    arrows.append(f'<path d="M {lpx} {py + ph:.1f} V {seg_bottom:.1f} '
                  f'H {lane:.1f} V {entry:.1f}" class="arrow" fill="none"/>')
    arrows.append(f'<line x1="{lane:.1f}" y1="{entry:.1f}" x2="{col_x(1) - 2:.1f}" '
                  f'y2="{entry:.1f}" class="arrow" marker-end="url(#head)"/>')

    total = max(bottoms.values()) + 10

    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {total:.0f}" '
        f'width="{WIDTH}" height="{total:.0f}" '
        f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
        STYLE,
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        *body, *arrows, "</svg>",
    ])


if __name__ == "__main__":
    with open(OUT, "w") as fh:
        fh.write(build())
    print(OUT)
