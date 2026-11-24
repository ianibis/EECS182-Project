"""Generate the causal GRU block diagram.

Emits reports/gru_block_diagram.svg (and, when rsvg-convert is available, the PDF that
the report includes). The left panel unrolls one 40-bin trial over time so the recurrent
hidden state is visible as the horizontal arrows; the right panel opens one cell.

Hyperparameters mirror `GRURegressor` in task4/utils.py and the sweep winner reported in
the writeup (one layer, 256 units, dropout 0), so the figure cannot silently drift.
"""

import os
import re
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SVG = os.path.join(ROOT, "reports", "gru_block_diagram.svg")
PDF = os.path.join(ROOT, "reports", "figures", "gru_block_diagram.pdf")

N_FEATURES, N_HIDDEN, N_OUT = 280, 256, 2

WIDTH, PAD = 744, 18
CENTERS = [68, 188, 292, 414]           # bins 1, 2, ellipsis, N
GRU_W, GRU_H, GRU_Y = 86, 54, 164
IN_W, IN_H, IN_Y = 86, 38, 252
OUT_W, OUT_H, OUT_Y = 74, 24, 52
BAR = (25, 90, 432, 38)
CARD = (494, 52, 238, 284)

COLORS = {
    "in": ("#e7f0ff", "#2f6fdb"),
    "cell": ("#f1e9ff", "#6b46c1"),
    "state": ("#f6f0ff", "#6b46c1"),
    "head": ("#ffe9e9", "#c62828"),
    "out": ("#e7f6e9", "#2e7d32"),
}

STYLE = """<style>
  svg .sect   { font-size: 11.5px; font-weight: 700; fill: #1f2328; }
  svg .name   { font-size: 11px; font-weight: 600; fill: #1f2328; }
  svg .small  { font-size: 8.6px; fill: #57606a; }
  svg .pill   { font-size: 9.5px; font-weight: 600; fill: #1f2328; }
  svg .eq     { font-size: 8.6px; fill: #1f2328;
             font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  svg .arrow  { stroke: #6e7781; stroke-width: 1.15; fill: none; }
  svg .rec    { stroke: #6b46c1; stroke-width: 1.7; fill: none; }
  svg .lbl    { font-size: 8.6px; fill: #6b46c1; font-weight: 600; }
</style>
<defs>
  <marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.5" markerHeight="4.5"
          orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e7781"/>
  </marker>
  <marker id="rhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.5" markerHeight="4.5"
          orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b46c1"/>
  </marker>
</defs>"""


def box(x, y, w, h, kind, rx=3.0):
    fill, stroke = COLORS[kind]
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')


def text(x, y, body, cls="name", anchor="start"):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{body}</text>'


def rich(x, y, body, cls="name", anchor="start", sub_size=7.2):
    """Text with `_{...}` subscripts rendered as real subscript spans."""
    spans, shift = [], 0.0
    for piece in re.split(r"(_\{[^}]*\})", body):
        if not piece:
            continue
        if piece.startswith("_{") and piece.endswith("}"):
            content, target, size = piece[2:-1], 2.8, sub_size
        else:
            content, target, size = piece, 0.0, None
        attrs = f' dy="{target - shift:.1f}"'
        if size:
            attrs += f' font-size="{size}"'
        spans.append(f"<tspan{attrs}>{content}</tspan>")
        shift = target
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" xml:space="preserve" '
            f'text-anchor="{anchor}">{"".join(spans)}</text>')


def arrow(x1, y1, x2, y2, cls="arrow", marker="head", dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'class="{cls}" marker-end="url(#{marker})"{extra}/>')


def path(d, cls="rec", marker="rhead", dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" class="{cls}" marker-end="url(#{marker})"{extra}/>'


def build():
    body = ['<rect width="100%" height="100%" fill="#ffffff"/>']

    # ---- left panel: unrolled over bins -------------------------------------
    body.append(rich(PAD, 14, "each column is one 50 ms bin", "sect", sub_size=7.4))
    body.append(rich(PAD, 27, "x_{b} is the 280-dimensional input for that bin", "sect",
                     sub_size=7.4))
    body.append(text(PAD, 44, "unrolled over bins b = 1\u2026N", "sect"))

    for cx, sub in zip(CENTERS, ("1", "2", None, "N")):
        if sub is None:
            body.append(text(cx, GRU_Y + 28, "\u22ef", "name", "middle"))
            body.append(text(cx, IN_Y + 25, "\u22ef", "name", "middle"))
            body.append(text(cx, OUT_Y + 17, "\u22ef", "name", "middle"))
            continue
        body.append(box(cx - OUT_W / 2, OUT_Y, OUT_W, OUT_H, "out"))
        body.append(rich(cx, OUT_Y + 16, f"v\u0302_{{{sub}}} (2)", "pill", "middle"))
        body.append(box(cx - GRU_W / 2, GRU_Y, GRU_W, GRU_H, "cell"))
        body.append(text(cx, GRU_Y + 23, "GRU", "name", "middle"))
        body.append(text(cx, GRU_Y + 37, f"{N_HIDDEN} units", "small", "middle"))
        body.append(box(cx - IN_W / 2, IN_Y, IN_W, IN_H, "in"))
        body.append(rich(cx, IN_Y + 17, f"x_{{{sub}}}", "name", "middle"))
        body.append(text(cx, IN_Y + 30, f"bin {sub} \u00b7 {N_FEATURES}", "small", "middle"))
        body.append(arrow(cx, IN_Y, cx, GRU_Y + GRU_H))
        body.append(arrow(cx, GRU_Y, cx, BAR[1] + BAR[3]))
        body.append(arrow(cx, BAR[1], cx, OUT_Y + OUT_H))
        body.append(rich(cx + 6, GRU_Y - 16, f"h_{{{sub}}} ({N_HIDDEN})", "small", sub_size=6.6))

    body.append(box(*BAR[:2], BAR[2], BAR[3], "head"))
    body.append(text(BAR[0] + BAR[2] / 2, BAR[1] + 17,
                     "linear head with weight sharing across bins", "name", "middle"))
    body.append(rich(BAR[0] + BAR[2] / 2, BAR[1] + 31,
                     f"v\u0302_{{b}} = W h_{{b}} + c   ({N_HIDDEN} \u2192 {N_OUT})",
                     "name", "middle", sub_size=7.2))

    # recurrence: the state leaves bin b and enters the cell of bin b + 1
    y = GRU_Y + GRU_H / 2
    body.append(arrow(CENTERS[0] + GRU_W / 2, y, CENTERS[1] - GRU_W / 2, y,
                      "rec", "rhead"))
    body.append(arrow(CENTERS[1] + GRU_W / 2, y, CENTERS[3] - GRU_W / 2, y,
                      "rec", "rhead", "4 3"))
    body.append(f'<rect x="282" y="{y - 11:.1f}" width="20" height="22" fill="#ffffff"/>')
    body.append(text(292, y + 5, "\u22ef", "name", "middle"))
    body.append(rich(258, y - 6, "h_{2}", "lbl", "middle", sub_size=6.4))
    body.append(rich(336, y - 6, "h_{N\u22121}", "lbl", "middle", sub_size=6.4))
    body.append(rich(128, y - 6, "h_{1}", "lbl", "middle", sub_size=6.4))

    # ---- right panel: one cell ---------------------------------------------
    body.append(text(498, 44, "GRU cell (one bin)", "sect"))
    body.append(f'<rect x="{CARD[0]}" y="{CARD[1]}" width="{CARD[2]}" height="{CARD[3]}" '
                'rx="4" fill="#fbfcfd" stroke="#eaeef2" stroke-width="1"/>')

    body.append(box(526, 64, 96, 34, "in"))
    body.append(rich(574, 79, "x_{b}", "name", "middle"))
    body.append(text(574, 92, f"{N_FEATURES}", "small", "middle"))
    body.append(arrow(574, 98, 574, 120))

    body.append(box(526, 120, 132, 60, "cell"))
    body.append(text(592, 139, "3 gates", "name", "middle"))
    body.append(text(592, 152, "z update, r reset", "small", "middle"))
    body.append(text(592, 165, "n candidate (\u03c3, tanh)", "small", "middle"))

    body.append(box(670, 133, 58, 34, "state"))
    body.append(rich(699, 148, "h_{b}", "name", "middle"))
    body.append(text(699, 161, f"{N_HIDDEN}", "small", "middle"))
    body.append(arrow(658, 150, 670, 150))

    body.append(box(526, 218, 112, 34, "state"))
    body.append(rich(582, 233, "h_{b\u22121}", "name", "middle"))
    body.append(text(582, 246, "previous state", "small", "middle"))
    body.append(arrow(574, 218, 574, 180, "rec", "rhead"))

    body.append(path("M 728 167 V 235 H 640", dash="4 3"))
    body.append(text(692, 202, "next bin", "lbl", "middle"))

    body.append(rich(504, 270, "h_{b} = GRU(x_{b}, h_{b\u22121})", "eq", sub_size=6.4))
    for i, line in enumerate([
        "z_{b} = \u03c3(W_{z} x_{b} + U_{z} h_{b\u22121} + b_{z})",
        "r_{b} = \u03c3(W_{r} x_{b} + U_{r} h_{b\u22121} + b_{r})",
        "n_{b} = tanh(W_{n} x_{b} + U_{n}(r_{b} \u2299 h_{b\u22121}) + b_{n})",
        "h_{b} = (1 \u2212 z_{b}) \u2299 h_{b\u22121} + z_{b} \u2299 n_{b}",
    ]):
        body.append(rich(504, 284 + 14 * i, line, "eq", sub_size=6.4))

    height = 348
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" '
        f'width="{WIDTH}" height="{height}" '
        'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
        STYLE,
        *body,
        "</svg>",
    ])


def main():
    with open(SVG, "w") as handle:
        handle.write(build())
    print(SVG)
    converter = shutil.which("rsvg-convert")
    if converter:
        os.makedirs(os.path.dirname(PDF), exist_ok=True)
        subprocess.run([converter, "-f", "pdf", "-o", PDF, SVG], check=True)
        print(PDF)


if __name__ == "__main__":
    main()
