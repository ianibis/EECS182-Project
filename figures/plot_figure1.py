#!/usr/bin/env python3
"""Render the experimental paradigm and the implemented Task 5 decoder."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BRAIN = HERE / "assets/brain_setup_cropped.png"
TRACES = HERE / "assets/figure1_on_traces_crop.png"
OUTPUT = PROJECT / "figures"
WIDTH, HEIGHT, DPI = 5.5, 3.86, 600

INK = "#17282e"
SECONDARY = "#3a4d55"
RULE = "#aab8ba"
NEURAL = "#245c58"
LABEL = "#88482f"
POOL = "#e1eeea"
QUERY = "#edf0f0"
NEURAL_WASH = "#edf4f2"
LABEL_WASH = "#f7f0eb"


def verify_project(repo):
    choice = json.loads((repo / "task5_results/improve_choice.json").read_text())["step2"]
    if (choice["name"], choice["jitter_bins"], choice["timing"]) != ("jitter", 2, "none"):
        raise ValueError(f"Figure assumes the final jitter setting, found {choice}")
    splits = json.loads((repo / "splits.json").read_text())
    days = [set(splits["days"][key]) for key in ("train", "val", "test")]
    if sum(map(len, days)) != len(set.union(*days)):
        raise ValueError("Recording days overlap between splits")
    for partition in splits["calibration"].values():
        pool, query = set(partition["pool"]), set(partition["query"])
        if len(pool) != 20 or len(query) != 30 or pool & query:
            raise ValueError("Unexpected calibration and query split")
    if "1-9-4-3-7" not in (repo / "README.md").read_text():
        raise ValueError("The illustrated reaching target order could not be verified")


class Figure:
    def __init__(self):
        self.fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0), aspect="auto")
        self.ax.set_axis_off()

    def text(self, x, y, value, *, bold=False, color=INK, size=8.4, ha="left", va="top"):
        return self.ax.text(x, y, value, fontsize=size, weight="bold" if bold else "normal",
                            color=color, ha=ha, va=va, linespacing=1.14, zorder=5)

    def panel(self, letter, y):
        self.text(.105, y, letter, bold=True, size=11)

    def link(self, points, *, color=NEURAL, arrow=True, dashed=False, lw=.76):
        xs, ys = zip(*points)
        style = (0, (2.3, 1.55)) if dashed else "-"
        if len(points)>2 or not arrow:
            stop = -1 if arrow else None
            self.ax.plot(xs[:stop], ys[:stop], color=color, lw=lw, ls=style, solid_capstyle="round",
                         zorder=3)
        if arrow:
            self.ax.annotate("", xy=points[-1], xytext=points[-2],
                             arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=style,
                                             shrinkA=0, shrinkB=0, mutation_scale=7), zorder=4)

    def rect(self, x, y, w, h, color, outline="none"):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=outline,
                                    linewidth=.6, zorder=2))

    def raster(self, path, left, top, width):
        with Image.open(path) as im:
            pixels = np.asarray(im.convert("RGBA" if "A" in im.getbands() else "RGB"))
        height = width * pixels.shape[0] / pixels.shape[1]
        self.ax.imshow(pixels, extent=(left, left+width, top+height, top),
                                origin="upper", interpolation="none", aspect="auto", zorder=1)
        return top+height

    def finish(self):
        self.fig.canvas.draw()
        page = self.fig.bbox
        renderer = self.fig.canvas.get_renderer()
        labels = self.ax.texts
        extents = [label.get_window_extent(renderer) for label in labels]
        errors = []
        for i, first in enumerate(extents):
            if labels[i].get_text() and (first.x0<0 or first.y0<0 or first.x1>page.x1 or first.y1>page.y1):
                errors.append(f"out of bounds: {labels[i].get_text()!r}")
            for j, second in enumerate(extents[i+1:], i+1):
                if labels[i].get_text() and labels[j].get_text() and first.overlaps(second):
                    errors.append(f"overlap: {labels[i].get_text()!r} / {labels[j].get_text()!r}")
        if errors:
            raise RuntimeError("\n".join(errors))


def experimental(fig, brain, traces):
    fig.panel("A", .065)
    fig.text(3.19, .068, "LFP (µV)", ha="center")
    fig.text(4.83, .068, "Touchscreen", ha="center")
    brain_end = fig.raster(brain, .107, .17, 2.00)
    fig.raster(traces, 2.16, .17, 2.00)
    # Repeat the symbol shapes outside the untouched raster as a compact key.
    key_y = brain_end + .12
    fig.rect(.212, key_y+.026, .073, .073, "#f4e95b", "#17282e")
    fig.text(.328, key_y, "M1 array", size=8.3)
    fig.ax.add_patch(Circle((1.016,key_y+.066),.037,facecolor="white",
                            edgecolor="#a23624",linewidth=.95,zorder=4))
    fig.text(1.097, key_y, "Epidural screws", size=8.3)

    points = {k: (4.495 + ((k-1)%3)*.335, .429 + ((k-1)//3)*.231) for k in range(1,10)}
    order = (1, 9, 4, 3, 7)
    fig.link([points[n] for n in order], arrow=False, lw=.84)
    for num, (x,y) in points.items():
        chosen = num in order
        fig.ax.add_patch(Circle((x,y), .0315, edgecolor=NEURAL if chosen else RULE,
                                facecolor=NEURAL if chosen else "white", lw=.6, zorder=4))
    fig.text(4.835, 1.021, "1 → 9 → 4 → 3 → 7", ha="center", size=8.1)
    fig.link([(4.737,1.603),(5.055,1.603)], color=INK)
    fig.link([(4.737,1.603),(4.737,1.325)], color=INK)
    fig.text(5.087, 1.543, r"$v_x$", size=9)
    fig.text(4.776, 1.313, r"$v_y$", size=9)


def partition(fig):
    fig.panel("B", 1.94)
    for i, name in enumerate(("Train", "Val", "Test")):
        x = .38+i*.429
        fig.rect(x,2.103,.406,.168,QUERY)
        fig.text(x+.203,2.128,name,ha="center",size=8.35)

    x, width, gap = 1.846, .683, .021
    pool_center = x+width+gap/2
    query_center = x+3.5*width+3*gap
    fig.text(pool_center,1.965,"Calibration (20)",ha="center",size=8.5)
    fig.text(query_center,1.965,"Query (30)",ha="center",size=8.5)
    for i in range(5):
        start = x+i*(width+gap)
        fig.rect(start,2.103,width,.168,POOL if i<2 else QUERY)
        fig.text(start+width/2,2.128,f"Block {i+1}",ha="center",size=8.15)
    fig.link([(x+.024,2.282),(x+2*width+gap-.024,2.282)],
             color=LABEL, arrow=False, dashed=True, lw=.67)
    fig.text(pool_center,2.315,r"$k$ labeled trials",ha="center",color=LABEL,size=8.1)


def decoder(fig):
    # Dashed paths carry context velocity or the template derived from it.
    fig.panel("C", 2.508)
    fig.text(.26,2.649,"Context velocity",size=8.1,color=LABEL)
    fig.text(.26,2.899,"Context LFP",size=8.1)
    fig.text(.26,3.263,"Query LFP",size=8.1)
    fig.rect(1.20,2.889,1.406,.527,NEURAL_WASH)
    fig.text(1.903,2.92,"EEGNet-style encoder",ha="center",size=8.2)
    fig.text(1.903,3.069,"Causal Transformer",ha="center",size=8.2)
    fig.text(1.903,3.254,"Frozen at test",ha="center",color=SECONDARY,size=8.0)
    fig.link([(.987,2.963),(1.20,2.963)])
    fig.link([(.987,3.325),(1.20,3.325)])

    fig.rect(3.033,2.623,1.085,.18,LABEL_WASH)
    fig.text(3.575,2.651,r"Template $m$",ha="center",size=8.25)
    fig.link([(1.238,2.715),(3.033,2.715)],color=LABEL,dashed=True)
    fig.ax.scatter(2.774,2.715,s=5,color=LABEL,zorder=4)
    fig.link([(2.774,2.715),(2.774,2.939),(3.036,2.939)],color=LABEL,dashed=True)
    fig.link([(3.674,2.803),(3.674,2.883)],color=LABEL,dashed=True)

    fig.rect(3.036,2.883,1.143,.363,NEURAL_WASH)
    fig.text(3.608,2.894,r"Ridge $\Delta W$",ha="center",size=8.3)
    fig.text(3.608,3.096,r"$W$ frozen at test",ha="center",color=SECONDARY,size=8.0)
    fig.link([(2.606,3.083),(3.036,3.083)])
    fig.text(2.65,2.965,r"$z_c$",size=8.5)

    fig.rect(4.272,3.156,1.123,.325,QUERY)
    fig.text(4.834,3.182,"Prediction",ha="center",size=8.3)
    fig.text(4.834,3.324,r"$m+z_q(W+\Delta W)$",ha="center",size=8.3)
    fig.link([(4.118,2.715),(5.08,2.715),(5.08,3.156)],color=LABEL,dashed=True)
    fig.link([(4.179,3.017),(4.471,3.017),(4.471,3.156)])
    fig.link([(2.606,3.325),(4.272,3.325)])
    fig.text(2.65,3.207,r"$z_q$",size=8.5)
    fig.link([(4.834,3.481),(4.834,3.603)],color=SECONDARY,lw=.65)
    fig.text(4.834,3.63,"Masked query MSE",ha="center",size=8.2)


def configure_fonts():
    plt.rcParams.update({"font.family":"DejaVu Sans", "mathtext.fontset":"stix",
                         "pdf.fonttype":42, "svg.fonttype":"none",
                         "svg.hashsalt":"eecs182-paradigm-compact"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--brain", type=Path, default=BRAIN)
    parser.add_argument("--traces", type=Path, default=TRACES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    verify_project(args.root)
    configure_fonts()
    diagram = Figure()
    experimental(diagram, args.brain, args.traces)
    partition(diagram)
    decoder(diagram)
    diagram.finish()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf","svg","png"):
        meta = ({"CreationDate":None,"ModDate":None} if suffix=="pdf" else
                {"Date":None} if suffix=="svg" else None)
        diagram.fig.savefig(args.output_dir/f"figure1.{suffix}", facecolor="white", dpi=DPI, metadata=meta)
    print(f"Artwork: {WIDTH} × {HEIGHT} in, PNG {DPI} dpi")


if __name__ == "__main__":
    main()
