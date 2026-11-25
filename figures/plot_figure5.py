#!/usr/bin/env python3
"""Plot matched Transformer ablations across archived calibration sizes."""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent

MODELS = ("original", "calibrated", "final")
RUNS = dict(original="final_in_context", calibrated="calibrated_a0", final="improve_jitter")
VARIANT = {"recorded_on": "recorded", "parrm_amp10": "amp10"}
METRICS = {"r2": r"$R^2$", "pearson_r": "Pearson’s r", "rmse_cm_s": "RMSE (cm/s)"}
FIELDS = [f"{metric}_{axis}" for metric in METRICS for axis in ("vx","vy","mean")]
KS = (0, 2, 5, 10, 20)
DRAW_IDS = {2: (0, 1, 2), 5: (0, 1, 2), 10: (0, 1, 2), 20: (0,)}
SEEDS = tuple(range(1, 6))
BOOT_SEED, BOOT_REPS = 18203, 10000
STYLE = {
    "original": ("Original Transformer", "#566877", "s", "-"),
    "calibrated": ("Transformer + ridge", "#8a5b83", "^", "--"),
    "final": ("Transformer + ridge + jitter", "#009E73", "o", "-."),
}



def day_interval(values):
    per_day = values.groupby(level="off_reference")
    sums = per_day.sum().to_numpy()
    counts = per_day.size().to_numpy()
    assert len(counts) == 5
    plan = np.random.default_rng(BOOT_SEED).integers(len(counts), size=(BOOT_REPS, len(counts)))
    resamples = sums[plan].sum(axis=1) / counts[plan].sum(axis=1)[:, None]
    return values.mean().to_numpy(), *np.percentile(resamples, [2.5, 97.5], axis=0)


def summarize(scores):
    keys = ["session", "off_reference", "query_data", "k"]
    sessions = (scores.groupby([*keys, "model"])[FIELDS].mean().reset_index())
    pairs = scores.pivot(index=[*keys, "draw", "seed"], columns="model", values=FIELDS)
    assert not pairs.isna().any().any()
    paired_sessions = {
        f"{after} minus {before}":
        (pairs.xs(after, axis=1, level="model")[FIELDS] -
         pairs.xs(before, axis=1, level="model")[FIELDS]).groupby(level=keys).mean()
        for after, before in (("calibrated", "original"), ("final", "calibrated"))
    }

    source, differences = [], []
    for (variant, k), group in sessions.groupby(["query_data", "k"], sort=False):
        by_model = {name: rows.set_index(["session", "off_reference"])[FIELDS]
                    for name, rows in group.groupby("model")}
        contrasts = {name: values.xs((variant, k), level=("query_data", "k"))
                     for name, values in paired_sessions.items()}
        for destination, kind, series in ((source, "condition", by_model),
                                          (differences, "paired difference", contrasts)):
            for name, values in series.items():
                assert len(values) == 14
                point, lo, hi = day_interval(values)
                for field, mean, lower, upper in zip(FIELDS, point, lo, hi):
                    destination.append(dict(query_data=variant, series=name, estimate_type=kind, metric=field,
                                            value=mean, ci_low=lower, ci_high=upper,
                                            n_sessions=len(values), n_days=5, n_seeds=len(SEEDS), k=k,
                                            n_draws_per_session=len(DRAW_IDS[k]),
                                            draw_ids="|".join(map(str, DRAW_IDS[k])),
                                            context_source="on", mask="common", test_status="exploratory",
                                            aggregation="matched individual scores averaged within session",
                                            n_boot=BOOT_REPS, bootstrap_seed=BOOT_SEED,
                                            plotted=kind=="condition" and field.endswith("_mean")))
    return sessions, pd.DataFrame(source), pd.DataFrame(differences)


def marker_line(model, label=False):
    name,color,marker,style=STYLE[model]
    return Line2D([0],[0],lw=1.05,linestyle=style,color=color,marker=marker,markersize=3.45,
                  markeredgewidth=.55,label=name if label else None)


def render(summary, out):
    for path in Path("/System/Library/Fonts/Supplemental").glob("Times New Roman*.ttf"):
        font_manager.fontManager.addfont(str(path))
    fonts={entry.name for entry in font_manager.fontManager.ttflist}
    face=next(name for name in ("Times New Roman","TeX Gyre Termes","Liberation Serif","STIXGeneral")
              if name in fonts)
    plt.rcParams.update({"font.family":face,"font.size":9,"axes.labelsize":9,
                         "xtick.labelsize":8,"ytick.labelsize":8,"pdf.fonttype":42,
                         "svg.fonttype":"none","svg.hashsalt":"eecs182-fig5",
                         "mathtext.fontset":"stix","text.color":"#18282d"})
    fig=plt.figure(figsize=(5.5,3.87),facecolor="white")
    fig.legend(handles=[marker_line(model,label=True) for model in MODELS],
               loc="upper center",bbox_to_anchor=(.513,.982),ncol=3,frameon=False,
               borderaxespad=0,labelspacing=.39,fontsize=8.0,handlelength=1.7,
               handletextpad=.38,columnspacing=1.15)
    for i,(variant,row_label) in enumerate(zip(VARIANT, ("Recorded stimulation", "10× artifact"))):
        y=(.540,.132)[i]
        fig.text(.515,y+.281,row_label,ha="center",fontsize=9.1)
        for j,(metric,label) in enumerate(METRICS.items()):
            left=.125+j*.305
            ax=fig.add_axes((left,y,.220,.236))
            values=summary[summary.metric.eq(metric+"_mean")]
            lo,hi=values.ci_low.min(),values.ci_high.max()
            pad=(hi-lo)*.055
            ax.set_ylim(lo-pad,hi+pad)
            points=values[values.query_data.eq(variant)]
            for model in MODELS:
                _,color,marker,style=STYLE[model]
                series=points[points.series.eq(model)].set_index("k").reindex(KS)
                selected=series.value.notna()
                x=series.index.to_numpy()
                ax.vlines(x[selected],series.loc[selected,"ci_low"],series.loc[selected,"ci_high"],
                          color=color,alpha=.54,lw=.69,zorder=2)
                ax.plot(x,series.value,color=color,marker=marker,linestyle=style,lw=1.05,
                        markersize=3.45,markeredgewidth=.55,zorder=3)
            ax.spines[["top","right"]].set_visible(False)
            for side in ("left","bottom"):
                ax.spines[side].set_color("#556368")
                ax.spines[side].set_linewidth(.55)
            ax.grid(axis="y",color="#e4e9e8",lw=.48)
            ax.set_axisbelow(True)
            ax.set_xticks(KS)
            ax.set_xlim(-.75,20.75)
            if metric=="r2" and lo<0<hi:
                ax.axhline(0,color="#a7b2b2",lw=.52,linestyle=":",zorder=1)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(length=2.5,width=.55,pad=2)
            ax.set_ylabel(label,labelpad=2)
            ax.text(-.055,1.059,chr(ord("A")+3*i+j),transform=ax.transAxes,
                    fontsize=10.5,weight="bold",ha="left")
    fig.text(.525,.040,r"Number of labeled calibration trials, $k$",ha="center",fontsize=9)
    fig.canvas.draw()
    b=fig.get_tightbbox(fig.canvas.get_renderer())
    assert -0.01<=b.x0<b.x1<=5.51 and -.01<=b.y0<b.y1<=3.88,b
    for extension in ("pdf","svg","png"):
        metadata=({"CreationDate":None,"ModDate":None} if extension=="pdf" else
                  {"Date":None} if extension=="svg" else None)
        fig.savefig(out/f"figure5.{extension}",dpi=600,facecolor="white",metadata=metadata)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=HERE / "source_data")
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "figures")
    args = parser.parse_args()
    source, out = args.source_dir, args.output_dir
    def read(name):
        path = source / name
        if not path.is_file():
            parser.error(f"Missing original figure data: {path}")
        return pd.read_csv(path, float_precision="round_trip")
    out.mkdir(parents=True, exist_ok=True)
    scores = read("figure5_curve_per_seed.csv")
    _, summary, _ = summarize(scores)
    render(summary, out)


if __name__ == "__main__":
    main()
