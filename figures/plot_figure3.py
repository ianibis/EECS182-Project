#!/usr/bin/env python3
"""Draw the principal EECS182 few-shot comparison from verified saved outputs."""
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

INPUTS = ("recorded_on", "parrm_amp10")
ROW_LABELS = ("Recorded stimulation", "10× artifact")
KS = (0, 2, 5, 10, 20)
METRICS = {"r2":r"$R^2$", "pearson_r":"Pearson’s r", "rmse_cm_s":"RMSE (cm/s)"}
FIELDS = [f"{m}_{axis}" for m in METRICS for axis in ("vx","vy","mean")]
KEY = ["session","off_reference","am_freq_hz","query_data","query_trial_ids","k","repeat",
       "context_source","context_session","context_data","context_trial_ids"]
BOOT_SEED, BOOT_REPS = 18203, 10000
POTENTIAL = ("gru_archived", "mlp_off", "transformer_final", "labels_only", "ridge_task5_refit")
# Color is supported by distinct marker shapes and patterns for grayscale.
STYLE = {
    "gru_archived": ("GRU ensemble", "#0072B2", "^", "-"),
    "mlp_off": ("MLP ensemble (OFF)", "#D55E00", "s", (0,(4,1.7))),
    "transformer_final": ("Final Transformer ensemble", "#009E73", "o", "-"),
    "labels_only": ("Labels-only", "#464646", None, (0,(3,2.1))),
    "ridge_task5_refit": ("Ridge (within ON)", "#7A5195", "D", (0,(4.8,1.4,1,1.4))),
}



def configure_fonts():
    for path in Path("/System/Library/Fonts/Supplemental").glob("Times New Roman*.ttf"):
        font_manager.fontManager.addfont(str(path))
    available = {item.name for item in font_manager.fontManager.ttflist}
    face = next(name for name in ("Times New Roman","TeX Gyre Termes","Liberation Serif","STIXGeneral")
                if name in available)
    plt.rcParams.update({"font.family":face,"font.size":9,"pdf.fonttype":42,
                         "svg.fonttype":"none","svg.hashsalt":"eecs182-principal-fig3",
                         "mathtext.fontset":"stix","axes.labelsize":9,
                         "xtick.labelsize":8,"ytick.labelsize":8,
                         "text.color":"#18282d","axes.labelcolor":"#18282d"})


def summarize(data):
    groups=["model","aggregation","query_data","k","context_source","session","off_reference","am_freq_hz"]
    per_session=data.groupby(groups)[FIELDS].mean()
    per_session["n_draws"]=data.groupby(groups).size()
    per_session=per_session.reset_index()
    out=[]
    for freq in ("pooled",3,20):
        use=per_session if freq=="pooled" else per_session[per_session.am_freq_hz.eq(freq)]
        days=sorted(use.off_reference.unique())
        plan=np.random.default_rng(BOOT_SEED).integers(len(days),size=(BOOT_REPS,len(days)))
        for (model,group,variant,k,context),rows in use.groupby(groups[:5]):
            daily=rows.groupby("off_reference")
            sums=daily[FIELDS].sum().reindex(days)
            size=daily.size().reindex(days)
            assert not sums.isna().any().any() and not size.isna().any()
            assert rows.n_draws.nunique()==1
            boot=sums.to_numpy()[plan].sum(axis=1)/size.to_numpy()[plan].sum(axis=1)[:,None]
            low,high=np.percentile(boot,[2.5,97.5],axis=0)
            for index,field in enumerate(FIELDS):
                out.append(dict(model=model,aggregation=group,frequency_hz=freq,
                                query_data=variant,k=k,context_source=context,metric=field,
                                value=rows[field].mean(),ci_low=low[index],ci_high=high[index],
                                n_sessions=len(rows),n_days=len(days),n_draws=int(rows.n_draws.iloc[0]),
                                day_ids="|".join(days),n_boot=BOOT_REPS,bootstrap_seed=BOOT_SEED,
                                mask="common",test_status="exploratory"))
    return per_session,pd.DataFrame(out)


def marker_line(model, label=False):
    name,color,marker,style=STYLE[model]
    return Line2D([0],[0],lw=1.05,linestyle=style,color=color,marker=marker,markersize=3.45,
                  markeredgewidth=.55,label=name if label else None)


def plot(summary,frequency,stem,out):
    models=[model for model in POTENTIAL if model in set(summary.model)]
    subset=summary[summary.frequency_hz.eq(frequency)]
    fig=plt.figure(figsize=(5.5,3.87),facecolor="white")
    fig.legend(handles=[marker_line(model,label=True) for model in models],
               loc="upper center",bbox_to_anchor=(.513,.982),ncol=3,frameon=False,
               borderaxespad=0,labelspacing=.39,fontsize=8.0,handlelength=1.7,
               handletextpad=.38,columnspacing=1.15)
    for i,(variant,row_label) in enumerate(zip(INPUTS,ROW_LABELS)):
        y=(.540,.132)[i]
        fig.text(.515,y+.281,row_label,ha="center",fontsize=9.1)
        for j,(metric,label) in enumerate(METRICS.items()):
            left=.125+j*.305
            ax=fig.add_axes((left,y,.220,.236))
            values=summary[summary.metric.eq(metric+"_mean")]
            lo,hi=values.ci_low.min(),values.ci_high.max()
            pad=(hi-lo)*.055
            ax.set_ylim(lo-pad,hi+pad)
            points=subset[subset.query_data.eq(variant)&subset.metric.eq(metric+"_mean")]
            for model in models:
                _,color,marker,style=STYLE[model]
                series=points[points.model.eq(model)].set_index("k").reindex(KS)
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
            if metric=="r2":
                ax.set_yticks([-.05,.10,.25,.40])
                ax.axhline(0,color="#a7b2b2",lw=.52,linestyle=":",zorder=1)
            else:
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
        fig.savefig(out/f"{stem}.{extension}",dpi=600,facecolor="white",metadata=metadata)
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
    draws = read("matched_per_draw.csv")
    _, summary = summarize(draws)
    configure_fonts()
    for freq, stem in (("pooled", "figure3"), (3, "figure3_3hz"), (20, "figure3_20hz")):
        plot(summary, freq, stem, out)


if __name__ == "__main__":
    main()
