#!/usr/bin/env python3
"""Plot a median-selected query example and matched frozen Transformer controls."""
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

HERE=Path(__file__).resolve().parent

INPUTS=("recorded_on","parrm_amp10")
INPUT_TO_OLD={"recorded_on":"recorded","parrm_amp10":"amp10"}
CONTROLS=("own","rotated","off","labels")
METRICS={"r2":r"$R^2$","pearson_r":"Pearson’s r","rmse_cm_s":"RMSE (cm/s)"}
FIELDS=[f"{m}_{a}" for m in METRICS for a in ("vx","vy","mean")]
BOOT_REPS,BOOT_SEED=10000,18203
STYLE={
    "own":("Correct ON","#009E73","o"),
    "rotated":("Rotated ON LFP","#626d7d","^"),
    "off":("Same-day OFF","#4c6170","s"),
    "labels":("Labels-only","#414141","D"),
}
TRACE={
    "actual":("Actual","#18282d","-"),
    "own":("Final Transformer ensemble","#009E73",(0,(5,1.5,1,1.5))),
    "labels":("Labels-only","#4c4c4c",(0,(3.3,2.1))),
}



def configure_fonts():
    for path in Path("/System/Library/Fonts/Supplemental").glob("Times New Roman*.ttf"):
        font_manager.fontManager.addfont(str(path))
    available={font.name for font in font_manager.fontManager.ttflist}
    face=next(name for name in ("Times New Roman","TeX Gyre Termes","Liberation Serif","STIXGeneral")
              if name in available)
    plt.rcParams.update({"font.family":face,"font.size":9,"pdf.fonttype":42,
                         "svg.fonttype":"none","svg.hashsalt":"eecs182-f4","mathtext.fontset":"stix",
                         "axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,"text.color":"#18282d"})


def interval(data, day="off_reference"):
    """Percentile whole-day bootstrap with equal session weighting."""
    grouped=data.groupby(level=day)
    sums=grouped.sum().to_numpy()
    sizes=grouped.size().to_numpy()
    choices=np.random.default_rng(BOOT_SEED).integers(len(sizes),size=(BOOT_REPS,len(sizes)))
    resample=sums[choices].sum(axis=1)/sizes[choices].sum(axis=1)[:,None]
    lo,hi=np.percentile(resample,[2.5,97.5],axis=0)
    return data.mean().to_numpy(),lo,hi


def aggregate(table,controls,fields):
    keyed=["session","off_reference"]
    result=[]
    for condition,group in table.groupby("query_data"):
        raw=group.pivot(index=keyed,columns="control",values=fields)
        assert len(raw)==14 and not raw.isna().any().any()
        for ctrl in controls:
            level=raw.xs(ctrl,axis=1,level="control")[fields]
            estimate={"condition":level}
            if ctrl!="own" and "own" in controls:
                estimate["control minus correct ON"]=level-raw.xs("own",axis=1,level="control")[fields]
            for kind,values in estimate.items():
                point,lo,hi=interval(values)
                for field,y,a,b in zip(fields,point,lo,hi):
                    result.append(dict(query_data=condition,control=ctrl,estimate_type=kind,metric=field,
                                       value=y,ci_low=a,ci_high=b,n_sessions=len(raw),
                                       n_days=group.off_reference.nunique(),bootstrap_seed=BOOT_SEED,
                                       n_boot=BOOT_REPS))
    return pd.DataFrame(result)


def decorate(ax,letter):
    ax.spines[["top","right"]].set_visible(False)
    for edge in ("left","bottom"):
        ax.spines[edge].set_linewidth(.55)
        ax.spines[edge].set_color("#617172")
    ax.tick_params(length=2.5,width=.55,pad=2)
    ax.text(-.09,1.065,letter,transform=ax.transAxes,weight="bold",fontsize=10.5)


def render(out,trace,summary):
    fig=plt.figure(figsize=(5.5,3.65),facecolor="white")
    top=[Line2D([0],[0],label=label,color=color,linestyle=style,lw=1.1)
         for label,color,style in TRACE.values()]
    fig.legend(handles=top,loc="upper center",bbox_to_anchor=(.53,.983),ncol=3,
               frameon=False,borderaxespad=0,fontsize=8,handlelength=1.9,
               columnspacing=1.2,handletextpad=.45)
    limit=np.abs(trace.loc[trace.valid,"velocity_cm_s"]).max()*1.08
    for j,axis in enumerate(("vx","vy")):
        ax=fig.add_axes((.123+j*.452,.624,.363,.248))
        for name,(_,color,style) in TRACE.items():
            rows=trace[trace.axis.eq(axis)&trace.series.eq(name)]
            ax.plot(rows.time_from_go_s,rows.velocity_cm_s.where(rows.valid),color=color,
                    ls=style,lw=1.07)
        ax.axvline(0,color="#8f9999",ls=":",lw=.58,zorder=-1)
        ax.set(xlim=(-.5,1.5),xticks=[-.5,0,.5,1,1.5],ylim=(-limit,limit))
        ax.set_ylabel(r"$v_x$ (cm/s)" if j==0 else r"$v_y$ (cm/s)",labelpad=2)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        decorate(ax,"AB"[j])
    fig.text(.530,.522,"Time from go (s)",ha="center",fontsize=9)

    handles=[Line2D([0],[0],label=label,color=color,marker=marker,linestyle="None",
                    markersize=3.45,markeredgewidth=.55) for label,color,marker in STYLE.values()]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.54,.445),ncol=4,
               frameon=False,borderaxespad=0,fontsize=8,handlelength=1.0,
               columnspacing=.98,handletextpad=.38)
    for j,(metric,label) in enumerate(METRICS.items()):
        ax=fig.add_axes((.124+j*.305,.107,.218,.246))
        vals=summary[summary.metric.eq(metric+"_mean") & summary.estimate_type.eq("condition")]
        for g,condition in enumerate(INPUTS):
            for d,ctrl in enumerate(CONTROLS):
                item=vals[vals.query_data.eq(condition)&vals.control.eq(ctrl)].iloc[0]
                _,color,marker=STYLE[ctrl]
                x=g+(d-1.5)*.185
                ax.vlines(x,item.ci_low,item.ci_high,color=color,lw=.73,alpha=.87)
                ax.plot(x,item.value,color=color,marker=marker,markersize=3.55,
                        markeredgewidth=.50,ls="None")
        lo,hi=vals.ci_low.min(),vals.ci_high.max()
        pad=(hi-lo)*.07
        ax.set(xlim=(-.44,1.44),ylim=(lo-pad,hi+pad),xticks=[0,1])
        ax.set_xticklabels(["Recorded","10× artifact"],fontsize=8)
        ax.set_ylabel(label,labelpad=2)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="y",color="#e7eaea",lw=.48)
        ax.set_axisbelow(True)
        decorate(ax,"CDE"[j])
    fig.canvas.draw()
    bbox=fig.get_tightbbox(fig.canvas.get_renderer())
    assert -.01<=bbox.x0<bbox.x1<=5.51 and -.01<=bbox.y0<bbox.y1<=3.66,bbox
    for ext in ("pdf","svg","png"):
        meta=({"CreationDate":None,"ModDate":None} if ext=="pdf" else {"Date":None} if ext=="svg" else None)
        fig.savefig(out/f"figure4.{ext}",dpi=600,metadata=meta,facecolor="white")


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
    trace = read("trace_source_data.csv")
    controls = read("control_per_session.csv")
    summary = aggregate(controls, CONTROLS, FIELDS)
    configure_fonts()
    render(out, trace, summary)


if __name__ == "__main__":
    main()
