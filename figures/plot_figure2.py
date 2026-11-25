#!/usr/bin/env python3
"""Assemble measured PARRM QC and frozen ridge results without refitting."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator

HERE = Path(__file__).resolve().parent
RIDGE = "ridge_task3_off_parrm"
OFF, ON, CLEAN, AMP = "recorded_off", "recorded_on", "parrm_cleaned_on", "parrm_amp10"
CONDITIONS = (OFF, ON, CLEAN)
LABEL = {OFF:"OFF",ON:"Recorded ON",CLEAN:"PARRM ON",AMP:"10× artifact"}
COLOR = {OFF:"#232a2c",ON:"#8c472b",CLEAN:"#20665e",AMP:"#5e4a79"}
STYLE = {OFF:(0,(1.1,1.6)),ON:"-",CLEAN:(0,(5,2.4)),AMP:(0,(5,2,1.1,2))}
MARKER = {OFF:"o",ON:"s",CLEAN:"^"}
METRICS = {"r2":"R²", "pearson_r":"Pearson’s r", "rmse_cm_s":"RMSE (cm/s)"}
FIELDS = [f"{key}_{axis}" for key in METRICS for axis in ("vx","vy","mean")]
SEED, N_BOOT = 18202, 10000
DURATION, SPECTRUM_MAX = 1.0, 150



def configure_fonts():
    for path in Path("/System/Library/Fonts/Supplemental").glob("Times New Roman*.ttf"):
        font_manager.fontManager.addfont(str(path))
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next(name for name in ("Times New Roman","TeX Gyre Termes","Liberation Serif","STIXGeneral")
                  if name in available)
    plt.rcParams.update({"font.family":family,"font.size":9,"axes.labelsize":9,
                         "xtick.labelsize":8,"ytick.labelsize":8,"pdf.fonttype":42,
                         "svg.fonttype":"none","svg.hashsalt":"eecs182-figure2",
                         "mathtext.fontset":"stix","text.color":"#17282e"})


def summarize_pairs(pairs):
    """Use the same session-weighted recording-day bootstrap as Figure 3."""
    contrast = {"ON minus OFF":(ON,OFF), "PARRM minus OFF":(CLEAN,OFF),
                "PARRM minus ON":(CLEAN,ON)}
    index = ["on_session","reference_day","date","frequency_hz"]
    wide = pairs.pivot(index=index,columns="condition",values=FIELDS)
    points = {condition:wide.xs(condition,axis=1,level=1) for condition in CONDITIONS}
    for name,(a,b) in contrast.items():
        points[name] = points[a]-points[b]
    estimates=[]
    for freq in ("pooled",3,20):
        group=wide if freq=="pooled" else wide[wide.index.get_level_values("frequency_hz")==freq]
        days=sorted(group.index.get_level_values("reference_day").unique())
        draws=np.random.default_rng(SEED).integers(len(days),size=(N_BOOT,len(days)))
        for name,point in points.items():
            data=point.loc[group.index,FIELDS]
            by_day=data.groupby(level="reference_day")
            sums=by_day.sum().reindex(days).to_numpy()
            counts=by_day.size().reindex(days).to_numpy()
            samples=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)[:,None]
            lo,hi=np.percentile(samples,[2.5,97.5],axis=0)
            for field, val, a, b in zip(FIELDS,data.mean(),lo,hi):
                estimates.append(dict(frequency_hz=freq,series=name,metric=field,estimate=val,
                                      ci_low=a,ci_high=b,n_paired_sessions=len(data),n_days=len(days),
                                      n_boot=N_BOOT,bootstrap_seed=SEED,k=0,mask="common",
                                      test_status="exploratory"))
    return pd.DataFrame(estimates)


def save(fig,directory,name,dpi):
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    if not (-.01 <= box.x0 and box.x1 <= width+.01 and -.01 <= box.y0 and box.y1 <= height+.01):
        raise ValueError(f"Artwork crosses the page edge: {name}, {box.bounds}")
    for ext in ("pdf","svg","png"):
        meta=({"CreationDate":None,"ModDate":None} if ext=="pdf" else {"Date":None} if ext=="svg" else None)
        fig.savefig(directory/f"{name}.{ext}",facecolor="white",dpi=dpi,metadata=meta)
    plt.close(fig)


def decorate(ax, letter):
    ax.spines[["top","right"]].set_visible(False)
    for edge in ("left","bottom"):
        ax.spines[edge].set_linewidth(.55)
        ax.spines[edge].set_color("#56686b")
    ax.tick_params(length=2.5,width=.55,pad=2)
    ax.text(-.15,1.06,letter,transform=ax.transAxes,fontsize=10.5,fontweight="bold")


def legend(fig, conditions, y):
    fig.legend(handles=[Line2D([0],[0],color=COLOR[c],ls=STYLE[c],lw=1.12,label=LABEL[c])
                        for c in conditions],frameon=False,loc="upper center",bbox_to_anchor=(.54,y),
               borderaxespad=0,handletextpad=.45,handlelength=2.35,columnspacing=1.45,
               fontsize=8.3,ncol=len(conditions))


def trace_panel(ax, trace, hz, letter, limits, conditions=CONDITIONS):
    for key in conditions:
        rows=trace[(trace.frequency_hz==hz)&trace.condition.eq(key)]
        ax.plot(rows.time_from_go_s,rows.lfp_uv,color=COLOR[key],ls=STYLE[key],lw=.62)
    ax.set(xlim=(0,DURATION),ylim=limits,xticks=[0,.5,1])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4,symmetric=True))
    ax.set_xlabel("Time from go (s)",labelpad=1)
    ax.set_ylabel("LFP (µV)",labelpad=1)
    ax.set_title(f"{hz} Hz AM",fontsize=9,pad=3)
    decorate(ax,letter)


def psd_panel(ax, spectra, selection, hz, letter, limits):
    for key in CONDITIONS:
        rows=spectra[(spectra.frequency_hz==hz)&spectra.condition.eq(key)&spectra.displayed]
        ax.plot(rows.frequency_bin_hz,rows.psd_uv2_per_hz,color=COLOR[key],ls=STYLE[key],lw=.77)
    ax.set(xlim=(0,SPECTRUM_MAX),ylim=limits,yscale="log",xticks=[0,50,100,150])
    ax.set_xlabel("Frequency (Hz)",labelpad=1)
    ax.set_ylabel("PSD (µV²/Hz)",labelpad=1)
    ax.yaxis.set_major_locator(LogLocator(numticks=4))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.axvline(hz,color="#8c9999",lw=.56,linestyle=(0,(2,2)),zorder=-1)
    change=100*(1-selection.loc[selection.frequency_hz.eq(hz),"am_clean_over_raw"].iat[0])
    ax.text(.965,.79,f"AM bin: −{change:.1f}%",transform=ax.transAxes,ha="right",fontsize=8)
    decorate(ax,letter)


def decoding_panel(ax,pairs,summary,metric,letter):
    field=metric+"_mean"
    # The thin paths keep same-day pairing visible. They show day means, not resamples.
    daily=pairs.groupby(["reference_day","condition"])[field].mean().unstack().reindex(columns=CONDITIONS)
    for _,vals in daily.iterrows():
        ax.plot(range(3),vals,color="#a1aaab",alpha=.75,lw=.53,zorder=1)
    row=summary[(summary.frequency_hz=="pooled") & summary.metric.eq(field)].set_index("series")
    for i,key in enumerate(CONDITIONS):
        result=row.loc[key]
        y=result.estimate
        ax.errorbar(i,y,yerr=[[y-result.ci_low],[result.ci_high-y]],color=COLOR[key],fmt=MARKER[key],
                    markersize=4,elinewidth=.82,capsize=2,capthick=.75,zorder=3)
    ax.set_xlim(-.24,2.24)
    ax.set_xticks(range(3),["OFF","ON","PARRM"])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_title(METRICS[metric],fontsize=9,pad=3)
    decorate(ax,letter)


def render(dest,trace,spectra,selection,pairs,summary,dpi):
    main_bound=np.abs(trace[trace.condition.isin(CONDITIONS)].lfp_uv).max()*1.025
    amp_bound=np.abs(trace.lfp_uv).max()*1.025
    psd_values=spectra.loc[spectra.displayed,"psd_uv2_per_hz"]
    p_limits=(psd_values.min()*.8,psd_values.max()*1.35)
    tracelims=(-main_bound,main_bound)

    fig=plt.figure(figsize=(5.5,4.06))
    for i,hz in enumerate((3,20)):
        x=.114+i*.46
        wave = fig.add_axes((x,.727,.389,.175))
        spectrum = fig.add_axes((x,.430,.389,.19))
        trace_panel(wave,trace,hz,"AB"[i],tracelims)
        psd_panel(spectrum,spectra,selection,hz,"CD"[i],p_limits)
        if i:
            for ax in (wave,spectrum):
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
    for j,metric in enumerate(METRICS):
        decoding_panel(fig.add_axes((.115+j*.308,.090,.236,.228)),pairs,summary,metric,"EFG"[j])
    fig.text(.037,.204,"Decoding performance",rotation=90,va="center",ha="center")
    legend(fig,CONDITIONS,.984)
    save(fig,dest,"figure2",dpi)

    panels=dest/"panels"
    panels.mkdir(exist_ok=True)
    for i,hz in enumerate((3,20)):
        for spec in (False,True):
            single,ax=plt.subplots(figsize=(3.20,1.76))
            single.subplots_adjust(left=.205,bottom=.255,right=.95,top=.77)
            if spec:
                psd_panel(ax,spectra,selection,hz,"CD"[i],p_limits)
                ax.set_title(f"{hz} Hz AM",fontsize=9,pad=3)
            else:
                trace_panel(ax,trace,hz,"AB"[i],tracelims)
            legend(single,CONDITIONS,.974)
            save(single,panels,f"panel_{'CD'[i] if spec else 'AB'[i]}",dpi)
    for j,metric in enumerate(METRICS):
        single,ax=plt.subplots(figsize=(1.9,1.54))
        single.subplots_adjust(left=.26,bottom=.23,right=.94,top=.79)
        decoding_panel(ax,pairs,summary,metric,"EFG"[j])
        save(single,panels,f"panel_{'EFG'[j]}",dpi)

    extra=plt.figure(figsize=(5.5,1.85))
    extra_conditions=(ON,CLEAN,AMP)
    for i,hz in enumerate((3,20)):
        ax=extra.add_axes((.115+i*.46,.26,.39,.49))
        trace_panel(ax,trace,hz,"AB"[i],(-amp_bound,amp_bound),extra_conditions)
        if i:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
    legend(extra,extra_conditions,.98)
    save(extra,dest,"figure2_10x_traces",dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=HERE / "source_data")
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "figures")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    source, out = args.source_dir, args.output_dir
    def read(name):
        path = source / name
        if not path.is_file():
            parser.error(f"Missing original figure data: {path}")
        return pd.read_csv(path, float_precision="round_trip")
    out.mkdir(parents=True, exist_ok=True)
    trace = read("lfp_traces.csv")
    spectra = read("saved_spectra.csv")
    selection = read("signal_examples.csv")
    pairs = read("ridge_pairs.csv")
    summary = summarize_pairs(pairs)
    configure_fonts()
    render(out, trace, spectra, selection, pairs, summary, args.dpi)


if __name__ == "__main__":
    main()
