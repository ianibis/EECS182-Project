"""Draw the synthetic tone augmentation used by the EEGNet + tones baseline.

The top panel is the operation; the lower panels show a real stimulation-off
training trial before and after a 3 Hz or 20 Hz tone is added.  The 20 Hz trace
uses the exact draw from ``task4.utils.add_stimulation_tones(train, seed=0)``;
the 3 Hz trace is the same trial with an illustrative low-frequency tone.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task4 import utils as U  # noqa: E402

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "figures"
)
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
    }
)

SESSION = U.STIMULATION_OFF_TRAINING_SESSIONS[0]
TRIAL = 0
CHANNEL = 28
FS = U.SAMPLE_RATE_HZ

dataset = U.load_lfp_dataset(SESSION, None)
clean = dataset["features"][TRIAL, CHANNEL].astype(np.float64)
n_samples = clean.size
t = np.arange(n_samples) / FS - 0.5  # seconds relative to the go cue
tone_time = np.arange(n_samples) / FS  # the code's time base

# Exact augmented trace for this trial and channel: the RNG starts at seed 0.
single = {key: value[TRIAL : TRIAL + 1] for key, value in dataset.items()}
actual = U.add_stimulation_tones(single, seed=0)["features"][0, CHANNEL].astype(
    np.float64
)
actual_tone = actual - clean

# Illustrative 3 Hz counterpart on the same trial (the real draw is 20 Hz).
A3, G3, PHI3 = 1.5, 1.0, 0.0
tone3 = A3 * G3 * np.sin(2 * np.pi * 3.0 * tone_time + PHI3)

# Recover the actual first draw so the annotation can report it.
gen = np.random.default_rng(0)
f_actual = float(gen.choice(U.STIMULATION_FREQ_HZ))
a_actual = float(gen.uniform(*U.STIMULATION_FREQ_MAG))
phi_actual = float(gen.uniform(0, 2 * np.pi))
n_channels = dataset["features"].shape[1]
gain_actual = float(gen.uniform(0.5, 1.5, size=(n_channels, 1))[CHANNEL, 0])

fig = plt.figure(figsize=(10.0, 7.2))
gs = fig.add_gridspec(
    3,
    2,
    height_ratios=[0.72, 1.0, 1.0],
    left=0.065,
    right=0.975,
    top=0.965,
    bottom=0.075,
    hspace=0.45,
    wspace=0.22,
)

# ---------------------------------------------------------------------------
# (a) Operation
# ---------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title(
    "(a) Synthetic tone augmentation: one draw per training trial",
    loc="left",
    fontweight="bold",
    pad=4,
)

box_style = dict(
    boxstyle="round,pad=0.0,rounding_size=0.025", linewidth=1.0, facecolor="white"
)
clean_box = FancyBboxPatch((0.015, 0.46), 0.235, 0.36, edgecolor="0.45", **box_style)
tone_box = FancyBboxPatch((0.305, 0.46), 0.245, 0.36, edgecolor="#d95f02", **box_style)
aug_box = FancyBboxPatch((0.615, 0.46), 0.285, 0.36, edgecolor="#1f77b4", **box_style)
param_box = FancyBboxPatch(
    (0.015, 0.0), 0.885, 0.40, edgecolor="0.65", linestyle="--", **box_style
)
for patch in (clean_box, tone_box, aug_box, param_box):
    ax.add_patch(patch)

ax.text(0.132, 0.755, "Clean training trial", ha="center", va="center", fontsize=8.5)
ax.text(0.132, 0.645, r"$x_c(t)$", ha="center", va="center", fontsize=12)
ax.text(
    0.132,
    0.525,
    r"$35\times2{,}000$",
    ha="center",
    va="center",
    fontsize=7.5,
    color="0.35",
)

ax.text(
    0.428,
    0.755,
    "Synthetic tone",
    ha="center",
    va="center",
    fontsize=8.5,
    color="#a04000",
)
ax.text(
    0.428,
    0.635,
    r"$A\,g_c\,\sin(2\pi f t+\phi)$",
    ha="center",
    va="center",
    fontsize=10.5,
    color="#a04000",
)
ax.text(
    0.428,
    0.525,
    r"$f\in\{3,20\}$ Hz",
    ha="center",
    va="center",
    fontsize=7.5,
    color="#a04000",
)

ax.text(0.758, 0.755, "Augmented trial", ha="center", va="center", fontsize=8.5)
ax.text(
    0.758,
    0.635,
    r"$x'_c(t)=x_c(t)+\mathrm{tone}$",
    ha="center",
    va="center",
    fontsize=10.5,
)
ax.text(
    0.758,
    0.525,
    r"$35\times2{,}000$",
    ha="center",
    va="center",
    fontsize=7.5,
    color="0.35",
)

ax.text(0.272, 0.640, r"$+$", ha="center", va="center", fontsize=18, color="0.25")
ax.text(0.584, 0.640, r"$=$", ha="center", va="center", fontsize=18, color="0.25")

ax.text(
    0.457,
    0.345,
    "Draw per training trial (one tone per trial)",
    ha="center",
    va="center",
    fontsize=8.0,
    fontweight="bold",
)
ax.text(
    0.250,
    0.255,
    r"$f\sim\mathrm{Uniform}\{3,20\}$ Hz",
    ha="center",
    va="center",
    fontsize=7.7,
)
ax.text(
    0.250,
    0.165,
    r"$A\sim\mathrm{Uniform}(0.5,3.0)$ z",
    ha="center",
    va="center",
    fontsize=7.7,
)
ax.text(
    0.665,
    0.255,
    r"$\phi\sim\mathrm{Uniform}(0,2\pi)$",
    ha="center",
    va="center",
    fontsize=7.7,
)
ax.text(
    0.665,
    0.165,
    r"$g_c\sim\mathrm{Uniform}(0.5,1.5)$ per channel",
    ha="center",
    va="center",
    fontsize=7.7,
)
ax.text(
    0.457,
    0.065,
    "Applied only to stimulation-off training trials; the same augmented set is reused for all five seeds.",
    ha="center",
    va="center",
    fontsize=7.0,
    color="0.35",
)

# ---------------------------------------------------------------------------
# Signal panels
# ---------------------------------------------------------------------------
display_t = -0.5, 1.5
go_kw = dict(color="0.55", linewidth=0.8, linestyle=":")
clean_kw = dict(color="0.45", linewidth=0.8, alpha=0.9)


def setup_signal_axis(ax, title, xlim=display_t):
    ax.set_xlim(*xlim)
    ax.set_title(title, loc="left")
    ax.set_xlabel("time from go cue (s)")
    ax.set_ylabel("robust $z$")
    ax.axvline(0, **go_kw)
    ax.grid(axis="y", color="0.92", linewidth=0.5)
    ax.set_axisbelow(True)


ax_clean = fig.add_subplot(gs[1, 0])
setup_signal_axis(ax_clean, "(b) Clean trial from the training split")
ax_clean.plot(t, clean, color="0.30", linewidth=0.75)
ax_clean.text(
    0.98,
    0.04,
    f"{SESSION} · trial {TRIAL} · channel {CHANNEL}",
    transform=ax_clean.transAxes,
    ha="right",
    va="bottom",
    fontsize=6.6,
    color="0.35",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
)
ax_clean.text(
    0.255, 0.94, "go cue", transform=ax_clean.transAxes, color="0.45", fontsize=6.8
)

ax3 = fig.add_subplot(gs[1, 1])
setup_signal_axis(ax3, "(c) Same trial + illustrative 3 Hz tone")
ax3.plot(t, clean, **clean_kw)
ax3.plot(t, clean + tone3, color="#1f77b4", linewidth=0.8, label="augmented")
ax3.plot(
    t,
    tone3,
    color="#1f77b4",
    linewidth=0.7,
    linestyle="--",
    alpha=0.55,
    label="added tone (offset 0)",
)
ax3.legend(
    loc="lower right",
    frameon=True,
    framealpha=0.88,
    edgecolor="none",
    ncol=2,
    handlelength=1.4,
    columnspacing=0.8,
    borderaxespad=0.2,
)
ax3.text(
    0.02,
    0.96,
    r"$A=1.5$ z, $g_c=1.0$, $f=3$ Hz",
    transform=ax3.transAxes,
    ha="left",
    va="top",
    fontsize=7.0,
    color="#1f77b4",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.5),
)

ax20 = fig.add_subplot(gs[2, 0])
setup_signal_axis(ax20, "(d) Actual draw for this trial: 20 Hz tone")
ax20.plot(t, clean, **clean_kw)
ax20.plot(t, actual, color="#d62728", linewidth=0.8, label="augmented")
ax20.plot(
    t,
    actual_tone,
    color="#d62728",
    linewidth=0.7,
    linestyle="--",
    alpha=0.55,
    label="added tone (offset 0)",
)
ax20.legend(
    loc="lower right",
    frameon=True,
    framealpha=0.88,
    edgecolor="none",
    ncol=2,
    handlelength=1.4,
    columnspacing=0.8,
    borderaxespad=0.2,
)
ax20.text(
    0.02,
    0.96,
    rf"$f={f_actual:.0f}$ Hz, $A={a_actual:.2f}$ z, $\phi={phi_actual:.2f}$ rad, $g_c={gain_actual:.2f}$",
    transform=ax20.transAxes,
    ha="left",
    va="top",
    fontsize=7.0,
    color="#b22222",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.5),
)


# ---------------------------------------------------------------------------
# Zoom: the tone is the only difference
# ---------------------------------------------------------------------------
axz = fig.add_subplot(gs[2, 1])
zoom = (t >= -0.10) & (t <= 0.15)
setup_signal_axis(axz, "(e) Zoom on 0.25 s: clean vs augmented", xlim=(-0.10, 0.15))
axz.plot(t[zoom], clean[zoom], color="0.30", linewidth=0.9, label="clean")
axz.plot(t[zoom], (clean + tone3)[zoom], color="#1f77b4", linewidth=1.0, label="+ 3 Hz")
axz.plot(t[zoom], actual[zoom], color="#d62728", linewidth=1.0, label="+ 20 Hz")
axz.legend(
    loc="lower right",
    frameon=True,
    framealpha=0.88,
    edgecolor="none",
    ncol=1,
    handlelength=1.2,
    borderaxespad=0.2,
)
axz.set_ylim(
    min(clean[zoom].min(), actual[zoom].min(), (clean + tone3)[zoom].min()) - 0.4,
    max(clean[zoom].max(), actual[zoom].max(), (clean + tone3)[zoom].max()) + 0.4,
)

fig.savefig(os.path.join(OUT, "tone_augmentation.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "tone_augmentation.pdf"), bbox_inches="tight")
print("wrote", os.path.join(OUT, "tone_augmentation.png"))
print("wrote", os.path.join(OUT, "tone_augmentation.pdf"))
