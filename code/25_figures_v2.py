"""Figure 1 (v2): what BOLD adds, per stage, in two preprocessing streams, and as BOLD quality grows.

Left: within-subject one-vs-rest AUC per stage for BOLD, the non-BOLD model and both (categorical
slots 1-3, validated all-pairs; shapes as secondary encoding), with time of night alone as a neutral
grey reference tick. Right: gain from adding BOLD to the non-BOLD model, within-subject mean with
95% subject-bootstrap CI, for the AFNI derivative (filled) and our minimal pipeline (open); the
non-superiority margin +/-0.03 is shaded; numbers are subjects with a positive gain. Panel c: on identical
epochs and an identical non-BOLD model (28_matched.py), the N3 and REM gains for four BOLD feature sets
ordered by quality; same colour rule as b (aqua = CI above zero), stage by marker and direct label.

    source <venv>/bin/activate
    python 25_figures_v2.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = Path(__file__).parent
OUT = BASE / "paper_isbi" / "figures"
STAGES = ["W", "N1", "N2", "N3", "REM"]
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"   # validated (dataviz validator)
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
MUTED = "#9a9893"     # own pipeline: secondary, so it does not compete with the derivative
DELTA = 0.03
plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def main():
    A = json.load(open(BASE / "results_v2_afni.json"))
    Rw = json.load(open(BASE / "results_v2_raw.json"))
    MT = json.load(open(BASE / "results_matched.json"))
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(7.0, 1.32), gridspec_kw=dict(width_ratios=[0.84, 0.88, 0.92]))   # draft 19: brain panel dropped (in the text as a number)
    ys = list(range(len(STAGES)))[::-1]
    auc = A["auc_ws"]

    # ---- left: AUC per stage
    series = [("BOLD", "bold", BLUE, "o"), ("Non-BOLD", "nonbold", ORANGE, "s"),
              ("Both", "nonbold+bold", VIOLET, "D")]   # green is reserved for "CI above zero"
    off = {"bold": 0.22, "nonbold": 0.0, "nonbold+bold": -0.22}   # separated enough that REM markers do not overlap
    for y, st in zip(ys, STAGES):
        ax.plot([auc["clock"][st]["mean"]] * 2, [y - 0.32, y + 0.32], color=INK2, lw=1.4, solid_capstyle="butt", zorder=2)
    for name, key, col, mk in series:
        ax.scatter([auc[key][st]["mean"] for st in STAGES], [y + off[key] for y in ys], s=24, marker=mk,
                   color=col, edgecolor="white", linewidth=0.7, zorder=3, label=name)
    ax.plot([], [], color=INK2, lw=0, marker="|", markersize=7, markeredgewidth=1.4, label="Time alone")
    ax.axvline(0.5, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.set_yticks(ys, STAGES)
    ax.set_xlim(0.47, 0.95)
    ax.set_ylim(-0.6, 4.75)
    ax.set_xlabel("Within-subject one-vs-rest AUC")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=7,
              handletextpad=0.3, columnspacing=0.8, borderaxespad=0.1)
    ax.text(-0.1, 1.2, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    # ---- right: increment forest
    bx.axvline(DELTA, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)   # the one-sided non-superiority margin
    bx.axvline(0, color=INK2, lw=0.6, zorder=1)
    for res, dy, filled, lab in [(A, 0.19, True, "Derivative (12)"), (Rw, -0.19, False, "Minimal pipeline (16)")]:
        inc = res["increment"]["nonbold+bold - nonbold"]
        for y, st in zip(ys, STAGES):
            r = inc[st]
            lo, hi = r["ws_ci"]
            sig = lo > 0
            col = AQUA if sig else (INK2 if filled else MUTED)
            bx.plot([lo, hi], [y + dy] * 2, color=col, lw=1.3 if filled else 0.9, solid_capstyle="round", zorder=2)
            bx.scatter([r["ws_mean"]], [y + dy], s=20 if filled else 14, marker="D", zorder=3, linewidth=0.9,
                       facecolor=col if filled else "white", edgecolor=col)
            bx.text(0.238 if filled else 0.306, y, f"{r['n_pos']}/{r['n_subj']}", va="center", ha="right", fontsize=7,
                    color=INK if filled else INK2, bbox=dict(facecolor="white", edgecolor="none", pad=0.3))   # side by side: derivative | minimal
        bx.scatter([], [], s=20 if filled else 14, marker="D", facecolor=INK2 if filled else "white",
                   edgecolor=INK2 if filled else MUTED, label=lab)
    bx.set_yticks(ys, STAGES)
    bx.set_xlim(-0.075, 0.308)
    bx.set_ylim(-0.6, 4.75)
    bx.set_xlabel(r"$\Delta$AUC, BOLD added to non-BOLD")
    bx.grid(axis="x", color=GRID, lw=0.5)
    bx.set_axisbelow(True)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    bx.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, frameon=False, fontsize=7,
              handletextpad=0.2, columnspacing=0.9, borderaxespad=0.1)
    bx.text(0.238, 4.42, "deriv.", ha="right", va="bottom", fontsize=7, color=INK2)    # n > 0 per pipeline
    bx.text(0.306, 4.42, "min.", ha="right", va="bottom", fontsize=7, color=INK2)
    bx.set_xticks([0.0, 0.1, 0.2])
    bx.text(DELTA + 0.004, 4.52, r"$\delta$", ha="left", va="center", fontsize=7, color=INK2)
    bx.text(-0.04, 1.2, "b", transform=bx.transAxes, fontsize=9, fontweight="bold", va="top")

    # ---- c: gain vs BOLD quality, identical epochs and non-BOLD model
    levels = [("raw84amp", "min."), ("rv84amp", "realign."), ("rv84phys", "+ phys."), ("afni84", "deriv."),
              ("afni", "200+FC"), ("raw", "min.+FC")]
    # the first three form the path amplitude -> derivative amplitude -> derivative 200+FC; "Own 84+FC" adds
    # FC to our pipeline, off that path, so it stands apart behind a separator
    xs = [0, 1, 2, 3, 4, 5.3]
    NP_ = 5                                    # steps on the path: minimal -> realigned -> + physiology -> derivative -> 200+FC
    cx.axhline(DELTA, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    cx.text(4.42, DELTA + 0.004, r"$\delta$", fontsize=7, color=INK2, ha="left", va="bottom")
    cx.axhline(0, color=INK2, lw=0.6, zorder=1)
    cx.axvline(4.65, color=INK2, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    for st, mk, ls, dx in [("N3", "D", "-", -0.07), ("REM", "o", (0, (2, 1.5)), 0.07)]:
        ms = [MT["increment"][k][st]["ws_mean"] for k, _ in levels]
        cx.plot([xs[j] + dx for j in range(NP_)], ms[:NP_], color=INK2, lw=0.9, ls=ls, zorder=2)
        for x, (k, _) in zip(xs, levels):
            r = MT["increment"][k][st]
            lo, hi = r["ws_ci"]
            col = AQUA if lo > 0 else INK2
            cx.plot([x + dx] * 2, [lo, hi], color=col, lw=1.3, solid_capstyle="round", zorder=3)
            cx.scatter([x + dx], [r["ws_mean"]], s=20, marker=mk, color=col, edgecolor="white", linewidth=0.6, zorder=4)
        if st == "N3":
            cx.text(4 + 0.2, ms[4] + 0.012, st, va="center", ha="left", fontsize=7, color=INK)
        else:
            cx.text(3.55, -0.047, st, va="center", ha="center", fontsize=7, color=INK)
    cx.set_xticks(xs, [l for _, l in levels], fontsize=7, rotation=30, ha="right", rotation_mode="anchor")
    cx.tick_params(axis="x", pad=1.5)
    cx.set_xlim(-0.4, 5.6)
    cx.set_ylim(-0.07, 0.185)

    cx.grid(axis="y", color=GRID, lw=0.5)
    cx.set_axisbelow(True)
    for sp in ("top", "right"):
        cx.spines[sp].set_visible(False)
    cx.set_title(r"Same epochs, $\Delta$AUC", fontsize=7, color=INK, loc="left", pad=3)
    cx.text(-0.22, 1.2, "c", transform=cx.transAxes, fontsize=9, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.27, top=0.77, wspace=0.42)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_gain_v2.pdf", dpi=600)       # embeds the brain renders at print resolution
    fig.savefig(OUT / "fig_gain_v2.png", dpi=300)
    print("saved fig_gain_v2.pdf/png")


if __name__ == "__main__":
    main()
