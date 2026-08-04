"""Figure 1: what the harness fixes, and what the controller is allowed to see."""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs"
FIXED, USER, EDGE, WARN = "#dbe5f1", "#f7e2c3", "#2f4356", "#a93226"


def box(ax, x, y, w, h, title, lines, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                                lw=1.1, edgecolor=EDGE, facecolor=fill))
    ax.text(x + w / 2, y + h - 0.075, title, ha="center", va="center",
            fontsize=9.5 if len(title) < 16 else 8.6, fontweight="bold", color=EDGE)
    step = 0.075
    top = y + h - 0.075 - 0.085
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, top - step * i, ln, ha="center",
                va="center", fontsize=7.3, color=EDGE)


def arrow(ax, p, q, ls="-", color=EDGE, lw=1.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", linestyle=ls,
                                 mutation_scale=12, color=color, lw=lw,
                                 shrinkA=1, shrinkB=1))


def main():
    fig, ax = plt.subplots(figsize=(10.6, 3.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    C = [0.000, 0.215, 0.420, 0.625, 0.845]
    W = [0.175, 0.170, 0.170, 0.175, 0.155]

    box(ax, C[0], 0.545, W[0], 0.40, "Plant",
        ["planar · spatial · dual", "checked against an",
         "independent derivation"], FIXED)
    box(ax, C[0], 0.055, W[0], 0.40, "Disturbance",
        ["Kaimal · Dryden", "exact variance, on a grid",
         "independent of the step"], FIXED)
    box(ax, C[1], 0.300, W[1], 0.40, "Uncertainty design",
        ["centred Latin hypercube", "drawn once and replayed",
         "for every controller"], FIXED)
    box(ax, C[2], 0.300, W[2], 0.40, "Controller",
        ["your code", "reset(plant, manoeuvre)", "__call__(t, x) → u"], USER)
    box(ax, C[3], 0.545, W[3], 0.40, "Metrics",
        ["frozen module; its source", "hash is written into",
         "every result file"], FIXED)
    box(ax, C[3], 0.055, W[3], 0.40, "Ledger",
        ["every seed, source hash,", "solver setting and",
         "library version"], FIXED)
    box(ax, C[4], 0.300, W[4], 0.40, "Paired contrast",
        ["bootstrap CI · Wilcoxon", "rank-biserial · McNemar"], FIXED)

    arrow(ax, (C[0] + W[0], 0.745), (C[1], 0.610))
    arrow(ax, (C[0] + W[0], 0.255), (C[1], 0.390))
    arrow(ax, (C[1] + W[1], 0.500), (C[2], 0.500))
    arrow(ax, (C[2] + W[2], 0.560), (C[3], 0.700))
    arrow(ax, (C[2] + W[2], 0.440), (C[3], 0.300))
    arrow(ax, (C[3] + W[3], 0.700), (C[4], 0.570))
    arrow(ax, (C[3] + W[3], 0.300), (C[4], 0.430))

    # the barrier: the controller's window on the experiment
    xc = C[2] + W[2] / 2
    ax.add_patch(FancyBboxPatch((C[2] - 0.016, 0.284), W[2] + 0.032, 0.432,
                                boxstyle="square,pad=0", lw=1.3, ls=(0, (4, 3)),
                                edgecolor=WARN, facecolor="none"))
    ax.annotate("sees the state and the reference, and nothing else:\n"
                "not the sampled parameters, not the wind\n"
                "record, not the metric module",
                xy=(xc, 0.720), xytext=(xc, 0.985), ha="center", va="top",
                fontsize=7.8, color=WARN,
                arrowprops=dict(arrowstyle="-", color=WARN, lw=0.9))

    ax.text(0.5, 0.012, "blue: fixed by the benchmark     amber: supplied by the user",
            ha="center", fontsize=8, color=EDGE)
    fig.savefig(OUT / "fig0_architecture.png", dpi=200, bbox_inches="tight")
    print("written", OUT / "fig0_architecture.png")


if __name__ == "__main__":
    main()
