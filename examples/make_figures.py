"""Figures for the paper.  Reads the batched campaign files, writes PNGs."""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
OUT = ROOT / "docs"

from cranebench.reference import Manoeuvre                     # noqa: E402
from cranebench.runner import Campaign, run_single             # noqa: E402
from cranebench.stats import paired_summary                    # noqa: E402
from cranebench.wind import KaimalParams                       # noqa: E402
from cranebench.wind.kaimal import kaimal_psd, synthesise      # noqa: E402
from summarise_batch import load                               # noqa: E402

ORDER = ["PD", "LQR", "ZVD", "SMC", "HSMC"]
C = dict(zip(ORDER, plt.rcParams["axes.prop_cycle"].by_key()["color"]))


def fig1_verification():
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.3))
    p, dt = KaimalParams(), 0.01
    rec = synthesise(2 ** 17, dt, p, np.random.default_rng(2))
    f, pxx = welch(rec, fs=1 / dt, nperseg=4096)
    b = (f > 1e-2) & (f < 5)
    ax[0].loglog(f[b], pxx[b], lw=0.7, label="realised (Welch)")
    ax[0].loglog(f[b], kaimal_psd(f[b], p), lw=1.8, label="Kaimal target")
    r = np.corrcoef(np.log(pxx[b]), np.log(kaimal_psd(f[b], p)))[0, 1]
    ax[0].set(xlabel="frequency [Hz]", ylabel=r"$S_u$ [m$^2$/s]",
              title=f"(a) disturbance model, log-PSD $r$ = {r:.3f}")
    ax[0].legend(fontsize=8, frameon=False)

    camp = Campaign(plant="planar", wind="none", dt=1e-2,
                    manoeuvre=Manoeuvre(distance=20, t_ramp=20, t_total=40)).build()
    for name, c in camp.controllers.items():
        _, (t, X, U) = run_single(c, camp, None, 7)
        ax[1].plot(t, np.degrees(X[:, 2]), lw=1.0, color=C[name], label=name)
    for s in (1, -1):
        ax[1].axhline(s * 4.8, ls="--", lw=0.8, color="k")
    ax[1].set(xlabel="time [s]", ylabel="swing [deg]", ylim=(-5.6, 5.6),
              title="(b) nominal manoeuvre, no wind")
    ax[1].legend(fontsize=8, ncol=3, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_verification.png", dpi=180)


def fig2_campaign():
    res = load("reference")
    n = len(res["PD"]["ise_pos"])
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))

    ax[0].boxplot([res[c]["residual_swing"] for c in ORDER],
                  tick_labels=ORDER, showfliers=False)
    ax[0].set(ylabel="residual swing [deg]",
              title=f"(a) reference campaign, n = {n}")

    for c in ORDER:
        ax[1].scatter(np.mean(res[c]["chatter"]),
                      np.mean(res[c]["residual_swing"]), s=45, color=C[c])
        ax[1].annotate(c, (np.mean(res[c]["chatter"]),
                           np.mean(res[c]["residual_swing"])),
                       textcoords="offset points", xytext=(7, 2), fontsize=8)
    ax[1].set(xlabel="command total variation [N]",
              ylabel="residual swing [deg]",
              title="(b) roughness against residual swing")

    others = [c for c in ORDER if c != "PD"]
    y = np.arange(len(others))
    for j, k in enumerate(("residual_swing", "peak_swing")):
        s = [paired_summary(res[c][k], res["PD"][k]) for c in others]
        ax[2].errorbar([d["mean_diff"] for d in s], y + (j - .5) * .22,
                       xerr=[[d["mean_diff"] - d["ci_low"] for d in s],
                             [d["ci_high"] - d["mean_diff"] for d in s]],
                       fmt="o", ms=4, capsize=3, label=k)
    ax[2].axvline(0, color="k", lw=0.8)
    ax[2].set_yticks(y, others)
    ax[2].set(xlabel="paired difference vs PD [deg]",
              title="(c) paired contrasts, 95 % CI")
    ax[2].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_campaign.png", dpi=180)


def fig3_operating_points():
    names = ["calm", "reference", "dryden", "stress"]
    label = ["no wind", "Kaimal", "Dryden", "Kaimal + fast\ntransfer"]
    data = {nm: load(nm) for nm in names}
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    x = np.arange(len(names))

    for c in ORDER:
        ax[0].plot(x, [np.mean(data[nm][c]["bound_ok"]) * 100 for nm in names],
                   "o-", ms=5, color=C[c], label=c)
    ax[0].set_xticks(x, label, fontsize=8)
    ax[0].set(ylabel="peak swing within 4.8 deg [%]",
              title="(a) bound satisfaction by operating point")
    ax[0].legend(fontsize=8, frameon=False, ncol=2)

    for c in ORDER:
        ax[1].plot(x, [np.mean(data[nm][c]["residual_swing"]) for nm in names],
                   "o-", ms=5, color=C[c])
    ax[1].set_xticks(x, label, fontsize=8)
    ax[1].set(ylabel="residual swing [deg]", yscale="log",
              title="(b) ranking is not preserved")

    r = data["stress"]
    ax[2].scatter([np.mean(r[c]["ise_pos"]) for c in ORDER],
                  [np.mean(r[c]["peak_swing"]) for c in ORDER],
                  s=45, c=[C[c] for c in ORDER])
    for c in ORDER:
        ax[2].annotate(c, (np.mean(r[c]["ise_pos"]), np.mean(r[c]["peak_swing"])),
                       textcoords="offset points", xytext=(7, 2), fontsize=8)
    ax[2].axhline(4.8, ls="--", lw=0.8, color="k")
    ax[2].set(xscale="log", xlabel="tracking error ISE [m$^2$s]",
              ylabel="peak swing [deg]",
              title="(c) stress campaign: the ZVD trade")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_operating_points.png", dpi=180)


if __name__ == "__main__":
    fig1_verification()
    fig2_campaign()
    fig3_operating_points()
    print("figures written to", OUT)
