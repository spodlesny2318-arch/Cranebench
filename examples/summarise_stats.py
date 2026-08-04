"""The statistics the manuscript reports, computed from the campaign files.

Everything here is a number the text quotes, so it lives in one script that can
be re-run rather than in a notebook that cannot.
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from cranebench.stats import (BOOTSTRAP, clopper_pearson,      # noqa: E402
                              kaplan_meier_rmst, mcnemar,
                              paired_summary, rank_agreement)
from summarise_batch import load                               # noqa: E402

ORDER = ["PD", "LQR", "ZVD", "SMC", "HSMC"]
POINTS = ["calm", "reference", "dryden", "stress"]
HORIZON = {"calm": 40.0, "reference": 40.0, "dryden": 40.0, "stress": 30.0}


def load_npz(path):
    z = np.load(path)
    out = {}
    for tag in z.files:
        c, k = tag.split("__")
        out.setdefault(c, {})[k] = z[tag]
    return out


def other_plants():
    """Spatial and dual campaigns: take the most recent file that exists.

    Results are written to whatever ``--outdir`` the runner was given, so
    picking a fixed path silently reported a stale campaign once already.
    """
    found, where = {}, {}
    for label, pattern in (("spatial", "spatial_paired.npz"),
                           ("dual", "dual_paired.npz")):
        cands = sorted(ROOT.rglob(pattern), key=lambda f: f.stat().st_mtime,
                       reverse=True)
        if cands:
            found[label] = load_npz(cands[0])
            where[label] = cands[0].relative_to(ROOT)
    for label, f in where.items():
        print(f"  reading {label} results from {f}")
    return found


def code_version():
    """Say which fixes are present, so a stale checkout is visible at once."""
    from cranebench.plants import DualCrane
    swing = "swing_state" in vars(DualCrane)
    gains = (ROOT / "run_retune" / "gains_dual.json").exists()
    print(f"dual plant declares its own swing coordinate: {swing}")
    print(f"gains tuned on the dual plant present:        {gains}")
    if not swing:
        print("  !! the hierarchical baseline will collapse onto the flat one")
    print()


def main():
    code_version()
    camps = {p: load(p) for p in POINTS}
    n = len(camps["reference"]["PD"]["peak_swing"])

    print(f"bootstrap: {BOOTSTRAP['resamples']} resamples, "
          f"{BOOTSTRAP['kind']}, seed {BOOTSTRAP['seed']}\n")

    print("=== rank agreement between operating points (Kendall tau) ===")
    for metric in ("residual_swing", "peak_swing", "bound_ok"):
        lower = metric != "bound_ok"
        print(f"\n  {metric}")
        for (a, b), (tau, p) in rank_agreement(camps, metric, ORDER, lower).items():
            print(f"    {a:10s} vs {b:10s}  tau = {tau:+.3f}   p = {p:.3f}")

    print("\n=== settling time under censoring (Kaplan-Meier RMST) ===")
    for point in ("reference", "stress"):
        h = HORIZON[point]
        print(f"\n  {point} (horizon {h:g} s)")
        for c in ORDER:
            t = np.asarray(camps[point][c]["settle_time"], float)
            km = kaplan_meier_rmst(t, t < h - 1e-6, h)
            print(f"    {c:5s} RMST {km['rmst']:5.1f} s   "
                  f"censored {km['censored_fraction']*100:5.1f} %   "
                  f"(naive mean {t.mean():5.1f} s)")

    print("\n=== bound satisfaction with exact binomial intervals ===")
    for point in ("reference", "stress"):
        print(f"\n  {point}")
        for c in ORDER:
            k = int(np.nansum(camps[point][c]["bound_ok"]))
            lo, hi = clopper_pearson(k, n)
            print(f"    {c:5s} {k:3d}/{n}   95 % CI [{lo*100:5.1f}, {hi*100:5.1f}] %")

    print("\n=== paired contrasts against PD, reference campaign ===")
    for c in ORDER[1:]:
        for k in ("residual_swing", "peak_swing"):
            s = paired_summary(camps["reference"][c][k], camps["reference"]["PD"][k])
            print(f"  {c:5s} {k:15s} diff {s['mean_diff']:+8.4f} "
                  f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]  "
                  f"r_rb {s['rank_biserial']:+.3f}  p {s['wilcoxon_p']:.2e}")
        mc = mcnemar(camps["reference"]["PD"]["bound_ok"],
                     camps["reference"][c]["bound_ok"])
        print(f"  {c:5s} {'bound_ok':15s} McNemar PD-only {mc['a_only']:3d}  "
              f"{c}-only {mc['b_only']:3d}  p {mc['p']:.2e}")


    others = other_plants()
    for label, d in others.items():
        m = len(d["PD"]["peak_swing"])
        keys = ["ise_pos", "peak_swing", "residual_swing", "final_pos_err",
                "effort", "chatter", "bound_ok"]
        if "peak_yaw" in d["PD"]:
            keys.insert(4, "peak_yaw")
        if "tension_ratio" in d["PD"]:
            keys.insert(4, "tension_ratio")
        print(f"\n=== {label} plant, n = {m} ===")
        print(f"{'ctrl':<6}" + "".join(f"{k:>15}" for k in keys))
        for c in ORDER:
            row = f"{c:<6}"
            for k in keys:
                v = float(np.nanmean(d[c][k]))
                row += f"{v:15.4g}" if abs(v) < 1e5 else f"{v:15.3e}"
            print(row)
        print(f"  bound satisfaction, exact 95 % CI")
        for c in ORDER:
            k = int(np.nansum(d[c]["bound_ok"]))
            lo, hi = clopper_pearson(k, m)
            print(f"    {c:5s} {k:3d}/{m}   [{lo*100:5.1f}, {hi*100:5.1f}] %")
        print("  paired contrasts against PD")
        for c in ORDER[1:]:
            for k in ("residual_swing", "peak_swing"):
                st = paired_summary(d[c][k], d["PD"][k])
                print(f"    {c:5s} {k:15s} diff {st['mean_diff']:+8.4f}  "
                      f"r_rb {st['rank_biserial']:+.3f}  p {st['wilcoxon_p']:.2e}")

    if not others:
        print("\n(no spatial or dual results found)")


if __name__ == "__main__":
    main()
