"""The reference campaign of the paper: five baselines, one paired design.

    python examples/run_reference_campaign.py --n 500

Produces ``results/reference_metrics.npz``, ``results/reference_ledger.json``
and the paired comparison table.  With ``--n 500`` the run takes about 45 min
on one core; the figures in the paper use that setting.
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cranebench.reference import Manoeuvre                      # noqa: E402
from cranebench.runner import Campaign, run_campaign            # noqa: E402
from cranebench.stats import paired_summary, running_mean_convergence  # noqa: E402

KEY = ["ise_pos", "peak_swing", "residual_swing", "effort", "chatter", "bound_ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--plant", default="planar")
    ap.add_argument("--wind", default="kaimal")
    ap.add_argument("--stress", action="store_true",
                    help="fast transfer, trapezoidal profile")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--only", default=None,
                    help="comma-separated controller subset (for parallel slices)")
    ap.add_argument("--dt", type=float, default=1e-2)
    ap.add_argument("--budget", type=float, default=None,
                    help="stop cleanly after this many seconds; resume by re-running")
    a = ap.parse_args()

    from cranebench.uncertainty import lhs_design
    man = (Manoeuvre(distance=20.0, t_ramp=10.0, t_total=30.0, kind="trapezoid")
           if a.stress else Manoeuvre(distance=20.0, t_ramp=20.0, t_total=40.0))
    camp = Campaign(name="stress" if a.stress else "reference",
                    plant=a.plant, wind=a.wind, dt=a.dt, manoeuvre=man)
    if a.only:
        from cranebench.controllers import BASELINES
        camp.controllers = {k: BASELINES[k]() for k in a.only.split(",")}
    design = lhs_design(n=a.n, seed=a.seed)
    res = run_campaign(camp, design, outdir=a.outdir, budget=a.budget)
    if res.get("_incomplete"):
        print("budget reached; re-run the same command to resume")
        return

    hdr = f"{'controller':<6}" + "".join(f"{k:>16}" for k in KEY)
    print("\n" + hdr)
    print("-" * len(hdr))
    for c, d in res.items():
        row = f"{c:<6}"
        for k in KEY:
            v = np.nanmean(d[k])
            row += f"{v:16.4g}" if abs(v) < 1e5 else f"{v:16.4e}"
        print(row)

    ref = "PD"
    print(f"\nPaired contrasts against {ref} (negative mean_diff = better)")
    for c in res:
        if c == ref:
            continue
        for k in ("residual_swing", "peak_swing", "effort"):
            s = paired_summary(res[c][k], res[ref][k])
            print(f"  {c:<5} {k:<16} diff={s['mean_diff']:+11.4g} "
                  f"CI[{s['ci_low']:+.4g},{s['ci_high']:+.4g}] "
                  f"p={s['wilcoxon_p']:.2e} win={s['win_rate']:.2f}")

    print("\nConvergence of the campaign mean (1 % band)")
    for c in res:
        n_req, ok = running_mean_convergence(res[c]["residual_swing"])
        print(f"  {c:<5} residual_swing converged at n = {n_req}  ({'ok' if ok else 'NOT converged'})")


if __name__ == "__main__":
    main()
