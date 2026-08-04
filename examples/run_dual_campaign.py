"""Campaign on the cooperative dual crane.

The third plant, and the only one whose falls can go slack: the closed-form fast
path is valid only while both carry tension, and the assembler takes over the
moment one does not. A campaign here therefore exercises a switching plant, not
just a third set of equations, and reports the load-sharing metrics that the
single-crane models have no analogue for.
"""

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cranebench.reference import Manoeuvre                # noqa: E402
from cranebench.runner import Campaign, run_campaign      # noqa: E402
from cranebench.uncertainty import lhs_design             # noqa: E402

FACTORS = {"m_beam": (0.80, 1.20), "k_cable": (0.50, 2.00),
           "c_cable": (0.50, 2.00), "b_x": (0.70, 1.30),
           "rest": (0.90, 1.10), "u_mean": (0.60, 1.40)}
ORDER = ["PD", "LQR", "ZVD", "SMC", "HSMC"]


def collect(root, n):
    done = collections.defaultdict(dict)
    for f in pathlib.Path(root).rglob("dual_runs.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["metrics"]:
                    done[r["controller"]][r["i"]] = r["metrics"]
    if any(c not in done for c in ORDER):
        print("incomplete:", {c: len(done[c]) for c in ORDER})
        return
    common = sorted(set.intersection(*[set(done[c]) for c in ORDER]))
    keys = list(next(iter(done["PD"].values())))
    arr = {f"{c}__{k}": np.array([done[c][i][k] for i in common], float)
           for c in ORDER for k in keys}
    out = pathlib.Path(root) / "dual_paired.npz"
    np.savez_compressed(out, **arr)
    print(f"merged {len(common)} paired samples -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "run_dual"))
    a = ap.parse_args()
    camp = Campaign(name="dual", plant="dual", wind="kaimal", dt=1e-2,
                    manoeuvre=Manoeuvre(distance=10.0, t_ramp=12.0, t_total=48.0))
    camp.build()
    # Use gains tuned on this plant when they exist.  Carrying the planar gains
    # across measures how well a tuning transfers, which is a different question
    # from how the controllers compare.
    gains = ROOT / "run_retune" / "gains_dual.json"
    if gains.exists():
        from cranebench.controllers import BASELINES
        g = json.loads(gains.read_text(encoding="utf-8"))
        for name, rec in g.items():
            kw = {rec["param_a"]: rec["a"], rec["param_b"]: rec["b"]}
            kw.update(rec.get("inherited", {}))
            camp.controllers[name] = BASELINES[name](**kw)
        horizons = {rec.get("horizon") for rec in g.values()}
        if horizons != {camp.manoeuvre.t_total}:
            print(f"REFUSING TO RUN: gains were tuned for horizon {horizons}, "
                  f"this campaign uses {camp.manoeuvre.t_total}. "
                  f"Re-run tools/retune.py.")
            sys.exit(1)
        print(f"using gains tuned on the dual plant ({gains.name})")
    else:
        print("WARNING: no gains_dual.json; falling back to the planar tuning")
    if a.only:
        from cranebench.controllers import BASELINES
        camp.controllers = {k: v for k, v in camp.controllers.items()
                            if k in a.only.split(",")}
    design = lhs_design(n=a.n, seed=20260729, factors=FACTORS)
    res = run_campaign(camp, design, outdir=a.outdir, budget=a.budget)
    if res.get("_incomplete"):
        print("budget reached; re-run to resume")
        return
    root = pathlib.Path(a.outdir)
    if not any(root.rglob("dual_runs.jsonl")):
        root = root.parent
    collect(root, a.n)


if __name__ == "__main__":
    main()
