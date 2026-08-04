"""Fifth campaign, on the three-dimensional plant.

Smaller than the planar campaigns (n = 150 rather than 500) because the spatial
plant is not batched: the point is to exercise the plant and the yaw metrics
end to end, not to match the statistical resolution of Section 3.2.
"""

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cranebench.reference import Manoeuvre                # noqa: E402
from cranebench.runner import Campaign, run_campaign      # noqa: E402
from cranebench.uncertainty import lhs_design             # noqa: E402

# c_theta does not exist on the spatial plant; its swing damping is c_swing
FACTORS = {"m_payload": (0.80, 1.20), "l0": (0.75, 1.25),
           "c_swing": (0.50, 2.00), "b_x": (0.70, 1.30),
           "k_psi": (0.50, 2.00), "u_mean": (0.60, 1.40)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "run_sp3"))
    a = ap.parse_args()

    camp = Campaign(name="spatial", plant="spatial", wind="kaimal", dt=1e-2,
                    manoeuvre=Manoeuvre(distance=15.0, t_ramp=15.0, t_total=30.0))
    camp.build()
    if a.only:
        from cranebench.controllers import BASELINES
        camp.controllers = {k: BASELINES[k]() for k in a.only.split(",")}
    design = lhs_design(n=a.n, seed=20260729, factors=FACTORS)
    res = run_campaign(camp, design, outdir=a.outdir, budget=a.budget)
    if res.get("_incomplete"):
        print("budget reached; re-run to resume")
        return
    # Write the merged file next to the checkpoints, wherever they actually are.
    # A campaign run in slices puts them in subdirectories of --outdir; a plain
    # run puts them in --outdir itself.  Taking the parent unconditionally, as
    # an earlier version did, dropped the file one level too high and the
    # manuscript checker then reported a table it had silently not verified.
    root = pathlib.Path(a.outdir)
    if not any(root.rglob("spatial_runs.jsonl")):
        root = root.parent
    collect(root, a.n)


def collect(root, n):
    """Merge every checkpoint under ``root`` into one paired result file.

    The campaign may be run in slices, on several directories, to fit a wall
    clock budget; the manuscript checker wants one array per controller, so the
    merge belongs here rather than in an ad-hoc script.
    """
    import collections
    import json
    done = collections.defaultdict(dict)
    for f in pathlib.Path(root).rglob("spatial_runs.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["metrics"]:
                    done[r["controller"]][r["i"]] = r["metrics"]
    order = ["PD", "LQR", "ZVD", "SMC", "HSMC"]
    have = [c for c in order if c in done]
    if len(have) < len(order):
        print("incomplete:", {c: len(done[c]) for c in order})
        return
    common = sorted(set.intersection(*[set(done[c]) for c in order]))
    keys = list(next(iter(done["PD"].values())))
    arr = {f"{c}__{k}": np.array([done[c][i][k] for i in common], float)
           for c in order for k in keys}
    out = pathlib.Path(root) / "spatial_paired.npz"
    np.savez_compressed(out, **arr)
    print(f"merged {len(common)} paired samples -> {out}")


if __name__ == "__main__":
    main()
