"""Ablation of two modelling choices, on the identical paired design.

Neither the relative-wind drag law nor the drive slew limit is optional physics:
a bluff body in a 12 m/s wind is aerodynamically damped, and no drive steps its
force. Both were absent from the first version of this package. Rather than fix
them silently, the harness can run with each switched off, so the size of the
error each one was causing is measured instead of asserted.

    python examples/run_ablation.py reference
    python examples/run_ablation.py stress
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cranebench.batch import run_campaign_batch          # noqa: E402
from cranebench.reference import Manoeuvre               # noqa: E402
from cranebench.runner import Campaign                   # noqa: E402
from cranebench.uncertainty import lhs_design            # noqa: E402

ORDER = ["PD", "LQR", "ZVD", "SMC", "HSMC"]
KEY = ["peak_swing", "residual_swing", "chatter", "bound_ok"]
VARIANTS = [("absolute wind, no slew limit", False, False),
            ("relative wind, no slew limit", True, False),
            ("relative wind + slew limit", True, True)]
MAN = {"reference": Manoeuvre(distance=20.0, t_ramp=20.0, t_total=40.0),
       "stress": Manoeuvre(distance=20.0, t_ramp=10.0, t_total=30.0,
                           kind="trapezoid")}


def main(name="reference", n=500):
    design = lhs_design(n=n, seed=20260729)
    cache = ROOT / "run_batch"
    cache.mkdir(exist_ok=True)
    out = {}
    for i, (label, rel, rate) in enumerate(VARIANTS):
        f = cache / f"{name}_ablation_{i}.npz"
        if f.exists():
            z = np.load(f)
            out[label] = {}
            for tag in z.files:
                c, k = tag.split("__")
                out[label].setdefault(c, {})[k] = z[tag]
            print(f"  cached: {label}", flush=True)
            continue
        camp = Campaign(name=name, plant="planar", wind="kaimal", dt=1e-2,
                        manoeuvre=MAN[name])
        r = run_campaign_batch(camp, design, progress=False,
                               relative=rel, rate_limit=rate)
        np.savez_compressed(f, **{f"{c}__{k}": v for c, d in r.items()
                                  for k, v in d.items()})
        out[label] = r
        print(f"  done: {label}", flush=True)

    print(f"\ncampaign '{name}', n = {n}")
    for k in KEY:
        print(f"\n{k}")
        print(f"  {'variant':32s}" + "".join(f"{c:>10s}" for c in ORDER))
        for label, _, _ in VARIANTS:
            row = f"  {label:32s}"
            for c in ORDER:
                v = float(np.nanmean(out[label][c][k]))
                row += f"{v:10.4g}" if abs(v) < 1e4 else f"{v:10.3e}"
            print(row)

    np.savez_compressed(ROOT / "run_batch" / f"{name}_ablation.npz",
                        **{f"{i}__{c}__{k}": out[lab][c][k]
                           for i, (lab, _, _) in enumerate(VARIANTS)
                           for c in ORDER for k in KEY})
    print(f"\nwritten run_batch/{name}_ablation.npz")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reference",
         int(sys.argv[2]) if len(sys.argv) > 2 else 500)
