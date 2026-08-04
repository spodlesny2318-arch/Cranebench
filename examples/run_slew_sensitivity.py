"""Sensitivity of the stress-campaign conclusion to the drive slew limit.

The limit is a declared property of the bench, not a measurement, so the
question is not what value is right but at what authority the reported rank
reversal survives.  The nominal 160 kN/s is exactly where the sliding baselines
begin to saturate, which makes it the least informative single choice.
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cranebench.batch import run_campaign_batch      # noqa: E402
from cranebench.reference import Manoeuvre           # noqa: E402
from cranebench.runner import Campaign               # noqa: E402
from cranebench.uncertainty import lhs_design        # noqa: E402

ORDER = ["PD", "LQR", "ZVD", "SMC", "HSMC"]
RATES = [(0.125, "20 kN/s"), (0.25, "40 kN/s"), (0.5, "80 kN/s"),
         (1.0, "160 kN/s"), (None, "unlimited")]


def main(n=500):
    design = lhs_design(n=n, seed=20260729)
    man = Manoeuvre(distance=20.0, t_ramp=10.0, t_total=30.0, kind="trapezoid")
    cache = ROOT / "run_batch"
    out = {}
    for i, (scale, label) in enumerate(RATES):
        f = cache / f"slew_{i}.npz"
        if f.exists():
            z = np.load(f)
            d = {}
            for tag in z.files:
                c, k = tag.split("__")
                d.setdefault(c, {})[k] = z[tag]
            out[label] = d
            print(f"  cached: {label}", flush=True)
            continue
        camp = Campaign(name="slew", plant="planar", wind="kaimal", dt=1e-2,
                        manoeuvre=man)
        r = run_campaign_batch(camp, design, progress=False, relative=True,
                               rate_limit=scale is not None,
                               rate_scale=scale or 1.0)
        np.savez_compressed(f, **{f"{c}__{k}": v for c, dd in r.items()
                                  for k, v in dd.items()})
        out[label] = r
        print(f"  done: {label}", flush=True)

    print(f"\nstress campaign, n = {n}: bound satisfaction out of {n} at 4.8 deg")
    print(f"  {'slew limit':14s}" + "".join(f"{c:>9s}" for c in ORDER))
    for _, label in RATES:
        row = f"  {label:14s}"
        for c in ORDER:
            row += f"{int(np.nansum(out[label][c]['bound_ok'])):9d}"
        print(row)
    print(f"\n  {'slew limit':14s}" + "".join(f"{c:>9s}" for c in ORDER)
          + "     (mean peak swing, deg)")
    for _, label in RATES:
        row = f"  {label:14s}"
        for c in ORDER:
            row += f"{float(np.nanmean(out[label][c]['peak_swing'])):9.2f}"
        print(row)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
