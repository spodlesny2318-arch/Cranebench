"""Stress campaign repeated with every baseline re-tuned on the stress nominal."""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cranebench import __version__                       # noqa: E402
from cranebench.batch import run_campaign_batch          # noqa: E402
from cranebench.controllers import HSMC, LQR, PD, SMC, ZVD  # noqa: E402
from cranebench.ledger import Ledger                     # noqa: E402
from cranebench.metrics import METRIC_HASH               # noqa: E402
from cranebench.reference import Manoeuvre               # noqa: E402
from cranebench.runner import Campaign                   # noqa: E402
from cranebench.uncertainty import lhs_design            # noqa: E402

CLS = {"PD": PD, "LQR": LQR, "ZVD": ZVD, "SMC": SMC, "HSMC": HSMC}


def retuned():
    g = json.loads((ROOT / "run_retune" / "gains_arr.json").read_text(
        encoding="utf-8"))
    out = {}
    for name, rec in g.items():
        kw = {rec["param_a"]: rec["a"], rec["param_b"]: rec["b"]}
        kw.update(rec.get("inherited", {}))
        out[name] = CLS[name](**kw)
    return {k: out[k] for k in ("PD", "LQR", "ZVD", "SMC", "HSMC")}


def main(n=500, seed=20260729):
    camp = Campaign(name="stress_retuned", plant="planar", wind="kaimal", dt=1e-2,
                    manoeuvre=Manoeuvre(distance=20.0, t_ramp=10.0,
                                        t_total=30.0, kind="trapezoid"))
    camp.controllers = retuned()
    design = lhs_design(n=n, seed=seed)
    res = run_campaign_batch(camp, design)
    out = ROOT / "run_batch"
    np.savez_compressed(out / "stress_retuned_metrics.npz",
                        **{f"{c}__{k}": v for c, d in res.items() for k, v in d.items()})
    Ledger(campaign="stress_retuned", plant="planar", wind="kaimal",
           controllers=list(camp.controllers), integrator="rk4-batched",
           dt=camp.dt, horizon=30.0, design_seed=seed, n_samples=n,
           wind_seeds=[int(s) for s in design.wind_seeds],
           metric_hash=METRIC_HASH, package_version=__version__,
           ).write(out / "stress_retuned_ledger.json")
    print("written", out / "stress_retuned_metrics.npz")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
