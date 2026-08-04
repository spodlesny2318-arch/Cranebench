"""Full-size campaigns on the batched path.

    python examples/run_batch_campaign.py reference
    python examples/run_batch_campaign.py stress
    python examples/run_batch_campaign.py dryden

Each writes ``run_batch/<name>_metrics.npz`` and a ledger.  The batched path is
verified against the scalar reference path in ``tests/test_batch.py``.
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cranebench import __version__                        # noqa: E402
from cranebench.batch import run_campaign_batch           # noqa: E402
from cranebench.ledger import Ledger                      # noqa: E402
from cranebench.metrics import METRIC_HASH                # noqa: E402
from cranebench.reference import Manoeuvre                # noqa: E402
from cranebench.runner import Campaign                    # noqa: E402
from cranebench.uncertainty import lhs_design             # noqa: E402

SPECS = {
    "reference": dict(wind="kaimal",
                      man=Manoeuvre(distance=20.0, t_ramp=20.0, t_total=40.0)),
    "stress":    dict(wind="kaimal",
                      man=Manoeuvre(distance=20.0, t_ramp=10.0, t_total=30.0,
                                    kind="trapezoid")),
    "dryden":    dict(wind="dryden",
                      man=Manoeuvre(distance=20.0, t_ramp=20.0, t_total=40.0)),
    "calm":      dict(wind="none",
                      man=Manoeuvre(distance=20.0, t_ramp=20.0, t_total=40.0)),
}


def main(name="reference", n=500, seed=20260729, outdir=ROOT / "run_batch",
         only=None):
    spec = SPECS[name]
    camp = Campaign(name=name, plant="planar", wind=spec["wind"], dt=1e-2,
                    manoeuvre=spec["man"]).build()
    design = lhs_design(n=n, seed=seed)
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Run in slices when the whole campaign does not fit a wall-clock budget:
    # each controller is cached separately and merged once all are present.
    for c in (only.split(",") if only else list(camp.controllers)):
        f = outdir / f"{name}_{c}.npz"
        if f.exists():
            continue
        one = Campaign(name=name, plant="planar", wind=spec["wind"], dt=1e-2,
                       manoeuvre=spec["man"])
        one.controllers = {c: camp.controllers[c]}
        r = run_campaign_batch(one, design, progress=False)
        np.savez_compressed(f, **{f"{c}__{k}": v for k, v in r[c].items()})
        print(f"  slice {c}: done", flush=True)
    parts = {c: outdir / f"{name}_{c}.npz" for c in camp.controllers}
    missing = [c for c, f in parts.items() if not f.exists()]
    if missing:
        print("still to run:", ",".join(missing))
        return None
    res = {}
    for c, f in parts.items():
        z = np.load(f)
        res[c] = {tag.split("__")[1]: z[tag] for tag in z.files}
    np.savez_compressed(outdir / f"{name}_metrics.npz",
                        **{f"{c}__{k}": v for c, d in res.items()
                           for k, v in d.items()})
    Ledger(campaign=name, plant="planar", wind=spec["wind"],
           controllers=list(camp.controllers), integrator="rk4-batched",
           dt=camp.dt, horizon=spec["man"].t_total, design_seed=seed,
           n_samples=n, wind_seeds=[int(s) for s in design.wind_seeds],
           metric_hash=METRIC_HASH, package_version=__version__,
           ).write(outdir / f"{name}_ledger.json")
    print(f"wrote {outdir / (name + '_metrics.npz')}")
    return res


if __name__ == "__main__":
    a = sys.argv[1:] or ["reference"]
    main(a[0], int(a[1]) if len(a) > 1 else 500,
         only=a[2] if len(a) > 2 else None)
