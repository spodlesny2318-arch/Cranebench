"""Campaign harness: one run, and a paired Monte Carlo over the design."""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import numpy as np

from . import __version__
from .controllers import BASELINES, Controller
from .integrate import drag_force, simulate
from .ledger import Ledger
from .metrics import METRIC_HASH, compute_metrics
from .plants import PLANTS
from .reference import Manoeuvre
from .uncertainty import UncertaintyDesign, apply_factors, lhs_design
from .wind import WIND_PARAMS, WINDS


@dataclass
class Campaign:
    name: str = "nominal"
    plant: str = "planar"
    wind: str = "kaimal"
    manoeuvre: Manoeuvre = field(default_factory=Manoeuvre)
    dt: float = 2.0e-3
    control_dt: float = 1.0e-2
    swing_bound_deg: float = 4.8
    controllers: Dict[str, Controller] = field(default_factory=dict)

    def build(self):
        if not self.controllers:
            self.controllers = {k: v() for k, v in BASELINES.items()}
        return self


def _make(plant_name, wind_name, factors, wind_seed, man, dt):
    plant_cls = PLANTS[plant_name]
    pp = plant_cls().p.__class__()
    wp = WIND_PARAMS.get(wind_name, WIND_PARAMS["kaimal"])()
    if factors:
        pp, wp = apply_factors(pp, wp, factors)
    plant = plant_cls(pp)
    wind = None
    if wind_name is not None and wind_name != "none":
        rng = np.random.default_rng(int(wind_seed))
        wind = WINDS[wind_name](man.t_total, wp, rng)
    return plant, wind


def run_single(controller: Controller, campaign: Campaign,
               factors: Optional[Dict[str, float]] = None,
               wind_seed: int = 0):
    """One closed-loop run; returns ``(metrics_dict, trajectory)``."""
    plant, wind = _make(campaign.plant, campaign.wind, factors, wind_seed,
                        campaign.manoeuvre, campaign.dt)
    t, X, U = simulate(plant, controller, campaign.manoeuvre, wind,
                       dt=campaign.dt, control_dt=campaign.control_dt,
                       force_fn=drag_force(plant))
    good = np.all(np.isfinite(X), axis=1)
    if not good.all():
        k = int(np.argmax(~good))
        t, X, U = t[:k], X[:k], U[:k]
    if t.size < 10:
        return None, (t, X, U)
    outs = [plant.outputs(row) for row in X]
    ref = np.array([campaign.manoeuvre.position(tt) for tt in t])
    m = compute_metrics(t, outs, U, ref,
                        horizontal_inputs=getattr(plant, "horizontal_inputs", (0,)),
                        swing_bound_deg=campaign.swing_bound_deg)
    return m.as_dict(), (t, X, U)


def run_campaign(campaign: Campaign, design: UncertaintyDesign | None = None,
                 outdir: str | pathlib.Path = "results",
                 progress: bool = True, budget: float | None = None
                 ) -> Dict[str, np.ndarray]:
    """Paired Monte Carlo: identical realisations replayed for each controller.

    Results are checkpointed to ``<name>_runs.jsonl`` after every run, and an
    interrupted campaign resumes from the checkpoint.  A long campaign on a
    shared machine should not have to start again because the job was evicted,
    and a reviewer re-running it should be able to do so in slices.  Pass
    ``budget`` (seconds) to stop cleanly after that much work and resume later.
    """
    campaign.build()
    design = design or lhs_design()
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / f"{campaign.name}_runs.jsonl"

    done: Dict[tuple, Optional[dict]] = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done[(rec["controller"], rec["i"])] = rec["metrics"]

    factor_dicts = design.as_dicts()
    t_start = time.time()
    stopped = False
    with ckpt.open("a", encoding="utf-8") as fh:
        for cname, ctrl in campaign.controllers.items():
            for i, fac in enumerate(factor_dicts):
                if (cname, i) in done or stopped:
                    continue
                if budget is not None and time.time() - t_start > budget:
                    stopped = True
                    continue
                m, _ = run_single(ctrl, campaign, fac, int(design.wind_seeds[i]))
                done[(cname, i)] = m
                fh.write(json.dumps({"controller": cname, "i": i,
                                     "metrics": m}) + "\n")
                fh.flush()
                if progress and (i + 1) % 25 == 0:
                    print(f"  {cname}: {i + 1}/{design.n}", flush=True)

    results: Dict[str, List[Optional[dict]]] = {
        c: [done.get((c, i)) for i in range(design.n)]
        for c in campaign.controllers}
    diverged = {c: sum(1 for r in rows if r is None and (c, i) in done)
                for c, rows in results.items() for i in [0]}
    n_done = sum(1 for k in done)
    if progress:
        print(f"[{n_done}/{design.n * len(campaign.controllers)} runs complete]",
              flush=True)
    if stopped:
        return {"_incomplete": True}

    metric_names = next(r for rows in results.values() for r in rows if r).keys()
    packed = {c: {k: np.array([(r[k] if r else np.nan) for r in rows], float)
                  for k in metric_names}
              for c, rows in results.items()}

    Ledger(campaign=campaign.name, plant=campaign.plant, wind=campaign.wind,
           controllers=list(campaign.controllers), integrator="rk4",
           dt=campaign.dt, horizon=campaign.manoeuvre.t_total,
           design_seed=design.seed, n_samples=design.n,
           wind_seeds=[int(s) for s in design.wind_seeds],
           metric_hash=METRIC_HASH, package_version=__version__,
           ).write(outdir / f"{campaign.name}_ledger.json")

    np.savez_compressed(outdir / f"{campaign.name}_metrics.npz",
                        **{f"{c}__{k}": v
                           for c, d in packed.items() for k, v in d.items()})
    (outdir / f"{campaign.name}_diverged.json").write_text(
        json.dumps(diverged, indent=2), encoding="utf-8")
    return packed
