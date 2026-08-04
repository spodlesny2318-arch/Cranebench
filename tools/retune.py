"""Re-tune every baseline on the stress nominal, under a declared tuning budget.

The campaigns of Section 3.2 use gains tuned once on the benign nominal.  That
protocol favours feedforward over feedback whenever the evaluation point moves,
so it is a candidate explanation for the rank reversal observed under the
aggressive transfer.  This script repeats the tuning at the stress nominal so
that the two can be told apart.

The budget is the point.  Every controller gets a grid over its two principal
gains with exactly ``NGRID`` evaluations, scored by one frozen objective on one
deterministic wind-free run.  Equal budget, equal objective, equal operating
point -- so any remaining difference is the controller, not the attention it
received.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                           # noqa: E402

from cranebench.batch import run_campaign_batch               # noqa: E402
from cranebench.controllers import HSMC, LQR, PD, SMC, ZVD    # noqa: E402
from cranebench.reference import Manoeuvre                    # noqa: E402
from cranebench.runner import Campaign, run_single             # noqa: E402
from cranebench.uncertainty import lhs_design                 # noqa: E402

np.seterr(over="ignore", invalid="ignore")

# Gains are scored on a small uncertainty design, not on one deterministic run.
# An earlier version tuned on a single nominal trajectory; the winning LQR gains
# then diverged across the evaluation design (mean peak swing 169 deg), because a
# single run cannot see instability that only appears at some plant samples.
# The tuning design uses a different seed from the evaluation design, so the
# gains are never chosen on the samples they are later scored on.
TUNE_N, TUNE_SEED = 8, 90210

NGRID = None        # per-controller budget = the full grid, reported per entry
# Minimise swing among gains that actually complete the transfer.  Scoring swing
# alone is degenerate: its global optimum is a controller that does not move, and
# an earlier version of this script duly drove PD to the lowest gains on the grid.
# Feasibility is the settling criterion already in the metric module -- the
# position error must enter and stay inside the 2 cm band before the horizon.
OBJECTIVE = ("minimise mean(peak_swing + 4 * residual_swing) [deg] over a "
             "held-out uncertainty design, subject to (i) every sample finite, "
             "(ii) the load arriving -- median |position error| over the last "
             "5 s below 1.25 % of the transfer -- and (iii) the command using "
             "no more than SLEW_BUDGET of the drive's available slew rate on "
             "average")
# Why (iii).  Scoring swing and arrival alone is indifferent to how violently
# the command moves.  On the dual crane that let the shaper win the swing bound
# with gains whose command sat on the drive's slew limit for the whole transfer:
# total variation 1.9e7 against 1.1e5 for every other baseline.  The bench has a
# chatter metric precisely to expose that, so the tuner should not be allowed to
# ignore it.  The cap is expressed as a fraction of what the declared slew rate
# would permit, which makes it a statement about the drive rather than a number
# chosen to get an agreeable answer.
SLEW_BUDGET = 0.05
ARRIVED = 0.25      # m; 1.25 % of the 20 m transfer

# Gains are tuned per plant.  Carrying the planar gains onto the dual crane
# measured the transfer of a tuning, not the controllers; the dual campaign
# showed it plainly, with LQR winning on peak swing while finishing 2.6 m short
# of the target on a 10 m move.
SETUPS = {
    "planar": dict(
        man=Manoeuvre(distance=20.0, t_ramp=10.0, t_total=30.0, kind="trapezoid"),
        wind="kaimal", tune_n=8, factors=None),
    "dual": dict(
        # The payload swings with a 7.5 s period and the model has no transverse
        # damping, so a 12 s settling window is 1.6 periods and nothing reaches
        # the arrival tolerance.  48 s gives about four periods, matching the
        # ratio the planar setup already had.
        man=Manoeuvre(distance=10.0, t_ramp=12.0, t_total=48.0),
        wind="kaimal", tune_n=4,
        factors={"m_beam": (0.80, 1.20), "k_cable": (0.50, 2.00),
                 "c_cable": (0.50, 2.00), "b_x": (0.70, 1.30),
                 "rest": (0.90, 1.10), "u_mean": (0.60, 1.40)}),
}
STRESS = SETUPS["planar"]["man"]

# The ranges are deliberately wider than the previous version, whose optima for
# two controllers fell on the grid boundary: an optimum on the edge means the
# budget, not the controller, decided the result.
# Ranges suited to each plant: the dual crane carries a 12 t beam on 60 kN
# drives, so its useful gains are nowhere near the planar ones.
GRIDS_DUAL = {
    "PD":   ("kp", [2e3, 6e3, 1.6e4, 4e4, 1.0e5, 2.5e5],
             "kd", [1e4, 3e4, 8e4, 2.0e5, 5.0e5]),
    "LQR":  ("q_swing", [1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
             "r", [2e-10, 2e-9, 2e-8, 2e-7, 2e-6]),
    "ZVD":  ("kp", [2e3, 6e3, 1.6e4, 4e4, 1.0e5, 2.5e5],
             "kd", [1e4, 3e4, 8e4, 2.0e5, 5.0e5]),
    "SMC":  ("c", [0.05, 0.1, 0.2, 0.4, 0.8, 1.6],
             "k", [2e3, 8e3, 3e4, 1.0e5, 3.0e5]),
    "HSMC": ("lam", [0.05, 0.15, 0.4, 1.0, 2.5, 6.0],
             "c_swing", [0.2, 0.5, 1.0, 2.0, 4.0]),
}

GRIDS = {
    "PD":   ("kp", [2.5e2, 5e2, 1e3, 2e3, 4e3, 6e3, 1.0e4, 1.6e4],
             "kd", [3e3, 6e3, 1.2e4, 2.4e4, 4.0e4, 6.0e4]),
    "LQR":  ("q_swing", [1e2, 4e2, 1.6e3, 6.4e3, 2.56e4, 1e5, 4e5, 1.6e6],
             "r", [2e-11, 2e-10, 2e-9, 2e-8, 2e-7, 2e-6, 2e-5, 2e-4]),
    "ZVD":  ("kp", [2.5e2, 5e2, 1e3, 2e3, 4e3, 6e3, 1.0e4, 1.6e4],
             "kd", [3e3, 6e3, 1.2e4, 2.4e4, 4.0e4, 6.0e4]),
    "SMC":  ("c", [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 1.1, 1.6, 2.2],
             "k", [2.5e2, 5e2, 1e3, 2e3, 4e3, 8e3, 1.5e4, 2.2e4]),
    "HSMC": ("lam", [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 1.2, 2.0, 3.0],
             "c_swing", [0.1, 0.2, 0.3, 0.5, 0.9, 1.3, 2.0, 3.0]),
}
CLS = {"PD": PD, "LQR": LQR, "ZVD": ZVD, "SMC": SMC, "HSMC": HSMC}


def build(name, a, b, smc_gains=None):
    ka, _, kb, _ = GRIDS[name]
    kw = {ka: a, kb: b}
    if name == "HSMC" and smc_gains:
        kw.update(smc_gains)
    return CLS[name](**kw)


def score(res, arrived_tol, chatter_cap=None):
    """Infeasible candidates sort after every feasible one."""
    if res is None:
        return (2, float("inf"))
    peak = np.asarray(res["peak_swing"], float)
    resid = np.asarray(res["residual_swing"], float)
    arrived = np.asarray(res["final_pos_err"], float)
    if not (np.all(np.isfinite(peak)) and np.all(np.isfinite(resid))
            and np.all(np.isfinite(arrived))):
        return (2, float("inf"))
    swing = float(np.mean(peak + 4.0 * resid))
    if float(np.median(arrived)) > arrived_tol:
        return (1, swing)
    if chatter_cap is not None:
        ch = np.asarray(res["chatter"], float)
        if not np.all(np.isfinite(ch)) or float(np.median(ch)) > chatter_cap:
            return (1, swing)
    return (0, swing)


def main(budget=35.0, plant="planar", out=None):
    setup = SETUPS[plant]
    grids = GRIDS if plant == "planar" else GRIDS_DUAL
    globals()["GRIDS"] = grids
    out = out or ROOT / "run_retune" / f"gains_{plant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    # Discard any entry tuned against a different manoeuvre: silently reusing
    # gains chosen under another horizon is the kind of stale-state error this
    # package exists to make impossible.
    stale = [k for k, v in done.items()
             if v.get("horizon") != setup["man"].t_total
             or "chatter_cap" not in v]
    for k in stale:
        del done[k]
    if stale:
        print("re-tuning (manoeuvre or objective changed):",
              ", ".join(stale), flush=True)
    # Checkpoint every grid point, not every controller: with a per-controller
    # checkpoint a grid that does not fit the wall-clock budget restarts from
    # scratch on every attempt and never finishes.
    ev_path = out.parent / f"tuning_evals_{plant}_{setup['man'].t_total:g}.json"
    evals = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else {}
    design = lhs_design(n=setup["tune_n"], seed=TUNE_SEED,
                        factors=setup["factors"])
    arrived = 0.0125 * setup["man"].distance
    # what the declared slew rate would permit over the whole run
    plant_cls = {"planar": "planar", "dual": "dual"}[plant]
    from cranebench.plants import PLANTS
    pl = PLANTS[plant_cls]()
    control_dt = 1.0e-2
    steps = setup["man"].t_total / control_dt
    per_step = sum(float(np.asarray(pl.p.u_rate, float)[i]) * control_dt
                   for i in pl.horizontal_inputs)
    chatter_cap = SLEW_BUDGET * steps * per_step
    print(f"slew budget: median chatter must stay below {chatter_cap:.3g} N "
          f"({SLEW_BUDGET:.0%} of the drive's available slew)", flush=True)
    t0 = time.time()
    for name in ("PD", "LQR", "ZVD", "SMC", "HSMC"):
        if name in done:
            continue
        smc_gains = None
        if name == "HSMC" and "SMC" in done:
            smc_gains = {"c": done["SMC"]["a"], "k": done["SMC"]["b"]}
        ka, avals, kb, bvals = GRIDS[name]
        best = None
        for a in avals:
            for b in bvals:
                tag = f"{name}|{a:g}|{b:g}"
                if tag in evals:
                    m = evals[tag]
                else:
                    if time.time() - t0 > budget:
                        ev_path.write_text(json.dumps(evals), encoding="utf-8")
                        print(f"budget reached inside {name} "
                              f"({sum(1 for k in evals if k.startswith(name + '|'))}"
                              f"/{len(avals) * len(bvals)} points); re-run to resume",
                              flush=True)
                        return
                    camp = Campaign(name="tune", plant=plant,
                                    wind=setup["wind"], dt=1e-2,
                                    manoeuvre=setup["man"])
                    camp.controllers = {name: build(name, a, b, smc_gains)}
                    if plant == "planar":
                        r = run_campaign_batch(camp, design, progress=False)[name]
                        m = {k: [float(x) for x in v] for k, v in r.items()}
                    else:
                        rows = [run_single(camp.controllers[name], camp, f,
                                           int(design.wind_seeds[j]))[0]
                                for j, f in enumerate(design.as_dicts())]
                        if any(x is None for x in rows):
                            m = None
                        else:
                            m = {k: [float(x[k]) for x in rows] for k in rows[0]}
                    evals[tag] = m
                sc = score(m, arrived, chatter_cap)
                if best is None or sc < best[0]:
                    best = (sc, a, b, m, sc[0] == 0)
        edge = (best[1] in (avals[0], avals[-1])) or (best[2] in (bvals[0], bvals[-1]))
        done[name] = {"param_a": ka, "a": best[1], "param_b": kb, "b": best[2],
                      "score": best[0][1], "feasible": bool(best[4]),
                      "evaluations": len(avals) * len(bvals),
                      "objective": OBJECTIVE, "optimum_on_grid_boundary": edge,
                      "tune_samples": setup["tune_n"], "tune_seed": TUNE_SEED,
                      "horizon": setup["man"].t_total,
                      "chatter_cap": chatter_cap,
                      "chatter": float(np.median(best[3]["chatter"])),
                      "peak_swing": float(np.mean(best[3]["peak_swing"])),
                      "residual_swing": float(np.mean(best[3]["residual_swing"]))}
        if smc_gains:
            done[name]["inherited"] = smc_gains
        out.write_text(json.dumps(done, indent=2), encoding="utf-8")
        ev_path.write_text(json.dumps(evals), encoding="utf-8")
        print(f"{name}: {ka}={best[1]:g} {kb}={best[2]:g}  score={best[0][1]:.3f}"
              + ("" if best[4] else "  [NO FEASIBLE GAINS: reports the fastest]")
              + ("   [ON GRID BOUNDARY]" if edge else ""), flush=True)
    print("tuning complete ->", out)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 35.0,
         sys.argv[2] if len(sys.argv) > 2 else "planar")
