"""The harness itself: determinism, pairing, step convergence, metric freezing."""

import numpy as np
import pytest

from cranebench.controllers import BASELINES, PD
from cranebench.metrics import METRIC_HASH
from cranebench.reference import Manoeuvre
from cranebench.runner import Campaign, run_single
from cranebench.stats import paired_summary, running_mean_convergence
from cranebench.uncertainty import lhs_design

MAN = Manoeuvre(distance=20.0, t_ramp=20.0, t_total=40.0)


def _camp(**kw):
    kw.setdefault("dt", 5e-3)
    return Campaign(plant="planar", wind="kaimal", manoeuvre=MAN, **kw).build()


def test_run_is_bitwise_reproducible():
    c = _camp()
    a, _ = run_single(PD(), c, None, 7)
    b, _ = run_single(PD(), c, None, 7)
    assert a == b


def test_metrics_are_converged_in_the_integrator_step():
    """Halving the step must not move any reported metric by more than 0.5 %."""
    ref, _ = run_single(PD(), _camp(dt=1e-3), None, 7)
    coarse, _ = run_single(PD(), _camp(dt=5e-3), None, 7)
    for k, v in ref.items():
        if k in ("bound_ok", "settle_time") or abs(v) < 1e-9:
            continue
        assert abs(coarse[k] - v) / abs(v) < 5e-3, k


def test_design_is_paired_and_reproducible():
    d1 = lhs_design(n=64, seed=11)
    d2 = lhs_design(n=64, seed=11)
    assert np.allclose(d1.samples, d2.samples)
    assert np.array_equal(d1.wind_seeds, d2.wind_seeds)
    # a Latin hypercube puts exactly one point in each 1-D stratum
    for j in range(d1.samples.shape[1]):
        lo, hi = list(d1.factors.values())[j]
        u = (d1.samples[:, j] - lo) / (hi - lo)
        assert np.array_equal(np.sort((u * 64).astype(int)), np.arange(64))


def test_every_baseline_runs_on_every_plant():
    for plant in ("planar", "spatial", "dual"):
        c = Campaign(plant=plant, wind="none", dt=2e-2,
                     manoeuvre=Manoeuvre(distance=3.0, t_ramp=5.0,
                                         t_total=8.0)).build()
        for name, ctrl in c.controllers.items():
            m, _ = run_single(ctrl, c, None, 3)
            assert m is not None, f"{name} diverged on {plant}"


def test_metric_hash_is_stable_within_a_session():
    assert len(METRIC_HASH) == 16
    from cranebench import metrics as m
    assert m.METRIC_HASH == METRIC_HASH


def test_paired_statistics_detect_a_known_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 400)
    a, b = base, base + 0.3          # b is worse by a constant
    s = paired_summary(a, b)
    assert s["mean_diff"] < 0
    assert s["ci_high"] < 0
    assert s["wilcoxon_p"] < 1e-6
    assert s["win_rate"] == 1.0


def test_convergence_helper():
    rng = np.random.default_rng(1)
    n, ok = running_mean_convergence(rng.normal(5.0, 0.1, 2000), tol=0.01)
    assert ok and n < 2000
