"""Cross-validation of the spatial and dual plants against an independent source.

Energy conservation and a symmetric positive-definite mass matrix are necessary
but not sufficient: a self-consistent but wrong Lagrangian passes both.  These
tests compare the numeric assembler against equations derived independently by
SymPy's ``LagrangesMethod`` (see ``tools/derive_symbolic.py``), which shares no
code with it.  The planar plant has the equivalent check in
``test_dynamics.py``, so after these three all plants are cross-validated.
"""

import numpy as np
import pytest

from cranebench.plants import DualCrane, SpatialCrane
from cranebench.plants import _generated as gen


def test_spatial_assembler_matches_symbolic():
    p = SpatialCrane(analytic=False)          # force the reference path
    pars = p._pars()
    rng = np.random.default_rng(3)
    worst_m = worst_a = 0.0
    for _ in range(25):
        q = np.array([rng.normal(0, 3), rng.normal(0, 3), rng.uniform(6, 20),
                      rng.normal(0, .5), rng.normal(0, .5), rng.normal(0, .6)])
        qd = rng.normal(0, 1, 6)
        u, d = rng.normal(0, 3e3, 3), rng.normal(0, 150, 3)
        M_num = p.mass_matrix(q)
        M_sym = gen.spatial_M(*q, *qd, *pars)
        worst_m = max(worst_m, np.max(np.abs(M_num - M_sym))
                      / max(1.0, np.max(np.abs(M_num))))
        Q = p.generalised_forces(0.0, q, qd, u, d)
        a_sym = np.linalg.solve(M_sym, gen.spatial_bias(*q, *qd, *pars) + Q)
        a_num = p.dynamics(0.0, np.concatenate([q, qd]), u, d)[6:]
        worst_a = max(worst_a, np.max(np.abs(a_num - a_sym))
                      / max(1.0, np.max(np.abs(a_num))))
    assert worst_m < 1e-12, f"mass matrix mismatch {worst_m:.2e}"
    assert worst_a < 1e-7, f"acceleration mismatch {worst_a:.2e}"


def test_dual_assembler_matches_symbolic_on_the_taut_branch():
    """The symbolic derivation assumes both falls carry tension; test only there."""
    p = DualCrane(analytic=False)
    pars = p._pars()
    rng = np.random.default_rng(5)
    x0 = p.initial_state()
    worst_m = worst_a = 0.0
    tested = 0
    for _ in range(400):
        q = x0[:5] + np.array([rng.normal(0, .4), rng.normal(0, .4),
                               rng.normal(0, .4), rng.normal(0, .2),
                               rng.normal(0, .06)])
        _, _, ln, _, _, _, _ = p._attach(q)
        if np.any(ln <= p.p.rest):
            continue
        tested += 1
        qd = rng.normal(0, .5, 5)
        u, d = rng.normal(0, 5e3, 2), np.zeros(3)
        M_num = p.mass_matrix(q)
        M_sym = gen.dual_M(*q, *qd, *pars)
        worst_m = max(worst_m, np.max(np.abs(M_num - M_sym))
                      / max(1.0, np.max(np.abs(M_num))))
        Q = p.generalised_forces(0.0, q, qd, u, d)
        a_sym = np.linalg.solve(M_sym, gen.dual_bias(*q, *qd, *pars) + Q)
        a_num = p.dynamics(0.0, np.concatenate([q, qd]), u, d)[5:]
        worst_a = max(worst_a, np.max(np.abs(a_num - a_sym))
                      / max(1.0, np.max(np.abs(a_num))))
        if tested >= 25:
            break
    assert tested >= 20, "not enough taut states sampled"
    assert worst_m < 1e-10, f"mass matrix mismatch {worst_m:.2e}"
    assert worst_a < 1e-4, f"acceleration mismatch {worst_a:.2e}"


@pytest.mark.parametrize("cls", [SpatialCrane, DualCrane])
def test_fast_path_matches_reference_path(cls):
    """The shipped fast path must reproduce the assembler it was checked against."""
    fast, ref = cls(analytic=True), cls(analytic=False)
    rng = np.random.default_rng(11)
    x0 = np.array(fast.initial_state(), float)
    worst = 0.0
    for _ in range(20):
        x = x0.copy()
        x[fast.unactuated[0]] += rng.normal(0, 0.15)
        x[fast.nq:] += rng.normal(0, 0.3, fast.nq)
        u = rng.normal(0, 3e3, fast.nu)
        a, b = fast.dynamics(0.0, x, u, np.zeros(3)), ref.dynamics(0.0, x, u, np.zeros(3))
        worst = max(worst, np.max(np.abs(a - b)) / max(1.0, np.max(np.abs(b))))
    assert worst < 1e-4, f"fast path deviates by {worst:.2e}"


def test_dual_fast_path_defers_to_the_assembler_when_a_fall_goes_slack():
    p = DualCrane(analytic=True)
    x = np.array(p.initial_state(), float)
    x[3] += 5.0                                # raise the beam: falls go slack
    _, _, ln, _, _, _, _ = p._attach(x[:5])
    assert np.any(ln <= p.p.rest), "test state is not slack"
    a = p.dynamics(0.0, x, np.zeros(2), np.zeros(3))
    b = DualCrane(analytic=False).dynamics(0.0, x, np.zeros(2), np.zeros(3))
    assert np.allclose(a, b)
