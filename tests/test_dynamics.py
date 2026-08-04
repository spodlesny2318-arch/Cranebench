"""Verification of the plant models.

These are the tests that make the modelling claims checkable.  A crane paper
normally asserts its equations of motion; here they are cross-validated against
an independently assembled model and against energy conservation.
"""

import numpy as np
import pytest

from cranebench.integrate import rk4
from cranebench.plants import DualCrane, PlanarCrane, SpatialCrane


def test_planar_analytic_matches_assembler():
    """The hand-derived planar equations must agree with the generic assembler."""
    pa, pn = PlanarCrane(analytic=True), PlanarCrane(analytic=False)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(40):
        x = np.concatenate([rng.normal(0, 3, 1), rng.uniform(5, 20, 1),
                            rng.normal(0, 0.4, 1), rng.normal(0, 1, 3)])
        u, d = rng.normal(0, 5e3, 2), rng.normal(0, 200, 3)
        fa, fn = pa.dynamics(0, x, u, d), pn.dynamics(0, x, u, d)
        worst = max(worst, np.max(np.abs(fa - fn)) / max(1.0, np.max(np.abs(fa))))
    assert worst < 1e-7, f"analytic/assembled mismatch {worst:.2e}"


@pytest.mark.parametrize("cls", [PlanarCrane, SpatialCrane, DualCrane])
def test_energy_is_conserved_without_forcing(cls):
    """With all damping zeroed and no input, energy must be conserved."""
    p = cls()
    for a in ("b_x", "b_y", "b_l", "c_theta", "c_swing", "c_psi", "c_cable"):
        if hasattr(p.p, a):
            setattr(p.p, a, 0.0)
    x = np.array(p.initial_state(), float)
    x[p.unactuated[0]] += 0.15
    e0 = p.energy(x)
    dt = 1e-3
    for k in range(2000):
        x = rk4(p.dynamics, k * dt, x, dt, np.zeros(p.nu), np.zeros(3))
    drift = abs(p.energy(x) - e0) / max(abs(e0), 1.0)
    assert drift < 1e-8, f"energy drift {drift:.2e}"


def test_mass_matrix_is_symmetric_positive_definite():
    for cls in (PlanarCrane, SpatialCrane, DualCrane):
        p = cls()
        q = np.array(p.initial_state()[: p.nq], float)
        M = p.mass_matrix(q)
        assert np.allclose(M, M.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(M) > 0)


def test_spatial_swing_parameterisation_preserves_rope_length():
    """The (theta, phi) map must have exact norm l, including at large swing."""
    p = SpatialCrane()
    for th, ph in [(0.0, 0.0), (0.4, -0.3), (1.2, 0.9)]:
        q = np.array([1.0, 2.0, 14.0, th, ph, 0.0])
        o = p.outputs(np.concatenate([q, np.zeros(6)]))
        r = np.hypot(np.hypot(o["payload_x"] - 1.0, o["payload_y"] - 2.0),
                     o["payload_z"])
        assert abs(r - 14.0) < 1e-12
