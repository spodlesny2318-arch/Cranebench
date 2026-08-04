"""Cooperative dual-crane lift of a rigid beam in the vertical plane.

Two trolleys travelling on a common runway at height ``H`` carry a rigid beam
through two visco-elastic falls.  Modelling the falls as spring-dampers rather
than as holonomic constraints keeps the system ODE (no index reduction, no
Lagrange multipliers) and exposes the load-sharing dynamics that a constrained
formulation hides.

Generalised coordinates ``q = [x1, x2, Xc, Zc, alpha]``: the two trolley
positions, the beam centre of mass and the beam pitch.  Two inputs drive five
degrees of freedom, so the system is underactuated by three.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from . import _generated as gen
from .base import LagrangianPlant

G = 9.80665


@dataclass
class DualParams:
    m_trolley: float = 1500.0
    m_beam: float = 12000.0
    length: float = 24.0        # m, beam length
    height: float = 30.0        # m, runway elevation
    k_cable: float = 4.0e6      # N/m, axial fall stiffness
    c_cable: float = 6.0e4      # N.s/m, axial fall damping
    rest: float = 14.0          # m, unstretched fall length
    u_rate: np.ndarray = field(default_factory=lambda: np.array([2.4e5, 2.4e5]))
    b_x: float = 250.0
    area: float = 40.0          # m^2, beam side area
    u_max: np.ndarray = field(default_factory=lambda: np.array([60.0e3, 60.0e3]))


class DualCrane(LagrangianPlant):
    nx = 10
    nu = 2
    actuated = (0, 1)
    unactuated = (2, 3, 4)
    horizontal_inputs = (0, 1)
    transfer_axes = (0, 1)   # both trolleys carry the beam, so both traverse
    antisway = (4,)
    state_names = ("x1", "x2", "Xc", "Zc", "alpha",
                   "x1d", "x2d", "Xcd", "Zcd", "alphad")

    def __init__(self, params: DualParams | None = None, analytic: bool = True):
        self.p = params or DualParams()
        super().__init__(nq=5)
        self.holomorphic = False   # unilateral falls: max(0, .) is not holomorphic
        self.analytic = analytic

    def _pars(self):
        p = self.p
        return (p.m_trolley, p.m_beam, p.length, p.height, p.k_cable, p.rest)

    def dynamics(self, t, x, u, d):
        """Closed-form fast path, valid only while both falls carry tension.

        The symbolic derivation assumes the smooth branch of the unilateral
        cable law, so the moment a fall goes slack the equations no longer
        describe the plant and the assembler takes over.  Silently using the
        taut equations on a slack fall would be exactly the kind of quiet error
        this package exists to make impossible.
        """
        if not self.analytic:
            return super().dynamics(t, x, u, d)
        q, qd = x[:5], x[5:]
        _, _, ln, _, _, _, _ = self._attach(q)
        if np.any(ln <= self.p.rest):
            return super().dynamics(t, x, u, d)
        M = gen.dual_M(*q, *qd, *self._pars())
        f = gen.dual_bias(*q, *qd, *self._pars())
        Q = self.generalised_forces(t, q, qd, u, d)
        return np.concatenate([qd, np.linalg.solve(M, f + Q)])

    # ---------------------------- geometry ---------------------------- #
    def _attach(self, q):
        """Fall geometry: attachment points, trolley points, lengths, units."""
        x1, x2, xc, zc, al = q
        r = 0.5 * self.p.length
        ca, sa = np.cos(al), np.sin(al)
        A = np.array([[xc - r * ca, zc - r * sa],
                      [xc + r * ca, zc + r * sa]])
        T = np.array([[x1, self.p.height], [x2, self.p.height]])
        dv = A - T
        ln = np.linalg.norm(dv, axis=1)
        e = dv / ln[:, None]
        return A, T, ln, e, r, ca, sa

    def reference_state(self, x0, pos, vel):
        """The beam travels with the trolleys, so its coordinate moves too."""
        xr = super().reference_state(x0, pos, vel)
        xr[2] = x0[2] + pos        # Xc
        xr[7] = vel                # Xcd
        return xr

    def swing_state(self, x):
        """Payload offset from the trolley midpoint, as an angle and its rate.

        Small-angle form about the vertical through the midpoint; the metric
        module uses the exact arctangent, which agrees with this to the order
        the anti-sway surface needs.
        """
        x1, x2, xc, zc = x[0], x[1], x[2], x[3]
        v1, v2, vc, vz = x[5], x[6], x[7], x[8]
        h = max(self.p.height - zc, 1e-6)
        mid, midd = 0.5 * (x1 + x2), 0.5 * (v1 + v2)
        theta = (xc - mid) / h
        rate = (vc - midd) / h + (xc - mid) * vz / (h * h)
        return float(theta), float(rate)

    def payload_velocity(self, x):
        return np.array([x[7], 0.0])          # beam centre of mass, horizontal

    def _dlen_dq(self, q):
        """d(fall length)/dq, shape (2, 5)."""
        _, _, _, e, r, ca, sa = self._attach(q)
        J = np.zeros((2, 5))
        for i, sgn in enumerate((-1.0, 1.0)):
            dAdq = np.zeros((2, 5))
            dAdq[:, i] = (-1.0, 0.0)             # trolley i
            dAdq[:, 2] = (1.0, 0.0)              # Xc
            dAdq[:, 3] = (0.0, 1.0)              # Zc
            dAdq[:, 4] = (-sgn * r * sa, sgn * r * ca)   # d/d(alpha)
            J[i] = e[i] @ dAdq
        return J

    # -------------------------- declaration --------------------------- #
    def point_masses(self, q):
        x1, x2, xc, zc, _ = q
        pos = np.array([[x1, 0.0, self.p.height],
                        [x2, 0.0, self.p.height],
                        [xc, 0.0, zc]])
        m = np.array([self.p.m_trolley, self.p.m_trolley, self.p.m_beam])
        return m, pos

    def added_inertia(self, q):
        A = np.zeros((5, 5))
        A[4, 4] = self.p.m_beam * self.p.length ** 2 / 12.0
        return A

    def potential(self, q):
        _, _, _, zc, _ = q
        _, _, ln, _, _, _, _ = self._attach(q)
        stretch = np.maximum(0.0, ln - self.p.rest)
        return (self.p.m_beam * G * zc
                + 0.5 * self.p.k_cable * float(np.sum(stretch ** 2)))

    def generalised_forces(self, t, q, qd, u, d):
        p = self.p
        u = np.clip(np.asarray(u, float), -p.u_max, p.u_max)
        d = np.atleast_1d(np.asarray(d, float))
        fw = float(d[0])
        J = self._dlen_dq(q)
        lnd = J @ qd                              # fall elongation rates
        _, _, ln, _, _, _, _ = self._attach(q)
        active = (ln > p.rest).astype(float)      # falls carry tension only
        Q = -(J.T @ (p.c_cable * lnd * active))
        Q[0] += u[0] - p.b_x * qd[0]
        Q[1] += u[1] - p.b_x * qd[1]
        Q[2] += fw                                 # wind on the beam
        return Q

    # ------------------------------ misc ------------------------------ #
    def initial_state(self):
        p = self.p
        sag = p.m_beam * G / (2.0 * p.k_cable)
        z = p.height - p.rest - sag
        return np.array([-0.5 * p.length, 0.5 * p.length, 0.0, z,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def outputs(self, x) -> Dict[str, float]:
        x1, x2, xc, zc, al = x[:5]
        _, _, ln, _, _, _, _ = self._attach(x[:5])
        stretch = np.maximum(0.0, ln - self.p.rest)
        tension = self.p.k_cable * stretch
        return {
            # The tracked output is the trolley midpoint, as it is the trolley
            # position on the single-crane plants.  Reporting the beam centre of
            # mass here instead made the arrival criterion unreachable by
            # construction: a steady wind hangs the payload 0.5 m downwind, and
            # no proportional law regulating the trolleys can remove that.  The
            # payload position is still reported, as "payload_x".
            "cart": float(0.5 * (x1 + x2)),
            "swing": float(np.arctan2(xc - 0.5 * (x1 + x2),
                                      max(self.p.height - zc, 1e-6))),
            "yaw": float(al),
            "pitch": float(al),
            "sync_err": float(x2 - x1 - self.p.length),
            "tension_1": float(tension[0]),
            "tension_2": float(tension[1]),
            "tension_ratio": float(tension.max() / max(tension.min(), 1.0)),
            "payload_x": float(xc),
            "payload_z": float(zc),
        }
