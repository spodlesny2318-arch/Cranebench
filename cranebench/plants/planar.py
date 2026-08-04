"""Planar overhead crane with hoisting -- the reference plant of the benchmark.

Generalised coordinates ``q = [x, l, theta]``: trolley travel, rope length and
swing angle from the vertical.  The payload is a point mass; the winch carries
an equivalent translational inertia ``m_w``.

The equations of motion are

    (M_t + m) xdd + m s ldd + m l c thdd + 2 m ld thd c - m l thd^2 s = Q_x
    m s xdd + (m + m_w) ldd - m l thd^2 - m g c                       = Q_l
    m l c xdd + m l^2 thdd + 2 m l ld thd + m g l s                   = Q_th

with ``s = sin(theta)``, ``c = cos(theta)``.  This closed form is provided as a
fast path; ``tests/test_planar_consistency.py`` checks it against the generic
assembler of :mod:`cranebench.plants.base` to 1e-8 over random states, so the
analytic derivation is verified rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .base import LagrangianPlant

G = 9.80665


@dataclass
class PlanarParams:
    m_trolley: float = 1500.0   # kg
    m_payload: float = 4000.0   # kg
    m_winch: float = 300.0      # kg, equivalent translational inertia
    u_rate: np.ndarray = field(
        default_factory=lambda: np.array([1.6e5, 3.2e5]))  # N/s, drive slew limit
    b_x: float = 250.0          # N.s/m, trolley drive viscous damping
    b_l: float = 120.0          # N.s/m, hoist viscous damping
    c_theta: float = 40.0       # N.m.s/rad, aerodynamic + joint swing damping
    l0: float = 12.0            # m, initial rope length
    area: float = 6.0           # m^2, payload frontal area (wind coupling)
    cd: float = 1.2             # drag coefficient
    u_max: np.ndarray = field(
        default_factory=lambda: np.array([40.0e3, 80.0e3]))  # N


class PlanarCrane(LagrangianPlant):
    """Planar trolley/hoist/payload crane."""

    nx = 6
    nu = 2
    actuated = (0, 1)
    unactuated = (2,)
    horizontal_inputs = (0,)
    transfer_axes = (0,)
    antisway = (2,)
    state_names = ("x", "l", "theta", "xd", "ld", "thetad")

    def __init__(self, params: PlanarParams | None = None, analytic: bool = True):
        self.p = params or PlanarParams()
        self.analytic = analytic
        super().__init__(nq=3)

    # -------------------------- declaration --------------------------- #
    def point_masses(self, q):
        x, l, th = q
        pos = np.array([
            [x, 0.0, 0.0],                          # trolley
            [x + l * np.sin(th), 0.0, -l * np.cos(th)],  # payload
        ])
        m = np.array([self.p.m_trolley, self.p.m_payload])
        return m, pos

    def added_inertia(self, q):
        A = np.zeros((3, 3))
        A[1, 1] = self.p.m_winch
        return A

    def potential(self, q):
        _, l, th = q
        return -self.p.m_payload * G * l * np.cos(th)

    def generalised_forces(self, t, q, qd, u, d):
        x, l, th = q
        xd, ld, thd = qd
        p = self.p
        u = np.clip(u, -p.u_max, p.u_max)
        fw = float(d[0]) if np.ndim(d) else float(d)   # horizontal wind force [N]
        # wind acts on the payload; project through the payload Jacobian
        Jw = np.array([1.0, np.sin(th), l * np.cos(th)])
        Q = np.array([u[0] - p.b_x * xd, u[1] - p.b_l * ld, -p.c_theta * thd])
        return Q + fw * Jw

    # ---------------------------- fast path --------------------------- #
    def dynamics(self, t, x, u, d):
        if not self.analytic:
            return super().dynamics(t, x, u, d)
        p = self.p
        q, qd = x[:3], x[3:]
        _, l, th = q
        _, ld, thd = qd
        s, c = np.sin(th), np.cos(th)
        m, mt, mw = p.m_payload, p.m_trolley, p.m_winch
        l = max(l, 1e-3)

        M = np.array([
            [mt + m, m * s,  m * l * c],
            [m * s,  m + mw, 0.0],
            [m * l * c, 0.0, m * l * l],
        ])
        cor = np.array([
            2.0 * m * ld * thd * c - m * l * thd * thd * s,
            -m * l * thd * thd,
            2.0 * m * l * ld * thd,
        ])
        grav = np.array([0.0, -m * G * c, m * G * l * s])
        Q = self.generalised_forces(t, q, qd, u, d)
        qdd = np.linalg.solve(M, Q - cor - grav)
        return np.concatenate([qd, qdd])

    # ------------------------------ misc ------------------------------ #
    def initial_state(self):
        return np.array([0.0, self.p.l0, 0.0, 0.0, 0.0, 0.0])

    def outputs(self, x) -> Dict[str, float]:
        xt, l, th = x[0], x[1], x[2]
        return {
            "cart": float(xt),
            "rope": float(l),
            "swing": float(th),
            "yaw": 0.0,
            "payload_x": float(xt + l * np.sin(th)),
            "payload_z": float(-l * np.cos(th)),
        }

    def payload_velocity(self, x):
        """Horizontal velocity of the payload, for the relative-wind drag law."""
        _, l, th = x[0], x[1], x[2]
        _, ld, thd = x[3], x[4], x[5]
        return np.array([ld * np.sin(th) + l * thd * np.cos(th) + x[3], 0.0])

    def wind_force(self, speed: float) -> float:
        """Quasi-steady drag on the payload for a given relative wind speed."""
        rho = 1.225
        return 0.5 * rho * self.p.cd * self.p.area * speed * abs(speed)
