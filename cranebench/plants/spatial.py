"""Three-dimensional overhead crane with a yawing rigid payload.

Generalised coordinates ``q = [x, y, l, theta, phi, psi]``: trolley travel,
bridge travel, rope length, the two spherical-pendulum swing angles and the
payload yaw about the vertical.  The payload hangs on a rigid rope of
controlled length; its yaw is restrained by the torsional stiffness of the
suspension, which is the standard two-fall/spreader configuration.

Two modelling choices are worth stating because they are conventions rather
than measurements.  The torsional stiffness of the suspension is *gravitational*
rather than elastic: two falls a distance ``S`` apart act as a bifilar pendulum,
giving ``k_psi = W (S/2)^2 / L``, which contains neither the rope modulus nor its
cross-section.  For the nominal configuration (4 t payload, 2 m fall spacing,
12 m rope) that is 3.3e3 N.m/rad, and the default is set accordingly.

The wind acts through a centre of pressure offset from the payload centroid by
``cp_ecc * width``, producing a yaw moment.  The default 0.15 follows the
torsional wind load case of ASCE 7, which pairs 75 % of the full wind load with
an equivalent eccentricity of 15 % of the width; 5 % is the value below which
building codes treat torsion as negligible.  A suspended payload is not a
building, so this is a declared convention and the parameter is exposed for a
sensitivity sweep, not a measured property of any particular load.

The swing parameterisation is

    p_rel = l * (sin(th) cos(ph), sin(ph), -cos(th) cos(ph)),

which has exact norm ``l`` for all angles and therefore does not degrade at
large swing, unlike the small-angle ``(l*th_x, l*th_y)`` form common in the
literature.

Scope note.  The rope is inextensible and the yaw axis is taken vertical.
Elastic-cable and full attitude (quaternion) variants are deliberately outside
the baseline: the benchmark fixes a common bench, not a state of the art.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from . import _generated as gen
from .base import LagrangianPlant

G = 9.80665


@dataclass
class SpatialParams:
    m_bridge: float = 4000.0
    m_trolley: float = 1500.0
    m_payload: float = 4000.0
    m_winch: float = 300.0
    izz: float = 2600.0        # kg.m^2, payload yaw inertia
    width: float = 2.0         # m, payload width across the wind
    area: float = 4.0          # m^2, projected frontal area
    cd: float = 1.2            # drag coefficient of the bluff payload
    cp_ecc: float = 0.15       # centre-of-pressure offset, fraction of width
    fall_spacing: float = 2.0  # m, distance between the two falls
    k_psi: float = 3.3e3       # N.m/rad, suspension torsional stiffness
    c_psi: float = 2.9e2       # N.m.s/rad
    u_rate: np.ndarray = field(
        default_factory=lambda: np.array([1.6e5, 1.8e5, 3.2e5]))
    b_x: float = 250.0
    b_y: float = 300.0
    b_l: float = 120.0
    c_swing: float = 40.0
    l0: float = 12.0
    u_max: np.ndarray = field(
        default_factory=lambda: np.array([40.0e3, 45.0e3, 80.0e3]))


class SpatialCrane(LagrangianPlant):
    nx = 12
    nu = 3
    actuated = (0, 1, 2)
    unactuated = (3, 4, 5)
    horizontal_inputs = (0, 1)
    transfer_axes = (0,)
    antisway = (3, 4, 5)
    state_names = ("x", "y", "l", "theta", "phi", "psi",
                   "xd", "yd", "ld", "thetad", "phid", "psid")

    def __init__(self, params: SpatialParams | None = None, analytic: bool = True):
        self.p = params or SpatialParams()
        self.analytic = analytic
        super().__init__(nq=6)

    def _pars(self):
        p = self.p
        return (p.m_bridge, p.m_trolley, p.m_payload, p.m_winch, p.izz, p.k_psi)

    def dynamics(self, t, x, u, d):
        """Closed-form fast path from the symbolic derivation.

        ``tools/derive_symbolic.py`` produced these equations with SymPy's
        ``LagrangesMethod``, which shares no code with the numeric assembler of
        :mod:`cranebench.plants.base`; ``tests/test_symbolic.py`` requires the
        two to agree, so the fast path is verified rather than trusted.
        """
        if not self.analytic:
            return super().dynamics(t, x, u, d)
        q, qd = x[:6], x[6:]
        M = gen.spatial_M(*q, *qd, *self._pars())
        f = gen.spatial_bias(*q, *qd, *self._pars())
        Q = self.generalised_forces(t, q, qd, u, d)
        return np.concatenate([qd, np.linalg.solve(M, f + Q)])

    # -------------------------- declaration --------------------------- #
    def point_masses(self, q):
        x, y, l, th, ph, _ = q
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        pos = np.array([
            [0.0, y, 0.0],                                       # bridge girder
            [x, y, 0.0],                                         # trolley
            [x + l * st * cp, y + l * sp, -l * ct * cp],         # payload
        ])
        m = np.array([self.p.m_bridge, self.p.m_trolley, self.p.m_payload])
        return m, pos

    def added_inertia(self, q):
        A = np.zeros((6, 6))
        A[2, 2] = self.p.m_winch
        A[5, 5] = self.p.izz
        return A

    def potential(self, q):
        _, _, l, th, ph, psi = q
        v = -self.p.m_payload * G * l * np.cos(th) * np.cos(ph)
        return v + 0.5 * self.p.k_psi * psi ** 2

    def payload_jacobian(self, q):
        """d p_payload / d q, shape (3, 6)."""
        _, _, l, th, ph, _ = q
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        J = np.zeros((3, 6))
        J[:, 0] = (1.0, 0.0, 0.0)
        J[:, 1] = (0.0, 1.0, 0.0)
        J[:, 2] = (st * cp, sp, -ct * cp)
        J[:, 3] = (l * ct * cp, 0.0, l * st * cp)
        J[:, 4] = (-l * st * sp, l * cp, l * ct * sp)
        return J

    def payload_velocity(self, x):
        v = self.payload_jacobian(x[:6]) @ x[6:]
        return np.array([v[0], v[1]])

    def generalised_forces(self, t, q, qd, u, d):
        p = self.p
        u = np.clip(np.asarray(u, float), -p.u_max, p.u_max)
        d = np.atleast_1d(np.asarray(d, float))
        fx, fy = (d[0], d[1]) if d.size >= 2 else (d[0], 0.0)
        mz = d[2] if d.size >= 3 else 0.0
        Q = np.zeros(6)
        Q[0] = u[0] - p.b_x * qd[0]
        Q[1] = u[1] - p.b_y * qd[1]
        Q[2] = u[2] - p.b_l * qd[2]
        Q[3] = -p.c_swing * qd[3]
        Q[4] = -p.c_swing * qd[4]
        Q[5] = -p.c_psi * qd[5] + mz
        Q += self.payload_jacobian(q).T @ np.array([fx, fy, 0.0])
        return Q

    # ------------------------------ misc ------------------------------ #
    def initial_state(self):
        x = np.zeros(12)
        x[2] = self.p.l0
        return x

    def outputs(self, x) -> Dict[str, float]:
        xt, yt, l, th, ph, psi = x[:6]
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        swing = float(np.arccos(np.clip(ct * cp, -1.0, 1.0)))
        return {
            "cart": float(xt),
            "bridge": float(yt),
            "rope": float(l),
            "swing": swing,
            "theta": float(th),
            "phi": float(ph),
            "yaw": float(psi),
            "payload_x": float(xt + l * st * cp),
            "payload_y": float(yt + l * sp),
            "payload_z": float(-l * ct * cp),
        }
