"""Boundary-layer SMC and hierarchical SMC -- the two sliding baselines.

Both are the textbook forms.  ``SMC`` slides on the actuated tracking error
only and therefore has no mechanism to damp the payload; ``HSMC`` folds the
unactuated coordinate into a composite surface, which is the standard
hierarchical construction.  Neither is a contribution of this package, and the
gains are exposed so that a user can re-tune them on their own bench.

The switching term uses ``tanh(s / phi)`` rather than ``sign(s)``.  This is not
cosmetic: with ``sign`` the measured control effort of a sliding controller
depends on the integrator step, which makes effort comparisons between
controllers meaningless.  The boundary layer is reported with the results.
"""

from __future__ import annotations

import numpy as np

from .base import Controller, trim


class SMC(Controller):
    """Sliding-mode control on the actuated coordinates, boundary layer phi."""

    name = "SMC"

    def __init__(self, c=1.1, k=2.2e4, eta=6.0e3, phi=0.06,
                 c_hoist=2.0, k_hoist=6.0e4, eta_hoist=1.0e4):
        self.c, self.k, self.eta, self.phi = c, k, eta, phi
        self.ch, self.kh, self.etah = c_hoist, k_hoist, eta_hoist

    def reset(self, plant, manoeuvre):
        super().reset(plant, manoeuvre)
        _, self.u_eq = trim(plant)
        self.nq = plant.nx // 2

    def _surfaces(self, t, x):
        p, man, nq = self.plant, self.man, self.nq
        transfer = getattr(p, "transfer_axes", (p.actuated[0],))
        s, gains = [], []
        for k, i in enumerate(p.actuated):
            if i in transfer:
                e = x[i] - (self.x0[i] + man.position(t))
                ed = x[nq + i] - man.velocity(t)
                s.append(ed + self.c * e)
                gains.append((self.k, self.eta))
            elif p.state_names[i] == "l":
                e = x[i] - man.rope(t, self.x0[i])
                ed = x[nq + i] - man.rope_rate(t)
                s.append(ed + self.ch * e)
                gains.append((self.kh, self.etah))
            else:
                e = x[i] - self.x0[i]
                s.append(x[nq + i] + self.c * e)
                gains.append((self.k, self.eta))
        return np.array(s), gains

    def __call__(self, t, x):
        s, gains = self._surfaces(t, x)
        u = np.array(self.u_eq, float)
        for j, (kj, etaj) in enumerate(gains):
            u[j] += -kj * s[j] - etaj * np.tanh(s[j] / self.phi)
        return u


class HSMC(SMC):
    """Hierarchical SMC: composite surface S = s_actuated + lam * s_swing.

    The swing sub-surface is ``s_sw = thetad + c_sw * theta``.  Only the first
    actuated axis carries the composite surface; the remaining axes keep their
    own first-layer surfaces, which is the usual arrangement for a crane whose
    hoist is not used for anti-sway.
    """

    name = "HSMC"

    def __init__(self, lam=0.40, c_swing=1.30, **kw):
        super().__init__(**kw)
        self.lam, self.c_swing = lam, c_swing

    def __call__(self, t, x):
        s, gains = self._surfaces(t, x)
        theta, theta_dot = self.plant.swing_state(x)
        s_sw = theta_dot + self.c_swing * theta
        s = s.copy()
        s[0] = s[0] + self.lam * s_sw
        u = np.array(self.u_eq, float)
        for j, (kj, etaj) in enumerate(gains):
            u[j] += -kj * s[j] - etaj * np.tanh(s[j] / self.phi)
        return u
