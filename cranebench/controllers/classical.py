"""PD, LQR and ZVD input shaping -- the three standard baselines."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

from .base import Controller, input_matrix, state_matrix, trim

G = 9.80665


class PD(Controller):
    """Position-only proportional-derivative control on the actuated axes.

    Carries no payload state, and is therefore the control experiment for any
    claim that a controller "needs" the swing state: whatever a richer design
    achieves must be measured against what this achieves without it.
    """

    name = "PD"

    def __init__(self, kp=6.0e3, kd=2.4e4, kp_hoist=4.0e4, kd_hoist=6.0e4):
        self.kp, self.kd = kp, kd
        self.kph, self.kdh = kp_hoist, kd_hoist

    def reset(self, plant, manoeuvre):
        super().reset(plant, manoeuvre)
        _, self.u_eq = trim(plant)

    def __call__(self, t, x):
        p, man = self.plant, self.man
        u = np.array(self.u_eq, float)
        nq = len(p.state_names) // 2
        transfer = getattr(p, "transfer_axes", (p.actuated[0],))
        for k, i in enumerate(p.actuated):
            if p.state_names[i] == "l":
                u[k] += (self.kph * (man.rope(t, self.x0[i]) - x[i])
                         + self.kdh * (man.rope_rate(t) - x[nq + i]))
            elif i in transfer:
                # axes that execute the transfer follow it from where they started
                u[k] += (self.kp * (self.x0[i] + man.position(t) - x[i])
                         + self.kd * (man.velocity(t) - x[nq + i]))
            else:
                u[k] += self.kp * (self.x0[i] - x[i]) + self.kd * (0.0 - x[nq + i])
        return u


class LQR(Controller):
    """Infinite-horizon LQR on the numerically linearised plant.

    The linearisation point, the weights and the solver are all recorded, so
    the baseline is reproducible rather than "an LQR we tuned".
    """

    name = "LQR"

    def __init__(self, q_pos=60.0, q_swing=400.0, q_rate=8.0, r=2.0e-7):
        self.q_pos, self.q_swing, self.q_rate, self.r = q_pos, q_swing, q_rate, r

    def reset(self, plant, manoeuvre):
        super().reset(plant, manoeuvre)
        x0, self.u_eq = trim(plant)
        A = state_matrix(plant, x0, self.u_eq)
        B = input_matrix(plant, x0, self.u_eq)
        nq = plant.nx // 2
        q = np.full(plant.nx, self.q_rate)
        for i in plant.actuated:
            q[i] = self.q_pos
        for i in plant.unactuated:
            q[i] = self.q_swing
        Q = np.diag(q)
        R = np.eye(plant.nu) * self.r
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ P)
        self.nq = nq

    def _xref(self, t):
        p, man = self.plant, self.man
        xr = p.reference_state(self.x0, man.position(t), man.velocity(t))
        for j in p.actuated:
            if p.state_names[j] == "l":
                xr[j] = man.rope(t, self.x0[j])
                xr[self.nq + j] = man.rope_rate(t)
        return xr

    def __call__(self, t, x):
        return self.u_eq - self.K @ (x - self._xref(t))


class ZVD(Controller):
    """Zero-vibration-derivative input shaping in cascade with PD tracking.

    The shaper is applied to the *reference*, not to the control signal, which
    is the standard command-shaping architecture.  It is tuned to the pendulum
    frequency at the initial rope length and is therefore expected to degrade
    when the rope length or the payload mass departs from nominal -- that
    degradation is part of what the benchmark is meant to expose.
    """

    name = "ZVD"

    def __init__(self, zeta=0.02, kp=6.0e3, kd=2.4e4, kp_hoist=4.0e4, kd_hoist=6.0e4):
        self.zeta = zeta
        self.inner = PD(kp, kd, kp_hoist, kd_hoist)

    def reset(self, plant, manoeuvre):
        super().reset(plant, manoeuvre)
        self.inner.reset(plant, manoeuvre)
        l0 = plant.initial_state()[list(plant.state_names).index("l")] \
            if "l" in plant.state_names else 12.0
        wn = np.sqrt(G / max(l0, 1e-3))
        z = self.zeta
        wd = wn * np.sqrt(max(1.0 - z * z, 1e-9))
        K = np.exp(-z * np.pi / np.sqrt(max(1.0 - z * z, 1e-9)))
        den = (1.0 + K) ** 2
        self.amp = np.array([1.0, 2.0 * K, K * K]) / den
        self.tau = np.array([0.0, np.pi / wd, 2.0 * np.pi / wd])

    def __call__(self, t, x):
        man = self.man
        pos = float(np.sum(self.amp * [man.position(t - d) for d in self.tau]))
        vel = float(np.sum(self.amp * [man.velocity(t - d) for d in self.tau]))

        class _Shaped:
            def position(_s, tt):
                return pos

            def velocity(_s, tt):
                return vel

            rope = man.rope
            rope_rate = man.rope_rate

        saved = self.inner.man
        self.inner.man = _Shaped()
        try:
            return self.inner(t, x)
        finally:
            self.inner.man = saved
