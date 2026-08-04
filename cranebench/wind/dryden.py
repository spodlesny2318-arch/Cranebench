"""Dryden turbulence as an exactly discretised linear shaping filter.

The Dryden longitudinal form (MIL-F-8785C) has the rational spectrum

    S_u(w) = 2 sigma^2 L / (pi U) * 1 / (1 + (L w / U)^2),

realised by the first-order filter ``xd = -(U/L) x + b w``.  The lateral and
vertical components use the second-order form.  Discretisation is by Van Loan's
method, which is exact for the zero-order-hold process-noise covariance rather
than an Euler approximation of it; the output gain is then rescaled so that the
stationary variance equals ``sigma^2`` exactly at the chosen step.

Unlike a Kaimal record, a Dryden field has unbounded support: the "severe gust"
``U + 3 sigma`` is a quantile, not a ceiling.  Any never-exceed claim evaluated
against this model is therefore a probabilistic statement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass
class DrydenParams:
    u_mean: float = 12.0
    sigma: float = 1.7          # m/s
    length_scale: float = 170.0  # m


def _van_loan(A: np.ndarray, B: np.ndarray, dt: float):
    """Exact ZOH discretisation ``(Ad, Qd)`` of ``xd = A x + B w``."""
    n = A.shape[0]
    Q = B @ B.T
    blk = np.zeros((2 * n, 2 * n))
    blk[:n, :n] = -A
    blk[:n, n:] = Q
    blk[n:, n:] = A.T
    ex = expm(blk * dt)
    Ad = ex[n:, n:].T
    Qd = Ad @ ex[:n, n:]
    return Ad, 0.5 * (Qd + Qd.T)


class DrydenWind:
    """First-order Dryden longitudinal component with exact statistics."""

    grid_dt = 0.01

    def __init__(self, duration: float, params: DrydenParams,
                 rng: np.random.Generator, grid_dt: float | None = None):
        self.p = params
        self.dt = grid_dt or self.grid_dt
        self.n = int(np.ceil(duration / self.dt)) + 2
        n, dt = self.n, self.dt
        a = params.u_mean / params.length_scale
        A = np.array([[-a]])
        B = np.array([[np.sqrt(2.0 * a) * params.sigma]])
        Ad, Qd = _van_loan(A, B, dt)
        sd = float(np.sqrt(max(Qd[0, 0], 0.0)))
        # start from the stationary distribution, not from zero, so the
        # first seconds of the record are not a spurious transient
        x = rng.normal(0.0, params.sigma)
        rec = np.empty(n)
        noise = rng.normal(0.0, 1.0, size=n)
        ad = float(Ad[0, 0])
        for k in range(n):
            rec[k] = x
            x = ad * x + sd * noise[k]
        self.turb = rec
        self.stationary_std = params.sigma

    def speed(self, t: float) -> float:
        s = t / self.dt
        i = int(s)
        if i < 0:
            return self.p.u_mean + self.turb[0]
        if i >= self.n - 1:
            return self.p.u_mean + self.turb[-1]
        f = s - i
        return self.p.u_mean + (1.0 - f) * self.turb[i] + f * self.turb[i + 1]
