"""Batched campaign path for the planar plant.

The scalar path of :mod:`cranebench.runner` is the reference implementation:
readable, general over all three plants, and the one the verification tests are
written against.  It is also slow -- roughly 0.8 s per 40 s run -- which puts a
2500-run campaign at about half an hour of single-core time and a 10^4-sample
campaign out of reach.

This module integrates the whole Monte Carlo ensemble at once: the state is an
``(N, nx)`` array, the plant parameters are ``(N,)`` arrays, and one RK4 step
advances every sample together.  The controllers are the same control laws
written elementwise.  Gains that require a per-sample setup (the LQR
linearisation and Riccati solution, the equilibrium input, the ZVD shaper
timing) are computed by calling the *scalar* code on a scalar plant built from
that sample's parameters, so the batched path cannot drift away from the
reference by re-deriving them.

``tests/test_batch.py`` asserts that the two paths agree to 1e-9 on every metric
over a full paired design.  Only the planar plant is batched; the assembled
spatial and dual plants keep the scalar path.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .controllers.base import input_matrix, state_matrix, trim
from .metrics import compute_metrics
from .plants import PlanarCrane
from .plants.planar import G
from .reference import Manoeuvre
from .uncertainty import apply_factors
from .wind import WIND_PARAMS, WINDS

FIELDS = ("m_trolley", "m_payload", "m_winch", "b_x", "b_l", "c_theta",
          "l0", "area", "cd")


# --------------------------------------------------------------------- #
# plant
# --------------------------------------------------------------------- #
@dataclass
class BatchPlanar:
    """Planar crane replicated over ``N`` parameter samples."""

    n: int
    par: Dict[str, np.ndarray]
    u_max: np.ndarray                      # (N, 2)

    @classmethod
    def from_samples(cls, plants: List[PlanarCrane]):
        par = {f: np.array([getattr(p.p, f) for p in plants], float)
               for f in FIELDS}
        u_max = np.array([p.p.u_max for p in plants], float)
        return cls(len(plants), par, u_max)

    def initial_state(self) -> np.ndarray:
        X = np.zeros((self.n, 6))
        X[:, 1] = self.par["l0"]
        return X

    def dynamics(self, t, X, U, fw):
        p = self.par
        l = np.maximum(X[:, 1], 1e-3)
        th = X[:, 2]
        xd, ld, thd = X[:, 3], X[:, 4], X[:, 5]
        s, c = np.sin(th), np.cos(th)
        m, mt, mw = p["m_payload"], p["m_trolley"], p["m_winch"]

        M = np.empty((self.n, 3, 3))
        M[:, 0, 0] = mt + m
        M[:, 0, 1] = M[:, 1, 0] = m * s
        M[:, 0, 2] = M[:, 2, 0] = m * l * c
        M[:, 1, 1] = m + mw
        M[:, 1, 2] = M[:, 2, 1] = 0.0
        M[:, 2, 2] = m * l * l

        rhs = np.empty((self.n, 3))
        Uc = np.clip(U, -self.u_max, self.u_max)
        # generalised forces: drives, damping, wind through the payload Jacobian
        rhs[:, 0] = Uc[:, 0] - p["b_x"] * xd + fw
        rhs[:, 1] = Uc[:, 1] - p["b_l"] * ld + fw * s
        rhs[:, 2] = -p["c_theta"] * thd + fw * l * c
        # Coriolis / centrifugal
        rhs[:, 0] -= 2.0 * m * ld * thd * c - m * l * thd * thd * s
        rhs[:, 1] -= -m * l * thd * thd
        rhs[:, 2] -= 2.0 * m * l * ld * thd
        # gravity
        rhs[:, 1] -= -m * G * c
        rhs[:, 2] -= m * G * l * s

        qdd = np.linalg.solve(M, rhs[:, :, None])[:, :, 0]
        return np.concatenate([X[:, 3:], qdd], axis=1)

    def payload_velocity_x(self, X):
        """Horizontal payload velocity for the relative-wind drag law."""
        l, th = X[:, 1], X[:, 2]
        return X[:, 3] + X[:, 4] * np.sin(th) + l * X[:, 5] * np.cos(th)

    def outputs(self, X):
        return {"cart": X[:, 0], "swing": X[:, 2],
                "yaw": np.zeros(X.shape[0])}


# --------------------------------------------------------------------- #
# controllers
# --------------------------------------------------------------------- #
class BatchController:
    name = "base"

    def setup(self, plants, man, batch):
        self.plants, self.man, self.b = plants, man, batch
        self.X0 = batch.initial_state()
        self.u_eq = np.array([trim(p)[1] for p in plants], float)

    def __call__(self, t, X):
        raise NotImplementedError


class BPD(BatchController):
    name = "PD"

    def __init__(self, ref):
        self.r = ref                       # the scalar controller, for its gains

    def _pd(self, t, X, pos, vel):
        r = self.r
        U = self.u_eq.copy()
        U[:, 0] += r.kp * (pos - X[:, 0]) + r.kd * (vel - X[:, 3])
        rope = self.X0[:, 1] + self.man.hoist * self.man._s(t)
        U[:, 1] += r.kph * (rope - X[:, 1]) + r.kdh * (self.man.rope_rate(t) - X[:, 4])
        return U

    def __call__(self, t, X):
        return self._pd(t, X, self.man.position(t), self.man.velocity(t))


class BZVD(BPD):
    name = "ZVD"

    def setup(self, plants, man, batch):
        super().setup(plants, man, batch)
        r = self.r
        z = r.zeta
        wn = np.sqrt(G / np.maximum(self.X0[:, 1], 1e-3))
        wd = wn * np.sqrt(max(1.0 - z * z, 1e-9))
        K = np.exp(-z * np.pi / np.sqrt(max(1.0 - z * z, 1e-9)))
        self.amp = np.stack([np.full(batch.n, 1.0), np.full(batch.n, 2.0 * K),
                             np.full(batch.n, K * K)], 1) / (1.0 + K) ** 2
        self.tau = np.stack([np.zeros(batch.n), np.pi / wd, 2.0 * np.pi / wd], 1)
        self.r = r.inner                   # inner PD carries the tracking gains
        self.r.kph, self.r.kdh = r.inner.kph, r.inner.kdh

    def __call__(self, t, X):
        td = t - self.tau
        pos = np.sum(self.amp * self.man.position_v(td), axis=1)
        vel = np.sum(self.amp * self.man.velocity_v(td), axis=1)
        return self._pd(t, X, pos, vel)


class BLQR(BatchController):
    name = "LQR"

    def __init__(self, ref):
        self.r = ref

    def setup(self, plants, man, batch):
        super().setup(plants, man, batch)
        from scipy.linalg import solve_continuous_are
        r = self.r
        Ks = []
        for i, p in enumerate(plants):
            x0, ue = p.initial_state(), self.u_eq[i]
            A = state_matrix(p, x0, ue)
            B = input_matrix(p, x0, ue)
            q = np.full(p.nx, r.q_rate)
            for j in p.actuated:
                q[j] = r.q_pos
            for j in p.unactuated:
                q[j] = r.q_swing
            R = np.eye(p.nu) * r.r
            P = solve_continuous_are(A, B, np.diag(q), R)
            Ks.append(np.linalg.solve(R, B.T @ P))
        self.K = np.array(Ks)

    def __call__(self, t, X):
        Xr = self.X0.copy()
        Xr[:, 0] = self.man.position(t)
        Xr[:, 3] = self.man.velocity(t)
        Xr[:, 1] = self.X0[:, 1] + self.man.hoist * self.man._s(t)
        Xr[:, 4] = self.man.rope_rate(t)
        return self.u_eq - np.einsum("nij,nj->ni", self.K, X - Xr)


class BSMC(BatchController):
    name = "SMC"

    def __init__(self, ref):
        self.r = ref

    def _surfaces(self, t, X):
        r, man = self.r, self.man
        s0 = (X[:, 3] - man.velocity(t)) + r.c * (X[:, 0] - man.position(t))
        rope = self.X0[:, 1] + man.hoist * man._s(t)
        s1 = (X[:, 4] - man.rope_rate(t)) + r.ch * (X[:, 1] - rope)
        return s0, s1

    def __call__(self, t, X):
        r = self.r
        s0, s1 = self._surfaces(t, X)
        U = self.u_eq.copy()
        U[:, 0] += -r.k * s0 - r.eta * np.tanh(s0 / r.phi)
        U[:, 1] += -r.kh * s1 - r.etah * np.tanh(s1 / r.phi)
        return U


class BHSMC(BSMC):
    name = "HSMC"

    def __call__(self, t, X):
        r = self.r
        s0, s1 = self._surfaces(t, X)
        s0 = s0 + r.lam * (X[:, 5] + r.c_swing * X[:, 2])
        U = self.u_eq.copy()
        U[:, 0] += -r.k * s0 - r.eta * np.tanh(s0 / r.phi)
        U[:, 1] += -r.kh * s1 - r.etah * np.tanh(s1 / r.phi)
        return U


BATCH_OF = {"PD": BPD, "LQR": BLQR, "ZVD": BZVD, "SMC": BSMC, "HSMC": BHSMC}


# --------------------------------------------------------------------- #
# campaign
# --------------------------------------------------------------------- #
def _build(design, campaign):
    plants, winds = [], []
    base_p = PlanarCrane().p.__class__()
    base_w = WIND_PARAMS.get(campaign.wind, WIND_PARAMS["kaimal"])()
    for i, fac in enumerate(design.as_dicts()):
        pp, wp = apply_factors(copy.deepcopy(base_p), copy.deepcopy(base_w), fac)
        plants.append(PlanarCrane(pp))
        if campaign.wind not in (None, "none"):
            rng = np.random.default_rng(int(design.wind_seeds[i]))
            winds.append(WINDS[campaign.wind](campaign.manoeuvre.t_total, wp, rng))
    return plants, winds


def run_campaign_batch(campaign, design, progress=True, relative=True,
                       rate_limit=True, rate_scale=1.0):
    """Integrate the whole ensemble at once; returns the same dict as the scalar path."""
    campaign.build()
    man = campaign.manoeuvre
    dt = campaign.dt
    plants, winds = _build(design, campaign)
    batch = BatchPlanar.from_samples(plants)
    n = design.n

    if winds:
        grid_dt = winds[0].dt
        turb = np.array([w.turb for w in winds])          # (N, ngrid)
        u_mean = np.array([w.p.u_mean for w in winds])
        ngrid = turb.shape[1]
    rho = 1.225
    kdrag = 0.5 * rho * batch.par["cd"] * batch.par["area"]

    def wind_force(t, X=None):
        if not winds:
            return np.zeros(n)
        s = t / grid_dt
        i = int(s)
        if i < 0:
            v = u_mean + turb[:, 0]
        elif i >= ngrid - 1:
            v = u_mean + turb[:, -1]
        else:
            f = s - i
            v = u_mean + (1 - f) * turb[:, i] + f * turb[:, i + 1]
        if X is not None and relative:
            v = v - batch.payload_velocity_x(X)
        return kdrag * v * np.abs(v)

    nt = int(round(man.t_total / dt)) + 1
    tgrid = np.arange(nt) * dt
    every = max(1, int(round(campaign.control_dt / dt)))

    out = {}
    for cname, scalar_ctrl in campaign.controllers.items():
        ctrl = BATCH_OF[cname](scalar_ctrl)
        ctrl.setup(plants, man, batch)
        X = batch.initial_state()
        cart = np.empty((n, nt))
        swing = np.empty((n, nt))
        Uh = np.empty((n, nt, 2))
        u_rate = (np.asarray(plants[0].p.u_rate, float) * rate_scale
                  if rate_limit else None)
        U = ctrl(0.0, X)
        for k in range(nt):
            if k % every == 0:
                cmd = ctrl(tgrid[k], X)
                if u_rate is not None:
                    lim = u_rate * campaign.control_dt
                    U = U + np.clip(cmd - U, -lim, lim)
                else:
                    U = cmd
            cart[:, k], swing[:, k] = X[:, 0], X[:, 2]
            Uh[:, k] = U
            if k == nt - 1:
                break
            t = tgrid[k]
            # the disturbance is frozen across the four RK4 stages, exactly as
            # the scalar reference path does it; re-evaluating it mid-stage
            # would be defensible but would no longer be the same experiment
            fw = wind_force(t, X)
            k1 = batch.dynamics(t, X, U, fw)
            k2 = batch.dynamics(t + .5 * dt, X + .5 * dt * k1, U, fw)
            k3 = batch.dynamics(t + .5 * dt, X + .5 * dt * k2, U, fw)
            k4 = batch.dynamics(t + dt, X + dt * k3, U, fw)
            X = X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ref = man.position_v(tgrid)
        rows = []
        for i in range(n):
            outs = [{"cart": cart[i, k], "swing": swing[i, k], "yaw": 0.0}
                    for k in range(nt)]
            rows.append(compute_metrics(tgrid, outs, Uh[i], ref,
                                        horizontal_inputs=(0,),
                                        swing_bound_deg=campaign.swing_bound_deg
                                        ).as_dict())
        out[cname] = rows
        if progress:
            print(f"  {cname}: {n} runs done", flush=True)
    keys = out[next(iter(out))][0].keys()
    return {c: {k: np.array([r[k] for r in rows], float) for k in keys}
            for c, rows in out.items()}
