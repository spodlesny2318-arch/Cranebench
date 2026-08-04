"""Fixed-step integrators.

A fixed step is used deliberately.  An adaptive solver silently changes the
effective sampling of a discontinuous or near-discontinuous control law, so two
controllers compared under an adaptive solver are not compared under the same
conditions.  The step is recorded in the ledger, and
``tests/test_convergence.py`` checks that the reported metrics are converged in
it.
"""

from __future__ import annotations

import numpy as np


def rk4(f, t, x, dt, *args):
    k1 = f(t, x, *args)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1, *args)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2, *args)
    k4 = f(t + dt, x + dt * k3, *args)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(plant, controller, manoeuvre, wind=None, dt=2.0e-3,
             control_dt=1.0e-2, force_fn=None, rate_limit=True):
    """Run one closed-loop simulation.

    The controller is evaluated on its own (slower) grid and held between
    updates, which is what a real drive does and which prevents a controller
    from buying performance with an unrealistically fast loop.
    """
    n = int(round(manoeuvre.t_total / dt)) + 1
    t = np.arange(n) * dt
    x = np.array(plant.initial_state(), float)
    controller.reset(plant, manoeuvre)

    X = np.empty((n, plant.nx))
    U = np.empty((n, plant.nu))
    every = max(1, int(round(control_dt / dt)))
    u_rate = getattr(plant.p, "u_rate", None) if rate_limit else None
    u = np.asarray(controller(0.0, x), float)

    for k in range(n):
        if k % every == 0:
            u = slew(np.asarray(controller(t[k], x), float), u, u_rate, control_dt)
        X[k] = x
        U[k] = u
        if k == n - 1:
            break
        d = force_fn(t[k], x, wind) if force_fn is not None else np.zeros(3)
        x = rk4(plant.dynamics, t[k], x, dt, u, d)
        if not np.all(np.isfinite(x)):
            X[k + 1:] = np.nan
            U[k + 1:] = np.nan
            break
    return t, X, U


RHO_AIR = 1.225


def drag_force(plant, relative=True):
    """Quasi-steady drag on the payload.

    The force is

        F = 0.5 rho C_D A |v_rel| v_rel,   v_rel = v_wind - v_payload,

    with the yaw moment ``F e`` of a centre of pressure offset by ``e`` from the
    centroid for plants that declare one.

    ``relative=True`` is the physically correct form and the default.  Using the
    absolute wind speed instead removes the aerodynamic damping entirely: for
    the nominal planar plant, ``d|F|/dv = rho C_D A U`` is 106 N.s/m against
    0.28 N.s/m for the modelled swing damping, so the absolute form omits the
    dominant dissipation by a factor of about 380 and overstates residual swing
    for every controller.  The flag exists so that the effect can be measured
    rather than argued about; ``examples/run_ablation.py`` does exactly that.

    The flow is treated as quasi-steady: no added mass, no vortex shedding, no
    dynamic stall.  That is a convention of the bench, not a claim about bluff
    bodies, and it is the first thing to revisit for a payload whose Strouhal
    frequency falls near a structural mode.
    """
    cd = getattr(plant.p, "cd", 1.2)
    area = getattr(plant.p, "area", 6.0)
    lever = getattr(plant.p, "cp_ecc", 0.0) * getattr(plant.p, "width", 0.0)
    k = 0.5 * RHO_AIR * cd * area

    def fn(t, x, wind):
        if wind is None:
            return np.zeros(3)
        v = wind.speed(t)
        if relative:
            v = v - plant.payload_velocity(x)[0]
        f = k * v * abs(v)
        return np.array([f, 0.0, f * lever])

    return fn


def slew(u_cmd, u_prev, u_rate, dt):
    """Limit how fast the drive command may change.

    A drive cannot step its force; without this the chatter metric rewards
    commands no actuator could follow, and the sliding baselines are compared
    on an authority they would not have.
    """
    if u_rate is None:
        return u_cmd
    lim = np.asarray(u_rate, float) * dt
    return u_prev + np.clip(u_cmd - u_prev, -lim, lim)
