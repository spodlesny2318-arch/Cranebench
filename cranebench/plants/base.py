"""Plant interface and a generic Lagrangian assembler.

A plant is declared by its *kinematics* (where the point masses are, as a
function of the generalised coordinates), an additive generalised inertia (for
rotational and drive-train degrees of freedom), a potential, and the
generalised forces.  The equations of motion are then assembled numerically:

    M(q) qdd + Mdot(q, qd) qd - dT/dq + dV/dq = Q(t, q, qd, u, d)

Assembling rather than hand-deriving is deliberate.  Hand-derived spatial crane
models are a well known source of silent errors, and the assembler lets every
plant be checked against energy conservation by the test suite (see
``tests/test_energy.py``).  Plants for which a closed form is cheap may
override ``dynamics`` with an analytic fast path; ``PlanarCrane`` does, and the
test suite cross-validates the two implementations against each other.

Kinematic Jacobians are taken by the complex-step derivative,
``J = Im[p(q + i h e_k)] / h``, which carries no subtractive cancellation and is
accurate to machine precision.  This matters here more than it usually does: the
Coriolis term needs a derivative *of* the mass matrix, so a central-difference
Jacobian would be differentiated a second time and its 1e-10 noise floor would
be amplified to 1e-4 in the accelerations.  Plants whose kinematics are not
holomorphic (a unilateral contact, a saturation) set ``holomorphic = False`` and
fall back to central differences.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


class Plant(ABC):
    """Minimal plant interface used by the benchmark harness."""

    #: number of state variables
    nx: int
    #: number of control inputs
    nu: int
    #: human-readable names of the states, used in result files
    state_names: tuple

    @abstractmethod
    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray,
                 d: np.ndarray) -> np.ndarray:
        """State derivative.  Must be free of side effects."""

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        """Nominal initial condition."""

    @abstractmethod
    def outputs(self, x: np.ndarray) -> Dict[str, float]:
        """Standardised observables consumed by the frozen metric module.

        Every plant must supply at least:

        ``pos_err_x``  horizontal position error of the actuated frame [m]
        ``swing``      total payload swing angle from vertical [rad]
        ``yaw``        payload rotation about the vertical axis [rad]
        """

    def reference_state(self, x0, pos, vel):
        """Full state the plant should be in when the transfer has reached ``pos``.

        Only a state-feedback law needs this.  The default moves the actuated
        transfer axes and leaves everything else at its initial value, which is
        right when the payload position is not itself a state.  On the dual
        crane it is: the beam centre of mass travels with the trolleys, and
        leaving it at its initial value told the regulator to move the trolleys
        ten metres while holding the beam still.  It obliged, and never arrived.
        """
        xr = np.array(x0, float)
        nq = self.nx // 2
        for i in getattr(self, "transfer_axes", (self.actuated[0],)):
            xr[i] = x0[i] + pos
            xr[nq + i] = vel
        return xr

    def swing_state(self, x: np.ndarray):
        """Return ``(angle, rate)`` of the coordinate an anti-sway law acts on.

        For a single crane this is a generalised coordinate and the default
        below is correct.  It is not always: on the dual crane the swing is the
        offset of the payload from the midpoint of the two trolleys, which is no
        single state, and the declared ``antisway`` coordinate -- the beam pitch
        -- is not excited by a synchronised transfer at all.  Building a
        hierarchical surface on it made the hierarchical baseline numerically
        identical to the flat one, which the dual campaign duly showed.
        """
        i = self.antisway[0]
        nq = self.nx // 2
        return float(x[i]), float(x[nq + i])

    def energy(self, x: np.ndarray) -> float:
        """Total mechanical energy; used by the conservation tests."""
        raise NotImplementedError


@dataclass
class LagrangianPlant(Plant):
    """Assembles the equations of motion from a kinematic declaration.

    Subclasses implement :meth:`point_masses`, :meth:`potential` and
    :meth:`generalised_forces`, and may implement :meth:`added_inertia`.
    """

    nq: int = 0
    eps: float = 1e-6
    cs_step: float = 1e-20
    holomorphic: bool = True
    _cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ #
    # declaration hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def point_masses(self, q: np.ndarray):
        """Return ``(masses (nb,), positions (nb, 3))`` in the inertial frame."""

    def added_inertia(self, q: np.ndarray) -> np.ndarray:
        """Additive generalised inertia, e.g. payload yaw inertia, winch inertia."""
        return np.zeros((self.nq, self.nq))

    @abstractmethod
    def potential(self, q: np.ndarray) -> float:
        """Potential energy (gravity plus any elastic terms)."""

    @abstractmethod
    def generalised_forces(self, t: float, q: np.ndarray, qd: np.ndarray,
                           u: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Non-conservative generalised forces: actuation, damping, disturbance."""

    # ------------------------------------------------------------------ #
    # assembler
    # ------------------------------------------------------------------ #
    def _jacobians(self, q: np.ndarray) -> np.ndarray:
        """dp_i/dq_k.  Shape ``(nb, 3, nq)``."""
        m, p0 = self.point_masses(np.asarray(q, float))
        J = np.empty((p0.shape[0], 3, self.nq))
        if self.holomorphic:
            h = self.cs_step
            qc = np.asarray(q, dtype=complex)
            for k in range(self.nq):
                qp = qc.copy(); qp[k] += 1j * h
                _, pp = self.point_masses(qp)
                J[:, :, k] = np.imag(pp) / h
            return J
        for k in range(self.nq):
            h = self.eps * max(1.0, abs(q[k]))
            qp = np.asarray(q, float).copy(); qp[k] += h
            qm = np.asarray(q, float).copy(); qm[k] -= h
            _, pp = self.point_masses(qp)
            _, pm = self.point_masses(qm)
            J[:, :, k] = (pp - pm) / (2.0 * h)
        return J

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        m, _ = self.point_masses(np.asarray(q, float))
        J = self._jacobians(q)
        M = np.einsum("i,ijk,ijl->kl", np.real(m), J, J)
        return M + self.added_inertia(q)

    def _velocities(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Body velocities ``v_i = J_i qd``, to machine precision."""
        if self.holomorphic:
            h = self.cs_step
            _, pp = self.point_masses(np.asarray(q, dtype=complex) + 1j * h * qd)
            return np.imag(pp) / h
        return self._jacobians(q) @ qd

    def kinetic(self, q: np.ndarray, qd: np.ndarray) -> float:
        m, _ = self.point_masses(np.asarray(q, float))
        v = self._velocities(q, qd)
        T = 0.5 * float(np.sum(m * np.sum(v * v, axis=1)))
        A = self.added_inertia(q)
        return T + 0.5 * float(qd @ A @ qd)

    def _dT_dq(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        g = np.empty(self.nq)
        for k in range(self.nq):
            h = self.eps * max(1.0, abs(q[k]))
            qp = q.copy(); qp[k] += h
            qm = q.copy(); qm[k] -= h
            g[k] = (self.kinetic(qp, qd) - self.kinetic(qm, qd)) / (2.0 * h)
        return g

    def _dV_dq(self, q: np.ndarray) -> np.ndarray:
        g = np.empty(self.nq)
        for k in range(self.nq):
            h = self.eps * max(1.0, abs(q[k]))
            qp = q.copy(); qp[k] += h
            qm = q.copy(); qm[k] -= h
            g[k] = (self.potential(qp) - self.potential(qm)) / (2.0 * h)
        return g

    def _Mdot_qd(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        h = self.eps
        Mp = self.mass_matrix(q + h * qd)
        Mm = self.mass_matrix(q - h * qd)
        return ((Mp - Mm) / (2.0 * h)) @ qd

    def dynamics(self, t, x, u, d):
        q, qd = x[: self.nq], x[self.nq:]
        M = self.mass_matrix(q)
        rhs = (self.generalised_forces(t, q, qd, u, d)
               - self._Mdot_qd(q, qd) + self._dT_dq(q, qd) - self._dV_dq(q))
        qdd = np.linalg.solve(M, rhs)
        return np.concatenate([qd, qdd])

    def energy(self, x):
        q, qd = x[: self.nq], x[self.nq:]
        return self.kinetic(q, qd) + self.potential(q)
