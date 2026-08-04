"""Controller interface and linearisation helpers.

Every controller in the baseline set is a *published classical* design.  The
benchmark deliberately contains no novel controller: its purpose is to fix the
bench, not to compete on it.  A new design is added by subclassing
:class:`Controller` in the user's own package and passing it to the runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    name: str = "base"

    def reset(self, plant, manoeuvre) -> None:
        self.plant = plant
        self.man = manoeuvre
        self.x0 = plant.initial_state()

    @abstractmethod
    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        """Control input at time ``t``.  Must not mutate ``x``."""


def input_matrix(plant, x0, u0, h=1e-4):
    """B = df/du by central differences (dynamics are affine in u, so exact)."""
    B = np.zeros((plant.nx, plant.nu))
    for j in range(plant.nu):
        up, um = np.array(u0, float), np.array(u0, float)
        up[j] += h
        um[j] -= h
        d = np.zeros(3)
        B[:, j] = (plant.dynamics(0.0, x0, up, d)
                   - plant.dynamics(0.0, x0, um, d)) / (2 * h)
    return B


def state_matrix(plant, x0, u0, h=1e-6):
    A = np.zeros((plant.nx, plant.nx))
    d = np.zeros(3)
    for j in range(plant.nx):
        xp, xm = np.array(x0, float), np.array(x0, float)
        s = h * max(1.0, abs(x0[j]))
        xp[j] += s
        xm[j] -= s
        A[:, j] = (plant.dynamics(0.0, xp, u0, d)
                   - plant.dynamics(0.0, xm, u0, d)) / (2 * s)
    return A


def trim(plant):
    """Equilibrium input at the nominal initial state (least squares)."""
    x0 = plant.initial_state()
    u0 = np.zeros(plant.nu)
    f0 = plant.dynamics(0.0, x0, u0, np.zeros(3))
    B = input_matrix(plant, x0, u0)
    u_eq, *_ = np.linalg.lstsq(B, -f0, rcond=None)
    return x0, u_eq
