"""Reference manoeuvres.

Two profiles are provided: a quintic (minimum-jerk) point-to-point transfer,
which is the benign case, and a trapezoidal-velocity transfer with a short ramp,
which is the stress case.  Both are declared here rather than inside a
controller so that every controller is driven by the identical reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Manoeuvre:
    distance: float = 20.0     # m, horizontal transfer
    t_ramp: float = 20.0       # s, duration of the transfer phase
    t_total: float = 40.0      # s, simulation horizon
    hoist: float = 0.0         # m, change of rope length (0 = no hoisting)
    kind: str = "quintic"      # "quintic" | "trapezoid"

    def position(self, t: float) -> float:
        return self.distance * self._s(t)

    def velocity(self, t: float) -> float:
        return self.distance * self._sd(t)

    def rope(self, t: float, l0: float) -> float:
        return l0 + self.hoist * self._s(t)

    def rope_rate(self, t: float) -> float:
        return self.hoist * self._sd(t)

    # ------------------------- vectorised ---------------------------- #
    def position_v(self, t):
        return self.distance * self._s_v(np.asarray(t, float))

    def velocity_v(self, t):
        return self.distance * self._sd_v(np.asarray(t, float))

    def _s_v(self, t):
        T = self.t_ramp
        u = np.clip(t / T, 0.0, 1.0)
        if self.kind == "quintic":
            s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        else:
            b = 0.15
            s = np.where(
                u < b, 0.5 * u * u / (b * (1 - b)),
                np.where(u < 1 - b, (u - 0.5 * b) / (1 - b),
                         1.0 - 0.5 * (1 - u) ** 2 / (b * (1 - b))))
        return np.where(t <= 0, 0.0, np.where(t >= T, 1.0, s))

    def _sd_v(self, t):
        T = self.t_ramp
        u = np.clip(t / T, 0.0, 1.0)
        if self.kind == "quintic":
            d = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / T
        else:
            b = 0.15
            d = np.where(u < b, (u / (b * (1 - b))) / T,
                         np.where(u < 1 - b, (1.0 / (1 - b)) / T,
                                  ((1 - u) / (b * (1 - b))) / T))
        return np.where((t <= 0) | (t >= T), 0.0, d)

    # ---------------------------------------------------------------- #
    def _s(self, t):
        T = self.t_ramp
        if t <= 0:
            return 0.0
        if t >= T:
            return 1.0
        u = t / T
        if self.kind == "quintic":
            return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        # trapezoid with 15 % blend on each side
        b = 0.15
        if u < b:
            return 0.5 * u * u / (b * (1 - b))
        if u < 1 - b:
            return (u - 0.5 * b) / (1 - b)
        v = 1 - u
        return 1.0 - 0.5 * v * v / (b * (1 - b))

    def _sd(self, t):
        T = self.t_ramp
        if t <= 0 or t >= T:
            return 0.0
        u = t / T
        if self.kind == "quintic":
            return (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / T
        b = 0.15
        if u < b:
            return (u / (b * (1 - b))) / T
        if u < 1 - b:
            return (1.0 / (1 - b)) / T
        v = 1 - u
        return (v / (b * (1 - b))) / T
