"""Kaimal spectral synthesis of a longitudinal turbulence record.

The one-sided Kaimal spectrum in the form used by IEC 61400-1 is

    f S(f) / sigma^2 = 4 (f L / U) / (1 + 6 f L / U)^(5/3)

A record is synthesised by inverse FFT with random phases, then rescaled so
that its realised variance equals ``sigma^2`` exactly.  The rescaling matters:
without it the realised turbulence intensity of a finite record is a random
variable, and two controllers evaluated on "the same" seed would in fact see
disturbances of different strength.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KaimalParams:
    u_mean: float = 12.0        # m/s, mean wind speed at payload height
    intensity: float = 0.14     # turbulence intensity sigma_u / U
    length_scale: float = 170.0  # m, integral length scale
    height: float = 20.0        # m (recorded for provenance only)


def kaimal_psd(f: np.ndarray, p: KaimalParams) -> np.ndarray:
    """One-sided PSD [m^2/s] at frequencies ``f`` [Hz]; ``S(0)`` set to 0."""
    sigma = p.intensity * p.u_mean
    fl = np.where(f > 0, f * p.length_scale / p.u_mean, np.inf)
    s = 4.0 * sigma ** 2 * (p.length_scale / p.u_mean) / (1.0 + 6.0 * fl) ** (5.0 / 3.0)
    return np.where(f > 0, s, 0.0)


def synthesise(n: int, dt: float, p: KaimalParams,
               rng: np.random.Generator) -> np.ndarray:
    """Return a turbulence record ``u'(t)`` of length ``n``, zero mean."""
    nfft = int(2 ** np.ceil(np.log2(max(n, 2) * 2)))
    f = np.fft.rfftfreq(nfft, dt)
    s = kaimal_psd(f, p)
    df = 1.0 / (nfft * dt)
    amp = np.sqrt(2.0 * s * df)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=f.size)
    phase[0] = 0.0
    spec = amp * np.exp(1j * phase) * nfft / 2.0
    spec[0] = 0.0
    rec = np.fft.irfft(spec, n=nfft)[:n]
    sigma = p.intensity * p.u_mean
    std = rec.std()
    if std > 0:
        rec *= sigma / std          # exact realised variance
    return rec - rec.mean()


class KaimalWind:
    """A frozen turbulence record on a grid independent of the solver step.

    The record is synthesised once on a fixed grid (``grid_dt``, 100 Hz by
    default) and linearly interpolated.  Tying the record to the integrator step
    instead would make the realisation change whenever the step changes, so a
    step-refinement study would be measuring a different disturbance at every
    step and could never converge.  With the record decoupled, refining the step
    refines only the integration.
    """

    grid_dt = 0.01

    def __init__(self, duration: float, params: KaimalParams,
                 rng: np.random.Generator, grid_dt: float | None = None):
        self.p = params
        self.dt = grid_dt or self.grid_dt
        self.n = int(np.ceil(duration / self.dt)) + 2
        self.turb = synthesise(self.n, self.dt, params, rng)

    def speed(self, t: float) -> float:
        s = t / self.dt
        i = int(s)
        if i < 0:
            return self.p.u_mean + self.turb[0]
        if i >= self.n - 1:
            return self.p.u_mean + self.turb[-1]
        f = s - i
        return self.p.u_mean + (1.0 - f) * self.turb[i] + f * self.turb[i + 1]
