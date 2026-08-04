"""Paired uncertainty design.

Plant uncertainty is drawn once, on a Latin hypercube, and the *same* sample
list is replayed for every controller.  This is the whole point: an unpaired
comparison of two controllers over two independent 500-sample clouds spends
most of its statistical power on the cloud, not on the controllers.  With a
paired design the difference is taken sample by sample and the plant variation
cancels out of the contrast.

The default design perturbs payload mass, rope length, swing damping, drive
damping and the wind seed.  Factors are declared as multiplicative ranges about
the nominal value so that the design is dimensionless and transfers between
plants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

DEFAULT_FACTORS: Dict[str, Tuple[float, float]] = {
    "m_payload": (0.80, 1.20),
    "l0": (0.75, 1.25),
    "c_theta": (0.50, 2.00),
    "b_x": (0.70, 1.30),
    "u_mean": (0.60, 1.40),
}


@dataclass
class UncertaintyDesign:
    factors: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_FACTORS))
    n: int = 500
    seed: int = 20260729
    samples: np.ndarray = field(default=None, repr=False)
    wind_seeds: np.ndarray = field(default=None, repr=False)

    @property
    def names(self) -> List[str]:
        return list(self.factors)

    def as_dicts(self) -> List[Dict[str, float]]:
        return [dict(zip(self.names, row)) for row in self.samples]


def lhs_design(n: int = 500, seed: int = 20260729,
               factors: Dict[str, Tuple[float, float]] | None = None
               ) -> UncertaintyDesign:
    """Centred Latin hypercube over the multiplicative factor ranges."""
    factors = dict(factors or DEFAULT_FACTORS)
    rng = np.random.default_rng(seed)
    k = len(factors)
    cut = (np.arange(n) + 0.5) / n
    u = np.empty((n, k))
    for j in range(k):
        u[:, j] = rng.permutation(cut)
    lo = np.array([v[0] for v in factors.values()])
    hi = np.array([v[1] for v in factors.values()])
    samples = lo + u * (hi - lo)
    wind_seeds = rng.integers(0, 2 ** 31 - 1, size=n)
    return UncertaintyDesign(factors=factors, n=n, seed=seed,
                             samples=samples, wind_seeds=wind_seeds)


def apply_factors(plant_params, wind_params, factors: Dict[str, float]):
    """Scale nominal parameters in place-free fashion; returns new objects."""
    import copy
    pp = copy.deepcopy(plant_params)
    wp = copy.deepcopy(wind_params)
    for name, mult in factors.items():
        if hasattr(pp, name):
            setattr(pp, name, getattr(pp, name) * mult)
        elif hasattr(wp, name):
            setattr(wp, name, getattr(wp, name) * mult)
        else:
            raise KeyError(f"factor {name!r} matches no parameter")
    return pp, wp
