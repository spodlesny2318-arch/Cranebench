"""
cranebench -- a reproducible benchmark for underactuated crane control.

The package deliberately contains *baseline* controllers and *infrastructure*
only. It fixes the plants, the disturbance models, the uncertainty design, the
frozen metric module and the seed ledger, so that any new controller can be
evaluated on the same bench against the same realisations.

Design rules (docs/DESIGN.md):

1. The metric module is frozen. Metrics are computed by a single function whose
   source hash is recorded in every result file.
2. Plants are pure: derivatives depend only on (t, state, input, disturbance,
   parameters). No controller may mutate plant state.
3. Every stochastic quantity comes from an explicitly seeded generator whose
   seed is written to the ledger.
4. Uncertainty realisations are drawn once and reused across controllers, so
   every comparison is paired.
"""

__version__ = "0.1.0"

from .metrics import METRIC_HASH, Metrics, compute_metrics
from .uncertainty import UncertaintyDesign, lhs_design
from .ledger import Ledger
from .runner import run_campaign, run_single

__all__ = [
    "__version__",
    "METRIC_HASH",
    "Metrics",
    "compute_metrics",
    "UncertaintyDesign",
    "lhs_design",
    "Ledger",
    "run_campaign",
    "run_single",
]
