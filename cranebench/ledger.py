"""Provenance ledger.

A result file that does not say what produced it is not reproducible, however
open the code is.  The ledger records: package version, metric-module hash, the
hash of every source file, the uncertainty design seed, every per-run wind seed,
the integrator and its step, and the interpreter and library versions.  It is
written next to the results as JSON.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np


def _source_hashes() -> Dict[str, str]:
    root = pathlib.Path(__file__).parent
    out = {}
    for f in sorted(root.rglob("*.py")):
        out[str(f.relative_to(root))] = hashlib.sha256(
            f.read_bytes()).hexdigest()[:16]
    return out


@dataclass
class Ledger:
    campaign: str
    plant: str
    wind: str
    controllers: List[str]
    integrator: str
    dt: float
    horizon: float
    design_seed: int
    n_samples: int
    wind_seeds: List[int] = field(default_factory=list)
    metric_hash: str = ""
    package_version: str = ""
    sources: Dict[str, str] = field(default_factory=_source_hashes)
    environment: Dict[str, Any] = field(default_factory=lambda: {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    })

    def write(self, path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True),
                        encoding="utf-8")
        return path
