"""The batched path must reproduce the scalar reference path exactly.

A second implementation of the same experiment is a liability unless it is
pinned to the first.  This test runs a full paired design through both paths
and requires every metric to agree to 1e-9 relative.
"""

import numpy as np

from cranebench.batch import run_campaign_batch
from cranebench.reference import Manoeuvre
from cranebench.runner import Campaign, run_single
from cranebench.uncertainty import lhs_design


def test_batch_matches_scalar_on_every_metric():
    design = lhs_design(n=6, seed=99)
    camp = Campaign(plant="planar", wind="kaimal", dt=1e-2,
                    manoeuvre=Manoeuvre(distance=20.0, t_ramp=20.0,
                                        t_total=40.0)).build()
    batch = run_campaign_batch(camp, design, progress=False)
    for cname, ctrl in camp.controllers.items():
        for i, fac in enumerate(design.as_dicts()):
            m, _ = run_single(ctrl, camp, fac, int(design.wind_seeds[i]))
            for k, v in m.items():
                got = batch[cname][k][i]
                assert abs(got - v) <= 1e-9 * max(abs(v), 1e-9), \
                    f"{cname}.{k}[{i}]: batch {got!r} vs scalar {v!r}"


def test_batch_is_deterministic():
    design = lhs_design(n=4, seed=7)
    camp = Campaign(plant="planar", wind="kaimal", dt=2e-2,
                    manoeuvre=Manoeuvre(t_total=20.0, t_ramp=12.0)).build()
    a = run_campaign_batch(camp, design, progress=False)
    b = run_campaign_batch(camp, design, progress=False)
    for c in a:
        for k in a[c]:
            assert np.array_equal(a[c][k], b[c][k])
