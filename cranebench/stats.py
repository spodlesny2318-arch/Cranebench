"""Paired statistics for the campaign results.

Only paired procedures are provided.  An unpaired t-test on two 500-sample
clouds drawn from the same uncertainty design would throw away the pairing that
the design was built to create, and would usually be less powerful by an order
of magnitude.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import stats as sps


BOOTSTRAP = {"resamples": 10000, "kind": "percentile", "seed": 12345}


def paired_summary(a: np.ndarray, b: np.ndarray,
                   n_boot: int = BOOTSTRAP["resamples"],
                   seed: int = BOOTSTRAP["seed"]) -> Dict[str, float]:
    """Compare metric arrays ``a`` and ``b`` sample by sample.

    The interval is a percentile bootstrap over ``n_boot`` resamples from a
    generator seeded with ``seed``; both are reported so the number is
    reproducible rather than merely plausible.  For strongly skewed differences
    such as ISE the percentile interval is biased and BCa would be preferable;
    that is why effect sizes, not intervals, carry the argument here.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        w = sps.wilcoxon(a, b, zero_method="zsplit")
        pval = float(w.pvalue)
    except ValueError:
        pval = float("nan")
    sd = d.std(ddof=1)
    return {
        "n": int(d.size),
        "rank_biserial": rank_biserial(a, b),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "wilcoxon_p": pval,
        "cohen_dz": float(d.mean() / sd) if sd > 0 else float("nan"),
        "win_rate": float(np.mean(a < b)),
    }


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation.

    The effect size that belongs with a Wilcoxon signed-rank test.  Cohen's
    ``d_z`` assumes normal differences, which the metric distributions here do
    not have; reporting a parametric effect size next to a non-parametric test
    is a mismatch a careful reader will notice.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return float("nan")
    r = sps.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def mcnemar(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Paired test for a binary outcome measured on the same samples."""
    a = np.asarray(a, float).astype(bool)
    b = np.asarray(b, float).astype(bool)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    return {"a_only": n01, "b_only": n10, "statistic": float(stat),
            "p": float(sps.chi2.sf(stat, 1))}


def bound_sensitivity(peak_swing: np.ndarray,
                      thresholds=(3.0, 4.0, 4.8, 6.0, 7.0, 10.0)) -> Dict[float, int]:
    """Bound satisfaction as a function of the declared limit.

    A benchmark whose headline metric depends on an undeclared threshold is
    one question away from collapsing; reporting the whole curve costs nothing.
    """
    x = np.asarray(peak_swing, float)
    return {float(t): int(np.sum(x <= t)) for t in thresholds}


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact binomial confidence interval for a count of successes.

    Reporting "0/500" without an interval reads as a claim of impossibility.
    It is not one: the exact upper bound at 95 % is 0.6 %.
    """
    lo = 0.0 if k == 0 else sps.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else sps.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def kaplan_meier_rmst(times, events, horizon):
    """Restricted mean survival time under right censoring.

    The mean of a right-censored sample estimates nothing.  Settling times are
    censored whenever the run ends before the position error settles, which under
    wind is most of them, so the summary has to acknowledge it.  ``events`` is
    True where the time was observed and False where the run was censored.
    """
    t = np.asarray(times, float)
    e = np.asarray(events, bool)
    order = np.argsort(t)
    t, e = t[order], e[order]
    n_at_risk = t.size
    surv, prev_t, rmst = 1.0, 0.0, 0.0
    for i, (ti, ei) in enumerate(zip(t, e)):
        rmst += surv * (min(ti, horizon) - prev_t)
        prev_t = min(ti, horizon)
        if ei:
            surv *= (n_at_risk - 1) / n_at_risk if n_at_risk > 1 else 0.0
        n_at_risk -= 1
        if prev_t >= horizon:
            break
    rmst += surv * max(horizon - prev_t, 0.0)
    return {"rmst": float(rmst), "censored_fraction": float(1.0 - e.mean()),
            "observed": int(e.sum()), "n": int(t.size)}


def rank_agreement(campaigns, metric, controllers, lower_is_better=True):
    """Kendall's tau between the controller rankings of two operating points.

    The claim that a ranking is not preserved between operating points is a
    statement about rank correlation, and is better made as one than as a list
    of examples.
    """
    names = list(campaigns)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            xa = [float(np.nanmean(campaigns[a][c][metric])) for c in controllers]
            xb = [float(np.nanmean(campaigns[b][c][metric])) for c in controllers]
            if not lower_is_better:
                xa, xb = [-v for v in xa], [-v for v in xb]
            r = sps.kendalltau(sps.rankdata(xa), sps.rankdata(xb))
            out[(a, b)] = (float(r.statistic), float(r.pvalue))
    return out


def running_mean_convergence(x: np.ndarray, tol: float = 0.01) -> Tuple[int, bool]:
    """Smallest prefix length whose mean is within ``tol`` (relative) of the full mean."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0, False
    full = x.mean()
    run = np.cumsum(x) / np.arange(1, x.size + 1)
    if full == 0:
        return x.size, True
    within = np.abs(run - full) <= tol * abs(full)
    tail = np.where(~within)[0]
    n_req = int(tail[-1] + 2) if tail.size else 1
    return min(n_req, x.size), n_req <= x.size
