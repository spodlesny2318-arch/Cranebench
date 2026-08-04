"""Merge campaign checkpoints and print the paired comparison table."""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cranebench.stats import paired_summary, running_mean_convergence  # noqa: E402

KEY = ["ise_pos", "peak_swing", "residual_swing", "settle_time",
       "effort", "chatter", "bound_ok"]
ORDER = ["PD", "LQR", "ZVD", "SMC", "HSMC"]


def load(paths, n):
    done = {}
    for p in paths:
        for f in pathlib.Path(p).rglob("*_runs.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done[(r["controller"], r["i"])] = r["metrics"]
    ctrls = sorted({c for c, _ in done}, key=lambda c: ORDER.index(c) if c in ORDER else 99)
    return {c: {k: np.array([(done[(c, i)][k] if done.get((c, i)) else np.nan)
                             for i in range(n)], float) for k in KEY}
            for c in ctrls}, done


def main(n=120, paths=("run",)):
    res, done = load(paths, n)
    print(f"paired campaign: n = {n}, runs on file = {len(done)}\n")
    hdr = f"{'ctrl':<6}" + "".join(f"{k:>16}" for k in KEY)
    print(hdr); print("-" * len(hdr))
    for c, d in res.items():
        row = f"{c:<6}"
        for k in KEY:
            v = float(np.nanmean(d[k]))
            row += f"{v:16.4g}" if abs(v) < 1e5 else f"{v:16.3e}"
        print(row)

    ref = "PD"
    print(f"\npaired contrasts against {ref}  (negative = better than {ref})")
    print(f"{'ctrl':<6}{'metric':<17}{'mean diff':>12}{'95% CI':>26}"
          f"{'wilcoxon p':>12}{'win rate':>10}")
    for c in res:
        if c == ref:
            continue
        for k in ("residual_swing", "peak_swing", "effort", "chatter"):
            s = paired_summary(res[c][k], res[ref][k])
            ci = f"[{s['ci_low']:+.4g}, {s['ci_high']:+.4g}]"
            print(f"{c:<6}{k:<17}{s['mean_diff']:+12.4g}{ci:>26}"
                  f"{s['wilcoxon_p']:12.2e}{s['win_rate']:10.2f}")

    print("\nbound satisfaction (peak swing <= 4.8 deg)")
    for c, d in res.items():
        ok = np.nansum(d["bound_ok"])
        print(f"  {c:<6} {int(ok):3d}/{n}")

    print("\nconvergence of the campaign mean (1 % band)")
    for c, d in res.items():
        nr, ok = running_mean_convergence(d["residual_swing"])
        print(f"  {c:<6} residual_swing stable from n = {nr:3d}  "
              f"({'converged' if ok else 'NOT converged'})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120,
         sys.argv[2:] or ("run",))
