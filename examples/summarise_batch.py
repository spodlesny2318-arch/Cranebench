"""Tables for the paper from the batched campaign files."""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cranebench.stats import paired_summary, running_mean_convergence  # noqa: E402

ORDER = ["PD", "LQR", "ZVD", "SMC", "HSMC"]
KEY = ["ise_pos", "peak_swing", "residual_swing", "final_pos_err",
       "effort", "chatter", "bound_ok"]


def load(name, outdir=ROOT / "run_batch"):
    z = np.load(pathlib.Path(outdir) / f"{name}_metrics.npz")
    res = {}
    for tag in z.files:
        c, k = tag.split("__")
        res.setdefault(c, {})[k] = z[tag]
    return {c: res[c] for c in ORDER if c in res}


def table(name):
    res = load(name)
    n = len(next(iter(res.values()))["ise_pos"])
    print(f"\n===== campaign '{name}', n = {n} =====")
    hdr = f"{'ctrl':<6}" + "".join(f"{k:>16}" for k in KEY)
    print(hdr); print("-" * len(hdr))
    for c, d in res.items():
        row = f"{c:<6}"
        for k in KEY:
            v = float(np.nanmean(d[k]))
            row += f"{v:16.4g}" if abs(v) < 1e5 else f"{v:16.3e}"
        print(row)
    print(f"{'':<6}" + "".join(f"{'(sd) ' + f'{float(np.nanstd(list(res.values())[0][k])):.3g}':>16}"
                               for k in KEY[:0]))

    ref = "PD"
    print(f"\npaired contrasts against {ref}")
    print(f"{'ctrl':<6}{'metric':<17}{'mean diff':>13}{'95% CI':>28}"
          f"{'wilcoxon p':>12}{'win':>7}{'d_z':>8}")
    for c in res:
        if c == ref:
            continue
        for k in ("residual_swing", "peak_swing", "effort", "chatter"):
            s = paired_summary(res[c][k], res[ref][k])
            ci = f"[{s['ci_low']:+.4g}, {s['ci_high']:+.4g}]"
            print(f"{c:<6}{k:<17}{s['mean_diff']:+13.4g}{ci:>28}"
                  f"{s['wilcoxon_p']:12.2e}{s['win_rate']:7.2f}{s['cohen_dz']:8.2f}")

    print("\nbound satisfaction (peak swing <= 4.8 deg) and convergence")
    for c, d in res.items():
        nr, ok = running_mean_convergence(d["residual_swing"])
        print(f"  {c:<6} {int(np.nansum(d['bound_ok'])):3d}/{n}   "
              f"mean stable from n = {nr:3d}")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["calm", "reference", "stress", "dryden"]):
        table(nm)
