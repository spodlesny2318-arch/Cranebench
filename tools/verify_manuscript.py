"""Check every number in the manuscript tables against the campaign files.

The declaration of AI use in this article states that the authors verified
every reported number by re-running the released code.  This script is what
makes that statement checkable rather than merely asserted: it parses the
result tables out of the manuscript, recomputes each cell from the stored
campaign data, and reports any cell that disagrees beyond tolerance.

    python tools/verify_manuscript.py            # check
    python tools/verify_manuscript.py --tol 5e-3 # tighter

Exit status is non-zero if any cell fails, so it can be run in CI.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

MD = ROOT / "PAPER_SoftwareX_draft.md"
ORDER = ["PD", "LQR", "ZVD", "SMC", "HSMC"]

# A result table is recognised by its own header row, not by the prose next to
# it: an earlier version matched on caption text, silently failed to find two
# tables, and reported success on the ones it did find.  A verification tool
# that cannot say how many tables it checked is not a verification tool.
COLUMN_TO_METRIC = {
    "ise": "ise_pos", "peak swing": "peak_swing",
    "residual swing": "residual_swing", "settling": "settle_time",
    "peak yaw": "peak_yaw", "rms yaw": "rms_yaw", "effort": "effort",
    "chatter": "chatter", "final error": "final_pos_err",
    "bound met": "bound_ok",
}
# context keyword -> campaign file, most specific first
CONTEXT_TO_CAMPAIGN = [
    ("frozen → re-tuned", ("stress", "stress_retuned")),
    ("six-factor design", ("spatial",)),
    ("spatial plant", ("spatial",)),
    ("dual plant", ("dual",)),
    ("stress campaign", ("stress",)),
    ("reference campaign", ("reference",)),
]

NUM = re.compile(r"^([<>]?)\s*([-+]?\d*\.?\d+)(?:·10⁻?([\d⁰¹²³⁴⁵⁶⁷⁸⁹]+))?")
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def parse_cell(txt):
    """Return a float, or None if the cell is not a plain number."""
    t = txt.strip()
    if not t:
        return None
    m = re.match(r"^(\d+)/(\d+)$", t)                     # 461/500
    if m:
        return int(m.group(1)) / int(m.group(2))
    t = t.replace(" c", "").replace(">", "").strip()      # censored marker
    m = re.match(r"^([-+]?\d*\.?\d+)·10(⁻?)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)$", t)
    if m:
        e = int(m.group(3).translate(SUP)) * (-1 if m.group(2) else 1)
        return float(m.group(1)) * 10.0 ** e
    m = re.match(r"^[-+]?\d*\.?\d+$", t)
    return float(t) if m else None


def _paired_file(kind):
    """Newest merged results for a non-batched plant, wherever it was written.

    The runners take an --outdir, so a fixed path silently reported a stale
    campaign more than once; taking the most recent match removes the trap.
    """
    cands = sorted(ROOT.rglob(f"{kind}_paired.npz"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    return cands[0] if cands else ROOT / f"{kind}_paired.npz"


def load_campaign(name):
    from summarise_batch import load as load_batch
    if name in ("reference", "stress", "dryden", "calm", "stress_retuned"):
        return load_batch(name)
    z = np.load(_paired_file(name))
    out = {}
    for tag in z.files:
        c, k = tag.split("__")
        out.setdefault(c, {})[k] = z[tag]
    return out


def tables_from(md):
    """Yield (caption, rows) for every markdown table in the manuscript."""
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|"):
            start = i
            while i < len(lines) and lines[i].strip().startswith("|"):
                i += 1
            caption = ""
            for j in range(start - 1, max(start - 4, -1), -1):
                if lines[j].strip():
                    caption = lines[j]
                    break
            yield caption, lines[start:i]
        i += 1


EXPECTED_CELLS = 65



CONTRAST = re.compile(
    r"([-+−]?\d*\.?\d+)\s*\[.*?\].*?@rrb@\s*=\s*([-+−]?\d*\.?\d+)")


def check_contrasts(tab, tol, failures):
    """Verify the paired-contrast table: mean difference and effect size.

    These cells hold differences and effect sizes rather than means, so the
    generic path cannot read them; leaving them unverified would have been the
    easy choice and the wrong one.
    """
    from cranebench.stats import paired_summary
    data = load_campaign("reference")
    header = [c.strip() for c in tab[0].strip().strip("|").split("|")]
    keys = [next((v for k, v in COLUMN_TO_METRIC.items() if k in h.lower()), None)
            for h in header[1:]]
    n = 0
    for row in tab[2:]:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if cells[0] not in ORDER:
            continue
        for key, cell in zip(keys, cells[1:]):
            m = CONTRAST.search(cell.replace("−", "-"))
            if key is None or m is None:
                continue
            s = paired_summary(data[cells[0]][key], data["PD"][key])
            for want, got, what in ((float(m.group(1)), s["mean_diff"], "diff"),
                                    (float(m.group(2)), s["rank_biserial"], "r_rb")):
                n += 1
                rel = abs(got - want) / max(abs(got), 1e-12)
                if rel > tol:
                    failures.append(
                        f"contrast/{cells[0]}/{key}/{what}: manuscript {want:g}, "
                        f"recomputed {got:g}  (rel {rel:.2%})")
    return n


def classify(context, header):
    """Decide which campaign a table belongs to from the text above it."""
    ctx = " ".join(context).lower()
    for key, camps in CONTEXT_TO_CAMPAIGN:
        if key.lower() in ctx or key.lower() in " ".join(header).lower():
            return camps
    return None


def check(tol):
    # Read as UTF-8 explicitly.  Python defaults to the locale encoding on
    # Windows, which mangles the superscripts in "5.79·10⁷" and the arrows in
    # the re-tuning table; the affected cells then fail to parse and were
    # silently skipped.  This was found by running the tool on Windows, not by
    # reasoning about it, which is the argument for the exercise.
    md = MD.read_text(encoding="utf-8")
    lines = md.split("\n")
    failures, skipped, per_table = [], [], []
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("|"):
            i += 1
            continue
        start = i
        while i < len(lines) and lines[i].strip().startswith("|"):
            i += 1
        tab = lines[start:i]
        header = [c.strip() for c in tab[0].strip().strip("|").split("|")]
        if header[0].strip().lower() != "controller":
            continue
        context = [l for l in lines[max(0, start - 6):start] if l.strip()]
        if any("paired contrast" in c.lower() for c in context):
            n = check_contrasts(tab, tol, failures)
            per_table.append(("reference (contrasts)", n))
            continue
        camps = classify(context, header)
        label = (context[-1][:60] if context else "?").strip()
        if camps is None:
            skipped.append(f"unclassified result table near: {label!r}")
            continue

        # map each column to a metric, and to which campaign each side belongs
        cols = []
        for h in header[1:]:
            key = next((v for k, v in COLUMN_TO_METRIC.items() if k in h.lower()), None)
            cols.append(key)
        if all(c is None for c in cols):
            skipped.append(f"no recognised metric columns in table near {label!r}")
            continue

        try:
            data = [load_campaign(c) for c in camps]
        except FileNotFoundError as e:
            skipped.append(f"{'/'.join(camps)}: result file missing ({e.filename})")
            continue

        n = 0
        for row in tab[2:]:
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if cells[0] not in ORDER:
                continue
            for key, cell in zip(cols, cells[1:]):
                if key is None:
                    continue
                parts = [q.strip() for q in cell.split("→")] if "→" in cell else [cell]
                for part, src in zip(parts, data if len(parts) > 1 else data[:1]):
                    want = parse_cell(part)
                    if want is None:
                        continue
                    if key not in src.get(cells[0], {}):
                        skipped.append(f"{'/'.join(camps)}: '{key}' absent from "
                                       f"the stored results -- re-run the campaign")
                        continue
                    got = float(np.nanmean(src[cells[0]][key]))
                    n += 1
                    rel = abs(got - want) / max(abs(got), 1e-12)
                    if rel > tol:
                        failures.append(
                            f"{'/'.join(camps)}/{cells[0]}/{key}: manuscript "
                            f"{want:g}, recomputed {got:g}  (rel {rel:.2%})")
        per_table.append(("+".join(camps), n))

    total = sum(n for _, n in per_table)
    print(f"tolerance {tol:.2%}   result tables found: "
          f"{len(per_table) + len(skipped)}, checked: {len(per_table)}")
    for name, n in per_table:
        print(f"  {name:24s} {n:3d} cells")
    if skipped:
        print("\nNOT CHECKED:")
        for sk in skipped:
            print("  -", sk)
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            print("  -", f)
    if failures or skipped:
        print("\nINCOMPLETE" if skipped and not failures else "")
        return 1
    print(f"\nall {total} cells in {len(per_table)} result tables match the "
          f"campaign files.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-2)
    sys.exit(check(ap.parse_args().tol))
