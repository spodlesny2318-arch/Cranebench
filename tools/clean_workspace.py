"""List — and on request delete — everything in the folder that is not needed.

Prints what it would remove and why, and refuses to touch anything a campaign
or the manuscript checker depends on.  Nothing is deleted without --yes.

    python tools/clean_workspace.py          # report only
    python tools/clean_workspace.py --yes    # actually delete
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Never remove these: the manuscript checker and the campaigns read them.
KEEP_EXACT = {
    "run_batch", "run_retune", "run_sp3", "run_dual6",
    "cranebench", "tools", "examples", "tests", "docs", ".git",
}
KEEP_NOTE = {
    "run_batch": "the four planar campaigns and their ledgers",
    "run_retune": "the tuning budget: chosen gains and every evaluated point",
    "run_sp3": "the spatial campaign",
    "run_dual6": "the dual campaign (the one the manuscript reports)",
}

# Superseded campaign directories, in the order they were produced.
SUPERSEDED = ["run_dual", "run_dual2", "run_dual3", "run_dual4", "run_dual5",
              "run_sp2", "run_spatial", "run_spatial_b", "run_spatial_c",
              "run", "results", "_pack", "_pack2"]

JUNK_DIRS = ["__pycache__", ".pytest_cache", "node_modules", "pv"]
JUNK_GLOBS = ["*.pyc", "spatial_paired.npz", "dual_paired.npz",
              "pg-*.jpg", "page-*.jpg", "*.tmp"]


def size(p: pathlib.Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="delete, do not just report")
    a = ap.parse_args()

    doomed = []
    for name in SUPERSEDED:
        p = ROOT / name
        if p.exists():
            doomed.append((p, "superseded campaign output"))
    for d in ROOT.rglob("*"):
        if d.is_dir() and (d.name in JUNK_DIRS or d.name.startswith("pv")):
            if not any(k in d.parts for k in (".git", ".venv")):
                doomed.append((d, "build or test artefact"))
    for pat in JUNK_GLOBS:
        for f in ROOT.glob(pat):
            doomed.append((f, "stray file at the project root"))

    doomed = [(p, why) for p, why in doomed if p.name not in KEEP_EXACT]
    seen, uniq = set(), []
    for p, why in doomed:
        if p not in seen and p.exists():
            seen.add(p); uniq.append((p, why))

    print("KEEP — the manuscript and the campaigns depend on these:")
    for k, note in KEEP_NOTE.items():
        p = ROOT / k
        if p.exists():
            print(f"  {k:14s} {size(p)/1e6:6.1f} MB   {note}")
    venv = ROOT / ".venv"
    if venv.exists():
        print(f"  {'.venv':14s} {size(venv)/1e6:6.1f} MB   the virtual environment; "
              f"delete only if you are finished")

    if not uniq:
        print("\nnothing to remove.")
        return 0
    total = sum(size(p) for p, _ in uniq)
    print(f"\nREMOVE — {len(uniq)} item(s), {total/1e6:.1f} MB:")
    for p, why in sorted(uniq, key=lambda t: -size(t[0])):
        print(f"  {str(p.relative_to(ROOT)):34s} {size(p)/1e6:7.2f} MB   {why}")

    if not a.yes:
        print("\nreport only. Re-run with --yes to delete.")
        return 0
    for p, _ in uniq:
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    print(f"\ndeleted {len(uniq)} item(s), {total/1e6:.1f} MB freed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
