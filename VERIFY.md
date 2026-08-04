# Author verification procedure

The manuscript states that the authors verified every reported number by
re-running the released code and checked every reference against the publisher
record. This file is the procedure that makes those two sentences true. It is
written for the authors, not for the reader, and takes about a day — most of it
unattended.

Run it from a clean checkout, in a fresh virtual environment, on a machine that
did not produce the shipped result files. That last point matters: it is the
only part of the exercise that tests anything the original run did not.

---

## 1. Environment and tests — 5 minutes

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Expect **24 passed**. The suite checks the models rather than stored outputs:
the planar equations against a numerically assembled Lagrangian, the spatial and
dual equations against independent SymPy derivations, energy conservation with
damping removed, the disturbance spectra against their targets, step
convergence, and the batched execution path against the scalar one.

Record the interpreter, NumPy and SciPy versions. If any test fails on your
machine but passes on ours, that is a finding and belongs in the paper.

## 2. Campaigns — about 4 minutes of compute, plus the spatial one

```bash
for c in calm reference dryden stress; do
    python examples/run_batch_campaign.py $c 500
done
python examples/run_retuned_stress.py 500          # needs run_retune/gains.json
python tools/retune.py 600                         # regenerate the gains first, if absent
python examples/run_spatial_campaign.py --n 150    # ~10 min, not batched
```

The Latin hypercube, the per-sample wind seeds and the metric-module hash are
all recorded in the `*_ledger.json` files. Compare the `metric_hash` field in
your ledgers with the ones in the shipped results: if they differ, the metric
module changed and no comparison between the two sets is meaningful.

## 3. The manuscript against the data — 1 minute

```bash
python tools/verify_manuscript.py
```

This parses the result tables out of `PAPER_SoftwareX_draft.md`, recomputes
every cell from the campaign files, and fails on any disagreement beyond 1 %.
It currently checks **65 cells**. At a 0.2 % tolerance a dozen cells "fail" on
rounding to three significant figures, which is the expected behaviour and not
a defect — 1 % is the meaningful setting.

What this does **not** check, and you must read by eye:

- the numbers quoted in running text rather than in tables (the verification
  table in Section 3.1, the censoring percentages, the threshold-sensitivity
  counts, the rank correlation, the McNemar counts);
- Table 1 in Section 3.1, whose entries come from the test suite;
- every claim of the form "X because Y".

## 4. References — half a day

Open `docs/reference_check.csv`. It lists all 36 entries with their provenance:

- **26 entries** were taken from the reference lists of the authors' own
  manuscripts. They were *not* independently re-verified during preparation.
  These are the ones the declaration commits you to checking.
- **9 entries** were located by search and confirmed against the publisher
  record during preparation. Confirm them anyway; it is faster than deciding
  which to trust.
- **1 entry** is the software self-citation and is complete once the Zenodo DOI
  exists.

For each row: resolve the DOI, confirm authors, title, journal, volume, issue,
year and page range, and tick `checked_by_author`. Two entries were completed
late from a Crossref record (Huang & Zhu 2021, McKay et al. 1979) and deserve a
second look.

## 5. What is still not verified by anything here

Stated so that nobody mistakes a green run for a complete check:

- **The models are consistent, not validated.** Three independent derivations
  agreeing tells you the algebra is right. It does not tell you the model
  describes a crane. That judgement needs a domain expert, and the default
  parameters — payload mass, rope length, frontal area, suspension stiffness,
  centre-of-pressure eccentricity — should be reviewed by one.
- **No hardware.** Every number in this package is a simulation result.
- **Cross-platform reproducibility: closed.** The full procedure was run
  independently on Windows 11 with CPython 3.14, NumPy 2.5 and SciPy 1.18,
  against the original Linux run on CPython 3.10, NumPy 2.2 and SciPy 1.15. All
  136 reported table cells agreed to six significant figures, and the fourteen
  cells that exceed a 0.2 % tolerance are the same fourteen on both platforms,
  with the same recomputed values — they are the manuscript's rounding to three
  significant figures, not a numerical difference. That run also found a real
  defect: the verification tool read the manuscript in the platform locale
  encoding, which mangles superscripts on Windows and caused it to skip cells
  silently. Fixed, with a regression test.
