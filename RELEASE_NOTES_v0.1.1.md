# cranebench 0.1.1 — SoftwareX submission release

This corrective release is the software state intended for the accompanying SoftwareX submission. It changes release/package metadata and repository hygiene only; benchmark algorithms and stored campaign outputs are unchanged from the 0.1.0 campaign state.

## Fixed release identity

- Package version: `0.1.1`
- Licence: BSD-3-Clause
- Repository: https://github.com/spodlesny2318-arch/Cranebench
- Intended immutable Git tag: `v0.1.1`
- Authors: Serhii Podliesnyi, Oleksii Sheremet, Bohdan Vorobiov
- Release date: 2026-09-05

## Verification performed

- `pytest -q`: 26 tests passed.
- `python tools/verify_manuscript.py`: 126/126 cells across five manuscript result tables matched the stored campaign files within the 1% checking tolerance.
- Public API smoke test executed successfully for the planar LQR example.

The reported campaign ledgers record CPython 3.14.2, NumPy 2.3.3, Windows 10 and the campaign package version `0.1.0`. Release `0.1.1` changes only release metadata and repository hygiene.

## Contents

The release contains the `cranebench` package, examples, tests, verification tools, benchmark campaign outputs, provenance ledgers, design documentation, software metadata, and the manuscript source required by `tools/verify_manuscript.py`. Submission templates, reference-audit spreadsheets, intermediate archives, and transient build/cache files are excluded.
