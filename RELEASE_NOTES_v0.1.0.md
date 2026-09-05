# cranebench 0.1.0 — SoftwareX release candidate

This release is the software state used for the accompanying SoftwareX submission.

## Fixed release identity

- Package version: `0.1.0`
- Licence: BSD-3-Clause
- Repository: https://github.com/spodlesny2318-arch/Cranebench
- Intended immutable Git tag: `v0.1.0`
- Authors: Serhii Podliesnyi, Oleksii Sheremet, Bohdan Vorobiov
- Release date: 2026-09-05

## Verification performed

- `pytest -q`: 26 tests passed.
- `python tools/verify_manuscript.py`: 126/126 cells across five manuscript result tables matched the stored campaign files within the 1% checking tolerance.
- Public API smoke test executed successfully for the planar LQR example.

The reported campaign ledgers record CPython 3.14.2, NumPy 2.3.3, Windows 10 and the package version `0.1.0`.

## Contents

The release contains the `cranebench` package, examples, tests, verification tools, benchmark campaign outputs, provenance ledgers, design documentation and publication-oriented software metadata. Transient Python caches and submission-only manuscript/template files are intentionally excluded.
