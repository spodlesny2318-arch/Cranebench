# What belongs in the public repository

Committed, because a reader must be able to reproduce every number:

    cranebench/          the package
    tests/               25 tests
    tools/               symbolic derivation, tuning, manuscript checker
    examples/            campaign runners and summarisers
    docs/                DESIGN.md, figures, reference_check.xlsx
    run_batch/*_metrics.npz, run_batch/*_ledger.json
    run_sp3/spatial_paired.npz, run_dual6/dual_paired.npz
    run_retune/gains_*.json
    README.md  VERIFY.md  RUN_ON_WINDOWS.md  LICENSE.txt  CITATION.cff
    pyproject.toml  references.bib
    PAPER_SoftwareX_draft.md   (the manuscript, so the checker has a target)

Not committed (see .gitignore), because it is intermediate and regenerated:

    *_runs.jsonl         per-run checkpoints
    run_batch/*_PD.npz   per-controller campaign slices
    tuning_evals*.json   every evaluated tuning point
    .venv/ __pycache__/ node_modules/ submission/

The last one is a judgement call: the tuning evaluations are the material
evidence for the declared tuning budget, and a reviewer may reasonably want
them. They are a few hundred kB; if in doubt, commit them.
