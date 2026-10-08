# Compatibility validation

The workflow `.github/workflows/tests.yml` defines four compatibility jobs plus two distribution-validation jobs:

| Profile | Python series | NumPy | SciPy | Systems |
|---|---|---|---|---|
| minimum | 3.10 | 1.24.0 | 1.10.0 | Ubuntu and Windows |
| revision | 3.14 | 2.3.3 | 1.18.0 | Ubuntu and Windows |

The additional Ubuntu and Windows jobs build a wheel and source distribution,
run strict metadata checks, then install each artifact in a separate clean
environment and execute the README example outside the checkout.

Dependency constraints apply to package installation and development extras.
`tools/ci_environment.py` asserts the runtime versions and records the actual
Python patch version, installed distributions and source SHA-256 hashes.

Each job checks installation, runs the full test suite, and saves JUnit and
line/branch coverage reports. Campaign assets must be present so the manuscript
test cannot silently skip merely because data were omitted. Test counts and
coverage are observations for that source revision and environment; they are
not proof of complete numerical or physical validation.

The workflow must reside at the repository root in the published GitHub
checkout. This local copy is prepared for that checkout. A local successful
test run does not establish successful GitHub Actions execution. The final
reviewer response should cite the actual workflow run and commit once available.

## Reproduce a profile locally

From the software root in an isolated Python 3.10 environment:

```console
python -m pip install -c ci/minimum.txt ".[dev]" pytest-cov
python -m pip check
python tools/ci_environment.py minimum --output test-results/environment.json
python -m pytest -q -p no:cacheprovider --junitxml=test-results/pytest.xml --cov=cranebench --cov-branch --cov-report=term-missing --cov-report=xml:test-results/coverage.xml --cov-report=json:test-results/coverage.json
```

Use Python 3.14 and replace `minimum` with `revision` for the second profile.

## Production release path

The TestPyPI trial for `0.1.2.dev0` succeeded, including clean-environment
installation of the wheel. The production release candidate is `0.1.2`; its
tag-triggered workflow validates the tagged source, builds both distributions,
and publishes through the protected GitHub `pypi` environment. The final
versioned GitHub release is intended to trigger the Zenodo archive. Do not
claim production publication or an archival DOI until those services confirm it.
