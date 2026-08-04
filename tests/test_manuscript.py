"""The manuscript tables must agree with the campaign files.

Marked slow because it needs the campaign result files; skipped when they are
absent, so a fresh checkout still gets a green suite.
"""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.mark.skipif(not (ROOT / "run_batch" / "reference_metrics.npz").exists(),
                    reason="campaign result files not present")
def test_manuscript_tables_match_the_campaign_files():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "verify_manuscript.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_checker_reads_the_manuscript_as_utf8():
    """Regression: the locale encoding on Windows mangles superscripts.

    Read with the platform default and the cells holding "5.79·10⁷" fail to
    parse, so they are skipped and the tool reports success on a subset. Found
    on Windows during author verification.
    """
    import tools.verify_manuscript as V  # noqa: F401
    src = (ROOT / "tools" / "verify_manuscript.py").read_text(encoding="utf-8")
    assert 'MD.read_text(encoding="utf-8")' in src, \
        "the manuscript must be read as UTF-8 regardless of platform locale"
