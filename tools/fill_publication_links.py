"""Fill the repository URL and the Zenodo DOI into every place that needs them.

There are four: the code-metadata table C2, reference [34], CITATION.cff and
README.md.  Typing the same URL into four files by hand is how they come to
disagree, and a manuscript whose metadata table and reference list point at
different places is exactly the failure this package is about.

    python tools/fill_publication_links.py                 # URL from git remote
    python tools/fill_publication_links.py --doi 10.5281/zenodo.1234567
    python tools/fill_publication_links.py --check         # report, change nothing
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "PAPER_SoftwareX_draft.md"
BIB = ROOT / "references.bib"
CFF = ROOT / "CITATION.cff"
README = ROOT / "README.md"


def repo_url():
    """https URL of the 'origin' remote, whatever form it was added in."""
    try:
        out = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    m = re.match(r"(?:https://github\.com/|git@github\.com:)(.+?)(?:\.git)?$", out)
    return f"https://github.com/{m.group(1)}" if m else None


def report(url, doi):
    print(f"repository: {url or 'NOT SET — add the origin remote first'}")
    print(f"DOI:        {doi or 'NOT SET — pass --doi after the Zenodo release'}")
    s = PAPER.read_text(encoding="utf-8")
    for label, pat in (("C2 in the metadata table", r"\| C2 \|[^|]*\|([^|]*)\|"),
                       ("reference [34]", r"^\[34\].*$")):
        m = re.search(pat, s, re.M)
        val = (m.group(1) if m and m.lastindex else m.group(0)) if m else "MISSING"
        print(f"  {label:26s} {val.strip()[:78]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--doi", default=None, help="e.g. 10.5281/zenodo.1234567")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    url = a.url or repo_url()
    if a.check:
        report(url, a.doi)
        return 0
    if not url:
        print("no origin remote and no --url given; nothing to fill")
        return 1

    n = 0
    s = PAPER.read_text(encoding="utf-8")
    s2 = re.sub(r"(\| C2 \| Permanent link to code/repository used for this code version \| )[^|]*(\|)",
                rf"\g<1>{url} \g<2>", s)
    if a.doi:
        s2 = re.sub(r"^\[34\] .*$",
                    f"[34] O. Sheremet, S. Podliesnyi, cranebench: a reproducible benchmark "
                    f"for underactuated crane control, version 1.0.0 [software], Zenodo, 2026. "
                    f"doi:{a.doi}.", s2, flags=re.M)
    if s2 != s:
        PAPER.write_text(s2, encoding="utf-8"); n += 1

    t = CFF.read_text(encoding="utf-8")
    t2 = re.sub(r'repository-code: ".*"', f'repository-code: "{url}"', t)
    if a.doi and "doi:" not in t2:
        t2 = t2.replace("version: 1.0.0", f"doi: {a.doi}\nversion: 1.0.0")
    if t2 != t:
        CFF.write_text(t2, encoding="utf-8"); n += 1

    if BIB.exists() and a.doi:
        b = BIB.read_text(encoding="utf-8")
        b2 = b.replace("doi     = {(to be inserted on release)}", f"doi     = {{{a.doi}}}")
        b2 = re.sub(r"(@\w+\{cranebench[^}]*?doi\s*=\s*\{)[^}]*(\})",
                    rf"\g<1>{a.doi}\g<2>", b2)
        if b2 != b:
            BIB.write_text(b2, encoding="utf-8"); n += 1

    if README.exists():
        r = README.read_text(encoding="utf-8")
        badge = f"\n[Repository]({url})"
        if url not in r:
            r = r.replace("# cranebench", f"# cranebench\n{badge}", 1)
            README.write_text(r, encoding="utf-8"); n += 1

    print(f"updated {n} file(s)")
    report(url, a.doi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
