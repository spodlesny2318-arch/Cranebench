"""Render the manuscript into the journal's own .docx template.

The journal accepts only submissions made in its template, and asks that the
template's formatting is not altered.  Rebuilding the document with our own
styles would satisfy the letter and not the intent, so this script keeps the
template's package -- styles, theme, numbering, headers, footers, page setup --
untouched and replaces only the body of word/document.xml, using the style ids
the template itself defines (Heading1..3, Body, Normal).

    python tools/build_from_template.py softwarex-osp-template.docx out.docx
"""

from __future__ import annotations

import pathlib
import re
import shutil
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "PAPER_SoftwareX_draft.md"
FIGS = ["docs/fig0_architecture.png", "docs/fig1_verification.png",
        "docs/fig2_campaign.png", "docs/fig3_operating_points.png"]
EMU = 9525                       # EMU per pixel at 96 dpi
TEXT_W = 5943600                 # 6.25 in of text width, in EMU


def esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def runs(text: str) -> str:
    """Inline **bold**, *italic*, `code`, ^superscript and the subscript tokens."""
    text = text.replace("\\*", "*").replace("\\_", "_")   # markdown escapes
    out, last = [], 0
    for m in re.finditer(r"(@dz@|@rrb@|\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|"
                         r"\^[A-Za-z0-9,*]+)", text):
        if m.start() > last:
            out.append(f"<w:r><w:t xml:space='preserve'>{esc(text[last:m.start()])}</w:t></w:r>")
        tok = m.group(0)
        if tok.startswith("^"):
            out.append(f"<w:r><w:rPr><w:vertAlign w:val='superscript'/></w:rPr>"
                       f"<w:t>{esc(tok[1:])}</w:t></w:r>")
            last = m.end(); continue
        if tok in ("@dz@", "@rrb@"):
            base, sub = ("d", "z") if tok == "@dz@" else ("r", "rb")
            out.append(f"<w:r><w:rPr><w:i/></w:rPr><w:t>{base}</w:t></w:r>"
                       f"<w:r><w:rPr><w:vertAlign w:val='subscript'/></w:rPr>"
                       f"<w:t>{sub}</w:t></w:r>")
        elif tok.startswith("**"):
            out.append(f"<w:r><w:rPr><w:b/></w:rPr><w:t xml:space='preserve'>"
                       f"{esc(tok[2:-2])}</w:t></w:r>")
        elif tok.startswith("`"):
            out.append(f"<w:r><w:rPr><w:rFonts w:ascii='Consolas' w:hAnsi='Consolas'/>"
                       f"<w:sz w:val='19'/></w:rPr><w:t xml:space='preserve'>"
                       f"{esc(tok[1:-1])}</w:t></w:r>")
        else:
            out.append(f"<w:r><w:rPr><w:i/></w:rPr><w:t xml:space='preserve'>"
                       f"{esc(tok[1:-1])}</w:t></w:r>")
        last = m.end()
    if last < len(text):
        out.append(f"<w:r><w:t xml:space='preserve'>{esc(text[last:])}</w:t></w:r>")
    return "".join(out) or "<w:r><w:t/></w:r>"


def para(text, style="Body", extra=""):
    return (f"<w:p><w:pPr><w:pStyle w:val='{style}'/>{extra}</w:pPr>"
            f"{runs(text)}</w:p>")


def table(rows, tbl_pr):
    cells = [r.strip().strip("|").split("|") for r in rows]
    ncol = len(cells[0])
    width = 9478
    col = width // ncol
    grid = "".join(f"<w:gridCol w:w='{col}'/>" for _ in range(ncol))
    body = []
    for i, row in enumerate(cells):
        if i == 1:
            continue                       # the markdown separator row
        tcs = []
        for c in row + [""] * (ncol - len(row)):
            txt = c.strip()
            bold = "<w:b/>" if i == 0 else ""
            tcs.append(f"<w:tc><w:tcPr><w:tcW w:w='{col}' w:type='dxa'/></w:tcPr>"
                       f"<w:p><w:pPr><w:pStyle w:val='Normal'/>"
                       f"{'<w:rPr>' + bold + '</w:rPr>' if bold else ''}</w:pPr>"
                       f"{runs(('**' + txt + '**') if i == 0 and txt else txt)}</w:p></w:tc>")
        trpr = "<w:trPr><w:tblHeader/></w:trPr>" if i == 0 else ""
        body.append(f"<w:tr>{trpr}{''.join(tcs)}</w:tr>")
    return f"<w:tbl>{tbl_pr}<w:tblGrid>{grid}</w:tblGrid>{''.join(body)}</w:tbl>"


def image(rid, px_w, px_h, idx):
    w = TEXT_W
    h = int(w * px_h / px_w)
    return (f"<w:p><w:pPr><w:pStyle w:val='Normal'/><w:jc w:val='center'/></w:pPr>"
            f"<w:r><w:drawing><wp:inline distT='0' distB='0' distL='0' distR='0'>"
            f"<wp:extent cx='{w}' cy='{h}'/><wp:docPr id='{100+idx}' name='Figure {idx+1}' "
                       f"descr='Figure {idx+1}: {['architecture of the benchmark harness', 'verification checks and nominal swing histories', 'reference-campaign paired performance comparisons', 'operating-point comparison of bound and performance metrics'][idx]}'/>"
            f"<a:graphic xmlns:a='http://schemas.openxmlformats.org/drawingml/2006/main'>"
            f"<a:graphicData uri='http://schemas.openxmlformats.org/drawingml/2006/picture'>"
            f"<pic:pic xmlns:pic='http://schemas.openxmlformats.org/drawingml/2006/picture'>"
            f"<pic:nvPicPr><pic:cNvPr id='{100+idx}' name='Figure {idx+1}'/><pic:cNvPicPr/></pic:nvPicPr>"
            f"<pic:blipFill><a:blip r:embed='{rid}'/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>"
            f"<pic:spPr><a:xfrm><a:off x='0' y='0'/><a:ext cx='{w}' cy='{h}'/></a:xfrm>"
            f"<a:prstGeom prst='rect'><a:avLst/></a:prstGeom></pic:spPr>"
            f"</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>")


def main(tpl, out):
    work = ROOT / "_tplbuild"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(exist_ok=True)
    with zipfile.ZipFile(tpl) as z:
        z.extractall(work)

    doc = (work / "word" / "document.xml").read_text(encoding="utf-8")
    tbl_pr = re.search(r"<w:tblPr>.*?</w:tblPr>", doc, re.S).group(0)
    sect = re.search(r"<w:sectPr.*?</w:sectPr>", doc, re.S).group(0)
    head = doc[:doc.index("<w:body>") + len("<w:body>")]

    # register the figures as relationships and package parts
    rels_p = work / "word" / "_rels" / "document.xml.rels"
    rels = rels_p.read_text(encoding="utf-8")
    media = work / "word" / "media"
    media.mkdir(exist_ok=True)
    fig_rid, fig_px = [], []
    import struct
    for i, f in enumerate(FIGS):
        src = ROOT / f
        dst = media / f"figure{i+1}.png"
        shutil.copy(src, dst)
        b = src.read_bytes()
        w, h = struct.unpack(">II", b[16:24])
        fig_px.append((w, h))
        rid = f"rIdFig{i+1}"
        fig_rid.append(rid)
        rels = rels.replace("</Relationships>",
                            f"<Relationship Id='{rid}' Type='http://schemas.openxmlformats.org/"
                            f"officeDocument/2006/relationships/image' "
                            f"Target='media/figure{i+1}.png'/></Relationships>")
    rels_p.write_text(rels, encoding="utf-8")
    ct = work / "[Content_Types].xml"
    c = ct.read_text(encoding="utf-8")
    if 'Extension="png"' not in c:
        c = c.replace("</Types>", "<Default Extension='png' ContentType='image/png'/></Types>")
        ct.write_text(c, encoding="utf-8")

    # ---- render the manuscript ----
    md = PAPER.read_text(encoding="utf-8")
    lines = md.split("\n")
    body, i, fig, buf = [], 0, 0, ""

    def flush():
        nonlocal buf
        if buf.strip():
            body.append(para(buf.strip()))
        buf = ""

    while i < len(lines):
        L = lines[i]
        if L.startswith("# "):
            flush(); body.append(para(L[2:].strip(), "Title")); i += 1; continue
        m = re.match(r"^(#{2,4})\s+(.*)$", L)
        if m:
            flush()
            lvl = min(len(m.group(1)) - 1, 3)
            body.append(para(re.sub(r"[*`]", "", m.group(2)), f"Heading{lvl}"))
            i += 1; continue
        if L.strip().startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i]); i += 1
            body.append(table(rows, tbl_pr))
            body.append(para(""))
            continue
        if L.startswith("```"):
            flush(); i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(para(lines[i] or " ", "Normal",
                                 "<w:ind w:left='284'/>"))
                i += 1
            i += 1; continue
        fm = re.match(r"^\*\((Figure \d+[:.][\s\S]*)\)\*$", L.strip())
        if fm and fig < len(FIGS):
            flush()
            body.append(image(fig_rid[fig], *fig_px[fig], fig))
            cap = re.sub(r"^Figure (\d+)[:.]", r"**Figure \1.**", fm.group(1))
            body.append(para(cap, "Normal"))
            fig += 1; i += 1; continue
        if re.match(r"^[-*]\s+", L):
            flush()
            body.append(para(re.sub(r"^[-*]\s+", "", L), "ListParagraph",
                             "<w:numPr><w:ilvl w:val='0'/><w:numId w:val='1'/></w:numPr>"))
            i += 1; continue
        if re.match(r"^\[\d+\]\s", L):
            flush()
            body.append(para(L, "Normal", "<w:ind w:left='284' w:hanging='284'/>"))
            i += 1; continue
        if L.strip() in ("", "---"):
            flush(); i += 1; continue
        buf += (" " if buf else "") + L.strip()
        i += 1
    flush()

    (work / "word" / "document.xml").write_text(
        head + "".join(body) + sect + "</w:body></w:document>", encoding="utf-8")

    out = pathlib.Path(out)
    try:
        out.unlink(missing_ok=True)
    except OSError:
        pass
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(work.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(work).as_posix())
    shutil.rmtree(work, ignore_errors=True)
    print(f"written {out}  ({out.stat().st_size//1024} kB, "
          f"{len(body)} block(s), {fig} figure(s))")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "cranebench_template.docx")
