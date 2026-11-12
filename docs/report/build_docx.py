"""Convert docs/report/REPORT.md into a formatted Word document (10 pt, single spacing).

Supported markdown: #/##/### headings, paragraphs, bullet lists (- ), numbered
lists (1. ), pipe tables, images ![caption](path), **bold** and *italic* runs,
and a line containing only '---' as a page break.
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent
MD = HERE / "REPORT.md"
OUT = HERE / "NIDS_Report.docx"
FONT = "Times New Roman"


def set_cell_border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "4"); el.set(qn("w:color"), "808080")
        borders.append(el)
    tcPr.append(borders)


def add_runs(par, text, size=10, bold_all=False):
    tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**"):
            run = par.add_run(tok[2:-2]); run.bold = True
        elif tok.startswith("`"):
            run = par.add_run(tok[1:-1]); run.font.name = "Consolas"
        elif tok.startswith("*"):
            run = par.add_run(tok[1:-1]); run.italic = True
        else:
            run = par.add_run(tok)
        run.font.size = Pt(size)
        if bold_all:
            run.bold = True
        if run.font.name is None:
            run.font.name = FONT


def style_doc(doc):
    st = doc.styles["Normal"]
    st.font.name = FONT; st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    pf = st.paragraph_format
    pf.space_after = Pt(4); pf.space_before = Pt(0); pf.line_spacing = 1.0
    for lvl, size in ((1, 14), (2, 12), (3, 11)):
        h = doc.styles[f"Heading {lvl}"]
        h.font.name = FONT; h.font.size = Pt(size); h.font.bold = True
        h.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        h.paragraph_format.space_before = Pt(10 if lvl == 1 else 6); h.paragraph_format.space_after = Pt(4)
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.9); s.left_margin = s.right_margin = Inches(1.0)


def add_table(doc, lines):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines if not re.match(r"^\s*\|?\s*:?-{2,}", ln)]
    ncol = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncol)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, r in enumerate(rows):
        for j in range(ncol):
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            add_runs(p, r[j] if j < len(r) else "", size=8.5, bold_all=(i == 0))
            p.paragraph_format.space_after = Pt(1)
            set_cell_border(cell)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def build():
    doc = Document(); style_doc(doc)
    lines = MD.read_text(encoding="utf-8").splitlines()
    i, para_buf = 0, []

    def flush():
        nonlocal para_buf
        if para_buf:
            p = doc.add_paragraph(); add_runs(p, " ".join(para_buf)); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            para_buf = []

    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "---":
            flush(); doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE); i += 1; continue
        if ln.startswith("#"):
            flush(); level = len(ln) - len(ln.lstrip("#"))
            text = ln.lstrip("#").strip()
            if level == 1 and i == 0:
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_runs(p, text, size=16, bold_all=True)
            else:
                doc.add_heading(text, level=min(level, 3))
            i += 1; continue
        m = re.match(r"!\[(.*?)\]\((.*?)\)", ln.strip())
        if m:
            flush(); cap, path = m.groups()
            img = (HERE / path).resolve()
            if img.exists():
                doc.add_picture(str(img), width=Inches(6.0))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(p, cap, size=9); p.runs[0].italic = True if p.runs else None
            i += 1; continue
        if ln.strip().startswith("|"):
            flush(); block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            add_table(doc, block); continue
        if re.match(r"^\s*[-*] ", ln):
            flush(); p = doc.add_paragraph(style="List Bullet"); add_runs(p, re.sub(r"^\s*[-*] ", "", ln)); i += 1; continue
        if re.match(r"^\s*\d+\. ", ln):
            flush(); p = doc.add_paragraph(style="List Number"); add_runs(p, re.sub(r"^\s*\d+\. ", "", ln)); i += 1; continue
        if re.match(r"^\s*\[\d+\]", ln):
            flush(); p = doc.add_paragraph(); add_runs(p, ln.strip(), size=9)
            p.paragraph_format.left_indent = Inches(0.35); p.paragraph_format.first_line_indent = Inches(-0.35)
            p.paragraph_format.space_after = Pt(2); i += 1; continue
        if ln.strip().startswith("```"):
            flush(); i += 1; code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            i += 1
            p = doc.add_paragraph(); r = p.add_run("\n".join(code)); r.font.name = "Consolas"; r.font.size = Pt(8)
            continue
        if not ln.strip():
            flush(); i += 1; continue
        para_buf.append(ln.strip()); i += 1
    flush()
    doc.save(OUT); print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
