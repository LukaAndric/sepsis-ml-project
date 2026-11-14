"""Convert docs/REPORT.md (+ figures/) into docs/SepsisWatch_Report.docx with a page break before each
numbered section (# n. ...), real tables and embedded pictures. Upload the .docx to Google Drive and
it opens as a Google Doc.   Usage: python docs/build_report_docx.py
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "REPORT.md"
OUT = ROOT / "docs" / "SepsisWatch_Report.docx"
BOLD = re.compile(r"(\*\*[^*]+\*\*)")


def add_runs(par, text, italic=False):
    for part in BOLD.split(text):
        if not part:
            continue
        bold = part.startswith("**") and part.endswith("**")
        run = par.add_run(part[2:-2] if bold else part)
        run.bold = bold
        run.italic = italic


def flush_table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(set(c) <= set("-: ") for c in r)]
    t = doc.add_table(rows=len(cells), cols=len(cells[0]))
    t.style = "Table Grid"
    for i, r in enumerate(cells):
        for j, c in enumerate(r):
            p = t.cell(i, j).paragraphs[0]
            add_runs(p, c)
            for run in p.runs:
                run.font.size = Pt(9)
                run.bold = run.bold or i == 0
    doc.add_paragraph()


def main():
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    lines = SRC.read_text(encoding="utf-8").splitlines()
    table, first_h1 = [], True
    for line in lines + [""]:
        if line.startswith("|"):
            table.append(line)
            continue
        if table:
            flush_table(doc, table); table = []
        if not line.strip():
            continue
        if line.startswith("# "):
            title = line[2:]
            if first_h1:
                doc.add_heading(title, 0); first_h1 = False
            else:
                doc.add_page_break()                       # every numbered section starts on a new page
                doc.add_heading(title, 1)
        elif m := re.match(r"!\[(.*?)\]\((.*?)\)", line):
            doc.add_picture(str(ROOT / m.group(2)), width=Inches(6.2))
        elif m := re.match(r"^(\s*)- (.*)", line):
            style = "List Bullet 2" if len(m.group(1)) >= 2 else "List Bullet"
            add_runs(doc.add_paragraph(style=style), m.group(2))
        elif m := re.match(r"^\s*(\d+)\. (.*)", line):
            add_runs(doc.add_paragraph(style="List Number"), m.group(2))
        elif line.startswith("Fig "):
            add_runs(doc.add_paragraph(), line, italic=True)
        else:
            add_runs(doc.add_paragraph(), line)
    doc.save(OUT)
    print("saved", OUT, f"({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
