"""Convert docs/REPORT.md (+ figures/) into docs/SepsisWatch_Report.docx.
Layout: Arial 10.5 pt, no forced page breaks (headings keep with the next paragraph), big headings,
figures scaled per figure so that they stay readable but short. Upload the .docx to Google Drive and it
opens as a Google Doc.   Usage: python docs/build_report_docx.py
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "REPORT.md"
OUT = ROOT / "docs" / "SepsisWatch_Report.docx"
FONT = "Arial"
BLUE = RGBColor(0x1F, 0x3A, 0x5F)
# picture width in inches, by figure file (page text width is 6.9 in)
WIDTH = {"fig1": 6.0, "fig2": 3.8, "fig3": 5.6, "fig5": 4.6, "fig11": 4.6, "fig12": 6.4, "fig10": 5.4,
         "fig6": 6.4, "fig13": 3.2, "fig8a": 4.2, "fig7": 5.0, "fig9": 4.2, "fig14": 6.4}


def set_font(style, size, bold=None, color=None):
    style.font.name = FONT
    style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Inches(0.8)
    sec.top_margin = sec.bottom_margin = Inches(0.8)
    set_font(doc.styles["Normal"], 10.5)
    pf = doc.styles["Normal"].paragraph_format
    pf.space_after = Pt(6); pf.line_spacing = 1.12
    set_font(doc.styles["Title"], 28, True, BLUE)
    set_font(doc.styles["Heading 1"], 22, True, BLUE)
    set_font(doc.styles["Heading 2"], 14, True, RGBColor(0x2E, 0x5E, 0x9E))
    for name, before, after in (("Heading 1", 22, 6), ("Heading 2", 12, 3)):
        f = doc.styles[name].paragraph_format
        f.space_before = Pt(before); f.space_after = Pt(after); f.keep_with_next = True

    est_inches = 0.0
    for line in SRC.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:], 0 if doc.paragraphs == [] else 1)
        elif line.startswith("## "):
            doc.add_heading(line[3:], 2)
        elif m := re.match(r"!\[(.*?)\]\((.*?)\)", line):
            key = Path(m.group(2)).stem.split("_")[0]
            w = WIDTH.get(key, 5.0)
            par = doc.add_paragraph()
            par.paragraph_format.keep_with_next = True      # picture stays with its caption
            par.add_run().add_picture(str(ROOT / m.group(2)), width=Inches(w))
            with Image.open(ROOT / m.group(2)) as im:
                est_inches += w * im.height / im.width
        elif line.startswith("Fig "):
            p = doc.add_paragraph()
            r = p.add_run(line); r.italic = True; r.font.size = Pt(9)
        else:
            doc.add_paragraph(line)
    doc.save(OUT)
    words = sum(len(p.text.split()) for p in doc.paragraphs)
    pages = (words / 16 * 0.2 * 1.25 + est_inches + 3.0) / 9.4        # rough estimate
    print(f"saved {OUT} ({OUT.stat().st_size/1e6:.1f} MB), {words} words, ~{pages:.0f} pages (rough estimate)")


if __name__ == "__main__":
    main()
