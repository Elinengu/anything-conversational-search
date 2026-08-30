"""Minimal Markdown -> PDF for the repo's docs/*.md -> *.pdf pairs.

Handles: ATX headings, paragraphs, - / * / 1. lists, ``` fenced code,
pipe tables, > blockquotes, --- rules, inline `code` **bold** *italic*.
"""
import re
import sys
import html

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, Table, TableStyle,
    HRFlowable, ListFlowable, ListItem,
)

_TRANS = {
    "─": "-", "│": "|", "┌": " ", "┐": " ", "└": " ",
    "┘": " ", "├": "|", "┤": "|", "┬": "-", "┴": "-",
    "▼": "v", "▲": "^", "→": "->", "←": "<-",
    "≤": "<=", "≥": ">=", "≈": "~",
    "≠": "!=", "∈": " in ", "Σ": "SUM", "…": "...",
    "−": "-",
}


def sanitize(s):
    return "".join(_TRANS.get(ch, ch) for ch in s)


SS = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=SS["BodyText"], fontSize=9.5, leading=13,
                      spaceAfter=6)
CODE = ParagraphStyle("code", parent=SS["Code"], fontSize=7.6, leading=9.5,
                      backColor=colors.HexColor("#f4f4f4"), borderPadding=5,
                      spaceAfter=8)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10, spaceAfter=0)
CELLH = ParagraphStyle("cellh", parent=CELL, fontName="Helvetica-Bold")
QUOTE = ParagraphStyle("quote", parent=BODY, leftIndent=14, textColor=colors.HexColor("#555555"),
                       fontName="Helvetica-Oblique")
H = {
    1: ParagraphStyle("h1", parent=SS["Heading1"], fontSize=17, spaceBefore=16, spaceAfter=8),
    2: ParagraphStyle("h2", parent=SS["Heading2"], fontSize=13.5, spaceBefore=13, spaceAfter=6),
    3: ParagraphStyle("h3", parent=SS["Heading3"], fontSize=11, spaceBefore=10, spaceAfter=4),
    4: ParagraphStyle("h4", parent=SS["Heading4"], fontSize=10, spaceBefore=8, spaceAfter=3),
}


def inline(text):
    text = sanitize(text).replace("\\|", "|")
    codes = []

    def stash(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\s][^*]*?)\*(?!\*)", r"<i>\1</i>", text)

    def restore(m):
        return f'<font face="Courier" size="8.5">{html.escape(codes[int(m.group(1))])}</font>'

    return re.sub(r"\x00(\d+)\x00", restore, text)


def cells(line):
    parts = re.split(r"(?<!\\)\|", line.strip())
    if parts and parts[0].strip() == "":
        parts = parts[1:]
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return [c.strip() for c in parts]


def build(md_path, pdf_path):
    lines = open(md_path, encoding="utf-8").read().split("\n")
    flow = []
    i = 0
    while i < len(lines):
        ln = lines[i]

        if ln.strip().startswith("```"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            flow.append(Preformatted(sanitize("\n".join(buf)) or " ", CODE))
            continue

        if re.match(r"^#{1,4}\s", ln):
            lvl = len(ln) - len(ln.lstrip("#"))
            flow.append(Paragraph(inline(ln.lstrip("#").strip()), H[min(lvl, 4)]))
            i += 1
            continue

        if re.match(r"^(-{3,}|\*{3,}|_{3,})\s*$", ln.strip()):
            flow.append(HRFlowable(width="100%", thickness=0.5,
                                   color=colors.HexColor("#cccccc"), spaceBefore=6, spaceAfter=10))
            i += 1
            continue

        if ln.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|?\s*$", lines[i + 1].strip()):
            header = cells(ln)
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(cells(lines[i]))
                i += 1
            data = [[Paragraph(inline(c), CELLH) for c in header]]
            for r in rows:
                r = (r + [""] * len(header))[:len(header)]
                data.append([Paragraph(inline(c), CELL) for c in r])
            avail = LETTER[0] - 1.5 * inch
            tbl = Table(data, colWidths=[avail / len(header)] * len(header), hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efefef")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            flow.append(tbl)
            flow.append(Spacer(1, 8))
            continue

        if ln.strip().startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            flow.append(Paragraph(inline(" ".join(buf)), QUOTE))
            continue

        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", ln)
        if m:
            items = []
            bullet = "1" if m.group(2)[0].isdigit() else "bullet"
            while i < len(lines):
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if not mm:
                    if lines[i].strip() == "":
                        break
                    items[-1] = items[-1] + " " + lines[i].strip()
                    i += 1
                    continue
                items.append(mm.group(3))
                i += 1
            flow.append(ListFlowable(
                [ListItem(Paragraph(inline(x), BODY), leftIndent=18) for x in items],
                bulletType=bullet, start="1" if bullet == "1" else None, leftIndent=12))
            continue

        if ln.strip() == "":
            i += 1
            continue

        buf = [ln]
        i += 1
        while i < len(lines) and lines[i].strip() != "" and not re.match(
                r"^(#{1,4}\s|```|\||>|\s*([-*]|\d+\.)\s|(-{3,}|\*{3,})\s*$)", lines[i]):
            buf.append(lines[i])
            i += 1
        flow.append(Paragraph(inline(" ".join(x.strip() for x in buf)), BODY))

    SimpleDocTemplate(pdf_path, pagesize=LETTER,
                      leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                      topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                      title=lines[0].lstrip("# ").strip()).build(flow)
    print("wrote", pdf_path)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
