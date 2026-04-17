"""Convert README.md to README.pdf using fpdf2's markdown support."""
import re
from fpdf import FPDF

# Font names after registration
SANS = "dsans"
MONO = "dmono"


class ReadmePDF(FPDF):
    """Custom PDF with header/footer and markdown rendering helpers."""

    def __init__(self):
        super().__init__()
        # Use built-in DejaVu Unicode fonts shipped with fpdf2
        self.add_font(SANS, "", "DejaVuSans.ttf", uni=True)
        self.add_font(SANS, "B", "DejaVuSans-Bold.ttf", uni=True)
        self.add_font(SANS, "I", "DejaVuSans-Oblique.ttf", uni=True)
        self.add_font(SANS, "BI", "DejaVuSans-BoldOblique.ttf", uni=True)
        self.add_font(MONO, "", "DejaVuSansMono.ttf", uni=True)
        self.add_font(MONO, "B", "DejaVuSansMono-Bold.ttf", uni=True)

    def header(self):
        self.set_font(SANS, "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 6, "YOLOv8n Adversarial Patch \u2014 Project Documentation", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font(SANS, "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title, level=1):
        sizes = {1: 18, 2: 14, 3: 12}
        self.set_font(SANS, "B", sizes.get(level, 11))
        self.set_text_color(30, 30, 30)
        self.ln(4 if level > 1 else 6)
        self.multi_cell(0, 7, title)
        if level <= 2:
            self.set_draw_color(200, 200, 200)
            self.line(self.l_margin, self.get_y() + 1,
                      self.w - self.r_margin, self.get_y() + 1)
            self.ln(3)
        else:
            self.ln(2)

    def body_text(self, text):
        self.set_font(SANS, "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def code_block(self, code):
        self.set_fill_color(244, 244, 244)
        self.set_font(MONO, "", 8.5)
        self.set_text_color(50, 50, 50)
        x = self.get_x()
        w = self.w - self.l_margin - self.r_margin
        for line in code.split("\n"):
            if self.get_y() > self.h - 20:
                self.add_page()
            self.cell(w, 4.5, "  " + line, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def inline_code(self, text):
        self.set_font(MONO, "", 9)
        self.set_text_color(50, 50, 50)

    def bullet(self, text, indent=0):
        self.set_font(SANS, "", 10)
        self.set_text_color(40, 40, 40)
        x = self.l_margin + indent
        self.set_x(x)
        self.cell(5, 5, chr(8226))
        self.multi_cell(self.w - self.r_margin - x - 5, 5, text)
        self.ln(0.5)

    def table(self, headers, rows):
        self.set_font(SANS, "", 8.5)
        available = self.w - self.l_margin - self.r_margin
        n = len(headers)
        # Calculate column widths proportionally based on content
        col_widths = []
        for i in range(n):
            max_len = len(headers[i])
            for row in rows:
                if i < len(row):
                    max_len = max(max_len, len(row[i]))
            col_widths.append(max_len)
        total = sum(col_widths) or 1
        col_widths = [max(w / total * available, 15) for w in col_widths]
        # Normalize to fit
        scale = available / sum(col_widths)
        col_widths = [w * scale for w in col_widths]

        # Header row
        self.set_font(SANS, "B", 8.5)
        self.set_fill_color(235, 235, 235)
        self.set_text_color(30, 30, 30)
        h = 5.5
        for i, header in enumerate(headers):
            self.cell(col_widths[i], h, " " + header, border=1, fill=True)
        self.ln()

        # Data rows
        self.set_font(SANS, "", 8.5)
        self.set_text_color(50, 50, 50)
        self.set_fill_color(255, 255, 255)
        for row in rows:
            # Check page break
            if self.get_y() > self.h - 20:
                self.add_page()
            row_h = h
            for i in range(n):
                val = row[i] if i < len(row) else ""
                self.cell(col_widths[i], row_h, " " + val, border=1)
            self.ln()
        self.ln(3)

    def blockquote(self, text):
        self.set_draw_color(3, 102, 214)
        self.set_fill_color(248, 249, 250)
        x = self.l_margin
        w = self.w - self.l_margin - self.r_margin
        y0 = self.get_y()
        self.set_x(x + 6)
        self.set_font(SANS, "I", 9.5)
        self.set_text_color(68, 68, 68)
        self.multi_cell(w - 6, 5, text)
        y1 = self.get_y()
        self.line(x + 1, y0, x + 1, y1)
        self.ln(3)

    def hr(self):
        self.set_draw_color(220, 220, 220)
        self.line(self.l_margin, self.get_y(),
                  self.w - self.r_margin, self.get_y())
        self.ln(4)


def parse_table(lines):
    """Parse markdown table lines into (headers, rows)."""
    headers = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:  # skip separator
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return headers, rows


def render_md_to_pdf(md_path, pdf_path):
    with open(md_path, encoding="utf-8") as f:
        content = f.read()

    pdf = ReadmePDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        # Heading
        m = re.match(r"^(#{1,3})\s+(.+)", line)
        if m:
            level = len(m.group(1))
            pdf.chapter_title(m.group(2), level)
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^---+\s*$", line):
            pdf.hr()
            i += 1
            continue

        # Table
        if "|" in line and i + 1 < len(lines) and re.match(r"^\|[-| :]+\|", lines[i + 1]):
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            headers, rows = parse_table(table_lines)
            pdf.table(headers, rows)
            continue

        # Fenced code block
        if line.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            pdf.code_block("\n".join(code_lines))
            continue

        # Blockquote
        if line.startswith("> ") or line.startswith(">"):
            bq_lines = []
            while i < len(lines) and (lines[i].startswith("> ") or lines[i].startswith(">")):
                bq_lines.append(lines[i].lstrip("> "))
                i += 1
            pdf.blockquote(" ".join(bq_lines))
            continue

        # Bullet
        m = re.match(r"^(\s*)-\s+(.+)", line)
        if m:
            indent = len(m.group(1))
            # Clean up markdown formatting for bullet text
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(2))
            text = re.sub(r"`(.+?)`", r"\1", text)
            pdf.bullet(text, indent)
            i += 1
            continue

        # Numbered list
        m = re.match(r"^(\d+)\.\s+(.+)", line)
        if m:
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(2))
            text = re.sub(r"`(.+?)`", r"\1", text)
            pdf.set_font(SANS, "", 10)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(6, 5, m.group(1) + ".")
            pdf.multi_cell(0, 5, text)
            pdf.ln(0.5)
            i += 1
            continue

        # Empty line
        if not line.strip():
            i += 1
            continue

        # Regular paragraph - collect consecutive lines
        para_lines = []
        while i < len(lines) and lines[i].strip() and \
              not lines[i].startswith("#") and \
              not lines[i].startswith("```") and \
              not lines[i].startswith("> ") and \
              not lines[i].startswith("---") and \
              not lines[i].startswith("- ") and \
              not re.match(r"^\d+\.\s", lines[i]) and \
              not ("|" in lines[i] and i + 1 < len(lines) and
                   re.match(r"^\|[-| :]+\|", lines[i + 1])):
            para_lines.append(lines[i])
            i += 1
        if para_lines:
            text = " ".join(para_lines)
            # Strip markdown formatting
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            text = re.sub(r"`(.+?)`", r"\1", text)
            text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
            text = re.sub(r"\$\$(.+?)\$\$", r"\1", text)
            text = re.sub(r"\$(.+?)\$", r"\1", text)
            pdf.body_text(text)

    pdf.output(pdf_path)
    print(f"Written: {pdf_path}")


if __name__ == "__main__":
    render_md_to_pdf("README.md", "README.pdf")
