import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def md_to_reportlab(md_filepath, pdf_filepath):
    with open(md_filepath, "r", encoding="utf-8") as f:
        content = f.read()

    doc = SimpleDocTemplate(
        pdf_filepath,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#312E81")   # Deep Indigo
    secondary_color = colors.HexColor("#4F46E5") # Indigo Accent
    text_color = colors.HexColor("#1F2937")      # Dark Slate
    code_bg = colors.HexColor("#F3F4F6")         # Light Grey

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=8,
        alignment=TA_CENTER
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=15,
        alignment=TA_CENTER
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=primary_color,
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=text_color,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#111827"),
        backColor=code_bg,
        borderColor=colors.HexColor("#E5E7EB"),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=8,
        spaceBefore=4
    )

    story = []

    lines = content.split("\n")
    in_code_block = False
    code_lines = []
    in_table = False
    table_lines = []

    def format_inline_markdown(text):
        # Convert bold **text** -> <b>text</b>
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        # Convert italic *text* -> <i>text</i>
        text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
        # Convert inline code `code` -> <font name="Courier">\1</font>
        text = re.sub(r"`(.*?)`", r'<font name="Courier" color="#374151" size="8.5">\1</font>', text)
        return text

    i = 0
    while i < len(lines):
        line = lines[i]

        # Handle Code Blocks
        if line.strip().startswith("```"):
            if in_code_block:
                # End of code block
                code_text = "\n".join(code_lines)
                # Replace special HTML chars in code
                code_text = code_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(code_text.replace("\n", "<br/>").replace(" ", "&nbsp;"), code_style))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Handle Markdown Tables
        if "|" in line and not in_code_block:
            if not in_table:
                in_table = True
                table_lines = [line]
            else:
                table_lines.append(line)
            
            # Check if next line is not table
            if i + 1 >= len(lines) or "|" not in lines[i+1]:
                # Render table
                in_table = False
                parsed_rows = []
                for tline in table_lines:
                    if re.match(r"^\|?\s*:?-+:?\s*\|", tline.strip()):
                        continue # separator row
                    cells = [c.strip() for c in tline.strip("|").split("|")]
                    parsed_rows.append([Paragraph(format_inline_markdown(c), body_style) for c in cells])
                
                if parsed_rows:
                    t = Table(parsed_rows, colWidths=None)
                    t.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2FF")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), primary_color),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 8))
                table_lines = []
            i += 1
            continue

        # Handle Headings
        if line.startswith("# "):
            story.append(Paragraph(format_inline_markdown(line[2:].strip()), title_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=10))
        elif line.startswith("## "):
            story.append(Spacer(1, 10))
            story.append(Paragraph(format_inline_markdown(line[3:].strip()), h1_style))
            story.append(HRFlowable(width="100%", thickness=0.75, color=secondary_color, spaceAfter=6))
        elif line.startswith("### "):
            story.append(Paragraph(format_inline_markdown(line[4:].strip()), h2_style))
        elif line.startswith("---"):
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceAfter=6))
        elif line.strip().startswith("* ") or line.strip().startswith("- "):
            bullet_text = format_inline_markdown(line.strip()[2:])
            story.append(Paragraph(f"• {bullet_text}", bullet_style))
        elif line.strip():
            story.append(Paragraph(format_inline_markdown(line.strip()), body_style))

        i += 1

    doc.build(story)
    print(f"PDF successfully generated at: {pdf_filepath}")

if __name__ == "__main__":
    md_file = r"d:\MadRocket\DoCopilot\docopilot_interview_master_guide.md"
    pdf_file = r"d:\MadRocket\DoCopilot\docopilot_interview_master_guide.pdf"
    md_to_reportlab(md_file, pdf_file)
