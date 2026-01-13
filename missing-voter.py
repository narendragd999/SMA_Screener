import streamlit as st
import pandas as pd
from collections import Counter
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import (
    SimpleDocTemplate,
    TableStyle,
    Paragraph,
    Spacer,
    LongTable
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO

# ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Missing + Duplicate Serial Analyzer")
st.title("Missing & Duplicate Voter Serial Numbers Analyzer")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

# ─────────────────────────────────────────────
def find_duplicates_in_ranges(ranges):
    all_serials = []
    for s, e in ranges:
        all_serials.extend(range(int(s), int(e) + 1))
    counter = Counter(all_serials)
    dups = sorted([num for num, cnt in counter.items() if cnt > 1])
    return dups, len(dups)

# ─────────────────────────────────────────────
def create_summary_row(part, total_voters, ranges):
    if not ranges:
        return {
            "Part": part,
            "Total Voters": total_voters,
            "Last Serial Used": 0,
            "Missing Serial Numbers": "None",
            "Missing Count": total_voters,
            "Duplicate Serial Numbers": "None",
            "Duplicate Count": 0
        }

    merged = []
    for r in sorted(ranges):
        if not merged or merged[-1][1] + 1 < r[0]:
            merged.append(list(r))
        else:
            merged[-1][1] = max(merged[-1][1], r[1])

    last_used = max(end for _, end in ranges)

    missing = []
    current = 1
    for start, end in merged:
        if current < start:
            missing.extend(range(current, start))
        current = end + 1

    if current <= total_voters:
        missing.extend(range(current, total_voters + 1))

    duplicates, dup_count = find_duplicates_in_ranges(ranges)

    return {
        "Part": part,
        "Total Voters": total_voters,
        "Last Serial Used": last_used,
        "Missing Serial Numbers": ",".join(map(str, missing)) if missing else "None",
        "Missing Count": len(missing),
        "Duplicate Serial Numbers": ",".join(map(str, duplicates)) if duplicates else "None",
        "Duplicate Count": dup_count
    }

# ─────────────────────────────────────────────
def create_pdf(df):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        leftMargin=0.25 * inch,
        rightMargin=0.25 * inch,
        topMargin=0.40 * inch,
        bottomMargin=0.35 * inch
    )

    elements = []
    styles = getSampleStyleSheet()

    elements.append(
        Paragraph(
            "<b>Voter Serial Numbers Analysis Report</b>",
            ParagraphStyle("title", fontSize=14, spaceAfter=6)
        )
    )

    elements.append(
        Paragraph(
            f"Generated on: {pd.Timestamp.now().strftime('%d-%m-%Y %H:%M')}",
            ParagraphStyle("date", fontSize=9, textColor=colors.grey)
        )
    )

    elements.append(Spacer(1, 10))

    # ─── Table Data ───────────────────────────
    data = []
    header = [Paragraph(f"<b>{c}</b>", styles["Normal"]) for c in df.columns]
    data.append(header)

    compact = ParagraphStyle(
        "compact",
        fontSize=6.8,
        leading=8,
        wordWrap="CJK"
    )

    right = ParagraphStyle(
        "right",
        fontSize=8,
        alignment=2
    )

    for _, row in df.iterrows():
        row_cells = []
        for col, val in zip(df.columns, row):
            if col in ["Missing Serial Numbers", "Duplicate Serial Numbers"]:
                row_cells.append(Paragraph(str(val), compact))
            elif col in ["Part", "Total Voters", "Last Serial Used", "Missing Count", "Duplicate Count"]:
                row_cells.append(Paragraph(str(val), right))
            else:
                row_cells.append(Paragraph(str(val), styles["Normal"]))
        data.append(row_cells)

    col_widths = [
        0.8 * inch,
        0.9 * inch,
        0.6 * inch,
        4.8 * inch,
        0.8 * inch,
        2.8 * inch,
        0.6 * inch
    ]

    table = LongTable(
        data,
        colWidths=col_widths,
        repeatRows=1,
        splitByRow=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),

        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 1), (-1, -1), "TOP"),

        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    elements.append(table)
    doc.build(elements)

    buffer.seek(0)
    return buffer

# ─────────────────────────────────────────────
if uploaded_file:
    try:
        df_input = pd.read_excel(uploaded_file)

        required = ["Part No", "Part Total Voter", "Start", "End"]
        if not all(col in df_input.columns for col in required):
            st.error(f"Excel must contain columns: {', '.join(required)}")
            st.stop()

        summary = []
        for part, group in df_input.groupby("Part No"):
            total = int(group["Part Total Voter"].iloc[0])
            ranges = list(zip(group["Start"], group["End"]))
            summary.append(create_summary_row(part, total, ranges))

        result_df = pd.DataFrame(summary)[[
            "Part",
            "Total Voters",
            "Last Serial Used",
            "Missing Serial Numbers",
            "Missing Count",
            "Duplicate Serial Numbers",
            "Duplicate Count"
        ]]

        st.subheader("Part-wise Summary")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        pdf = create_pdf(result_df)
        st.download_button(
            "📄 Download PDF Report",
            pdf,
            file_name="Voter_Serial_Analysis.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(str(e))
