import streamlit as st
import pandas as pd
from collections import Counter
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO

st.set_page_config(layout="wide", page_title="Missing + Duplicate Serial Analyzer")
st.title("Missing & Duplicate Voter Serial Numbers Analyzer")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])


def find_duplicates_in_ranges(ranges):
    all_serials = []
    for s, e in ranges:
        all_serials.extend(range(s, e + 1))
    counter = Counter(all_serials)
    dups = [num for num, cnt in counter.items() if cnt > 1]
    dups.sort()
    return dups, len(dups)


def create_summary_row(part, total_voters, ranges):
    # Merge overlapping/adjacent ranges
    merged = []
    for r in sorted(ranges):
        if not merged or merged[-1][1] + 1 < r[0]:
            merged.append(list(r))
        else:
            merged[-1][1] = max(merged[-1][1], r[1])

    # Find missing numbers
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
        'Part': part,
        'Total Voters': total_voters,
        'Missing Serial Numbers': ','.join(map(str, missing)) if missing else 'None',
        'Missing Count': len(missing),
        'Duplicate Serial Numbers': ','.join(map(str, duplicates)) if duplicates else 'None',
        'Duplicate Count': dup_count
    }


def create_pdf(df):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        leftMargin=0.20*inch,    # ← intentionally small left margin
        rightMargin=0.25*inch,
        topMargin=0.40*inch,
        bottomMargin=0.35*inch
    )

    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=14, spaceAfter=8)
    elements.append(Paragraph("Voter Serial Numbers Analysis Report", title_style))

    date_style = ParagraphStyle('Date', parent=styles['Normal'], fontSize=9, textColor=colors.grey)
    elements.append(Paragraph(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", date_style))
    elements.append(Spacer(1, 0.18*inch))

    # ─── Prepare table data ───────────────────────────────
    data = []
    header = [Paragraph(f"<b>{col}</b>", styles['Normal']) for col in df.columns]
    data.append(header)

    compact_style = ParagraphStyle(
        name='Compact',
        fontSize=7.2,
        leading=8.8,
        alignment=0,  # left
        spaceAfter=1,
        wordWrap='CJK'
    )

    normal_right = ParagraphStyle(
        name='Right',
        parent=styles['Normal'],
        alignment=2,  # right
        fontSize=8.5
    )

    for _, row in df.iterrows():
        row_cells = []
        for col_name, val in zip(df.columns, row):
            if col_name in ['Missing Serial Numbers', 'Duplicate Serial Numbers'] and isinstance(val, str) and len(str(val)) > 35:
                text = str(val).replace(',', ', ')
                row_cells.append(Paragraph(text, compact_style))
            elif col_name == 'Part':
                row_cells.append(Paragraph(str(val), normal_right))
            else:
                row_cells.append(Paragraph(str(val), styles['Normal']))
        data.append(row_cells)

    # Column widths - total ~10.6–10.8 inch (should fit landscape letter)
    col_widths = [
        0.95*inch,   # Part           ← increased width + right align
        0.90*inch,   # Total Voters
        5.20*inch,   # Missing Serial Numbers   ← main column
        0.95*inch,   # Missing Count
        2.90*inch,   # Duplicate Serial Numbers
        0.95*inch    # Duplicate Count
    ]

    table = Table(data, colWidths=col_widths, repeatRows=1)

    table.setStyle(TableStyle([
        # Header
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('VALIGN', (0,0), (-1,0), 'MIDDLE'),

        # Body
        ('GRID', (0,0), (-1,-1), 0.45, colors.lightgrey),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('VALIGN', (0,1), (-1,-1), 'TOP'),

        # Alignments
        ('ALIGN', (0,1), (0,-1), 'RIGHT'),     # Part → RIGHT
        ('ALIGN', (1,1), (1,-1), 'RIGHT'),
        ('ALIGN', (2,1), (2,-1), 'LEFT'),
        ('ALIGN', (3,1), (3,-1), 'RIGHT'),
        ('ALIGN', (4,1), (4,-1), 'LEFT'),
        ('ALIGN', (5,1), (5,-1), 'RIGHT'),

        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),

        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9f9f9')]),
    ]))

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        df_input = pd.read_excel(uploaded_file)

        required = ['Part No', 'Part Total Voter', 'Start', 'End']
        if not all(col in df_input.columns for col in required):
            st.error(f"Excel must contain columns: {', '.join(required)}")
            st.stop()

        summary_rows = []
        for part, group in df_input.groupby('Part No'):
            total = int(group['Part Total Voter'].iloc[0])
            ranges = list(zip(group['Start'], group['End']))
            summary_rows.append(create_summary_row(part, total, ranges))

        if not summary_rows:
            st.info("No data to analyze.")
            st.stop()

        result_df = pd.DataFrame(summary_rows)

        st.subheader("Part-wise Summary")
        st.dataframe(
            result_df,
            column_config={
                "Part": st.column_config.TextColumn("Part", width="small"),
                "Total Voters": st.column_config.NumberColumn("Total", width="small"),
                "Missing Serial Numbers": st.column_config.TextColumn("Missing Numbers", width="large"),
                "Missing Count": st.column_config.NumberColumn("Miss. Count", width="small"),
                "Duplicate Serial Numbers": st.column_config.TextColumn("Duplicates", width="medium"),
                "Duplicate Count": st.column_config.NumberColumn("Dup. Count", width="small")
            },
            use_container_width=True,
            hide_index=True
        )

        pdf_buffer = create_pdf(result_df)

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name="Voter_Serial_Analysis.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Error processing file:\n{str(e)}")