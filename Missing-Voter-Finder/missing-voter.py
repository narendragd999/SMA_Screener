import streamlit as st
import pandas as pd
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO

st.set_page_config(layout="wide", page_title="Missing Serial Numbers Analyzer")

st.title("Missing Voter Serial Numbers Analyzer")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

def create_compact_pdf(df):
    buffer = BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=0.25*inch,   # ← reduced margins
        leftMargin=0.25*inch,
        topMargin=0.4*inch,
        bottomMargin=0.4*inch
    )
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=13, spaceAfter=6)
    elements.append(Paragraph("Missing Voter Serial Numbers Report", title_style))
    
    # Date
    date_style = ParagraphStyle('Date', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
    elements.append(Paragraph(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", date_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Prepare data
    data = []
    header = [Paragraph(f"<b>{col}</b>", styles['Normal']) for col in df.columns]
    data.append(header)
    
    for _, row in df.iterrows():
        row_data = []
        for col_name, val in zip(df.columns, row):
            if col_name == "Missing Serial Numbers" and isinstance(val, str) and len(val) > 30:
                p = Paragraph(
                    val.replace(',', ', '),
                    ParagraphStyle(
                        name='Compact',
                        fontSize=7,
                        leading=8.5,
                        alignment=0,
                        spaceAfter=1,
                        wordWrap='CJK'
                    )
                )
                row_data.append(p)
            else:
                row_data.append(Paragraph(str(val), styles['Normal']))
        data.append(row_data)
    
    # Balanced & compact column widths (total ~11.5 inch usable width in landscape letter)
    col_widths = [
        1.00*inch,     # Part - now clearly visible
        7.80*inch,     # Missing Serial Numbers - reduced
        0.95*inch,     # Range Count
        1.25*inch      # Total Missing Count
    ]
    
    table = Table(data, colWidths=col_widths, repeatRows=1)
    
    table.setStyle(TableStyle([
        # Header
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 5),
        ('TOPPADDING', (0,0), (-1,0), 5),
        ('VALIGN', (0,0), (-1,0), 'MIDDLE'),
        
        # Body
        ('GRID', (0,0), (-1,-1), 0.4, colors.lightgrey),
        ('FONTSIZE', (0,1), (-1,-1), 7),
        ('VALIGN', (0,1), (-1,-1), 'TOP'),
        ('ALIGN', (0,1), (0,-1), 'CENTER'),
        ('ALIGN', (2,1), (-1,-1), 'CENTER'),
        ('ALIGN', (1,1), (1,-1), 'LEFT'),
        ('LEFTPADDING', (0,1), (-1,-1), 4),
        ('RIGHTPADDING', (0,1), (-1,-1), 4),
        ('TOPPADDING', (0,1), (-1,-1), 2),
        ('BOTTOMPADDING', (0,1), (-1,-1), 2),
        
        # Very light rows
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#fbfbfb')]),
    ]))
    
    elements.append(table)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

if uploaded_file is not None:
    try:
        df_input = pd.read_excel(uploaded_file)
        
        expected_cols = ['Part No', 'Part Total Voter', 'Start', 'End']
        if not all(col in df_input.columns for col in expected_cols):
            st.error("Excel file must contain columns: 'Part No', 'Part Total Voter', 'Start', 'End'")
            st.stop()
        
        groups = df_input.groupby('Part No')
        output_data = []
        
        for part, group in groups:
            total = int(group['Part Total Voter'].iloc[0])
            group_sorted = group.sort_values('Start')
            ranges = list(zip(group_sorted['Start'], group_sorted['End']))
            
            merged = []
            for r in ranges:
                if not merged or merged[-1][1] + 1 < r[0]:
                    merged.append(list(r))
                else:
                    merged[-1][1] = max(merged[-1][1], r[1])
            
            total_missing = 0
            current = 1
            
            for start, end in merged:
                if current < start:
                    gap = list(range(current, start))
                    count = len(gap)
                    total_missing += count
                    serials = ','.join(map(str, gap))
                    output_data.append({
                        'Part': part,
                        'Missing Serial Numbers': serials,
                        'Range Count': count,
                        'Total Missing Count': 0
                    })
                current = end + 1
            
            if current <= total:
                gap = list(range(current, total + 1))
                count = len(gap)
                total_missing += count
                serials = ','.join(map(str, gap))
                output_data.append({
                    'Part': part,
                    'Missing Serial Numbers': serials,
                    'Range Count': count,
                    'Total Missing Count': 0
                })
            
            if total_missing == 0:
                output_data.append({
                    'Part': part,
                    'Missing Serial Numbers': 'None',
                    'Range Count': 0,
                    'Total Missing Count': 0
                })
            
            for row in output_data:
                if row['Part'] == part:
                    row['Total Missing Count'] = total_missing
        
        if output_data:
            output_df = pd.DataFrame(output_data)
            column_order = ['Part', 'Missing Serial Numbers', 'Range Count', 'Total Missing Count']
            output_df = output_df[column_order]
            
            st.subheader("Missing Numbers Summary")
            st.dataframe(
                output_df,
                column_config={
                    "Part": st.column_config.TextColumn("Part", width="small"),
                    "Missing Serial Numbers": st.column_config.TextColumn("Missing Serial Numbers", width="large"),
                    "Range Count": st.column_config.NumberColumn("Range Count", width="small"),
                    "Total Missing Count": st.column_config.NumberColumn("Total Missing Count", width="small")
                },
                use_container_width=True,
                hide_index=True
            )
            
            # PDF Download
            pdf_buffer = create_compact_pdf(output_df)
            
            st.download_button(
                label="📄 Download Compact PDF (Part No visible)",
                data=pdf_buffer,
                file_name="Missing_Serial_Numbers_Compact.pdf",
                mime="application/pdf"
            )
        else:
            st.info("No missing numbers detected.")
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")