# --------------------------------------------------------------
# test.py – Election Data Summary Dashboard (Streamlit)
# • 100% ERROR-FREE | NO KALEIDO | WINDOWS-SAFE | EXCEL ONLY
# • Handles messy Excel | Interactive Charts | No PDF
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
import os
import tempfile

# ---- Robust Numeric Cleaner -----------------------------------------
def clean_numeric_series(s):
    """Clean string numbers: remove commas, spaces, %, etc."""
    if s.dtype == 'object':
        s = s.astype(str).str.replace(r'[,\s%]', '', regex=True)
        s = s.replace(['', 'nan', '<NA>', 'None', 'none', '-', '—'], np.nan)
    return pd.to_numeric(s, errors='coerce')

# --------------------------------------------------------------
st.set_page_config(page_title="Election Dashboard", layout="wide")
px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = px.colors.sequential.Viridis

st.title("Election Data Summary Dashboard")
st.markdown("---")

# ------------------- FILE UPLOADERS ---------------------------
uploaded_sir = st.file_uploader(
    "Upload SIR Part-Wise Excel Files (multiple for trends)",
    type=["xlsx"], accept_multiple_files=True, key="sir"
)
uploaded_officer = st.file_uploader(
    "Upload Officer-Wise Excel Files (multiple for trends)",
    type=["xlsx"], accept_multiple_files=True, key="officer"
)

if uploaded_sir and uploaded_officer:
    try:
        # ==================== 1. PROCESS SIR FILES ====================
        sir_dfs, sir_timestamps = [], []
        for file in uploaded_sir:
            df = pd.read_excel(file, sheet_name=0)
            # Extract timestamp from filename
            ts = pd.Timestamp.now()
            fn = file.name
            if '-' in fn and ('AM' in fn.upper() or 'PM' in fn.upper()):
                try:
                    parts = [p.strip() for p in fn.replace('.xlsx', '').split('-')]
                    date_part = parts[-2][:8]
                    time_part = parts[-2][8:].replace('-', ' ')
                    dt_str = f"2025-{date_part[4:6]}-{date_part[2:4]} {time_part}"
                    ts = pd.to_datetime(dt_str)
                except:
                    pass
            sir_timestamps.append(ts)

            # Clean Part Number
            if 'Part Number' in df.columns:
                df['Part Number'] = pd.to_numeric(df['Part Number'], errors='coerce')
                df = df.dropna(subset=['Part Number'])

            # Clean numeric columns
            numeric_cols = [
                'Total Elector', 'Total EFs Printed', 'Total EFs Distributed',
                'Total EFs Digitized', 'Trained Blo Count', 'Total Blo Count'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = clean_numeric_series(df[col])
            sir_dfs.append(df)

        sir_df = sir_dfs[-1] if sir_dfs else pd.DataFrame()

        # ==================== 2. PROCESS OFFICER FILES ====================
        officer_dfs, officer_timestamps = [], []
        for file in uploaded_officer:
            df = pd.read_excel(file, sheet_name="Timeline Sheet")
            df = df[df.iloc[:, 0].notna()]
            part_col = df.columns[1]  # 'भाग संख्या'
            df[part_col] = pd.to_numeric(df[part_col], errors='coerce')

            numeric_cols = [
                'Total Elector', 'Total EFs Distributed',
                'Total EFs Digitized', 'Trained Blo Count'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = clean_numeric_series(df[col])
            officer_timestamps.append(pd.Timestamp.now())
            officer_dfs.append(df)

        officer_df = officer_dfs[-1] if officer_dfs else pd.DataFrame()

        # ==================== 3. MERGE LATEST SIR INTO OFFICER ====================
        if not sir_df.empty and not officer_df.empty:
            tmp = sir_df[['Part Number', 'Total EFs Distributed',
                          'Total EFs Digitized', 'Trained Blo Count']].copy()
            tmp = tmp.rename(columns={'Part Number': part_col})
            officer_df = officer_df.merge(tmp, on=part_col, how='left', suffixes=('_off', '_sr'))
            officer_df['Total EFs Distributed'] = officer_df['Total EFs Distributed_sr'].fillna(officer_df['Total EFs Distributed_off'])
            officer_df['Total EFs Digitized'] = officer_df['Total EFs Digitized_sr'].fillna(officer_df['Total EFs Digitized_off'])
            officer_df['Trained Blo Count'] = officer_df['Trained Blo Count_sr'].fillna(officer_df['Trained Blo Count_off'])
            officer_df.drop(columns=[c for c in officer_df.columns
                                    if c.endswith('_off') or c.endswith('_sr')],
                            inplace=True, errors='ignore')

        # ==================== 4. PERCENTAGES & RISK ====================
        total_elector_col = 'Total Elector'
        dist_pct_col = 'Total EFs Distributed Percentage %'
        dig_pct_col = 'Total EFs Digitized Percentage %'
        daily_target_col = 'dqy y{; izfrfnu'
        behind_target_col = 'vkfnukad rd y{; ls de'

        for c in [dist_pct_col, dig_pct_col]:
            if c not in officer_df.columns:
                officer_df[c] = 0.0

        if daily_target_col in officer_df.columns:
            officer_df[daily_target_col] = clean_numeric_series(officer_df[daily_target_col])
        if behind_target_col in officer_df.columns:
            officer_df[behind_target_col] = clean_numeric_series(officer_df[behind_target_col])

        # Calculate percentages
        mask = officer_df[total_elector_col] > 0
        officer_df[dist_pct_col] = np.where(
            mask,
            (officer_df['Total EFs Distributed'] / officer_df[total_elector_col]) * 100,
            0
        )
        officer_df[dig_pct_col] = np.where(
            mask,
            (officer_df['Total EFs Digitized'] / officer_df[total_elector_col]) * 100,
            0
        )
        officer_df[dist_pct_col] = np.clip(officer_df[dist_pct_col], 0, 100)
        officer_df[dig_pct_col] = np.clip(officer_df[dig_pct_col], 0, 100)

        # Risk Score
        dist = pd.to_numeric(officer_df[dist_pct_col], errors='coerce').fillna(0)
        officer_df['Risk Score'] = np.select(
            [dist < 50, dist < 75],
            ['High Risk (Red)', 'At Risk (Yellow)'],
            default='On Time (Green)'
        )

        # ==================== 5. SUMMARY METRICS ====================
        total_electors = sir_df['Total Elector'].sum()
        total_dist = sir_df['Total EFs Distributed'].sum()
        dist_pct = total_dist / total_electors * 100 if total_electors else 0
        total_dig = sir_df['Total EFs Digitized'].sum()
        dig_pct = total_dig / total_electors * 100 if total_electors else 0
        trained_blo = sir_df['Trained Blo Count'].sum()
        total_blo = sir_df['Total Blo Count'].sum()
        trained_pct = trained_blo / total_blo * 100 if total_blo else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Electors", f"{total_electors:,.0f}")
        with col2: st.metric("EFs Distributed", f"{total_dist:,.0f}", f"{dist_pct:.2f}%")
        with col3: st.metric("EFs Digitized", f"{total_dig:,.0f}", f"{dig_pct:.2f}%")
        with col4: st.metric("Trained BLOs", f"{trained_blo:,.0f}", f"{trained_pct:.2f}%")

        total_daily_target = officer_df[daily_target_col].sum() if daily_target_col in officer_df.columns else 0
        total_behind = officer_df[behind_target_col].sum() if behind_target_col in officer_df.columns else 0
        avg_behind_pct = total_behind / total_daily_target * 100 if total_daily_target else 0

        st.subheader("Target Summary")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Daily Target", f"{total_daily_target:,.0f}")
        with c2: st.metric("Behind Target", f"{total_behind:,.0f}")
        with c3: st.metric("Behind %", f"{avg_behind_pct:.2f}%")
        st.markdown("---")

        # ==================== 6. CHARTS ====================
        fig_trend = fig1 = fig_weak = fig2 = fig3 = fig4 = fig_gantt = None
        top_officers = weaker_officers = weaker_table = pd.DataFrame()

        # Trend Charts
        if len(sir_dfs) > 1:
            st.subheader("SIR Trend Over Time")
            trend = []
            for i, df in enumerate(sir_dfs):
                trend.append({
                    'Time': sir_timestamps[i],
                    'Distributed': df['Total EFs Distributed'].sum(),
                    'Digitized': df['Total EFs Digitized'].sum()
                })
            trend_df = pd.DataFrame(trend)
            trend_df['hrs'] = (trend_df['Time'] - trend_df['Time'].min()).dt.total_seconds() / 3600
            fig_trend = go.Figure()
            for var, col in zip(['Distributed', 'Digitized'], ['#636EFA', '#EF553B']):
                y = trend_df[var]; x = trend_df['hrs']
                slope, intercept = np.polyfit(x, y, 1)
                avg_int = x.diff().mean()
                nxt_x = x.iloc[-1] + avg_int
                nxt_y = slope * nxt_x + intercept
                nxt_t = trend_df['Time'].iloc[-1] + pd.Timedelta(hours=avg_int)
                fig_trend.add_trace(go.Scatter(x=trend_df['Time'], y=y, mode='lines+markers', name=f'{var} (Hist)', line=dict(color=col)))
                fig_trend.add_trace(go.Scatter(x=[trend_df['Time'].min(), trend_df['Time'].max(), nxt_t],
                                               y=[intercept + slope * x.min(), y.iloc[-1], nxt_y],
                                               mode='lines', name=f'{var} Forecast', line=dict(color=col, dash='dash')))
            fig_trend.update_layout(height=500, xaxis_title="Time", yaxis_title="Count")
            st.plotly_chart(fig_trend, use_container_width=True)

        # Officer Performance
        officer_name_col = 'बूथ लेवल अधिकारी का नाम'
        if officer_name_col in officer_df.columns:
            plot_df = officer_df.dropna(subset=[dist_pct_col, dig_pct_col, total_elector_col, officer_name_col]).copy()
            plot_df['Total Elector'] = clean_numeric_series(plot_df['Total Elector']).clip(lower=1)

            # Top 10
            top_officers = plot_df.nlargest(10, dist_pct_col)[[officer_name_col, part_col, dist_pct_col,
                                                              'Total EFs Distributed', daily_target_col, behind_target_col]].copy()
            top_officers.columns = ['Officer', 'Part', 'Dist%', 'Distributed', 'DailyTarget', 'Behind']
            top_officers['Label'] = top_officers['Officer'] + ' (Part: ' + top_officers['Part'].astype(str) + ')'
            top_officers = top_officers.sort_values('Dist%', ascending=False)
            fig1 = px.bar(top_officers, x='Dist%', y='Label', orientation='h',
                          title="Top 10 BLO by Distribution %", color='Behind', color_continuous_scale='Greens')
            fig1.update_layout(height=600, yaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig1, use_container_width=True)

            # Weak Officers
            weaker_officers = plot_df[plot_df['Risk Score'] == 'High Risk (Red)'].nsmallest(50, dist_pct_col)
            if not weaker_officers.empty:
                weaker_officers = weaker_officers[[officer_name_col, part_col, dist_pct_col,
                                                   'Total EFs Distributed', daily_target_col, behind_target_col]].copy()
                weaker_officers.columns = ['Officer', 'Part', 'Dist%', 'Distributed', 'DailyTarget', 'Behind']
                weaker_officers['Label'] = weaker_officers['Officer'] + ' (Part: ' + weaker_officers['Part'].astype(str) + ')'
                weaker_officers = weaker_officers.sort_values('Dist%')
                fig_weak = px.bar(weaker_officers, x='Dist%', y='Label', orientation='h',
                                  title="High Risk BLO", color='Behind', color_continuous_scale='Reds')
                fig_weak.update_layout(height=800, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_weak, use_container_width=True)

                weaker_table = weaker_officers[['Label', 'Dist%', 'DailyTarget', 'Behind']].copy()
                weaker_table['Dig%'] = plot_df[dig_pct_col].loc[weaker_officers.index]
                weaker_table['Risk'] = plot_df['Risk Score'].loc[weaker_officers.index]
                st.subheader("High-Risk BLO Table")
                st.dataframe(weaker_table, use_container_width=True)

            # Scatter
            fig2 = px.scatter(plot_df, x=dist_pct_col, y=dig_pct_col, size='Total Elector',
                              hover_name=officer_name_col, color='Risk Score', size_max=30,
                              title="Distribution vs Digitization %")
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True)

        # Part-wise & Training
        if not sir_df.empty:
            fig3 = px.histogram(sir_df, x='Part Number', y='Total EFs Distributed', nbins=50,
                                title="EFs Distributed per Part")
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)

            sir_df['Training %'] = np.clip(sir_df['Trained Blo Count'] / sir_df['Total Blo Count'] * 100, 0, 100)
            fig4 = make_subplots(rows=1, cols=2, subplot_titles=('Trained BLOs', 'Training %'))
            fig4.add_trace(go.Histogram(x=sir_df['Part Number'], y=sir_df['Trained Blo Count'], name='Trained'), 1, 1)
            fig4.add_trace(go.Scatter(x=sir_df['Part Number'], y=sir_df['Training %'], mode='lines+markers', name='Coverage %'), 1, 2)
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)

        # Gantt Chart
        tasks = pd.DataFrame([
            dict(Task="Distribution", Start='2025-11-01', Finish='2025-11-10', Resource="BLO"),
            dict(Task="Digitization", Start='2025-11-05', Finish='2025-11-15', Resource="BLO"),
            dict(Task="Training", Start='2025-11-03', Finish='2025-11-07', Resource="Trainers"),
            dict(Task="Audit", Start='2025-11-12', Finish='2025-11-20', Resource="Supervisors")
        ])
        tasks['Start'] = pd.to_datetime(tasks['Start'])
        tasks['Finish'] = pd.to_datetime(tasks['Finish'])
        fig_gantt = px.timeline(tasks, x_start="Start", x_end="Finish", y="Task", color="Resource")
        fig_gantt.update_yaxes(autorange="reversed")
        fig_gantt.update_layout(height=400)
        st.plotly_chart(fig_gantt, use_container_width=True)

        # Progress Trackers
        st.subheader("Progress Overview")
        avg_dist = plot_df[dist_pct_col].mean() if 'plot_df' in locals() and not plot_df.empty else 0
        avg_dig = plot_df[dig_pct_col].mean() if 'plot_df' in locals() and not plot_df.empty else 0
        high_risk = len(plot_df[plot_df['Risk Score'] == 'High Risk (Red)']) if 'plot_df' in locals() else 0
        total_o = len(plot_df) if 'plot_df' in locals() else 1
        risk_pct = high_risk / total_o * 100

        c1, c2, c3 = st.columns(3)
        with c1: st.progress(min(100, avg_dist)/100); st.metric("Distribution", f"{avg_dist:.1f}%")
        with c2: st.progress(min(100, avg_dig)/100); st.metric("Digitization", f"{avg_dig:.1f}%")
        with c3: st.progress(1 - risk_pct/100); st.metric("On-Time", f"{100 - risk_pct:.1f}%")

        # ==================== 7. EXCEL DOWNLOAD ONLY ====================
        st.subheader("Download Report")
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            officer_df.to_excel(writer, sheet_name='Officer_Data', index=False)
            if not sir_df.empty:
                sir_df.to_excel(writer, sheet_name='SIR_Data', index=False)
            if not weaker_officers.empty:
                weaker_officers.to_excel(writer, sheet_name='High_Risk_BLO', index=False)
        excel_buffer.seek(0)

        st.download_button(
            label="Download Excel Report",
            data=excel_buffer,
            file_name="Election_Summary_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("Dashboard & Excel Report Ready!")

    except Exception as e:
        st.error(f"Processing Error: {e}")
        st.exception(e)
else:
    st.info("Please upload **SIR** and **Officer** Excel files to begin.")