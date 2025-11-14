import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
from io import BytesIO
import base64
import re

# Page configuration
st.set_page_config(
    page_title="Election Data Reporting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .report-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class ElectionDataProcessor:
    def __init__(self):
        self.sir_data = None
        self.officer_data = None
        self.merged_data = None
        self.historical_data = None
        
    def clean_numeric_data(self, series):
        """Clean numeric data by removing formulas and converting to numbers"""
        cleaned_series = series.copy()
        
        for i, value in enumerate(cleaned_series):
            if pd.isna(value):
                continue
                
            # If it's a string that looks like a formula, extract the numeric part
            if isinstance(value, str):
                # Remove formula indicators and extract numbers
                if '=' in value:
                    # Try to extract numbers from formulas like "=I2/15"
                    numbers = re.findall(r'\d+\.?\d*', value)
                    if numbers:
                        # Take the first number found
                        value = float(numbers[0])
                    else:
                        value = 0
                else:
                    # Try to convert regular string to number
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        value = 0
            
            # Ensure it's a numeric value
            try:
                cleaned_series.iloc[i] = float(value) if pd.notna(value) else 0
            except (ValueError, TypeError):
                cleaned_series.iloc[i] = 0
        
        return pd.to_numeric(cleaned_series, errors='coerce').fillna(0)
    
    def load_sir_data(self, file):
        """Load Rajasthan SIR data"""
        try:
            self.sir_data = pd.read_excel(file, sheet_name='EnumFormTracking')
            
            # Clean numeric columns in SIR data
            numeric_columns = ['Total Elector', 'Total EFs Printed', 'Total EFs Distributed', 
                             'Total EFs Digitized', 'Trained Blo Count', 'Total Blo Count']
            
            for col in numeric_columns:
                if col in self.sir_data.columns:
                    self.sir_data[col] = self.clean_numeric_data(self.sir_data[col])
            
            st.success(f"✅ SIR Data loaded: {self.sir_data.shape[0]} rows, {self.sir_data.shape[1]} columns")
            
            # Show data types for debugging
            with st.expander("SIR Data Types"):
                st.write(self.sir_data.dtypes)
                
            return True
        except Exception as e:
            st.error(f"Error loading SIR data: {str(e)}")
            return False
    
    def load_officer_data(self, file):
        """Load Officer-wise data"""
        try:
            self.officer_data = pd.read_excel(file, sheet_name='Timeline Sheet')
            
            # Clean numeric columns in Officer data
            # Column indices: I=8 (Total Elector), L=11 (Total EFs Distributed), N=13 (Total EFs Digitized), P=15 (Trained Blo Count)
            numeric_indices = [8, 11, 13, 15]
            
            for idx in numeric_indices:
                if idx < len(self.officer_data.columns):
                    self.officer_data.iloc[:, idx] = self.clean_numeric_data(self.officer_data.iloc[:, idx])
            
            st.success(f"✅ Officer Data loaded: {self.officer_data.shape[0]} rows, {self.officer_data.shape[1]} columns")
            
            # Show data types for debugging
            with st.expander("Officer Data Types"):
                st.write(self.officer_data.dtypes)
                
            return True
        except Exception as e:
            st.error(f"Error loading Officer data: {str(e)}")
            return False
    
    def load_historical_data(self, file):
        """Load historical SIR data for comparison"""
        try:
            self.historical_data = pd.read_excel(file, sheet_name='EnumFormTracking')
            
            # Clean numeric columns
            numeric_columns = ['Total Elector', 'Total EFs Printed', 'Total EFs Distributed', 
                             'Total EFs Digitized', 'Trained Blo Count', 'Total Blo Count']
            
            for col in numeric_columns:
                if col in self.historical_data.columns:
                    self.historical_data[col] = self.clean_numeric_data(self.historical_data[col])
            
            st.success(f"✅ Historical Data loaded: {self.historical_data.shape[0]} rows")
            return True
        except Exception as e:
            st.error(f"Error loading historical data: {str(e)}")
            return False
    
    def merge_and_update_data(self):
        """Merge and update data based on common part numbers"""
        if self.sir_data is None or self.officer_data is None:
            st.error("Please load both SIR and Officer data files first")
            return False
        
        try:
            # Create a copy of officer data
            self.merged_data = self.officer_data.copy()
            
            # Convert part numbers to same type for merging
            self.sir_data['Part Number'] = self.sir_data['Part Number'].astype(str).str.strip()
            self.merged_data['Part Number'] = self.merged_data.iloc[:, 1].astype(str).str.strip()  # Column B
            
            # Create mapping from SIR data
            sir_mapping = {}
            for _, row in self.sir_data.iterrows():
                part_num = str(row['Part Number'])
                sir_mapping[part_num] = {
                    'distributed': row['Total EFs Distributed'],
                    'digitized': row['Total EFs Digitized'],
                    'trained': row['Trained Blo Count']
                }
            
            # Update officer data with SIR data
            update_count = 0
            for idx in range(len(self.merged_data)):
                part_num = str(self.merged_data.iloc[idx, 1])  # Column B
                
                if part_num in sir_mapping:
                    # Update columns L, N, P (indices 11, 13, 15)
                    self.merged_data.iloc[idx, 11] = sir_mapping[part_num]['distributed']
                    self.merged_data.iloc[idx, 13] = sir_mapping[part_num]['digitized']
                    self.merged_data.iloc[idx, 15] = sir_mapping[part_num]['trained']
                    update_count += 1
            
            # Apply sorting: first by Officer (Column E), then by Total EFs Distributed (Column L)
            self.merged_data = self.merged_data.sort_values(
                by=[self.merged_data.columns[4], self.merged_data.columns[11]],  # Columns E and L
                ascending=[True, False]
            )
            
            # Ensure numeric columns are properly typed
            numeric_indices = [8, 11, 13, 15]
            for idx in numeric_indices:
                self.merged_data.iloc[:, idx] = self.clean_numeric_data(self.merged_data.iloc[:, idx])
            
            st.success(f"✅ Data merged and sorted successfully! Updated {update_count} records")
            return True
            
        except Exception as e:
            st.error(f"Error merging data: {str(e)}")
            return False
    
    def get_download_link(self, df, filename):
        """Generate download link for DataFrame"""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Updated_Report')
        processed_data = output.getvalue()
        
        b64 = base64.b64encode(processed_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Updated Excel File</a>'
        return href

class ElectionReportGenerator:
    def __init__(self, data_processor):
        self.dp = data_processor
    
    def safe_sum(self, series):
        """Safely sum a series with proper error handling"""
        try:
            cleaned = self.dp.clean_numeric_data(series)
            return cleaned.sum()
        except:
            return 0
    
    def safe_mean(self, series):
        """Safely calculate mean with proper error handling"""
        try:
            cleaned = self.dp.clean_numeric_data(series)
            return cleaned.mean()
        except:
            return 0
    
    def show_data_overview(self):
        """Display data overview"""
        st.header("📋 Data Overview")
        
        if self.dp.merged_data is not None:
            df = self.dp.merged_data
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_parts = len(df)
                st.metric("Total Parts", f"{total_parts:,}")
            
            with col2:
                total_electors = self.safe_sum(df.iloc[:, 8])  # Column I
                st.metric("Total Electors", f"{total_electors:,.0f}")
            
            with col3:
                total_distributed = self.safe_sum(df.iloc[:, 11])  # Column L
                st.metric("Total EFs Distributed", f"{total_distributed:,.0f}")
            
            with col4:
                total_digitized = self.safe_sum(df.iloc[:, 13])  # Column N
                st.metric("Total EFs Digitized", f"{total_digitized:,.0f}")
            
            # Additional metrics
            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_electors = self.safe_sum(df.iloc[:, 8])
                total_distributed = self.safe_sum(df.iloc[:, 11])
                avg_distribution = (total_distributed / total_electors * 100) if total_electors > 0 else 0
                st.metric("Average Distribution %", f"{avg_distribution:.1f}%")
            
            with col2:
                total_trained = self.safe_sum(df.iloc[:, 15])  # Column P
                trained_ratio = (total_trained / len(df)) * 100 if len(df) > 0 else 0
                st.metric("Trained BLO Ratio", f"{trained_ratio:.1f}%")
            
            with col3:
                digitization_rate = (total_digitized / total_distributed * 100) if total_distributed > 0 else 0
                st.metric("Digitization Rate", f"{digitization_rate:.1f}%")
            
            with col4:
                avg_electors_per_part = self.safe_mean(df.iloc[:, 8])
                st.metric("Avg Electors/Part", f"{avg_electors_per_part:.0f}")
    
    def generate_officer_wise_report(self):
        """Generate officer-wise performance report"""
        st.header("👨‍💼 Officer-wise Performance Report")
        
        if self.dp.merged_data is None:
            st.warning("Please load and merge data first")
            return
        
        df = self.dp.merged_data
        
        # Officer selection
        officers = df.iloc[:, 4].unique()  # Column E
        selected_officer = st.selectbox("Select Officer:", officers)
        
        if selected_officer:
            officer_data = df[df.iloc[:, 4] == selected_officer]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_parts = len(officer_data)
                st.metric("Total Parts", total_parts)
            
            with col2:
                total_electors = self.safe_sum(officer_data.iloc[:, 8])  # Column I
                st.metric("Total Electors", f"{total_electors:,}")
            
            with col3:
                total_distributed = self.safe_sum(officer_data.iloc[:, 11])  # Column L
                st.metric("EFs Distributed", f"{total_distributed:,}")
            
            with col4:
                distribution_rate = (total_distributed / total_electors * 100) if total_electors > 0 else 0
                st.metric("Distribution Rate", f"{distribution_rate:.1f}%")
            
            # Display officer's parts data
            st.subheader(f"Parts under {selected_officer}")
            
            # Create a clean display dataframe
            display_data = officer_data.copy()
            display_data['Distribution %'] = (display_data.iloc[:, 11] / display_data.iloc[:, 8] * 100).round(1)
            
            display_cols = [
                df.columns[1],  # B - Part Number
                df.columns[8],  # I - Total Electors
                df.columns[11], # L - EFs Distributed
                'Distribution %',
                df.columns[13], # N - EFs Digitized
                df.columns[15]  # P - Trained BLO
            ]
            
            st.dataframe(display_data[display_cols], use_container_width=True)
    
    def generate_comparison_report(self):
        """Generate comparison reports"""
        st.header("📊 Comparison Analysis")
        
        if self.dp.merged_data is None:
            st.warning("Please load and merge data first")
            return
        
        df = self.dp.merged_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Officer performance comparison
            officer_stats = []
            for officer in df.iloc[:, 4].unique():  # Column E
                officer_df = df[df.iloc[:, 4] == officer]
                total_electors = self.safe_sum(officer_df.iloc[:, 8])
                total_distributed = self.safe_sum(officer_df.iloc[:, 11])
                distribution_rate = (total_distributed / total_electors * 100) if total_electors > 0 else 0
                
                officer_stats.append({
                    'Officer': officer,
                    'Total_Parts': len(officer_df),
                    'Distribution_Rate': distribution_rate
                })
            
            officer_stats_df = pd.DataFrame(officer_stats)
            
            if not officer_stats_df.empty:
                fig = px.bar(officer_stats_df, 
                            x='Officer', 
                            y='Distribution_Rate',
                            title="EF Distribution Rate by Officer",
                            color='Distribution_Rate')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Part size distribution
            elector_data = self.dp.clean_numeric_data(df.iloc[:, 8])
            if len(elector_data) > 0:
                fig = px.histogram(x=elector_data, 
                                 title="Distribution of Part Sizes (Total Electors)",
                                 nbins=20,
                                 labels={'x': 'Number of Electors', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
    
    def generate_custom_report(self):
        """Generate custom reports based on user criteria"""
        st.header("🔍 Custom Report Generator")
        
        if self.dp.merged_data is None:
            st.warning("Please load and merge data first")
            return
        
        df = self.dp.merged_data
        
        # Filter options
        st.subheader("Filter Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Officer filter
            officers = ['All'] + list(df.iloc[:, 4].unique())  # Column E
            selected_officer = st.selectbox("Filter by Officer:", officers)
        
        with col2:
            # Part number range
            part_numbers = self.dp.clean_numeric_data(df.iloc[:, 1])  # Column B
            min_part = int(part_numbers.min()) if len(part_numbers) > 0 else 1
            max_part = int(part_numbers.max()) if len(part_numbers) > 0 else 100
            part_range = st.slider("Part Number Range:", min_part, max_part, (min_part, max_part))
        
        with col3:
            # Distribution percentage filter
            min_dist = st.slider("Minimum Distribution %:", 0, 100, 0)
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_officer != 'All':
            filtered_df = filtered_df[filtered_df.iloc[:, 4] == selected_officer]
        
        # Filter by part number range
        part_nums = self.dp.clean_numeric_data(filtered_df.iloc[:, 1])
        filtered_df = filtered_df[
            (part_nums >= part_range[0]) & 
            (part_nums <= part_range[1])
        ]
        
        # Calculate distribution percentage and filter
        electors = self.dp.clean_numeric_data(filtered_df.iloc[:, 8])
        distributed = self.dp.clean_numeric_data(filtered_df.iloc[:, 11])
        distribution_pct = (distributed / electors * 100).fillna(0)
        filtered_df = filtered_df[distribution_pct >= min_dist]
        
        # Display results
        st.subheader(f"Filtered Results: {len(filtered_df)} parts")
        
        # Select columns to display
        all_columns = list(df.columns)
        default_cols = [1, 4, 8, 11, 13, 15]  # B, E, I, L, N, P
        selected_columns = st.multiselect(
            "Select columns to display:",
            options=all_columns,
            default=[all_columns[i] for i in default_cols]
        )
        
        if selected_columns:
            # Add calculated distribution percentage
            display_df = filtered_df[selected_columns].copy()
            if df.columns[8] in selected_columns and df.columns[11] in selected_columns:
                electors_idx = selected_columns.index(df.columns[8])
                distributed_idx = selected_columns.index(df.columns[11])
                display_df['Distribution %'] = (display_df.iloc[:, distributed_idx] / display_df.iloc[:, electors_idx] * 100).round(1)
            
            st.dataframe(display_df, use_container_width=True)
        
        # Export option
        if st.button("Export Filtered Data"):
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="filtered_election_data.csv",
                mime="text/csv"
            )
    
    def generate_historical_comparison(self):
        """Compare current data with historical data"""
        st.header("📈 Historical Comparison")
        
        if self.dp.merged_data is None or self.dp.historical_data is None:
            st.warning("Please load both current and historical data")
            return
        
        current_df = self.dp.merged_data
        historical_df = self.dp.historical_data
        
        # Clean part numbers for comparison
        current_df['Part Number Clean'] = self.dp.clean_numeric_data(current_df.iloc[:, 1])
        historical_df['Part Number Clean'] = self.dp.clean_numeric_data(historical_df['Part Number'])
        
        # Merge for comparison
        comparison_data = []
        
        for _, current_row in current_df.iterrows():
            part_num = current_row['Part Number Clean']
            historical_match = historical_df[historical_df['Part Number Clean'] == part_num]
            
            if not historical_match.empty:
                current_dist = current_row.iloc[11]  # Column L
                historical_dist = historical_match['Total EFs Distributed'].values[0]
                
                comparison_data.append({
                    'Part Number': part_num,
                    'Officer': current_row.iloc[4],  # Column E
                    'Current_Distributed': current_dist,
                    'Historical_Distributed': historical_dist,
                    'Change': current_dist - historical_dist,
                    'Change_Percentage': ((current_dist - historical_dist) / historical_dist * 100) if historical_dist > 0 else 0
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribution Changes")
                # Format the display
                display_comparison = comparison_df.copy()
                display_comparison['Change_Percentage'] = display_comparison['Change_Percentage'].round(1)
                st.dataframe(display_comparison, use_container_width=True)
            
            with col2:
                # Progress chart
                if len(comparison_df) > 0:
                    fig = px.scatter(comparison_df, 
                                   x='Historical_Distributed', 
                                   y='Current_Distributed',
                                   hover_data=['Part Number', 'Officer'],
                                   title="Progress: Current vs Historical Distribution",
                                   trendline="lowess")
                    fig.add_shape(type="line", x0=0, y0=0, 
                                x1=max(comparison_df[['Historical_Distributed', 'Current_Distributed']].max()), 
                                y1=max(comparison_df[['Historical_Distributed', 'Current_Distributed']].max()), 
                                line=dict(dash="dash", color="red"))
                    st.plotly_chart(fig, use_container_width=True)

def main():
    st.markdown('<div class="main-header">Election Data Reporting & Analysis System</div>', unsafe_allow_html=True)
    
    # Initialize data processor
    dp = ElectionDataProcessor()
    report_gen = ElectionReportGenerator(dp)
    
    # Sidebar for file uploads
    st.sidebar.header("📁 Data Upload")
    
    sir_file = st.sidebar.file_uploader("Upload Rajasthan SIR Data", type=['xlsx'])
    officer_file = st.sidebar.file_uploader("Upload Officer-wise Data", type=['xlsx'])
    historical_file = st.sidebar.file_uploader("Upload Historical Data (Optional)", type=['xlsx'])
    
    # Load data
    if sir_file:
        dp.load_sir_data(sir_file)
    
    if officer_file:
        dp.load_officer_data(officer_file)
    
    if historical_file:
        dp.load_historical_data(historical_file)
    
    # Data processing section
    st.sidebar.header("🔄 Data Processing")
    
    if st.sidebar.button("Merge and Update Data"):
        if dp.merge_and_update_data():
            st.sidebar.success("Data processing completed!")
    
    # Download processed data
    if dp.merged_data is not None:
        st.sidebar.markdown("---")
        st.sidebar.header("💾 Export Data")
        download_link = dp.get_download_link(dp.merged_data, "Updated_Election_Report.xlsx")
        st.sidebar.markdown(download_link, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "👨‍💼 Officer Reports", 
        "📈 Comparisons", 
        "🔍 Custom Reports", 
        "📈 Historical Analysis"
    ])
    
    with tab1:
        report_gen.show_data_overview()
        
        if dp.merged_data is not None:
            st.subheader("Data Preview")
            # Show first 10 rows with key columns
            preview_cols = [dp.merged_data.columns[i] for i in [1, 4, 8, 11, 13, 15]]  # B, E, I, L, N, P
            preview_data = dp.merged_data[preview_cols].copy()
            
            # Add distribution percentage for preview
            preview_data['Distribution %'] = (preview_data.iloc[:, 3] / preview_data.iloc[:, 2] * 100).round(1)
            
            st.dataframe(preview_data.head(10), use_container_width=True)
    
    with tab2:
        report_gen.generate_officer_wise_report()
    
    with tab3:
        report_gen.generate_comparison_report()
    
    with tab4:
        report_gen.generate_custom_report()
    
    with tab5:
        report_gen.generate_historical_comparison()

if __name__ == "__main__":
    main()