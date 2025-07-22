import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import requests
from bs4 import BeautifulSoup
import os
import pyperclip
import difflib
import re
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Stock SMA & Financial Screener with V20 Strategy", layout="wide")
st.title("📈 Stock SMA & Financial Screener with V20 Strategy")
st.markdown("""
Upload a CSV file with stock symbols and type (v200, v40, v40next, ML) to screen stocks based on:
- Simple Moving Averages (20, 50, 200 days)
- V20 strategy (20%+ gain with green candles, with v200 types below 200 SMA)
- Financial performance (QoQ and YoY Net Profit/Actual Income, adjusted and raw, highest historical, and ascending checks)
Stocks with current price within 2% of V20 strategy low price are highlighted in the V20 table.
""")

# Initialize session state
if 'quarterly_data' not in st.session_state:
    st.session_state.quarterly_data = {}
if 'yearly_data' not in st.session_state:
    st.session_state.yearly_data = {}
if 'finance_override' not in st.session_state:
    st.session_state.finance_override = False
if 'enable_same_quarter' not in st.session_state:
    st.session_state.enable_same_quarter = False
if 'force_refresh' not in st.session_state:
    st.session_state.force_refresh = False

# Function to scrape data from Screener.in
@st.cache_data
def scrape_screener_data(ticker):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    urls = [
        f"https://www.screener.in/company/{ticker}/consolidated/",
        f"https://www.screener.in/company/{ticker}/"
    ]
    
    quarterly_data = None
    yearly_data = None
    error = None
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            tables = soup.find_all('table', class_='data-table')
            data_found = False
            
            for table in tables:
                tbody = table.find('tbody')
                if tbody and any(tr.find_all('td') for tr in tbody.find_all('tr')):
                    section = table.find_parent('section')
                    if section:
                        if section.get('id') == 'quarters':
                            quarterly_data = parse_table(table)
                            if (quarterly_data is not None and 
                                not quarterly_data.empty and
                                quarterly_data.shape[1] >= 2 and 
                                quarterly_data.shape[0] >= 1 and 
                                str(quarterly_data.iloc[:, 1].values[0]).strip() != ""):
                                data_found = True
                        elif section.get('id') == 'profit-loss':
                            yearly_data = parse_table(table)
                            if (yearly_data is not None and 
                                not yearly_data.empty and
                                yearly_data.shape[0] >= 1):
                                data_found = True
            
            if data_found:
                break
           
        except requests.RequestException as e:
            error = f"Error fetching {url} for ticker {ticker}: {e}"
        
        if url == urls[1] and quarterly_data is None and yearly_data is None:
            error = f"No financial data tables with valid data found for ticker {ticker}"
    
    return quarterly_data, yearly_data, error

# Function to parse table data into a DataFrame
def parse_table(table):
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]
    if not headers or len(headers) < 2:
        return None
    headers[0] = ''
    rows = []
    for tr in table.find('tbody').find_all('tr'):
        cells = [td.text.strip() for td in tr.find_all('td')]
        if cells and len(cells) == len(headers):
            rows.append(cells)
    df = pd.DataFrame(rows, columns=headers) if rows else None
    return df

# Function to load data from CSV files
def load_from_csv(ticker):
    output_dir = 'output_tables'
    quarterly_file = os.path.join(output_dir, f'{ticker}_quarterly_results.csv')
    yearly_file = os.path.join(output_dir, f'{ticker}_profit_loss.csv')
    
    quarterly_data = None
    yearly_data = None
    error = None
    
    try:
        if os.path.exists(quarterly_file):
            quarterly_data = pd.read_csv(quarterly_file)
            if quarterly_data.empty or quarterly_data.shape[1] < 2:
                quarterly_data = None
                error = f"Invalid or empty quarterly CSV data for {ticker}"
            else:
                if quarterly_data.columns[0] != '':
                    quarterly_data.columns = [''] + quarterly_data.columns[1:].tolist()
        else:
            error = f"Quarterly CSV file not found for {ticker}"
    except Exception as e:
        error = f"Error loading quarterly CSV for {ticker}: {e}"
    
    try:
        if os.path.exists(yearly_file):
            yearly_data = pd.read_csv(yearly_file)
            if yearly_data.empty or yearly_data.shape[1] < 2:
                yearly_data = None
                error = f"Invalid or empty yearly CSV data for {ticker}"
            else:
                if yearly_data.columns[0] != '':
                    yearly_data.columns = [''] + yearly_data.columns[1:].tolist()
        else:
            error = f"Yearly CSV file not found for {ticker}" if not error else error
    except Exception as e:
        error = f"Error loading yearly CSV for {ticker}: {e}" if not error else error
    
    return quarterly_data, yearly_data, error

# Function to determine if company is finance or non-finance
def is_finance_company(quarterly_data):
    if quarterly_data is None or quarterly_data.empty:
        return False
    return "Financing Profit" in quarterly_data.iloc[:, 0].values

# Function to find row by partial, case-insensitive, or fuzzy match
def find_row(data, row_name, threshold=0.8):
    possible_names = [row_name, row_name.replace(" ", ""), "Consolidated " + row_name, row_name + " (Consolidated)"]
    for name in possible_names:
        for index in data.index:
            if name.lower() in index.lower():
                return index
    matches = difflib.get_close_matches(row_name.lower(), [idx.lower() for idx in data.index], n=1, cutoff=threshold)
    return matches[0] if matches else None

# Function to clean numeric data
def clean_numeric(series):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[0]
    elif not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    series = series.astype(str).str.replace(',', '', regex=False).str.replace('[^0-9.-]', '', regex=True)
    return pd.to_numeric(series, errors='coerce').fillna(0)

# Function to adjust Net Profit and Actual Income
def adjust_non_finance(data, is_finance):
    if is_finance:
        net_profit_row = find_row(data, "Net Profit") or find_row(data, "Profit after tax")
        actual_income_row = find_row(data, "Actual Income") or net_profit_row
        net_profit = clean_numeric(data.loc[net_profit_row].iloc[1:]) if net_profit_row and net_profit_row in data.index else None
        actual_income = clean_numeric(data.loc[actual_income_row].iloc[1:]) if actual_income_row and actual_income_row in data.index else None
        return net_profit, actual_income

    net_profit_row = find_row(data, "Net Profit") or find_row(data, "Profit after tax")
    actual_income_row = find_row(data, "Actual Income") or net_profit_row
    other_income_row = find_row(data, "Other Income")

    net_profit = clean_numeric(data.loc[net_profit_row].iloc[1:]) if net_profit_row and net_profit_row in data.index else None
    actual_income = clean_numeric(data.loc[actual_income_row].iloc[1:]) if actual_income_row and actual_income_row in data.index else net_profit
    other_income = clean_numeric(data.loc[other_income_row].iloc[1:]) if other_income_row and other_income_row in data.index else pd.Series(0, index=net_profit.index if net_profit is not None else [])

    adjusted_net_profit = net_profit - other_income if net_profit is not None else None
    adjusted_actual_income = actual_income - other_income if actual_income is not None else adjusted_net_profit

    return adjusted_net_profit, adjusted_actual_income

# Function to check if data is in ascending order
def is_ascending(series):
    if series is None or series.empty:
        return False
    return all(series[i] <= series[i + 1] for i in range(len(series) - 1))

# Function to extract quarter and year from column name
def extract_quarter_year(column):
    patterns = [
        r'(\w+)\s+(\d{4})',  # e.g., "Mar 2025"
        r'(\w+)-(\d{2})',    # e.g., "Mar-25"
        r'(\w+)\s*\'(\d{2})' # e.g., "Mar'25"
    ]
    column = column.strip()
    for pattern in patterns:
        match = re.match(pattern, column)
        if match:
            quarter, year = match.groups()
            year = int(year) if len(year) == 4 else int("20" + year)
            return quarter, year
    return None, None

# Function to check same-quarter comparison
def check_same_quarter_comparison(data, enable_same_quarter):
    if not enable_same_quarter or data is None or data.empty:
        return {
            'Same Quarter Net Profit (Adjusted)': 'N/A',
            'Same Quarter Net Profit (Raw)': 'N/A'
        }
    
    if '' not in data.columns:
        return {
            'Same Quarter Net Profit (Adjusted)': 'N/A',
            'Same Quarter Net Profit (Raw)': 'N/A'
        }
    
    data = data.set_index('')
    adjusted_net_profit, _ = adjust_non_finance(data, is_finance_company(data))
    raw_net_profit = clean_numeric(data.loc[find_row(data, "Net Profit") or find_row(data, "Profit after tax")].iloc[1:]) if find_row(data, "Net Profit") or find_row(data, "Profit after tax") else None

    results = {
        'Same Quarter Net Profit (Adjusted)': 'N/A',
        'Same Quarter Net Profit (Raw)': 'N/A'
    }

    if adjusted_net_profit is None or raw_net_profit is None:
        return results

    try:
        latest_column = data.columns[-1]
        latest_quarter, latest_year = extract_quarter_year(latest_column)
        if latest_quarter is None or latest_year is None:
            return results

        prev_year_column = None
        for col in data.columns[:-1]:
            quarter, year = extract_quarter_year(col)
            if quarter == latest_quarter and year == latest_year - 1:
                prev_year_column = col
                break

        if prev_year_column:
            latest_adj_np = adjusted_net_profit[latest_column]
            prev_adj_np = adjusted_net_profit[prev_year_column]
            latest_raw_np = raw_net_profit[latest_column]
            prev_raw_np = raw_net_profit[prev_year_column]

            results['Same Quarter Net Profit (Adjusted)'] = 'PASS' if latest_adj_np >= prev_adj_np else 'FAIL'
            results['Same Quarter Net Profit (Raw)'] = 'PASS' if latest_raw_np >= prev_raw_np else 'FAIL'

    except Exception:
        pass

    return results

# Function to check if latest quarter/year is highest and in ascending order
def check_highest_historical(data, is_quarterly, is_finance):
    if data is None or data.empty:
        return {}
    
    if '' not in data.columns:
        return {
            'Net Profit (Adjusted)': 'N/A',
            'Actual Income (Adjusted)': 'N/A',
            'Raw Net Profit (Raw)': 'N/A',
            'Raw Actual Income (Raw)': 'N/A',
            'Net Profit (Adjusted) Ascending': 'N/A',
            'Actual Income (Adjusted) Ascending': 'N/A',
            'Raw Net Profit (Raw) Ascending': 'N/A',
            'Raw Actual Income (Raw) Ascending': 'N/A'
        }
    
    data = data.set_index('')
    adjusted_net_profit, adjusted_actual_income = adjust_non_finance(data, is_finance)
    raw_net_profit = clean_numeric(data.loc[find_row(data, "Net Profit") or find_row(data, "Profit after tax")].iloc[1:]) if find_row(data, "Net Profit") or find_row(data, "Profit after tax") else None
    raw_actual_income = clean_numeric(data.loc[find_row(data, "Actual Income") or find_row(data, "Net Profit") or find_row(data, "Profit after tax")].iloc[1:]) if find_row(data, "Actual Income") or find_row(data, "Net Profit") or find_row(data, "Profit after tax") else None

    results = {}
    for metric, values, prefix in [
        ("Net Profit (Adjusted)", adjusted_net_profit, ""),
        ("Actual Income (Adjusted)", adjusted_actual_income, ""),
        ("Net Profit (Raw)", raw_net_profit, "Raw "),
        ("Actual Income (Raw)", raw_actual_income, "Raw ")
    ]:
        if values is None or values.empty:
            results[f"{prefix}{metric}"] = "N/A"
            results[f"{prefix}{metric} Ascending"] = "N/A"
            continue
        try:
            latest_value = values.iloc[-1]
            historical_values = values.iloc[:-1]
            if historical_values.empty:
                results[f"{prefix}{metric}"] = "N/A"
                results[f"{prefix}{metric} Ascending"] = "N/A"
            else:
                is_highest = latest_value >= historical_values.max()
                results[f"{prefix}{metric}"] = "PASS" if is_highest else "FAIL"
                is_asc = is_ascending(values)
                results[f"{prefix}{metric} Ascending"] = "PASS" if is_asc else "FAIL"
        except Exception:
            results[f"{prefix}{metric}"] = "N/A"
            results[f"{prefix}{metric} Ascending"] = "N/A"
    return results

# Function to calculate SMAs, V20 strategy, and financial metrics
def calculate_sma_and_screen(symbols_df, start_date, end_date):
    bullish_stocks = []
    bearish_stocks = []
    v20_stocks = []
    total_stocks = len(symbols_df)
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    processed_stocks = 0
    
    for index, row in symbols_df.iterrows():
        symbol = row['Symbol']
        stock_type = row.get('Type', '').lower()
        ticker = symbol.replace('.NS', '')  # For Screener.in scraping
        
        try:
            # Append .NS if no exchange suffix is present
            if not symbol.endswith('.NS'):
                symbol = symbol + '.NS'
            
            # Download historical stock data
            stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if stock_data.empty:
                st.warning(f"No stock data found for {symbol}. Skipping...")
                processed_stocks += 1
                progress_bar.progress(min(processed_stocks / total_stocks, 1.0))
                continue
                
            # Ensure enough data for 200 SMA
            if len(stock_data) < 200:
                st.warning(f"Insufficient data points ({len(stock_data)}) for {symbol} to calculate 200 SMA. Skipping...")
                processed_stocks += 1
                progress_bar.progress(min(processed_stocks / total_stocks, 1.0))
                continue
                
            # Handle multi-level columns if present
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)
            
            # Verify 'Close' column exists
            if 'Close' not in stock_data.columns:
                st.warning(f"No 'Close' column found for {symbol}. Columns: {list(stock_data.columns)}. Skipping...")
                processed_stocks += 1
                progress_bar.progress(min(processed_stocks / total_stocks, 1.0))
                continue
                
            # Calculate SMAs
            stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
            stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
            stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
            
            # Get the latest values
            latest_data = stock_data.iloc[-1]
            
            # Extract scalar values
            try:
                close_price = float(latest_data['Close']) if not isinstance(latest_data['Close'], pd.Series) else float(latest_data['Close'].iloc[0])
                sma_20 = float(latest_data['SMA_20']) if not isinstance(latest_data['SMA_20'], pd.Series) else float(latest_data['SMA_20'].iloc[0])
                sma_50 = float(latest_data['SMA_50']) if not isinstance(latest_data['SMA_50'], pd.Series) else float(latest_data['SMA_50'].iloc[0])
                sma_200 = float(latest_data['SMA_200']) if not isinstance(latest_data['SMA_200'], pd.Series) else float(latest_data['SMA_200'].iloc[0])
            except (KeyError, IndexError, ValueError) as e:
                st.warning(f"Error extracting scalar values for {symbol}: {str(e)}. Columns: {list(stock_data.columns)}. Skipping...")
                processed_stocks += 1
                progress_bar.progress(min(processed_stocks / total_stocks, 1.0))
                continue
                
            # Check if SMAs are valid (not NaN)
            if pd.isna(close_price) or pd.isna(sma_20) or pd.isna(sma_50) or pd.isna(sma_200):
                st.warning(f"Missing values for {symbol} (Close: {close_price}, SMA_20: {sma_20}, SMA_50: {sma_50}, SMA_200: {sma_200}). Skipping...")
                processed_stocks += 1
                progress_bar.progress(min(processed_stocks / total_stocks, 1.0))
                continue
                
            # Financial data processing
            quarterly_data, yearly_data, error = None, None, None
            if not st.session_state.force_refresh:
                quarterly_data, yearly_data, error = load_from_csv(ticker)
            else:
                quarterly_data, yearly_data, error = None, None, None

            if error or quarterly_data is None or yearly_data is None:
                quarterly_data, yearly_data, error = scrape_screener_data(ticker)
                if not error:
                    save_to_csv(quarterly_data, yearly_data, ticker)
                    st.session_state.quarterly_data[ticker] = quarterly_data
                    st.session_state.yearly_data[ticker] = yearly_data
                else:
                    st.warning(f"Financial data error for {ticker}: {error}")

            is_finance = st.session_state.finance_override or is_finance_company(quarterly_data)
            qoq_results = check_highest_historical(quarterly_data, True, is_finance)
            same_quarter_results = check_same_quarter_comparison(quarterly_data, st.session_state.enable_same_quarter)
            yoy_results = check_highest_historical(yearly_data, False, is_finance)

            # Base stock data
            stock_info = {
                'Symbol': symbol,
                'Type': stock_type,
                'Close Price': round(close_price, 2),
                '20 SMA': round(sma_20, 2),
                '50 SMA': round(sma_50, 2),
                '200 SMA': round(sma_200, 2),
                'Company Type': 'Finance' if is_finance else 'Non-Finance',
                'QOQ Net Profit (Adjusted)': qoq_results.get('Net Profit (Adjusted)', 'N/A'),
                'QOQ Actual Income (Adjusted)': qoq_results.get('Actual Income (Adjusted)', 'N/A'),
                'QOQ Net Profit (Raw)': qoq_results.get('Raw Net Profit (Raw)', 'N/A'),
                'QOQ Actual Income (Raw)': qoq_results.get('Raw Actual Income (Raw)', 'N/A'),
                'QOQ Net Profit Ascending (Adjusted)': qoq_results.get('Net Profit (Adjusted) Ascending', 'N/A'),
                'QOQ Actual Income Ascending (Adjusted)': qoq_results.get('Actual Income (Adjusted) Ascending', 'N/A'),
                'QOQ Net Profit Ascending (Raw)': qoq_results.get('Raw Net Profit (Raw) Ascending', 'N/A'),
                'QOQ Actual Income Ascending (Raw)': qoq_results.get('Raw Actual Income (Raw) Ascending', 'N/A'),
                'Same Quarter Net Profit (Adjusted)': same_quarter_results.get('Same Quarter Net Profit (Adjusted)', 'N/A'),
                'Same Quarter Net Profit (Raw)': same_quarter_results.get('Same Quarter Net Profit (Raw)', 'N/A'),
                'YOY Net Profit (Adjusted)': yoy_results.get('Net Profit (Adjusted)', 'N/A'),
                'YOY Actual Income (Adjusted)': yoy_results.get('Actual Income (Adjusted)', 'N/A'),
                'YOY Net Profit (Raw)': yoy_results.get('Raw Net Profit (Raw)', 'N/A'),
                'YOY Actual Income (Raw)': yoy_results.get('Raw Actual Income (Raw)', 'N/A'),
                'YOY Net Profit Ascending (Adjusted)': yoy_results.get('Net Profit (Adjusted) Ascending', 'N/A'),
                'YOY Actual Income Ascending (Adjusted)': yoy_results.get('Actual Income (Adjusted) Ascending', 'N/A'),
                'YOY Net Profit Ascending (Raw)': yoy_results.get('Raw Net Profit (Raw) Ascending', 'N/A'),
                'YOY Actual Income Ascending (Raw)': yoy_results.get('Raw Actual Income (Raw) Ascending', 'N/A'),
                'Error': error or 'None'
            }
            
            # Screen for bullish trend: 200 SMA > 50 SMA > 20 SMA
            if sma_200 > sma_50 and sma_50 > sma_20:
                bullish_stocks.append(stock_info)
                
            # Screen for bearish trend: 200 SMA < 50 SMA < 20 SMA
            elif sma_200 < sma_50 and sma_50 < sma_20:
                bearish_stocks.append(stock_info)
                
            # V20 Strategy: Check for 20%+ gain with green candles or single green candle
            if stock_type in ['v200', 'v40', 'v40next']:
                momentum_period, momentum_gain, start_date_momentum, end_date_momentum, v20_low_price = check_v20_strategy(stock_data, stock_type, sma_200)
                if momentum_gain >= 20:
                    near_v20_low = abs(close_price - v20_low_price) / v20_low_price <= 0.02 if v20_low_price else False
                    v20_info = stock_info.copy()
                    v20_info.update({
                        'Momentum Gain (%)': round(momentum_gain, 2),
                        'Momentum Duration': f"{start_date_momentum} to {end_date_momentum}",
                        'V20 Low Price': round(v20_low_price, 2) if v20_low_price else None,
                        'Near V20 Low (Within 2%)': 'Yes' if near_v20_low else 'No'
                    })
                    v20_stocks.append(v20_info)
                
            processed_stocks += 1
            progress_bar.progress(min(processed_stocks / total_stocks, 1.0))
                
        except Exception as e:
            st.warning(f"Error processing {symbol}: {str(e)}")
            processed_stocks += 1
            progress_bar.progress(min(processed_stocks / total_stocks, 1.0))
    
    return pd.DataFrame(bullish_stocks), pd.DataFrame(bearish_stocks), pd.DataFrame(v20_stocks)

# Function to check V20 strategy
def check_v20_strategy(stock_data, stock_type, latest_sma_200):
    if len(stock_data) < 2:
        return None, 0, None, None, None
    
    stock_data['Gain'] = ((stock_data['Close'] - stock_data['Open']) / stock_data['Open']) * 100
    stock_data['Is_Green'] = stock_data['Close'] > stock_data['Open']
    
    for i in range(len(stock_data)):
        if stock_data['Gain'].iloc[i] >= 20 and stock_data['Is_Green'].iloc[i]:
            candle_date = stock_data.index[i].strftime('%Y-%m-%d')
            if stock_type == 'v200':
                if stock_data['High'].iloc[i] < stock_data['SMA_200'].iloc[i]:
                    return "Single Candle", stock_data['Gain'].iloc[i], candle_date, candle_date, stock_data['Open'].iloc[i]
            else:
                return "Single Candle", stock_data['Gain'].iloc[i], candle_date, candle_date, stock_data['Open'].iloc[i]
    
    max_gain = 0
    start_idx = 0
    current_start = 0
    cumulative_gain = 0
    is_green_sequence = True
    
    for i in range(1, len(stock_data)):
        daily_gain = ((stock_data['Close'].iloc[i] - stock_data['Open'].iloc[i]) / stock_data['Open'].iloc[i]) * 100
        if stock_data['Is_Green'].iloc[i]:
            if is_green_sequence:
                cumulative_gain = ((stock_data['Close'].iloc[i] - stock_data['Open'].iloc[current_start]) / stock_data['Open'].iloc[current_start]) * 100
                if cumulative_gain >= 20:
                    if stock_type == 'v200':
                        if stock_data['High'].iloc[current_start:i+1].max() < stock_data['SMA_200'].iloc[current_start:i+1].min():
                            max_gain = cumulative_gain
                            start_idx = current_start
                    else:
                        max_gain = cumulative_gain
                        start_idx = current_start
        else:
            is_green_sequence = False
            cumulative_gain = 0
            current_start = i
        if not is_green_sequence:
            is_green_sequence = stock_data['Is_Green'].iloc[i]
            if is_green_sequence:
                current_start = i
                cumulative_gain = 0
    
    if max_gain >= 20:
        start_date = stock_data.index[start_idx].strftime('%Y-%m-%d')
        end_date = stock_data.index[-1].strftime('%Y-%m-%d')
        return "Continuous Green Candles", max_gain, start_date, end_date, stock_data['Low'].iloc[start_idx]
    
    return None, 0, None, None, None

# Function to save tables to CSV
def save_to_csv(quarterly_data, yearly_data, ticker):
    output_dir = 'output_tables'
    os.makedirs(output_dir, exist_ok=True)

    if quarterly_data is not None and not quarterly_data.empty:
        quarterly_data.to_csv(os.path.join(output_dir, f'{ticker}_quarterly_results.csv'), index=False)

    if yearly_data is not None and not yearly_data.empty:
        yearly_data.to_csv(os.path.join(output_dir, f'{ticker}_profit_loss.csv'), index=False)

# Function to copy DataFrame to clipboard
def copy_to_clipboard(df, table_name):
    try:
        df_string = df.to_csv(sep='\t', index=False)
        pyperclip.copy(df_string)
        st.success(f"{table_name} copied to clipboard! Paste into Excel or Google Sheets.")
    except Exception as e:
        st.error(f"Error copying {table_name} to clipboard: {e}. Please try again or check clipboard permissions.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with stock symbols and type (v200, v40, v40next, ML)", type=["csv"])
st.checkbox("Override: Treat as Finance Company", key="finance_override")
st.checkbox("Enable Same Quarter Year-over-Year Net Profit Comparison (e.g., Mar 2025 vs. Mar 2024)", key="enable_same_quarter")
st.checkbox("Force Refresh Financial Data (re-scrape all tickers instead of using CSVs)", key="force_refresh")

# Load stock symbols from CSV
if uploaded_file is not None:
    symbols_df = pd.read_csv(uploaded_file)
    if 'Symbol' not in symbols_df.columns or 'Type' not in symbols_df.columns:
        st.error("CSV must contain 'Symbol' and 'Type' columns.")
        symbols_df = pd.DataFrame()
else:
    st.error("Please upload a CSV file with stock symbols and types.")
    symbols_df = pd.DataFrame()

# Date range selection
st.sidebar.subheader("Date Range for SMA Calculation (Minimum 1.5 Years)")
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=548)
start_date = st.sidebar.date_input("Start Date", value=start_date, min_value=end_date - datetime.timedelta(days=548))
end_date = st.sidebar.date_input("End Date", value=end_date)

# Screen stocks when button is clicked
if st.button("Screen Stocks"):
    if not symbols_df.empty:
        with st.spinner("Fetching stock and financial data, calculating SMAs and V20 strategy..."):
            bullish_df, bearish_df, v20_df = calculate_sma_and_screen(symbols_df, start_date, end_date)
        
        # Display Bullish Stocks
        st.subheader("Bullish Stocks (200 SMA > 50 SMA > 20 SMA)")
        if not bullish_df.empty:
            st.dataframe(bullish_df, use_container_width=True)
            csv = bullish_df.to_csv(index=False)
            st.download_button(
                label="Download Bullish Stocks as CSV",
                data=csv,
                file_name="bullish_stocks.csv",
                mime="text/csv"
            )
            if st.button("Copy Bullish Stocks to Clipboard"):
                copy_to_clipboard(bullish_df, "Bullish Stocks")
        else:
            st.warning("No stocks match the bullish criteria.")
        
        # Display Bearish Stocks
        st.subheader("Bearish Stocks (200 SMA < 50 SMA < 20 SMA)")
        if not bearish_df.empty:
            st.dataframe(bearish_df, use_container_width=True)
            csv = bearish_df.to_csv(index=False)
            st.download_button(
                label="Download Bearish Stocks as CSV",
                data=csv,
                file_name="bearish_stocks.csv",
                mime="text/csv"
            )
            if st.button("Copy Bearish Stocks to Clipboard"):
                copy_to_clipboard(bearish_df, "Bearish Stocks")
        else:
            st.warning("No stocks match the bearish criteria.")
        
        # Display V20 Strategy Stocks
        st.subheader("V20 Strategy Stocks (20%+ Gain with Green Candles)")
        if not v20_df.empty:
            def highlight_near_v20_low(row):
                if row['Near V20 Low (Within 2%)'] == 'Yes':
                    return ['background-color: #90EE90'] * len(row)
                return [''] * len(row)
            
            st.dataframe(v20_df.style.apply(highlight_near_v20_low, axis=1), use_container_width=True)
            csv = v20_df.to_csv(index=False)
            st.download_button(
                label="Download V20 Strategy Stocks as CSV",
                data=csv,
                file_name="v20_stocks.csv",
                mime="text/csv"
            )
            if st.button("Copy V20 Strategy Stocks to Clipboard"):
                copy_to_clipboard(v20_df, "V20 Strategy Stocks")
        else:
            st.warning("No stocks match the V20 strategy criteria.")
        
        # Display Tickers with All PASS Criteria
        pass_columns = [
            'QOQ Net Profit (Adjusted)', 'QOQ Actual Income (Adjusted)', 'QOQ Net Profit (Raw)', 'QOQ Actual Income (Raw)',
            'QOQ Net Profit Ascending (Adjusted)', 'QOQ Actual Income Ascending (Adjusted)', 'QOQ Net Profit Ascending (Raw)', 'QOQ Actual Income Ascending (Raw)',
            'YOY Net Profit (Adjusted)', 'YOY Actual Income (Adjusted)', 'YOY Net Profit (Raw)', 'YOY Actual Income (Raw)',
            'YOY Net Profit Ascending (Adjusted)', 'YOY Actual Income Ascending (Adjusted)', 'YOY Net Profit Ascending (Raw)', 'YOY Actual Income Ascending (Raw)'
        ]
        if st.session_state.enable_same_quarter:
            pass_columns.extend(['Same Quarter Net Profit (Adjusted)', 'Same Quarter Net Profit (Raw)'])
        
        for df, title in [(bullish_df, "Bullish Stocks with All PASS Criteria"), (bearish_df, "Bearish Stocks with All PASS Criteria"), (v20_df, "V20 Stocks with All PASS Criteria")]:
            if not df.empty:
                pass_df = df[df[pass_columns].eq('PASS').all(axis=1)]
                if not pass_df.empty:
                    st.subheader(title)
                    display_columns = ['Symbol', 'Type', 'Close Price', '20 SMA', '50 SMA', '200 SMA', 'Company Type'] + pass_columns
                    if title == "V20 Stocks with All PASS Criteria":
                        display_columns.extend(['Momentum Gain (%)', 'Momentum Duration', 'V20 Low Price', 'Near V20 Low (Within 2%)'])
                    st.dataframe(pass_df[display_columns].style.apply(highlight_near_v20_low, axis=1) if title == "V20 Stocks with All PASS Criteria" else pass_df[display_columns], use_container_width=True)
                    csv = pass_df[display_columns].to_csv(index=False)
                    st.download_button(
                        label=f"Download {title} as CSV",
                        data=csv,
                        file_name=f"{title.lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                    if st.button(f"Copy {title} to Clipboard"):
                        copy_to_clipboard(pass_df[display_columns], title)
                else:
                    st.info(f"No {title.lower()} meet all PASS criteria.")
    else:
        st.error("No valid stock symbols provided. Please upload a valid CSV file.")