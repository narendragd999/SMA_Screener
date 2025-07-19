import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Stock SMA Screener with V20 Strategy", layout="wide")
st.title("📈 Stock SMA Screener with V20 Strategy")
st.markdown("Upload a CSV file with stock symbols and type (v200, v40, v40next, ML) to screen stocks based on Simple Moving Averages (20, 50, 200 days) and V20 strategy.")

# Function to calculate SMAs and screen stocks
def calculate_sma_and_screen(symbols_df, start_date, end_date):
    bullish_stocks = []
    bearish_stocks = []
    v20_stocks = []
    
    for index, row in symbols_df.iterrows():
        symbol = row['Symbol']
        stock_type = row.get('Type', '').lower()  # Get company type, default to empty string
        
        try:
            # Append .NS if no exchange suffix is present
            if not symbol.endswith('.NS'):
                symbol = symbol + '.NS'
            
            # Download historical data
            stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if stock_data.empty:
                st.warning(f"No data found for {symbol}. Skipping...")
                continue
                
            # Ensure enough data for 200 SMA
            if len(stock_data) < 200:
                st.warning(f"Insufficient data points ({len(stock_data)}) for {symbol} to calculate 200 SMA. Skipping...")
                continue
                
            # Handle multi-level columns if present
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)
            
            # Verify 'Close' column exists
            if 'Close' not in stock_data.columns:
                st.warning(f"No 'Close' column found for {symbol}. Columns: {list(stock_data.columns)}. Skipping...")
                continue
                
            # Calculate SMAs
            stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
            stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
            stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
            
            # Get the latest values
            latest_data = stock_data.iloc[-1]
            
            # Extract scalar values, handling potential Series
            try:
                close_price = float(latest_data['Close']) if not isinstance(latest_data['Close'], pd.Series) else float(latest_data['Close'].iloc[0])
                sma_20 = float(latest_data['SMA_20']) if not isinstance(latest_data['SMA_20'], pd.Series) else float(latest_data['SMA_20'].iloc[0])
                sma_50 = float(latest_data['SMA_50']) if not isinstance(latest_data['SMA_50'], pd.Series) else float(latest_data['SMA_50'].iloc[0])
                sma_200 = float(latest_data['SMA_200']) if not isinstance(latest_data['SMA_200'], pd.Series) else float(latest_data['SMA_200'].iloc[0])
            except (KeyError, IndexError, ValueError) as e:
                st.warning(f"Error extracting scalar values for {symbol}: {str(e)}. Columns: {list(stock_data.columns)}. Skipping...")
                continue
                
            # Check if SMAs are valid (not NaN)
            if pd.isna(close_price) or pd.isna(sma_20) or pd.isna(sma_50) or pd.isna(sma_200):
                st.warning(f"Missing values for {symbol} (Close: {close_price}, SMA_20: {sma_20}, SMA_50: {sma_50}, SMA_200: {sma_200}). Skipping...")
                continue
                
            # Screen for bullish trend: 200 SMA > 50 SMA > 20 SMA
            if sma_200 > sma_50 and sma_50 > sma_20:
                bullish_stocks.append({
                    'Symbol': symbol,
                    'Type': stock_type,
                    'Close Price': round(close_price, 2),
                    '20 SMA': round(sma_20, 2),
                    '50 SMA': round(sma_50, 2),
                    '200 SMA': round(sma_200, 2)
                })
                
            # Screen for bearish trend: 200 SMA < 50 SMA < 20 SMA
            elif sma_200 < sma_50 and sma_50 < sma_20:
                bearish_stocks.append({
                    'Symbol': symbol,
                    'Type': stock_type,
                    'Close Price': round(close_price, 2),
                    '20 SMA': round(sma_20, 2),
                    '50 SMA': round(sma_50, 2),
                    '200 SMA': round(sma_200, 2)
                })
                
            # V20 Strategy: Check for 20%+ gain with green candles or single green candle
            # Apply V20 strategy only for v200, v40, v40next types
            if stock_type in ['v200', 'v40', 'v40next']:
                momentum_period, momentum_gain, start_date_momentum, end_date_momentum = check_v20_strategy(stock_data, stock_type, sma_200)
                if momentum_gain >= 20:
                    v20_stocks.append({
                        'Symbol': symbol,
                        'Type': stock_type,
                        'Momentum Gain (%)': round(momentum_gain, 2),
                        'Momentum Duration': f"{start_date_momentum} to {end_date_momentum}",
                        'Close Price': round(close_price, 2),
                        '20 SMA': round(sma_20, 2),
                        '50 SMA': round(sma_50, 2),
                        '200 SMA': round(sma_200, 2)
                    })
                
        except Exception as e:
            st.warning(f"Error processing {symbol}: {str(e)}")
    
    return pd.DataFrame(bullish_stocks), pd.DataFrame(bearish_stocks), pd.DataFrame(v20_stocks)

# Function to check V20 strategy
def check_v20_strategy(stock_data, stock_type, latest_sma_200):
    # Ensure enough data
    if len(stock_data) < 2:
        return None, 0, None, None
    
    # Check for single candle with 20%+ gain
    stock_data['Gain'] = ((stock_data['Close'] - stock_data['Open']) / stock_data['Open']) * 100
    stock_data['Is_Green'] = stock_data['Close'] > stock_data['Open']
    
    # Single candle check
    for i in range(len(stock_data)):
        if stock_data['Gain'].iloc[i] >= 20 and stock_data['Is_Green'].iloc[i]:
            candle_date = stock_data.index[i].strftime('%Y-%m-%d')
            # For v200 type, ensure candle range is below 200 SMA
            if stock_type == 'v200':
                if stock_data['High'].iloc[i] < stock_data['SMA_200'].iloc[i]:
                    return "Single Candle", stock_data['Gain'].iloc[i], candle_date, candle_date
            else:  # v40 and v40next can be above or below 200 SMA
                return "Single Candle", stock_data['Gain'].iloc[i], candle_date, candle_date
    
    # Check for continuous green candles with cumulative 20%+ gain
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
                    # For v200 type, ensure range is below 200 SMA
                    if stock_type == 'v200':
                        if stock_data['High'].iloc[current_start:i+1].max() < stock_data['SMA_200'].iloc[current_start:i+1].min():
                            max_gain = cumulative_gain
                            start_idx = current_start
                    else:  # v40 and v40next can be above or below 200 SMA
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
        return "Continuous Green Candles", max_gain, start_date, end_date
    
    return None, 0, None, None

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with stock symbols and type (v200, v40, v40next, ML)", type=["csv"])

# Load stock symbols from CSV
if uploaded_file is not None:
    symbols_df = pd.read_csv(uploaded_file)
    # Ensure CSV has 'Symbol' and 'Type' columns
    if 'Symbol' not in symbols_df.columns or 'Type' not in symbols_df.columns:
        st.error("CSV must contain 'Symbol' and 'Type' columns.")
        symbols_df = pd.DataFrame()  # Empty DataFrame to prevent further processing
else:
    st.error("Please upload a CSV file with stock symbols and types.")
    symbols_df = pd.DataFrame()  # Empty DataFrame to prevent further processing

# Date range selection
st.sidebar.subheader("Date Range for SMA Calculation (Minimum 1.5 Years)")
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=548)  # 1.5 years (approximately 548 days)
start_date = st.sidebar.date_input("Start Date", value=start_date, min_value=end_date - datetime.timedelta(days=548))
end_date = st.sidebar.date_input("End Date", value=end_date)

# Screen stocks when button is clicked
if st.button("Screen Stocks"):
    if not symbols_df.empty:
        with st.spinner("Fetching data and calculating SMAs and V20 strategy..."):
            bullish_df, bearish_df, v20_df = calculate_sma_and_screen(symbols_df, start_date, end_date)
        
        # Display Bullish Stocks (200 SMA > 50 SMA > 20 SMA)
        st.subheader("Bullish Stocks (200 SMA > 50 SMA > 20 SMA)")
        if not bullish_df.empty:
            st.dataframe(bullish_df)
            # Option to download bullish stocks as CSV
            csv = bullish_df.to_csv(index=False)
            st.download_button(
                label="Download Bullish Stocks as CSV",
                data=csv,
                file_name="bullish_stocks.csv",
                mime="text/csv"
            )
        else:
            st.warning("No stocks match the bullish criteria.")
        
        # Display Bearish Stocks (200 SMA < 50 SMA < 20 SMA)
        st.subheader("Bearish Stocks (200 SMA < 50 SMA < 20 SMA)")
        if not bearish_df.empty:
            st.dataframe(bearish_df)
            # Option to download bearish stocks as CSV
            csv = bearish_df.to_csv(index=False)
            st.download_button(
                label="Download Bearish Stocks as CSV",
                data=csv,
                file_name="bearish_stocks.csv",
                mime="text/csv"
            )
        else:
            st.warning("No stocks match the bearish criteria.")
        
        # Display V20 Strategy Stocks
        st.subheader("V20 Strategy Stocks (20%+ Gain with Green Candles)")
        if not v20_df.empty:
            st.dataframe(v20_df)
            # Option to download V20 stocks as CSV
            csv = v20_df.to_csv(index=False)
            st.download_button(
                label="Download V20 Strategy Stocks as CSV",
                data=csv,
                file_name="v20_stocks.csv",
                mime="text/csv"
            )
        else:
            st.warning("No stocks match the V20 strategy criteria.")
    else:
        st.error("No valid stock symbols provided. Please upload a valid CSV file.")