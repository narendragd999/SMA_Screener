import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Stock SMA Screener", layout="wide")
st.title("📈 Stock SMA Screener")
st.markdown("Upload a CSV file with stock symbols or use the default list to screen stocks based on Simple Moving Averages (20, 50, 200 days).")

# Function to calculate SMAs and screen stocks
def calculate_sma_and_screen(symbols, start_date, end_date):
    bullish_stocks = []
    bearish_stocks = []
    
    for symbol in symbols:
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
                
            # Debug output for problematic stocks
            # if symbol in ['BEL.NS', 'ITC.NS', 'ASHOKLEY.NS', 'BOSCHLTD.NS', 'TATAMOTORS.NS', 'HDFCBANK.NS', 'UNITEDSPIR.NS']:
            #     st.write(f"Debug for {symbol}: Close={close_price}, SMA_20={sma_20}, SMA_50={sma_50}, SMA_200={sma_200}, Columns={list(stock_data.columns)}")
                
            # Screen for bullish trend: 200 SMA > 50 SMA > 20 SMA
            if sma_200 > sma_50 and sma_50 > sma_20:
                bullish_stocks.append({
                    'Symbol': symbol,
                    'Close Price': round(close_price, 2),
                    '20 SMA': round(sma_20, 2),
                    '50 SMA': round(sma_50, 2),
                    '200 SMA': round(sma_200, 2)
                })
                
            # Screen for bearish trend: 200 SMA < 50 SMA < 20 SMA
            elif sma_200 < sma_50 and sma_50 < sma_20:
                bearish_stocks.append({
                    'Symbol': symbol,
                    'Close Price': round(close_price, 2),
                    '20 SMA': round(sma_20, 2),
                    '50 SMA': round(sma_50, 2),
                    '200 SMA': round(sma_200, 2)
                })
                
        except Exception as e:
            st.warning(f"Error processing {symbol}: {str(e)}")
    
    return pd.DataFrame(bullish_stocks), pd.DataFrame(bearish_stocks)

# Default stock list from provided Excel (already includes .NS)
default_stocks = [
    'BEL.NS', 'ITC.NS', 'ASHOKLEY.NS', 'BOSCHLTD.NS', 'TATAMOTORS.NS', 'HDFCBANK.NS', 
    'UNITDSPR.NS', 'CDSL.NS', 'ULTRACEMCO.NS', 'SRF.NS', 'QUESS.NS', 'LT.NS', 
    'COALINDIA.NS', 'HAVELLS.NS', 'RELIANCE.NS', 'HINDUNILVR.NS', 'JINDALSTEL.NS', 
    'ABB.NS', 'PVRINOX.NS', 'JSWSTEEL.NS', 'UPL.NS', 'BAJFINANCE.NS', 'PAYTM.NS', 
    'NESTLEIND.NS', 'GUJGASLTD.NS', 'POLYMED.NS', 'APOLLOHOSP.NS', 'BHARATFORG.NS', 
    'CUMMINSIND.NS', 'LICI.NS', 'NETWEB.NS', 'TCS.NS', 'HCLTECH.NS', 'INDHOTEL.NS', 
    'NETWORK18.NS', 'ADANIENT.NS', 'HINDALCO.NS', 'SHREECEM.NS', 'NAUKRI.NS', 
    'ADANIGREEN.NS', 'JKPAPER.NS', 'DABUR.NS', 'SUNPHARMA.NS', 'NTPC.NS', 'DBCORP.NS', 
    'DLF.NS', 'TRENT.NS', 'HFCL.NS', 'BHARTIARTL.NS', 'PAGEIND.NS', 'ADANIPORTS.NS', 
    'INDIGO.NS', 'HAL.NS', 'GODREJCP.NS', 'ESCORTS.NS', 'MOTHERSUMI.NS', 'MARUTI.NS', 
    'ICICIBANK.NS', 'VBL.NS', 'BSE.NS', 'AMBUJACEM.NS', 'DEEPAKNTR.NS', 'VSTIND.NS', 
    'TEAMLEASE.NS', 'IOC.NS', 'CROMPTON.NS', 'SIEMENS.NS', 'TATATECH.NS', 'SAREGAMA.NS', 
    'TATASTEEL.NS', 'COROMANDEL.NS', 'JIOFIN.NS', 'ZOMATO.NS', 'BRITANNIA.NS', 'IGL.NS', 
    'MAXHEALTH.NS', 'COLPAL.NS', 'AIAENG.NS', 'SKFINDIA.NS', 'ICICIGI.NS', 'TEJASNET.NS', 
    'INFY.NS', 'WIPRO.NS', 'JUBLFOOD.NS', 'SUNTV.NS', 'MMTC.NS', 'VEDL.NS', 'ONGC.NS', 
    'ACC.NS', 'JUSTDIAL.NS', 'WSTCSTPAPR.NS', 'DRREDDY.NS', 'ADANIPOWER.NS', 
    'NAVNETEDUL.NS', 'GODREJPROP.NS', 'DMART.NS', 'IDEA.NS', 'KPRMILL.NS', 'GMRINFRA.NS', 
    'CONCOR.NS'
]

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with stock symbols", type=["csv"])

# Load stock symbols
symbols = default_stocks
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Assuming the CSV has a column named 'Symbol' or similar
    if 'Symbol' in df.columns:
        symbols = df['Symbol'].dropna().tolist()
    else:
        st.error("CSV must contain a 'Symbol' column.")
        symbols = default_stocks

# Date range selection
st.sidebar.subheader("Date Range for SMA Calculation")
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365)  # 1 year of data
start_date = st.sidebar.date_input("Start Date", value=start_date)
end_date = st.sidebar.date_input("End Date", value=end_date)

# Screen stocks when button is clicked
if st.button("Screen Stocks"):
    if symbols:
        with st.spinner("Fetching data and calculating SMAs..."):
            bullish_df, bearish_df = calculate_sma_and_screen(symbols, start_date, end_date)
        
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
    else:
        st.error("No valid stock symbols provided.")