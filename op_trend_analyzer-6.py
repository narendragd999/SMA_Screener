import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import requests
from bs4 import BeautifulSoup
import re
import difflib
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
import os

# ───────────────────────────────────────────────
# CONFIG & FOLDER SETUP
# ───────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ───────────────────────────────────────────────
# LOAD NSE TICKERS FROM LOCAL CSV FILE
# ───────────────────────────────────────────────
TICKER_CSV_PATH = "NSE_ALL_TICKERS_LIST.csv"  # Place this file in the same directory

@st.cache_data
def load_nse_tickers() -> pd.DataFrame:
    if not os.path.exists(TICKER_CSV_PATH):
        st.error(f"File **{TICKER_CSV_PATH}** not found. Place the CSV in the app directory.")
        st.stop()
    df = pd.read_csv(TICKER_CSV_PATH, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"symbol", "name of company"}
    if not required.issubset(df.columns):
        st.error(f"CSV must contain columns: {required}")
        st.stop()
    df = df.rename(columns={"name of company": "company_name"})
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["company_name"] = df["company_name"].str.strip()
    df["display"] = df["symbol"] + " – " + df["company_name"]
    return df

nse_df = load_nse_tickers()
ticker_options = nse_df["display"].tolist()
ticker_to_symbol = dict(zip(nse_df["display"], nse_df["symbol"]))

# ───────────────────────────────────────────────
# DARK MODE & STYLING
# ───────────────────────────────────────────────
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)
theme = "dark" if dark_mode else "light"

st.markdown(f"""
<style>
    .main {{
        padding: 2rem;
        background-color: {'#0e1117' if dark_mode else '#f8f9fa'};
        color: {'#fafafa' if dark_mode else '#212529'};
    }}
    .stApp {{
        background-color: {'#0e1117' if dark_mode else '#f8f9fa'};
    }}
    .title {{
        font-size: 3rem !important;
        font-weight: 700;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }}
    .subtitle {{
        font-size: 1.9rem;
        text-align: center;
        color: {'#aaa' if dark_mode else '#666'};
        margin-bottom: 2rem;
    }}
    .card {{
        background-color: {'#1e1e2e' if dark_mode else 'white'};
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border: 1px solid {'#333' if dark_mode else '#e0e0e0'};
    }}
    .metric-label {{
        font-size: 1.9rem;
        color: {'#ccc' if dark_mode else '#666'};
        font-weight: 500;
    }}
    .metric-value {{
        font-size: 2.8rem !important;
        font-weight: 700;
        color: #1f77b4;
    }}
    .section-title {{
        font-size: 3rem !important;
        font-weight: 600;
        color: #1f77b4;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1f77b4;
    }}
    .insight-title {{
        font-size: 2.8rem !important;
        font-weight: 600;
        color: #2ca02c;
        margin: 1.5rem 0 1rem 0;
    }}
    .stButton > button {{
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        border: none;
    }}
    .stButton > button:hover {{
        background-color: #155a8a;
    }}
    .stDownloadButton > button {{
        background-color: #2ca02c;
        color: white;
    }}
    .stDownloadButton > button:hover {{
        background-color: #1e7b1e;
    }}
    .footer {{
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: {'#888' if dark_mode else '#aaa'};
        font-size: 1.9rem;
    }}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="OP & Sales Analyzer", layout="wide")
st.markdown("<h1 class='title'>OP & Sales vs Stock Price Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'><strong style='color: #2ca02c;'>P&L saved locally — no re-scrape!</strong></p>", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# SCRAPING & CACHING (FIXED FOR BAYERCROP & OTHERS)
# ───────────────────────────────────────────────
def scrape_screener_data(ticker: str, debug: bool = False) -> pd.DataFrame | None:
    ticker = ticker.strip().upper()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    urls = [
        f"https://www.screener.in/company/{ticker}/consolidated/",
        f"https://www.screener.in/company/{ticker}/"
    ]
    for url in urls:
        try:
            if debug:
                st.info(f"Trying URL: {url}")
            r = requests.get(url, headers=headers, timeout=15)
            if debug:
                st.info(f"Status: {r.status_code}")
            if r.status_code != 200:
                if debug:
                    st.warning(f"Status {r.status_code} for {url}")
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            if debug:
                st.info(f"Soup title: {soup.title.text if soup.title else 'No title'}")

            # PRIMARY: Look for section with id="profit-loss"
            section = soup.find("section", {"id": "profit-loss"})
            if not section:
                # FALLBACK: Look for h2 with "Profit & Loss" and next table
                h2 = soup.find("h2", string=re.compile(r"Profit\s*&?\s*L[oss]?", re.I))
                if h2:
                    section = h2.find_parent() or h2
                    if debug:
                        st.success("Fallback: Found via h2 'Profit & Loss'")
                else:
                    if debug:
                        st.error("No Profit & Loss section or h2 found")
                    continue

            table = section.find("table", class_="data-table") or soup.find("table", class_="data-table")
            if not table:
                if debug:
                    st.error("No .data-table found")
                continue

            df = parse_table(table)
            if df is not None and not df.empty:
                if debug:
                    st.success(f"Parsed table with {len(df)} rows from {url}")
                return df
        except Exception as e:
            if debug:
                st.error(f"Error for {url}: {e}")
            continue
    return None

def parse_table(table) -> pd.DataFrame | None:
    try:
        thead = table.find("thead")
        if not thead:
            return None
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]
        if len(headers) < 2:
            return None
        # Always set first column as "Metric" even if empty
        if not headers[0]:
            headers[0] = "Metric"
        rows = []
        tbody = table.find("tbody")
        if not tbody:
            return None
        for tr in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) == len(headers):
                rows.append(cells)
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=headers)
        df.set_index("Metric", inplace=True)
        return df
    except Exception as e:
        if st.session_state.get("debug_mode", False):
            st.error(f"Parse error: {e}")
        return None

def clean_numeric(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d.-]", "", regex=True)
        .str.strip()
        .replace("", "0")
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

def find_row(df: pd.DataFrame, name: str) -> tuple[str | None, str]:
    candidates = {
        "profit": [
            ("Operating Profit", "Operating Profit"),
            ("OP", "Operating Profit"),
            ("EBIT", "Operating Profit"),
            ("EBITDA", "Operating Profit"),
            ("Financing Profit", "Financing Profit"),
            ("Interest Income", "Financing Profit"),
        ],
        "sales": [
            ("Sales", "Sales"),
            ("Revenue", "Sales"),
            ("Net Sales", "Sales"),
            ("Total Income", "Sales"),
            ("Interest Earned", "Interest Earned"),
        ],
    }
    target = name.lower()
    idx_lower = [i.lower().strip() for i in df.index]
    search_list = candidates["profit"] if target == "profit" else candidates["sales"]
    for keyword, display in search_list:
        for idx in df.index:
            if keyword.lower() in idx.lower():
                return idx, display
    matches = difflib.get_close_matches(target, idx_lower, n=1, cutoff=0.7)
    if matches:
        matched_idx = df.index[idx_lower.index(matches[0])]
        return matched_idx, "Operating/Financing Profit" if target == "profit" else "Sales"
    return None, "Unknown"

# ───────────────────────────────────────────────
# LOAD OR SCRAPE P&L (WITH TICKER VALIDATION & DEBUG)
# ───────────────────────────────────────────────
def is_valid_ticker(ticker: str) -> bool:
    return ticker in nse_df["symbol"].values

def get_pl_data(ticker: str, force_scrape: bool = False, debug: bool = False) -> pd.DataFrame | None:
    if not is_valid_ticker(ticker):
        if debug:
            st.error(f"Invalid ticker: {ticker} (not in NSE list)")
        return None
    file_path = os.path.join(DATA_DIR, f"{ticker}_pl.csv")
    if not force_scrape and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, index_col=0)
            st.success(f"Loaded P&L for **{ticker}** from local cache")
            return df
        except:
            st.warning(f"Cache corrupt for {ticker}. Re-scraping...")
    with st.spinner(f"Scraping P&L for **{ticker}**..."):
        df = scrape_screener_data(ticker, debug=debug)
        if df is not None:
            df.to_csv(file_path)
            st.success(f"Saved P&L for **{ticker}** to `{file_path}`")
        else:
            st.error(f"Failed to scrape **{ticker}**")
    return df

# ───────────────────────────────────────────────
# CORE ANALYSIS
# ───────────────────────────────────────────────
def analyze_single_ticker(ticker: str, fy_start: int, force_scrape: bool = False, debug: bool = False):
    ticker = ticker.strip().upper()
    pl_df = get_pl_data(ticker, force_scrape, debug)
    if pl_df is None or pl_df.empty:
        return {"ticker": ticker, "error": "No P&L data"}
    profit_row_idx, profit_label = find_row(pl_df, "profit")
    if not profit_row_idx:
        return {"ticker": ticker, "error": "No Profit row"}
    sales_row_idx, sales_label = find_row(pl_df, "sales")

    def extract_metric(row_idx):
        if row_idx is None:
            return pd.Series()
        series = pl_df.loc[row_idx].iloc[1:]
        raw_cols = pl_df.columns[1:]
        years, vals = [], []
        for i, col in enumerate(raw_cols):
            col_str = col.strip()
            if col_str.upper() == "TTM":
                continue
            m = re.search(r"\d{4}", col_str)
            if m:
                yr = int(m.group())
                if 2000 <= yr <= 2100:
                    years.append(yr)
                    vals.append(series.iloc[i])
                    continue
            try:
                yr = int(col_str)
                if 2000 <= yr <= 2100:
                    years.append(yr)
                    vals.append(series.iloc[i])
            except:
                continue
        if not years:
            return pd.Series()
        clean = clean_numeric(pd.Series(vals, index=years))
        return clean[clean != 0].dropna()

    profit_clean = extract_metric(profit_row_idx)
    sales_clean = extract_metric(sales_row_idx) if sales_row_idx else pd.Series()
    if profit_clean.empty:
        return {"ticker": ticker, "error": "No Profit data"}

    try:
        start_date = f"{fy_start}-04-01"
        end_date = f"{datetime.date.today().year + 1}-03-31"
        price_df = yf.download(f"{ticker}.NS", start=start_date, end=end_date, progress=False)
    except Exception as e:
        return {"ticker": ticker, "error": f"Price error: {e}"}

    if price_df.empty:
        return {"ticker": ticker, "error": "No price data"}
    price_df = price_df.copy()
    price_df["FY_Year"] = price_df.index.year
    price_df.loc[price_df.index.month <= 3, "FY_Year"] -= 1
    avg_price = price_df.groupby("FY_Year")["Close"].mean().round(2)
    current_price = round(float(price_df["Close"].iloc[-1]), 2) if len(price_df) > 0 else 0.0

    common_years_profit = profit_clean.index.intersection(avg_price.index)
    if len(common_years_profit) < 2:
        return {"ticker": ticker, "error": "Need >=2 years"}
    profit_clean = profit_clean.loc[common_years_profit]
    avg_price_profit = avg_price.loc[common_years_profit]

    has_sales = False
    merged_sales = pd.DataFrame()
    if not sales_clean.empty:
        common_years_sales = sales_clean.index.intersection(avg_price.index)
        if len(common_years_sales) >= 2:
            sales_clean = sales_clean.loc[common_years_sales]
            avg_price_sales = avg_price.loc[common_years_sales]
            has_sales = True

    profit_1d = np.array(profit_clean).flatten()
    price_1d = np.array(avg_price_profit).flatten()
    ratio_profit_1d = np.round(price_1d / profit_1d, 4)
    merged_profit = pd.DataFrame(
        {
            f"{profit_label} (Cr)": profit_1d,
            "Avg Stock Price": price_1d,
            f"Price/{profit_label.split()[0]}": ratio_profit_1d,
        },
        index=common_years_profit.astype(int),
    ).sort_index()

    if has_sales:
        sales_1d = np.array(sales_clean).flatten()
        price_sales_1d = np.array(avg_price_sales).flatten()
        ratio_sales_1d = np.round(price_sales_1d / sales_1d, 4)
        merged_sales = pd.DataFrame(
            {
                f"{sales_label} (Cr)": sales_1d,
                "Avg Stock Price": price_sales_1d,
                f"Price/{sales_label.split()[0]}": ratio_sales_1d,
            },
            index=common_years_sales.astype(int),
        ).sort_index()

    X = merged_profit[f"{profit_label} (Cr)"].values.reshape(-1, 1)
    y = merged_profit["Avg Stock Price"].values
    model = LinearRegression().fit(X, y)
    latest_profit = float(merged_profit[f"{profit_label} (Cr)"].iloc[-1])
    pred_price = round(float(model.predict([[latest_profit]])[0]), 2)
    gain_pct = round(((pred_price - current_price) / current_price) * 100, 2) if current_price > 0 else 0.0
    r2 = round(model.score(X, y), 3)
    b1 = round(model.coef_[0], 6)
    b0 = round(model.intercept_, 2)
    eq = f"Price = {b0} + {b1} × {profit_label.split()[0]}"
    next_fy = merged_profit.index[-1] + 1

    return {
        "ticker": ticker,
        "profit_label": profit_label,
        "current_price": current_price,
        "forecasted_price": pred_price,
        "gain_pct": gain_pct,
        "r2": r2,
        "avg_ratio": round(merged_profit[f"Price/{profit_label.split()[0]}"].mean(), 4),
        "latest_ratio": round(merged_profit[f"Price/{profit_label.split()[0]}"].iloc[-1], 4),
        "years_count": len(common_years_profit),
        "merged_profit": merged_profit,
        "merged_sales": merged_sales,
        "eq": eq,
        "latest_profit": latest_profit,
        "next_fy": next_fy,
    }

# ───────────────────────────────────────────────
# SIDEBAR INPUTS
# ───────────────────────────────────────────────
st.sidebar.markdown("<h2 style='color: #1f77b4;'>Input Mode</h2>", unsafe_allow_html=True)
input_mode = st.sidebar.radio("Choose:", ["Single Ticker", "Multi-Ticker", "Upload CSV"])
debug_mode = st.sidebar.checkbox("Debug Mode (Show scraping details)", value=False)
st.session_state.debug_mode = debug_mode
fy_start = st.sidebar.number_input(
    "Start FY (Apr–Mar)",
    min_value=2000,
    max_value=datetime.date.today().year,
    value=2014,
)

tickers = []

if input_mode == "Single Ticker":
    # AUTOCOMPLETE SEARCH
    search = st.sidebar.text_input(
        "Search ticker (type to filter)",
        placeholder="e.g. ITC, RELIANCE, BAYERCROP",
        help="Start typing symbol or company name – a dropdown will appear."
    ).strip().upper()

    if search:
        matches = [
            opt for opt in ticker_options
            if search in opt.upper()
        ][:20]
        selected_display = st.sidebar.selectbox(
            "Select a ticker",
            options=[""] + matches,
            format_func=lambda x: x if x else "Type to search"
        )
    else:
        selected_display = ""

    ticker_input = ticker_to_symbol.get(selected_display, "").strip()
    tickers = [ticker_input] if ticker_input else []

elif input_mode == "Multi-Ticker":
    ticker_input = st.sidebar.text_area(
        "Enter Tickers (comma or newline)", "ITC,RELIANCE,HDFCBANK", height=100
    ).strip().upper()
    tickers = [t.strip() for t in re.split(r'[, \n]+', ticker_input) if t.strip()]

elif input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV with 'Symbol' column", type="csv")
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if "Symbol" in df_upload.columns:
                tickers = (
                    df_upload["Symbol"]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .dropna()
                    .unique()
                    .tolist()
                )
                st.sidebar.success(f"Loaded {len(tickers)} tickers")
            else:
                st.sidebar.error("CSV must have 'Symbol' column.")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# Force Re-Scrape
force_scrape = st.sidebar.checkbox("Re-scrape P&L data (ignore cache)", value=False)
if st.sidebar.button("Re-Scrape All P&L Data", type="secondary"):
    with st.spinner("Deleting cache..."):
        for f in os.listdir(DATA_DIR):
            os.remove(os.path.join(DATA_DIR, f))
    st.success("Cache cleared! Next run will re-scrape.")
    st.rerun()

# ───────────────────────────────────────────────
# ANALYZE
# ───────────────────────────────────────────────
if st.sidebar.button("Analyze All", type="primary"):
    if not tickers:
        st.error("No tickers provided.")
        st.stop()
    progress = st.progress(0)
    results = []
    for i, t in enumerate(tickers):
        with st.spinner(f"Analyzing {t}..."):
            res = analyze_single_ticker(t, fy_start, force_scrape, debug=debug_mode)
            results.append(res)
        progress.progress((i + 1) / len(tickers))
    progress.empty()

    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        st.error("No valid data found.")
        st.stop()

    st.session_state.analysis_results = {r["ticker"]: r for r in valid_results}
    st.session_state.summary_df = pd.DataFrame(valid_results)
    st.success(f"Analyzed {len(valid_results)} tickers")

# ───────────────────────────────────────────────
# DISPLAY RESULTS
# ───────────────────────────────────────────────
if "summary_df" in st.session_state:
    df = st.session_state.summary_df
    st.markdown("## Batch Summary")
    display_df = df[
        ["ticker", "current_price", "forecasted_price", "gain_pct", "r2", "years_count"]
    ].copy()
    display_df = display_df.round(
        {"current_price": 2, "forecasted_price": 2, "gain_pct": 1, "r2": 3}
    )
    st.dataframe(
        display_df.style.format(
            {
                "current_price": "Rs.{:.2f}",
                "forecasted_price": "Rs.{:.2f}",
                "gain_pct": "{:+.1f}%",
            }
        ),
        use_container_width=True,
    )

    high_gainers = df[df["gain_pct"] > 20].sort_values("gain_pct", ascending=False)
    if not high_gainers.empty:
        st.markdown("## Top Gainers (>20% Upside)")
        hg = high_gainers[
            ["ticker", "current_price", "forecasted_price", "gain_pct", "r2"]
        ].round(2)
        st.dataframe(
            hg.style.format(
                {
                    "current_price": "Rs.{:.2f}",
                    "forecasted_price": "Rs.{:.2f}",
                    "gain_pct": "{:+.1f}%",
                }
            ),
            use_container_width=True,
        )

    st.markdown("## View Full Analysis")
    valid_tickers = sorted(df["ticker"].tolist())
    selected = st.selectbox(
        "Select a ticker for detailed charts & forecast",
        options=[""] + valid_tickers,
        index=0,
    )

    if selected and selected in st.session_state.analysis_results:
        res = st.session_state.analysis_results[selected]
        st.markdown(f"## {selected} — Full Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price", f"Rs.{res['current_price']:,.2f}")
            st.metric(
                "Forecasted Price",
                f"Rs.{res['forecasted_price']:,.2f}",
                f"{res['gain_pct']:+.1f}%",
            )
        with col2:
            st.metric("R² (Model Fit)", f"{res['r2']:.3f}")
            st.caption(res["eq"])

        # Profit vs Price Chart
        chart_df_p = res["merged_profit"][
            [f"{res['profit_label']} (Cr)", "Avg Stock Price"]
        ].reset_index()
        chart_df_p.columns = ["Year", "Profit", "Price"]
        chart_long_p = chart_df_p.melt("Year", var_name="Metric", value_name="Value")
        base_p = alt.Chart(chart_long_p).encode(x=alt.X("Year:O", title="FY"))
        line_profit = base_p.mark_line(color="#1f77b4", strokeWidth=3).transform_filter(
            alt.datum.Metric == "Profit"
        ).encode(y=alt.Y("Value:Q", title=f"{res['profit_label']} (Cr)"))
        line_price = base_p.mark_line(color="#ff7f0e", strokeWidth=3).transform_filter(
            alt.datum.Metric == "Price"
        ).encode(y=alt.Y("Value:Q", title="Avg Price"))
        chart_p = alt.layer(line_profit, line_price).resolve_scale(y="independent").properties(
            width=700, height=400
        )
        st.altair_chart(chart_p, use_container_width=True)

        # Sales vs Price Chart
        if not res["merged_sales"].empty:
            chart_df_s = res["merged_sales"].iloc[:, [0, 1]].reset_index()
            chart_df_s.columns = ["Year", "Sales", "Price"]
            chart_long_s = chart_df_s.melt("Year", var_name="Metric", value_name="Value")
            base_s = alt.Chart(chart_long_s).encode(x=alt.X("Year:O"))
            line_sales = base_s.mark_line(color="#2ca02c", strokeWidth=3).transform_filter(
                alt.datum.Metric == "Sales"
            ).encode(y=alt.Y("Value:Q", title="Sales (Cr)"))
            line_price_s = base_s.mark_line(color="#ff7f0e", strokeWidth=3).transform_filter(
                alt.datum.Metric == "Price"
            ).encode(y=alt.Y("Value:Q", title="Avg Price"))
            chart_s = alt.layer(line_sales, line_price_s).resolve_scale(y="independent").properties(
                width=700, height=400
            )
            st.altair_chart(chart_s, use_container_width=True)

        # Valuation Insight
        ratio_col = f"Price/{res['profit_label'].split()[0]}"
        if ratio_col in res["merged_profit"].columns:
            avg_ratio = res["merged_profit"][ratio_col].mean()
            latest_ratio = res["merged_profit"][ratio_col].iloc[-1]
            change = ((latest_ratio - avg_ratio) / avg_ratio) * 100
            status = (
                "Overvalued"
                if change > 20
                else "Undervalued"
                if change < -20
                else "Fairly Valued"
            )
            st.markdown(
                f"### Valuation Insight\n**Avg Ratio**: `{avg_ratio:.4f}` | **Latest**: `{latest_ratio:.4f}` → **{change:+.1f}%** → **{status}**"
            )

        # Download Data
        combined = res["merged_profit"].copy()
        if not res["merged_sales"].empty:
            combined = combined.join(
                res["merged_sales"].drop(columns="Avg Stock Price", errors="ignore"), how="left"
            )
        csv = combined.to_csv().encode()
        st.download_button(
            "Download Full Data",
            data=csv,
            file_name=f"{selected}_analysis.csv",
            mime="text/csv",
        )

    # Export All
    export_df = df.copy()
    export_df["profit_data"] = export_df["merged_profit"].apply(
        lambda x: x.to_csv(index=True) if not x.empty else ""
    )
    export_df["sales_data"] = export_df["merged_sales"].apply(
        lambda x: x.to_csv(index=True) if not x.empty else ""
    )
    csv_all = export_df.drop(columns=["merged_profit", "merged_sales"]).to_csv(index=False)
    st.download_button(
        "Download All Results",
        data=csv_all,
        file_name=f"analysis_batch_{datetime.date.today()}.csv",
        mime="text/csv",
    )

st.caption(f"Data cached in `{DATA_DIR}/` • Re-scrape with button above • Fixed for BAYERCROP & others!")