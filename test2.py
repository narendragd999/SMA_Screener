# test2.py
import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import re
import difflib
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
import os
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# ───────────────────────────────────────────────
# PAGE CONFIG (must be before other st.* calls)
# ───────────────────────────────────────────────
st.set_page_config(page_title="OP & Sales Analyzer", layout="wide", initial_sidebar_state="collapsed")

# ───────────────────────────────────────────────
# CSS / THEME - Premium Dashboard (glass-style cards)
# ───────────────────────────────────────────────
st.markdown("""
<style>

:root {
    --primary-blue: #2D82F4;
    --primary-blue-dark: #1d63c5;
    --accent-purple: #7C4DFF;
    --accent-teal: #14B8A6;
    --text-dark: #1e293b;
    --text-muted: #64748b;
    --bg-main: #e3e9f2;       /* solid background */
    --card-bg: #ffffff;       /* strong solid white */
    --rounded: 14px;
    --shadow-1: 0 6px 20px rgba(0,0,0,0.08);
    --shadow-2: 0 12px 35px rgba(0,0,0,0.12);
}

/* PAGE BACKGROUND (solid color, visible always) */
.stApp {
    background: var(--bg-main) !important;
}

/* Header */
.app-header {
    text-align: center;
    padding-top: 15px;
    padding-bottom: 4px;
}
.app-title {
    font-size: 34px;
    font-weight: 800;
    color: var(--primary-blue);
    margin: 0;
}
.app-sub {
    margin: 0;
    color: var(--text-muted);
    font-size: 14px;
}

/* MAIN CARD — Premium Gradient */
.main-card {
    background: linear-gradient(135deg, #ffffff 0%, #e8eeff 40%, #dfe4fd 100%);
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 12px 35px rgba(88, 105, 141, 0.18);
    border: 1px solid rgba(0,0,0,0.05);
}

/* SMALL CARD — More colorful with subtle gradient */
.small-card {
    background: linear-gradient(145deg, #ffffff 0%, #eef3ff 60%, #e2e7ff 100%);
    border-radius: 14px;
    padding: 16px;
    margin-top: 12px;
    box-shadow: 0 8px 22px rgba(90, 110, 150, 0.15);
    border: 1px solid rgba(0,0,0,0.04);
}


/* Labels */
.control-label {
    font-weight: 600;
    color: var(--text-dark);
    margin-bottom: 6px;
}
.muted {
    color: var(--text-muted);
    font-size: 13px;
}

/* BUTTON STYLES */
.stButton>button {
    border-radius: 12px !important;
    padding: 10px 16px !important;
    border: none !important;
    font-weight: 600;
}

/* Primary button */
.stButton.primary-btn>button {
    background: linear-gradient(90deg, var(--primary-blue), var(--accent-purple)) !important;
    color: white !important;
    box-shadow: 0 5px 16px rgba(45,130,244,0.35);
}
.stButton.primary-btn>button:hover {
    background: linear-gradient(90deg, var(--primary-blue-dark), var(--accent-purple)) !important;
}

/* Secondary button */
.stButton.secondary-btn>button {
    background: #f1f5f9 !important;
    color: var(--text-dark) !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
}

/* Download button */
.stDownloadButton>button {
    background: linear-gradient(90deg, var(--accent-teal), var(--primary-blue)) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 8px 14px !important;
    font-weight: 700;
    border: none;
}

/* Metrics */
.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: var(--primary-blue-dark);
}
.metric-label {
    color: var(--text-muted);
    font-size: 13px;
}

/* Data Table */
.stDataFrame table {
    background: white !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0,0,0,0.05) !important;
    overflow: hidden !important;
}

/* Tabs - colorful and visible */
.stTabs [role="tab"] {
    background: #f3f4f6;
    padding: 8px 16px;
    border-radius: 8px;
    margin-right: 5px;
    color: var(--text-dark);
}
.stTabs [aria-selected="true"] {
    background: var(--primary-blue) !important;
    color: white !important;
    font-weight: 600;
}

/* Fix all select/dropdowns */
.css-1x8cf1d, .stSelectbox select {
    background-color: white !important;
}

</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────
# DATA DIR
# ───────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ───────────────────────────────────────────────
# LOAD NSE TICKERS
# ───────────────────────────────────────────────
TICKER_CSV_PATH = "NSE_ALL_TICKERS_LIST.csv"

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
# Helper functions (Playwright scraping, parsing etc.)
# ───────────────────────────────────────────────
@st.cache_data(ttl=60*60*24)
def _playwright_page_source(url: str, debug: bool = False) -> str | None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080}
        )
        page = context.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_selector("section#profit-loss, div#profit-loss, table.data-table", timeout=20000)
            html = page.content()
            if debug:
                st.success(f"Fetched: {url}")
            return html
        except Exception as e:
            if debug:
                st.error(f"Playwright error: {e}")
            return None
        finally:
            browser.close()

def scrape_screener_data(ticker: str, debug: bool = False) -> pd.DataFrame | None:
    ticker = ticker.strip().upper()
    urls = [
        f"https://www.screener.in/company/{ticker}/consolidated/",
        f"https://www.screener.in/company/{ticker}/"
    ]
    for url in urls:
        html = _playwright_page_source(url, debug)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        container = soup.find("section", {"id": "profit-loss"}) or soup.find("div", {"id": "profit-loss"})
        if not container:
            h2 = soup.find("h2", string=re.compile(r"Profit\s*&?\s*L[oss]?", re.I))
            if h2:
                container = h2.find_next("table")
        if not container:
            continue
        table = container.find("table", class_="data-table") or soup.find("table", class_="data-table")
        if not table:
            continue
        df = parse_table(table)
        if df is not None and not df.empty:
            return df
    return None

def parse_table(table) -> pd.DataFrame | None:
    try:
        thead = table.find("thead")
        if not thead:
            return None
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]
        if len(headers) < 2:
            return None
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
        "other_income": [
            ("Other Income", "Other Income"),
            ("Other Operating Income", "Other Income"),
            ("Other Inc", "Other Income"),
        ],
    }
    target = name.lower()
    search_list = candidates.get(target, [])
    for keyword, display in search_list:
        for idx in df.index:
            if keyword.lower() in idx.lower():
                return idx, display
    idx_lower = [i.lower().strip() for i in df.index]
    matches = difflib.get_close_matches(target, idx_lower, n=1, cutoff=0.6)
    if matches:
        matched_idx = df.index[idx_lower.index(matches[0])]
        return matched_idx, "Operating Profit" if target == "profit" else "Sales" if target == "sales" else "Other Income"
    return None, "Unknown"

def is_bank_or_finance(ticker: str) -> bool:
    ticker = ticker.upper()
    row = nse_df[nse_df["symbol"] == ticker]
    if row.empty:
        return False
    name = row["company_name"].iloc[0].upper()
    return any(k in name for k in ["BANK", "FINANCE", "NBFC", "FINANCIAL", "LENDING", "MICROFINANCE"])

def is_valid_ticker(ticker: str) -> bool:
    return ticker in nse_df["symbol"].values

def get_pl_data(ticker: str, force_scrape: bool = False, debug: bool = False) -> pd.DataFrame | None:
    if not is_valid_ticker(ticker):
        if debug:
            st.error(f"Invalid ticker: {ticker}")
        return None
    file_path = os.path.join(DATA_DIR, f"{ticker}_pl.csv")
    if not force_scrape and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, index_col=0)
            st.success(f"Loaded P&L for **{ticker}** from cache")
            return df
        except:
            st.warning(f"Corrupt cache for {ticker}. Re-scraping...")
    with st.spinner(f"Scraping P&L for **{ticker}**..."):
        df = scrape_screener_data(ticker, debug=debug)
        if df is not None:
            df.to_csv(file_path)
            st.success(f"Saved P&L for **{ticker}** → `{file_path}`")
        else:
            st.error(f"Failed to scrape **{ticker}**")
    return df

def analyze_single_ticker(ticker: str, fy_start: int, force_scrape: bool = False, debug: bool = False, include_other_income: bool = False):
    ticker = ticker.strip().upper()
    pl_df = get_pl_data(ticker, force_scrape, debug)
    if pl_df is None or pl_df.empty:
        return {"ticker": ticker, "error": "No P&L data"}

    profit_row_idx, profit_label = find_row(pl_df, "profit")
    if not profit_row_idx:
        return {"ticker": ticker, "error": "No Profit row"}
    sales_row_idx, sales_label = find_row(pl_df, "sales")

    other_income_row_idx = None
    other_income_label = None
    if include_other_income and is_bank_or_finance(ticker):
        other_income_row_idx, other_income_label = find_row(pl_df, "other_income")

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
    other_income_clean = extract_metric(other_income_row_idx) if other_income_row_idx else pd.Series()
    sales_clean = extract_metric(sales_row_idx) if sales_row_idx else pd.Series()

    # COMBINE OP + OTHER INCOME
    if include_other_income and not other_income_clean.empty and is_bank_or_finance(ticker):
        common_years = profit_clean.index.intersection(other_income_clean.index)
        if len(common_years) >= 2:
            profit_clean = profit_clean.loc[common_years] + other_income_clean.loc[common_years]
            profit_label = f"{profit_label} + {other_income_label}"
        else:
            st.warning(f"Not enough overlapping years for Other Income in {ticker}")

    if profit_clean.empty:
        return {"ticker": ticker, "error": "No Profit data"}

    # ── PRICE DATA ───────────────────────────────────────
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

    # ── ALIGN WITH FUNDAMENTALS ───────────────────────
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

    # ── OP MODEL ───────────────────────────────────────
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

    X = merged_profit[f"{profit_label} (Cr)"].values.reshape(-1, 1)
    y = merged_profit["Avg Stock Price"].values
    model = LinearRegression().fit(X, y)
    latest_profit = float(merged_profit[f"{profit_label} (Cr)"].iloc[-1])
    pred_price = round(float(model.predict([[latest_profit]])[0]), 2)
    gain_pct = round(((pred_price - current_price) / current_price) * 100, 2) if current_price > 0 else 0.0
    r2 = round(model.score(X, y), 3)
    b1 = round(model.coef_[0], 6)
    b0 = round(model.intercept_, 2)
    eq = f"Price = {b0} + {b1} × {profit_label.split(' ')[0]}"

    # ── SALES MODEL (optional) ───────────────────────
    model_sales = None
    r2_sales = None
    eq_sales = None
    pred_price_sales = None
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

        X_sales = merged_sales[f"{sales_label} (Cr)"].values.reshape(-1, 1)
        y_sales = merged_sales["Avg Stock Price"].values
        model_sales = LinearRegression().fit(X_sales, y_sales)
        latest_sales = float(merged_sales[f"{sales_label} (Cr)"].iloc[-1])
        pred_price_sales = round(float(model_sales.predict([[latest_sales]])[0]), 2)
        r2_sales = round(model_sales.score(X_sales, y_sales), 3)
        b1_sales = round(model_sales.coef_[0], 6)
        b0_sales = round(model_sales.intercept_, 2)
        eq_sales = f"Price = {b0_sales} + {b1_sales} × {sales_label.split()[0]}"

    # ── HISTORICAL VALUATION (OP) ─────────────────────
    historical_op = {}
    for yr in merged_profit.index:
        op_val = merged_profit.loc[yr, f"{profit_label} (Cr)"]
        fair = round(float(model.predict([[op_val]])[0]), 2)
        actual = merged_profit.loc[yr, "Avg Stock Price"]
        misprice = round(((actual - fair) / fair) * 100, 1)
        status = "Overvalued" if misprice > 20 else "Undervalued" if misprice < -20 else "Fair"
        historical_op[yr] = {"OP": op_val, "Fair": fair, "Actual": actual, "Misprice": misprice, "Status": status}

    # ── HISTORICAL VALUATION (SALES) ───────────────────
    historical_sales = {}
    if has_sales:
        for yr in merged_sales.index:
            sales_val = merged_sales.loc[yr, f"{sales_label} (Cr)"]
            fair = round(float(model_sales.predict([[sales_val]])[0]), 2)
            actual = merged_sales.loc[yr, "Avg Stock Price"]
            misprice = round(((actual - fair) / fair) * 100, 1)
            status = "Overvalued" if misprice > 20 else "Undervalued" if misprice < -20 else "Fair"
            historical_sales[yr] = {"Sales": sales_val, "Fair": fair, "Actual": actual, "Misprice": misprice, "Status": status}

    return {
        "ticker": ticker,
        "profit_label": profit_label,
        "sales_label": sales_label if has_sales else None,
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
        "include_other_income": include_other_income,
        "has_sales": has_sales,
        "model_sales": model_sales,
        "eq_sales": eq_sales,
        "r2_sales": r2_sales,
        "pred_price_sales": pred_price_sales,
        "historical_op": historical_op,
        "historical_sales": historical_sales,
        "price_series": price_df["Close"].reset_index()
    }

# ───────────────────────────────────────────────
# PAGE HEADER
# ───────────────────────────────────────────────
st.markdown('<div class="app-header"><h1 class="app-title">OP & Sales vs Stock Price Analyzer</h1>'
            '<div class="app-sub">Clean, local P&L cache • No re-scrape unless requested</div></div>', unsafe_allow_html=True)

# ───────────────────────────────────────────────
# INPUT CARD (single, always visible)
# ───────────────────────────────────────────────
container = st.container()
with container:
    # centralize the card using columns (left, center, right)
    left_col, center_col, right_col = st.columns([1, 2.2, 1])
    with center_col:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown("### Analysis Settings", unsafe_allow_html=True)

        # --- Input mode
        input_mode = st.selectbox("Input Mode", options=["Single Ticker", "Multi-Ticker", "Upload CSV"], index=0)

        # --- Debug and FY
        cols = st.columns([1, 1, 1])
        with cols[0]:
            debug_mode = st.checkbox("Debug Mode", value=False)
            st.session_state.debug_mode = debug_mode
        with cols[1]:
            fy_start = st.number_input("Start FY (Apr–Mar)", min_value=2000, max_value=datetime.date.today().year, value=2014)
        with cols[2]:
            force_scrape = st.checkbox("Re-scrape P&L (ignore cache)", value=False)

        # --- Ticker input logic (mirrors original but on-page)
        tickers = []
        include_other_income_global = False

        if input_mode == "Single Ticker":
            st.markdown('<div class="small-card" style="margin-top:10px;">', unsafe_allow_html=True)
            st.markdown("**Search & Select Ticker**", unsafe_allow_html=True)
            search = st.text_input("Search ticker (symbol or company)", placeholder="e.g. ICICIBANK, HDFCBANK").strip().upper()
            selected_display = ""
            if search:
                matches = [opt for opt in ticker_options if search in opt.upper()][:30]
                selected_display = st.selectbox("Matches", options=[""] + matches, format_func=lambda x: x if x else "Type to search")
            else:
                selected_display = st.selectbox("Recent / Popular", options=[""] + ticker_options[:6], format_func=lambda x: x if x else "Type to search")
            ticker_input = ticker_to_symbol.get(selected_display, "").strip()
            tickers = [ticker_input] if ticker_input else []

            if tickers and is_bank_or_finance(tickers[0]):
                include_other_income_global = st.checkbox("Include Other Income in OP (Banks/NBFC)", value=True)
                st.caption("For banks: OP = Operating Profit + Other Income")
            st.markdown('</div>', unsafe_allow_html=True)

        elif input_mode == "Multi-Ticker":
            st.markdown('<div class="small-card" style="margin-top:10px;">', unsafe_allow_html=True)
            st.markdown("**Multi-Ticker Input**", unsafe_allow_html=True)
            ticker_input_area = st.text_area("Enter tickers (comma / newline)", "ICICIBANK,AXISBANK,HDFCBANK", height=100).strip().upper()
            tickers = [t.strip() for t in re.split(r'[, \n]+', ticker_input_area) if t.strip()]
            include_other_income_global = st.checkbox("Include Other Income in OP for Banks", value=True)
            st.markdown('</div>', unsafe_allow_html=True)

        elif input_mode == "Upload CSV":
            st.markdown('<div class="small-card" style="margin-top:10px;">', unsafe_allow_html=True)
            st.markdown("**Upload CSV**", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload CSV with 'Symbol' column", type="csv")
            if uploaded_file:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    if "Symbol" in df_upload.columns:
                        tickers = df_upload["Symbol"].astype(str).str.upper().str.strip().dropna().unique().tolist()
                        st.success(f"Loaded {len(tickers)} tickers")
                    else:
                        st.error("CSV must have 'Symbol' column.")
                except Exception as e:
                    st.error(f"Error: {e}")
            include_other_income_global = st.checkbox("Include Other Income in OP for Banks", value=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- Action Buttons (centered)
        st.write("")  # spacing
        btn_cols = st.columns([1, 1, 1])
        with btn_cols[0]:
            if st.button("Clear Cache & Re-scrape All", key="clear_cache"):
                with st.spinner("Clearing cache..."):
                    for f in os.listdir(DATA_DIR):
                        os.remove(os.path.join(DATA_DIR, f))
                st.success("Cache cleared!")
                st.experimental_rerun()
        with btn_cols[1]:
            analyze_pressed = st.button("Analyze All", key="analyze_all")
        with btn_cols[2]:
            st.download_button("Download Template CSV", data="Symbol\nICICIBANK\nHDFCBANK\n", file_name="tickers_template.csv", mime="text/csv")

        st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# ANALYZE action (mirrors the previous behavior)
# ───────────────────────────────────────────────
if analyze_pressed:
    if not tickers:
        st.error("No tickers provided.")
        st.stop()
    progress = st.progress(0)
    results = []
    for i, t in enumerate(tickers):
        include_other = include_other_income_global and is_bank_or_finance(t)
        with st.spinner(f"Analyzing {t}..."):
            res = analyze_single_ticker(t, fy_start, force_scrape, debug=debug_mode, include_other_income=include_other)
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
# DISPLAY RESULTS (full-width)
# ───────────────────────────────────────────────
if "summary_df" in st.session_state:
    df = st.session_state.summary_df
    st.markdown("## Batch Summary")
    display_df = df[["ticker", "current_price", "forecasted_price", "gain_pct", "r2", "years_count"]].copy()
    display_df = display_df.round({"current_price": 2, "forecasted_price": 2, "gain_pct": 1, "r2": 3})
    st.dataframe(
        display_df.style.format({"current_price": "Rs.{:.2f}", "forecasted_price": "Rs.{:.2f}", "gain_pct": "{:+.1f}%"}),
        use_container_width=True,
    )

    high_gainers = df[df["gain_pct"] > 20].sort_values("gain_pct", ascending=False)
    if not high_gainers.empty:
        st.markdown("## Top Gainers (>20% Upside)")
        hg = high_gainers[["ticker", "current_price", "forecasted_price", "gain_pct", "r2"]].round(2)
        st.dataframe(hg.style.format({"current_price": "Rs.{:.2f}", "forecasted_price": "Rs.{:.2f}", "gain_pct": "{:+.1f}%"}), use_container_width=True)

    st.markdown("## View Full Analysis")
    valid_tickers = sorted(df["ticker"].tolist())
    selected = st.selectbox("Select a ticker for detailed charts & forecast", options=[""] + valid_tickers, index=0)

    if selected and selected in st.session_state.analysis_results:
        res = st.session_state.analysis_results[selected]
        st.markdown(f"## {selected} — Full Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price", f"Rs.{res['current_price']:,.2f}")
            st.metric("Forecasted Price", f"Rs.{res['forecasted_price']:,.2f}", f"{res['gain_pct']:+.1f}%")
        with col2:
            st.metric("R² (Model Fit)", f"{res['r2']:.3f}")
            st.caption(res["eq"])
            if res.get("include_other_income"):
                st.caption("**OP includes Other Income**")

        # FUNDAMENTAL VS PRICE CHARTS
        chart_df_p = res["merged_profit"][[f"{res['profit_label']} (Cr)", "Avg Stock Price"]].reset_index()
        chart_df_p.columns = ["Year", "Profit", "Price"]
        chart_long_p = chart_df_p.melt("Year", var_name="Metric", value_name="Value")
        base_p = alt.Chart(chart_long_p).encode(x=alt.X("Year:O", title="FY"))
        line_profit = base_p.mark_line(color="#1f77b4", strokeWidth=3).transform_filter(alt.datum.Metric == "Profit").encode(y=alt.Y("Value:Q", title=f"{res['profit_label']} (Cr)"))
        line_price = base_p.mark_line(color="#ff7f0e", strokeWidth=3).transform_filter(alt.datum.Metric == "Price").encode(y=alt.Y("Value:Q", title="Avg Price"))
        chart_p = alt.layer(line_profit, line_price).resolve_scale(y="independent").properties(width="container", height=380)
        st.altair_chart(chart_p, use_container_width=True)

        if not res["merged_sales"].empty:
            chart_df_s = res["merged_sales"].iloc[:, [0, 1]].reset_index()
            chart_df_s.columns = ["Year", "Sales", "Price"]
            chart_long_s = chart_df_s.melt("Year", var_name="Metric", value_name="Value")
            base_s = alt.Chart(chart_long_s).encode(x=alt.X("Year:O"))
            line_sales = base_s.mark_line(color="#2ca02c", strokeWidth=3).transform_filter(alt.datum.Metric == "Sales").encode(y=alt.Y("Value:Q", title="Sales (Cr)"))
            line_price_s = base_s.mark_line(color="#ff7f0e", strokeWidth=3).transform_filter(alt.datum.Metric == "Price").encode(y=alt.Y("Value:Q", title="Avg Price"))
            chart_s = alt.layer(line_sales, line_price_s).resolve_scale(y="independent").properties(width="container", height=380)
            st.altair_chart(chart_s, use_container_width=True)

        # VALUATION INSIGHT
        ratio_col = f"Price/{res['profit_label'].split()[0]}"
        if ratio_col in res["merged_profit"].columns:
            avg_ratio = res["merged_profit"][ratio_col].mean()
            latest_ratio = res["merged_profit"][ratio_col].iloc[-1]
            change = ((latest_ratio - avg_ratio) / avg_ratio) * 100
            status = "Overvalued" if change > 20 else "Undervalued" if change < -20 else "Fairly Valued"
            st.markdown(f"### Valuation Insight\n**Avg Ratio**: `{avg_ratio:.4f}` | **Latest**: `{latest_ratio:.4f}` → **{change:+.1f}%** → **{status}**")

        # HISTORICAL VALUATION CHECKER
        st.markdown("### Historical Valuation Checker")
        tab_op, tab_sales = st.tabs(["Operating Profit Model", "Sales Model"])

        with tab_op:
            years = sorted(res["historical_op"].keys())
            selected_year = st.slider("Select FY Year (OP)", min_value=years[0], max_value=years[-1], value=years[-1], key="op_year")
            hf = res["historical_op"][selected_year]
            colA, colB, colC = st.columns(3)
            with colA: st.metric(f"FY {selected_year} {res['profit_label'].split()[0]}", f"Rs.{hf['OP']:,.0f} Cr")
            with colB: st.metric("Fair Price", f"Rs.{hf['Fair']:,.0f}")
            with colC: st.metric("Actual Price", f"Rs.{hf['Actual']:,.0f}", f"{hf['Misprice']:+.1f}%")
            st.markdown(f"**Status**: **{hf['Status']}**")

            # FULL HISTORIC PRICE CHART
            price_series = res["price_series"]
            price_series.columns = ["Date", "Close"]
            price_series["Date"] = pd.to_datetime(price_series["Date"])

            chart_price = alt.Chart(price_series).mark_line(
                color="#ff7f0e", strokeWidth=2
            ).encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Close:Q", title="Close Price (Rs.)"),
                tooltip=["Date:T", "Close:Q"]
            ).properties(
                width="container",
                height=350,
                title="Historical Stock Price (Daily Close)"
            ).interactive()

            st.altair_chart(chart_price, use_container_width=True)

        if res["has_sales"]:
            with tab_sales:
                years_s = sorted(res["historical_sales"].keys())
                selected_year_s = st.slider("Select FY Year (Sales)", min_value=years_s[0], max_value=years_s[-1], value=years_s[-1], key="sales_year")
                hf_s = res["historical_sales"][selected_year_s]
                colA, colB, colC = st.columns(3)
                with colA: st.metric(f"FY {selected_year_s} {res['sales_label'].split()[0]}", f"Rs.{hf_s['Sales']:,.0f} Cr")
                with colB: st.metric("Fair Price", f"Rs.{hf_s['Fair']:,.0f}")
                with colC: st.metric("Actual Price", f"Rs.{hf_s['Actual']:,.0f}", f"{hf_s['Misprice']:+.1f}%")
                st.markdown(f"**Status**: **{hf_s['Status']}**")
                st.caption(res["eq_sales"])

                chart_price_s = alt.Chart(price_series).mark_line(
                    color="#ff7f0e", strokeWidth=2
                ).encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Close:Q", title="Close Price (Rs.)"),
                    tooltip=["Date:T", "Close:Q"]
                ).properties(
                    width="container",
                    height=350,
                    title="Historical Stock Price (Daily Close)"
                ).interactive()

                st.altair_chart(chart_price_s, use_container_width=True)

        # DOWNLOAD per ticker
        combined = res["merged_profit"].copy()
        if not res["merged_sales"].empty:
            combined = combined.join(res["merged_sales"].drop(columns="Avg Stock Price", errors="ignore"), how="left")
        csv = combined.to_csv().encode()
        st.download_button("Download Full Data", data=csv, file_name=f"{selected}_analysis.csv", mime="text/csv")

    # EXPORT ALL
    export_df = df.copy()
    export_df["profit_data"] = export_df["merged_profit"].apply(lambda x: x.to_csv(index=True) if not x.empty else "")
    export_df["sales_data"] = export_df["merged_sales"].apply(lambda x: x.to_csv(index=True) if not x.empty else "")
    csv_all = export_df.drop(columns=["merged_profit", "merged_sales"]).to_csv(index=False)
    st.download_button("Download All Results", data=csv_all, file_name=f"analysis_batch_{datetime.date.today()}.csv", mime="text/csv")

st.caption(f"Data cached in `{DATA_DIR}/` • Fixed for **ICICIBANK**, **AXISBANK**, **Other Income** support!")
