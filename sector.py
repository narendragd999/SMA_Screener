# sector_fixed_full.py
# Complete Streamlit app (FULL SCRIPT) — includes:
# - Sector Selection (multi-sector) + CSV support
# - Batch processing of tickers
# - Caching & Playwright scraping of Screener P&L tables
# - OP vs Price linear model, Sales model, historical valuation
# - Distinct chart colors (Profit = purple #6a11cb, Price = blue #3498db)
# - Fixed NameError by always reading results from st.session_state for exports
#
# Save as: sector_fixed_full.py
# Run: streamlit run sector_fixed_full.py
#
# Requirements:
# streamlit, pandas, yfinance, beautifulsoup4, playwright, scikit-learn, altair, numpy
# (Playwright must be installed and `playwright install` run on host)

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
# PAGE CONFIG
# ───────────────────────────────────────────────
st.set_page_config(page_title="OP & Sales Analyzer", layout="wide", initial_sidebar_state="collapsed")

# ───────────────────────────────────────────────
# CSS / THEME (kept as requested)
# ───────────────────────────────────────────────
st.markdown("""
<style>

:root {
    --primary-blue: #3498db;
    --primary-blue-dark: #2D82F4;
    --accent-purple: #6a11cb;
    --text-dark: #1e293b;
    --text-muted: #64748b;
    --bg-main: linear-gradient(180deg, #6a11cb 0%, #ffffff 100%);
    --card-bg: #ffffff;
    --rounded: 12px;
}

/* Page background */
.stApp {
    background: linear-gradient(180deg, #6a11cb 0%, #ffffff 100%) !important;
}

/* Header */
.app-header { text-align: center; padding-top: 14px; padding-bottom: 8px; }
.app-title { font-size: 30px; font-weight: 800; color: white; margin: 0; text-shadow: 0 1px 4px rgba(0,0,0,0.2); }
.app-sub { margin: 0; color: rgba(255,255,255,0.9); font-size: 13px; }

/* Main card */
.main-card {
    background: white;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.10);
    border: none;
}

/* Small card */
.small-card {
    background: white;
    border-radius: 10px;
    padding: 12px;
    margin-top: 10px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}

/* Buttons */
.stButton>button { border-radius: 10px !important; padding: 8px 14px !important; font-weight: 700; }
.stButton.primary-btn>button { background: #3498db !important; color: white !important; box-shadow: 0 6px 18px rgba(52,152,219,0.18); }
.stButton.secondary-btn>button { background: #6a11cb !important; color: white !important; }

/* Input cards & tables */
.stDataFrame table { background: white !important; border-radius: 8px !important; }

/* Smaller controls */
input, textarea { font-size: 14px !important; }

</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# DATA DIR
# ───────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ───────────────────────────────────────────────
# Load optional NSE tickers csv for search/autocomplete
# ───────────────────────────────────────────────
TICKER_CSV_PATH = "NSE_ALL_TICKERS_LIST.csv"

@st.cache_data
def load_nse_tickers() -> pd.DataFrame:
    if not os.path.exists(TICKER_CSV_PATH):
        return pd.DataFrame(columns=["symbol", "company_name", "display"])
    df = pd.read_csv(TICKER_CSV_PATH, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    if "name of company" in df.columns:
        df = df.rename(columns={"name of company": "company_name"})
    if "symbol" not in df.columns or "company_name" not in df.columns:
        return pd.DataFrame(columns=["symbol", "company_name", "display"])
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["company_name"] = df["company_name"].str.strip()
    df["display"] = df["symbol"] + " – " + df["company_name"]
    return df

nse_df = load_nse_tickers()
ticker_options = nse_df["display"].tolist() if not nse_df.empty else []
ticker_to_symbol = dict(zip(nse_df.get("display", []), nse_df.get("symbol", [])))

# ───────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────
@st.cache_data(ttl=60*60*24)
def _playwright_page_source(url: str, debug: bool = False) -> str | None:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                viewport={"width": 1366, "height": 768}
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            try:
                page.wait_for_selector("section#profit-loss, div#profit-loss, table.data-table", timeout=20000)
            except:
                pass
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        if st.session_state.get("debug_mode", False):
            st.error(f"Playwright error: {e}")
        return None

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
    if not nse_df.empty:
        row = nse_df[nse_df["symbol"] == ticker]
        if not row.empty:
            name = row["company_name"].iloc[0].upper()
            return any(k in name for k in ["BANK", "FINANCE", "NBFC", "FINANCIAL", "LENDING", "MICROFINANCE"])
    return False

def is_valid_ticker(ticker: str) -> bool:
    return isinstance(ticker, str) and len(ticker) > 0

def get_pl_data(ticker: str, force_scrape: bool = False, debug: bool = False) -> pd.DataFrame | None:
    ticker = ticker.strip().upper()
    file_path = os.path.join(DATA_DIR, f"{ticker}_pl.csv")
    if not force_scrape and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, index_col=0)
            if st.session_state.get("debug_mode", False):
                st.success(f"Loaded P&L for {ticker} from cache")
            return df
        except Exception:
            st.warning(f"Cache corrupt for {ticker}, re-scraping.")
    with st.spinner(f"Scraping P&L for {ticker}..."):
        df = scrape_screener_data(ticker, debug=debug)
        if df is not None:
            try:
                df.to_csv(file_path)
            except Exception:
                pass
            st.success(f"Saved P&L for {ticker}")
        else:
            st.error(f"Failed to scrape P&L for {ticker}")
    return df

def analyze_single_ticker(ticker: str, fy_start: int, force_scrape: bool = False, debug: bool = False, include_other_income: bool = False):
    ticker = ticker.strip().upper()
    if not is_valid_ticker(ticker):
        return {"ticker": ticker, "error": "Invalid ticker"}

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

    # Combine OP + Other Income for banks if requested
    if include_other_income and not other_income_clean.empty and is_bank_or_finance(ticker):
        common_years = profit_clean.index.intersection(other_income_clean.index)
        if len(common_years) >= 2:
            profit_clean = profit_clean.loc[common_years] + other_income_clean.loc[common_years]
            profit_label = f"{profit_label} + {other_income_label}"
        else:
            st.warning(f"Not enough overlapping years for Other Income in {ticker}")

    if profit_clean.empty:
        return {"ticker": ticker, "error": "No Profit data"}

    # Price data
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

    # Align with fundamentals
    common_years_profit = profit_clean.index.intersection(avg_price.index)
    if len(common_years_profit) < 2:
        return {"ticker": ticker, "error": "Need >=2 years of overlapping data"}
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

    # OP Model
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

    # Sales model (optional)
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

    # Historical valuation (OP)
    historical_op = {}
    for yr in merged_profit.index:
        op_val = merged_profit.loc[yr, f"{profit_label} (Cr)"]
        fair = round(float(model.predict([[op_val]])[0]), 2)
        actual = merged_profit.loc[yr, "Avg Stock Price"]
        misprice = round(((actual - fair) / fair) * 100, 1) if fair != 0 else 0.0
        status = "Overvalued" if misprice > 20 else "Undervalued" if misprice < -20 else "Fair"
        historical_op[yr] = {"OP": op_val, "Fair": fair, "Actual": actual, "Misprice": misprice, "Status": status}

    historical_sales = {}
    if has_sales:
        for yr in merged_sales.index:
            sales_val = merged_sales.loc[yr, f"{sales_label} (Cr)"]
            fair = round(float(model_sales.predict([[sales_val]])[0]), 2)
            actual = merged_sales.loc[yr, "Avg Stock Price"]
            misprice = round(((actual - fair) / fair) * 100, 1) if fair != 0 else 0.0
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
            '<div class="app-sub">Batch analysis • Sector CSV support • Cached P&L</div></div>', unsafe_allow_html=True)

# ───────────────────────────────────────────────
# INPUT CARD (central)
# ───────────────────────────────────────────────
container = st.container()
with container:
    left_col, center_col, right_col = st.columns([1, 2.2, 1])
    with center_col:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown("### Analysis Settings", unsafe_allow_html=True)

        input_mode = st.selectbox("Input Mode", options=["Single Ticker", "Multi-Ticker", "Upload CSV"], index=0)

        cols = st.columns([1, 1, 1])
        with cols[0]:
            debug_mode = st.checkbox("Debug Mode", value=False)
            st.session_state.debug_mode = debug_mode
        with cols[1]:
            fy_start = st.number_input("Start FY (Apr–Mar)", min_value=2000, max_value=datetime.date.today().year, value=2014)
        with cols[2]:
            force_scrape = st.checkbox("Re-scrape P&L (ignore cache)", value=False)

        # Sector selection card
        st.markdown('<div class="small-card" style="margin-top:10px;">', unsafe_allow_html=True)
        st.markdown("### Sector Filtering (Optional)", unsafe_allow_html=True)
        st.markdown("Upload a CSV with columns `Symbol` and `Sector`. Then pick sector(s) to process.", unsafe_allow_html=True)
        uploaded_sector_file = st.file_uploader("Upload Sector CSV (Symbol, Sector)", type="csv", key="sector_csv")
        sector_df = pd.DataFrame()
        sector_list = []
        selected_sectors = []
        filtered_tickers = []

        if uploaded_sector_file:
            try:
                tmp = pd.read_csv(uploaded_sector_file)
                tmp.columns = [c.strip().title() for c in tmp.columns]
                if {"Symbol", "Sector"}.issubset(tmp.columns):
                    tmp["Symbol"] = tmp["Symbol"].astype(str).str.upper().str.strip()
                    tmp["Sector"] = tmp["Sector"].astype(str).str.strip()
                    sector_df = tmp[["Symbol", "Sector"]].dropna().drop_duplicates().reset_index(drop=True)
                    sector_list = sorted(sector_df["Sector"].unique().tolist())
                    if sector_list:
                        selected_sectors = st.multiselect("Select sector(s) to analyze", options=sector_list, default=[])
                        if selected_sectors:
                            filtered_tickers = sector_df[sector_df["Sector"].isin(selected_sectors)]["Symbol"].unique().tolist()
                            st.success(f"{len(filtered_tickers)} tickers selected from {len(selected_sectors)} sector(s).")
                else:
                    st.error("Sector CSV must include columns 'Symbol' and 'Sector'.")
            except Exception as e:
                st.error(f"Error reading sector CSV: {e}")
        else:
            st.info("No sector CSV uploaded — sector filtering disabled.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Ticker input logic
        tickers = []
        include_other_income_global = False

        if input_mode == "Single Ticker":
            st.markdown('<div class="small-card" style="margin-top:10px;">', unsafe_allow_html=True)
            st.markdown("**Search & Select Ticker**", unsafe_allow_html=True)
            search = st.text_input("Search ticker (symbol or company)", placeholder="e.g. ICICIBANK, HDFCBANK").strip().upper()
            selected_display = ""
            if search:
                if ticker_options:
                    matches = [opt for opt in ticker_options if search in opt.upper()][:30]
                    selected_display = st.selectbox("Matches", options=[""] + matches, format_func=lambda x: x if x else "Type to search")
                    ticker_input = ticker_to_symbol.get(selected_display, "").strip() if selected_display else search
                else:
                    ticker_input = search
                tickers = [ticker_input] if ticker_input else []
            else:
                if ticker_options:
                    selected_display = st.selectbox("Recent / Popular", options=[""] + ticker_options[:8], format_func=lambda x: x if x else "Type to search")
                    ticker_input = ticker_to_symbol.get(selected_display, "").strip()
                    tickers = [ticker_input] if ticker_input else []
                else:
                    tickers = []

            if tickers and is_bank_or_finance(tickers[0]):
                include_other_income_global = st.checkbox("Include Other Income in OP (Banks/NBFC)", value=True)
                st.caption("For banks: OP = Operating Profit + Other Income")
            st.markdown('</div>', unsafe_allow_html=True)

        elif input_mode == "Multi-Ticker":
            st.markdown('<div class="small-card" style="margin-top:10px;">', unsafe_allow_html=True)
            st.markdown("**Multi-Ticker Input**", unsafe_allow_html=True)
            ticker_input_area = st.text_area("Enter tickers (comma / newline)", "ICICIBANK,AXISBANK,HDFCBANK", height=120).strip().upper()
            tickers = [t.strip() for t in re.split(r'[, \n]+', ticker_input_area) if t.strip()]
            include_other_income_global = st.checkbox("Include Other Income in OP for Banks", value=True)
            st.markdown('</div>', unsafe_allow_html=True)

        elif input_mode == "Upload CSV":
            st.markdown('<div class="small-card" style="margin-top:10px;">', unsafe_allow_html=True)
            st.markdown("**Upload CSV**", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload CSV with 'Symbol' column (optional 'Sector')", type="csv", key="upload_csv")
            if uploaded_file:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    df_upload.columns = [c.strip().title() for c in df_upload.columns]
                    if "Symbol" in df_upload.columns:
                        tickers = df_upload["Symbol"].astype(str).str.upper().str.strip().dropna().unique().tolist()
                        st.success(f"Loaded {len(tickers)} tickers from uploaded CSV")
                        if "Sector" in df_upload.columns:
                            tmp = df_upload[["Symbol","Sector"]].dropna().drop_duplicates().reset_index(drop=True)
                            if sector_df.empty:
                                sector_df = tmp
                            else:
                                sector_df = pd.concat([sector_df, tmp], ignore_index=True).drop_duplicates().reset_index(drop=True)
                            sector_list = sorted(sector_df["Sector"].unique().tolist())
                    else:
                        st.error("CSV must have a 'Symbol' column.")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
            include_other_income_global = st.checkbox("Include Other Income in OP for Banks", value=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # If sector filter active override tickers
        if filtered_tickers:
            tickers = filtered_tickers
            st.info(f"Overriding ticker list: using {len(tickers)} tickers from selected sector(s).")

        # Action buttons
        st.write("")  # spacing
        btn_cols = st.columns([1, 1, 1])
        with btn_cols[0]:
            if st.button("Clear Cache & Re-scrape All", key="clear_cache"):
                with st.spinner("Clearing cache..."):
                    for f in os.listdir(DATA_DIR):
                        try:
                            os.remove(os.path.join(DATA_DIR, f))
                        except Exception:
                            pass
                st.success("Cache cleared!")
                st.experimental_rerun()
        with btn_cols[1]:
            analyze_pressed = st.button("Analyze All", key="analyze_all")
        with btn_cols[2]:
            template_csv = "Symbol,Sector\nICICIBANK,Banking\nAXISBANK,Banking\nTCS,IT\n"
            st.download_button("Download Template CSV", data=template_csv, file_name="tickers_sector_template.csv", mime="text/csv")

        st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# ANALYZE action
# ───────────────────────────────────────────────
if 'analyze_pressed' in locals() and analyze_pressed:
    if not tickers:
        st.error("No tickers provided. Provide tickers via Single, Multi, Upload CSV, or Sector selection.")
        st.stop()
    tickers = [t.strip().upper() for t in tickers if t and isinstance(t, str)]
    tickers = sorted(list(dict.fromkeys(tickers)))
    progress = st.progress(0)
    results = []
    failed = []
    for i, t in enumerate(tickers):
        include_other = include_other_income_global and is_bank_or_finance(t)
        with st.spinner(f"Analyzing {t}..."):
            res = analyze_single_ticker(t, fy_start, force_scrape, debug=debug_mode, include_other_income=include_other)
            if res and "error" not in res:
                results.append(res)
            else:
                failed.append({"ticker": t, "error": res.get("error") if isinstance(res, dict) else "Unknown error"})
        progress.progress(int(((i + 1) / len(tickers)) * 100))
    progress.empty()

    if not results:
        st.error("No valid data found for selected tickers.")
    else:
        st.session_state.analysis_results = {r["ticker"]: r for r in results}
        summary_rows = []
        for r in results:
            summary_rows.append({
                "ticker": r["ticker"],
                "current_price": r["current_price"],
                "forecasted_price": r["forecasted_price"],
                "gain_pct": r["gain_pct"],
                "r2": r["r2"],
                "years_count": r["years_count"],
            })
        st.session_state.summary_df = pd.DataFrame(summary_rows)
        st.success(f"Analyzed {len(results)} tickers successfully. {len(failed)} failed.")

    if failed:
        st.markdown("### Failed / Skipped Tickers")
        df_fail = pd.DataFrame(failed)
        st.dataframe(df_fail, use_container_width=True)

# ───────────────────────────────────────────────
# DISPLAY RESULTS (full-width) — reads from session_state safely
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

    if selected and "analysis_results" in st.session_state and selected in st.session_state.analysis_results:
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
        st.markdown("## Red Stock Price and Blue Profit")
        # FUNDAMENTAL VS PRICE CHART (distinct colors)
        chart_df_p = res["merged_profit"][[f"{res['profit_label']} (Cr)", "Avg Stock Price"]].reset_index()
        chart_df_p.columns = ["Year", "Profit", "Price"]
        chart_long_p = chart_df_p.melt("Year", var_name="Metric", value_name="Value")
        base_p = alt.Chart(chart_long_p).encode(x=alt.X("Year:O", title="FY"))

        # Profit (purple) and Price (blue)
        line_profit = base_p.mark_line(color="#6a11cb", strokeWidth=3).transform_filter(alt.datum.Metric == "Profit").encode(
            y=alt.Y("Value:Q", title=f"{res['profit_label']} (Cr)")
        )
        line_price  = base_p.mark_line(color="#FF0000", strokeWidth=3).transform_filter(alt.datum.Metric == "Price").encode(
            y=alt.Y("Value:Q", title="Avg Price")
        )

        chart_p = alt.layer(line_profit, line_price).resolve_scale(y="independent").properties(width="container", height=380)
        st.altair_chart(chart_p, use_container_width=True)

        if not res["merged_sales"].empty:
            chart_df_s = res["merged_sales"].iloc[:, [0, 1]].reset_index()
            chart_df_s.columns = ["Year", "Sales", "Price"]
            chart_long_s = chart_df_s.melt("Year", var_name="Metric", value_name="Value")
            base_s = alt.Chart(chart_long_s).encode(x=alt.X("Year:O"))
            line_sales = base_s.mark_line(strokeWidth=3).transform_filter(alt.datum.Metric == "Sales").encode(y=alt.Y("Value:Q", title="Sales (Cr)"))
            line_price_s = base_s.mark_line(color="#FF0000", strokeWidth=3).transform_filter(alt.datum.Metric == "Price").encode(y=alt.Y("Value:Q", title="Avg Price"))
            chart_s = alt.layer(line_sales, line_price_s).resolve_scale(y="independent").properties(width="container", height=380)
            st.altair_chart(chart_s, use_container_width=True)

        # VALUATION INSIGHT
        ratio_col = f"Price/{res['profit_label'].split()[0]}"
        if ratio_col in res["merged_profit"].columns:
            avg_ratio = res["merged_profit"][ratio_col].mean()
            latest_ratio = res["merged_profit"][ratio_col].iloc[-1]
            change = ((latest_ratio - avg_ratio) / avg_ratio) * 100 if avg_ratio != 0 else 0.0
            status = "Overvalued" if change > 20 else "Undervalued" if change < -20 else "Fairly Valued"
            st.markdown(f"### Valuation Insight\n**Avg Ratio**: `{avg_ratio:.4f}` | **Latest**: `{latest_ratio:.4f}` → **{change:+.1f}%** → **{status}**")

        # HISTORICAL VALUATION CHECKER
        st.markdown("### Historical Valuation Checker")
        tab_op, tab_sales = st.tabs(["Operating Profit Model", "Sales Model"])

        with tab_op:
            years = sorted(res["historical_op"].keys())
            if years:
                selected_year = st.slider("Select FY Year (OP)", min_value=years[0], max_value=years[-1], value=years[-1], key="op_year")
                hf = res["historical_op"][selected_year]
                colA, colB, colC = st.columns(3)
                with colA: st.metric(f"FY {selected_year} {res['profit_label'].split()[0]}", f"Rs.{hf['OP']:,.0f} Cr")
                with colB: st.metric("Fair Price", f"Rs.{hf['Fair']:,.0f}")
                with colC: st.metric("Actual Price", f"Rs.{hf['Actual']:,.0f}", f"{hf['Misprice']:+.1f}%")
                st.markdown(f"**Status**: **{hf['Status']}**")
            else:
                st.info("No historical OP data available to inspect.")

            # FULL HISTORIC PRICE CHART (blue)
            price_series = res["price_series"]
            price_series.columns = ["Date", "Close"]
            price_series["Date"] = pd.to_datetime(price_series["Date"])

            chart_price = alt.Chart(price_series).mark_line(color="#3498db", strokeWidth=2).encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Close:Q", title="Close Price (Rs.)"),
                tooltip=["Date:T", "Close:Q"]
            ).properties(width="container", height=350, title="Historical Stock Price (Daily Close)").interactive()
            st.altair_chart(chart_price, use_container_width=True)

        if res["has_sales"]:
            with tab_sales:
                years_s = sorted(res["historical_sales"].keys())
                if years_s:
                    selected_year_s = st.slider("Select FY Year (Sales)", min_value=years_s[0], max_value=years_s[-1], value=years_s[-1], key="sales_year")
                    hf_s = res["historical_sales"][selected_year_s]
                    colA, colB, colC = st.columns(3)
                    with colA: st.metric(f"FY {selected_year_s} {res['sales_label'].split()[0]}", f"Rs.{hf_s['Sales']:,.0f} Cr")
                    with colB: st.metric("Fair Price", f"Rs.{hf_s['Fair']:,.0f}")
                    with colC: st.metric("Actual Price", f"Rs.{hf_s['Actual']:,.0f}", f"{hf_s['Misprice']:+.1f}%")
                    st.markdown(f"**Status**: **{hf_s['Status']}**")
                    st.caption(res["eq_sales"])
                else:
                    st.info("No historical Sales data available to inspect.")

                chart_price_s = alt.Chart(price_series).mark_line(color="#3498db", strokeWidth=2).encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Close:Q", title="Close Price (Rs.)"),
                    tooltip=["Date:T", "Close:Q"]
                ).properties(width="container", height=350, title="Historical Stock Price (Daily Close)").interactive()
                st.altair_chart(chart_price_s, use_container_width=True)

        # DOWNLOAD per ticker
        combined = res["merged_profit"].copy()
        if not res["merged_sales"].empty:
            combined = combined.join(res["merged_sales"].drop(columns="Avg Stock Price", errors="ignore"), how="left")
        csv = combined.to_csv().encode()
        st.download_button("Download Full Data (selected ticker)", data=csv, file_name=f"{selected}_analysis.csv", mime="text/csv")

    # EXPORT ALL - use st.session_state safely
    export_df = []
    results_from_state = []
    if "analysis_results" in st.session_state and isinstance(st.session_state.analysis_results, dict):
        results_from_state = list(st.session_state.analysis_results.values())

    if results_from_state:
        for r in results_from_state:
            export_df.append({
                "ticker": r["ticker"],
                "current_price": r["current_price"],
                "forecasted_price": r["forecasted_price"],
                "gain_pct": r["gain_pct"],
                "r2": r["r2"],
                "years_count": r["years_count"],
                "profit_label": r["profit_label"],
                "merged_profit_csv": r["merged_profit"].to_csv() if (r.get("merged_profit") is not None and not r["merged_profit"].empty) else "",
                "merged_sales_csv": r["merged_sales"].to_csv() if (r.get("merged_sales") is not None and not r["merged_sales"].empty) else ""
            })
        export_all_df = pd.DataFrame(export_df)
        csv_all = export_all_df.to_csv(index=False)
        st.download_button("Download All Results (batch)", data=csv_all, file_name=f"analysis_batch_{datetime.date.today()}.csv", mime="text/csv")
    else:
        st.info("No analysis results available to export.")

st.caption(f"Data cached in `{DATA_DIR}/` • Upload Sector CSV with `Symbol, Sector` to enable sector-based batch runs.")
