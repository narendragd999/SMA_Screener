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

# ───────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────
st.set_page_config(page_title="OP & Sales Analyzer", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #1f77b4;'>Operating / Financing Profit & Sales vs Stock-Price Trend Analyzer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size: 16px; color: #666;'>Data: Screener.in | Yahoo Finance | FY = Apr–Mar | Forecasts are linear models—use with caution.</p>",
    unsafe_allow_html=True
)

# ───────────────────────────────────────────────
# CACHING & SCRAPING
# ───────────────────────────────────────────────
@st.cache_data(ttl=3600)
def scrape_screener_data(ticker: str) -> pd.DataFrame | None:
    ticker = ticker.strip().upper()
    headers = {"User-Agent": "Mozilla/5.0"}
    urls = [
        f"https://www.screener.in/company/{ticker}/consolidated/",
        f"https://www.screener.in/company/{ticker}/"
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            section = soup.find("section", {"id": "profit-loss"})
            if not section:
                continue
            table = section.find("table", class_="data-table")
            if not table:
                continue
            df = parse_table(table)
            if df is not None and not df.empty:
                return df
        except:
            continue
    return None


def parse_table(table) -> pd.DataFrame | None:
    try:
        headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
        if len(headers) < 2:
            return None
        headers[0] = "Metric"
        rows = []
        for tr in table.find("tbody").find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) == len(headers):
                rows.append(cells)
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=headers)
        df.set_index("Metric", inplace=True)
        return df
    except:
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


def find_row(df: pd.DataFrame, name: str, threshold: float = 0.7) -> tuple[str | None, str]:
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
    matches = difflib.get_close_matches(target, idx_lower, n=1, cutoff=threshold)
    if matches:
        matched_idx = df.index[idx_lower.index(matches[0])]
        return matched_idx, "Operating/Financing Profit" if target == "profit" else "Sales"
    return None, "Unknown"


# ───────────────────────────────────────────────
# CORE ANALYSIS FUNCTION
# ───────────────────────────────────────────────
def analyze_single_ticker(ticker: str, fy_start: int):
    ticker = ticker.strip().upper()
    if not ticker:
        return {"ticker": ticker, "error": "Empty ticker"}

    pl_df = scrape_screener_data(ticker)
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

    # Fetch price
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

    # Align years
    common_years_profit = profit_clean.index.intersection(avg_price.index)
    if len(common_years_profit) < 2:
        return {"ticker": ticker, "error": "Need ≥2 years"}
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

    # Profit Table
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

    # Forecast
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
# SIDEBAR INPUT
# ───────────────────────────────────────────────
st.sidebar.markdown("<h2 style='color: #1f77b4;'>Input Mode</h2>", unsafe_allow_html=True)
input_mode = st.sidebar.radio("Choose:", ["Single Ticker", "Multi-Ticker", "Upload CSV"])

fy_start = st.sidebar.number_input(
    "Start FY (Apr–Mar)",
    min_value=2000,
    max_value=datetime.date.today().year,
    value=2015,
)

tickers = []

if input_mode == "Single Ticker":
    ticker_input = st.sidebar.text_input("Enter Ticker", "ITC").strip().upper()
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

# ───────────────────────────────────────────────
# ANALYZE BUTTON
# ───────────────────────────────────────────────
if st.sidebar.button("Analyze All", type="primary"):
    if not tickers:
        st.error("No tickers provided.")
        st.stop()

    with st.spinner("Analyzing..."):
        progress = st.progress(0)
        results = []
        for i, t in enumerate(tickers):
            res = analyze_single_ticker(t, fy_start)
            results.append(res)
            progress.progress((i + 1) / len(tickers))
        progress.empty()

    # Filter valid results
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        st.error("No valid data found.")
        st.stop()

    # Store in session state
    st.session_state.analysis_results = {r["ticker"]: r for r in valid_results}
    st.session_state.summary_df = pd.DataFrame(valid_results)
    st.success(f"Analyzed {len(valid_results)} tickers")

# ───────────────────────────────────────────────
# DISPLAY SUMMARY
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
                "current_price": "₹{:.2f}",
                "forecasted_price": "₹{:.2f}",
                "gain_pct": "{:+.1f}%",
            }
        ),
        use_container_width=True,
    )

    # Top Gainers
    high_gainers = df[df["gain_pct"] > 20].sort_values("gain_pct", ascending=False)
    if not high_gainers.empty:
        st.markdown("## Top Gainers (>20% Upside)")
        hg = high_gainers[
            ["ticker", "current_price", "forecasted_price", "gain_pct", "r2"]
        ].round(2)
        st.dataframe(
            hg.style.format(
                {
                    "current_price": "₹{:.2f}",
                    "forecasted_price": "₹{:.2f}",
                    "gain_pct": "{:+.1f}%",
                }
            ),
            use_container_width=True,
        )

    # FULL ANALYSIS
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
            st.metric("Current Price", f"₹{res['current_price']:,.2f}")
            st.metric(
                "Forecasted Price",
                f"₹{res['forecasted_price']:,.2f}",
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
        base_p = alt.Chart(chart_long_p).encode(x=alt.X("Year:O", title="FY (Apr–Mar)"))
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

        # Sales Chart
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

        # Download
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

st.caption("Built with Streamlit • Data: Screener.in & Yahoo Finance • FY = Apr–Mar")