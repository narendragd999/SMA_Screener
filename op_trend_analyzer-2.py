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
# Page Config
# ───────────────────────────────────────────────
st.set_page_config(page_title="OP & Sales vs Stock-Price Analyzer", layout="centered")
st.title("Operating / Financing Profit & Sales vs Stock-Price Trend Analyzer")

# ───────────────────────────────────────────────
# Utility Functions
# ───────────────────────────────────────────────
@st.cache_data(ttl=3600)
def scrape_screener_data(ticker: str) -> pd.DataFrame | None:
    ticker = ticker.strip().upper()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    urls = [
        f"https://www.screener.in/company/{ticker}/consolidated/",
        f"https://www.screener.in/company/{ticker}/"
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=15)
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
        except Exception:
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
    except Exception:
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
    """
    Returns (row_index, display_name)
    """
    candidates = {
        "profit": [
            ("Operating Profit", "Operating Profit"),
            ("OP", "Operating Profit"),
            ("EBIT", "Operating Profit"),
            ("EBITDA", "Operating Profit"),
            ("Operating Income", "Operating Profit"),
            ("Financing Profit", "Financing Profit"),
            ("Finance Profit", "Financing Profit"),
            ("Interest Income", "Financing Profit"),
            ("Net Interest Income", "Financing Profit"),
        ],
        "sales": [
            ("Sales", "Sales"),
            ("Revenue", "Sales"),
            ("Net Sales", "Sales"),
            ("Total Income", "Sales"),
            ("Income", "Sales"),
            ("Interest Earned", "Interest Earned"),
        ]
    }

    target = name.lower()
    idx_lower = [i.lower().strip() for i in df.index]
    search_list = candidates["profit"] if target == "profit" else candidates["sales"]

    for keyword, display in search_list:
        for idx in df.index:
            if keyword.lower() in idx.lower():
                return idx, display

    # Fuzzy match fallback
    matches = difflib.get_close_matches(target, idx_lower, n=1, cutoff=threshold)
    if matches:
        matched_idx = df.index[idx_lower.index(matches[0])]
        if target == "profit":
            return matched_idx, "Operating/Financing Profit"
        else:
            return matched_idx, "Sales"

    return None, "Unknown"


# ───────────────────────────────────────────────
# Sidebar
# ───────────────────────────────────────────────
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input(
    "Enter NSE Ticker",
    value="BAJAJHF",
    help="e.g. RELIANCE, TCS, BAJAJHF, HDFCBANK"
).strip().upper()

fy_start = st.sidebar.number_input(
    "Start Fiscal Year (Apr–Mar)",
    min_value=2000,
    max_value=datetime.date.today().year,
    value=2020,
    help="Data from April of this year will be included"
)

if st.sidebar.button("Analyze Trend"):
    if not ticker:
        st.error("Please enter a ticker.")
        st.stop()

    # ───── Step 1: Scrape P&L ─────
    with st.spinner(f"Fetching financials for **{ticker}**..."):
        pl_df = scrape_screener_data(ticker)

    if pl_df is None or pl_df.empty:
        st.error(f"Could not fetch data for **{ticker}**.")
        st.info("Try: RELIANCE, TCS, BAJAJHF, HDFCBANK")
        st.stop()

    # ───── Extract Profit Row (Operating OR Financing) ─────
    profit_row_idx, profit_label = find_row(pl_df, "profit")
    if not profit_row_idx:
        st.error("Could not locate 'Operating Profit' or 'Financing Profit' row.")
        st.stop()

    # Extract Sales
    sales_row_idx, sales_label = find_row(pl_df, "sales")
    if not sales_row_idx:
        st.warning("Could not locate 'Sales' or 'Revenue' row. Skipping Sales chart.")

    # ───── Step 2: Parse Profit & Sales (skip TTM) ─────
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
            except ValueError:
                continue
        if not years:
            return pd.Series()
        clean = clean_numeric(pd.Series(vals, index=years))
        return clean[clean != 0].dropna()

    profit_clean = extract_metric(profit_row_idx)
    sales_clean = extract_metric(sales_row_idx) if sales_row_idx else pd.Series()

    if profit_clean.empty:
        st.error(f"No valid {profit_label} data.")
        st.stop()

    st.success(f"Fetched **{profit_label}** for **{len(profit_clean)}** years.")

    # ───── Step 3: Fetch price (Apr–Mar FY) ─────
    with st.spinner(f"Downloading **{ticker}.NS** prices..."):
        try:
            start_date = f"{fy_start}-04-01"
            end_date = f"{datetime.date.today().year + 1}-03-31"
            price_df = yf.download(
                f"{ticker}.NS",
                start=start_date,
                end=end_date,
                progress=False,
            )
        except Exception as e:
            st.error(f"Yahoo Finance error: {e}")
            st.stop()

    if price_df.empty:
        st.error(f"No price data for **{ticker}.NS**.")
        st.stop()

    price_df = price_df.copy()
    price_df["FY_Year"] = price_df.index.year
    price_df.loc[price_df.index.month <= 3, "FY_Year"] -= 1
    avg_price = price_df.groupby("FY_Year")["Close"].mean().round(2)

    # ───── Step 4: Align years ─────
    common_years_profit = profit_clean.index.intersection(avg_price.index)
    if len(common_years_profit) == 0:
        st.error("No overlapping fiscal years between profit and price.")
        st.stop()

    profit_clean = profit_clean.loc[common_years_profit]
    avg_price_profit = avg_price.loc[common_years_profit]

    if not sales_clean.empty:
        common_years_sales = sales_clean.index.intersection(avg_price.index)
        if len(common_years_sales) > 0:
            sales_clean = sales_clean.loc[common_years_sales]
            avg_price_sales = avg_price.loc[common_years_sales]
        else:
            sales_clean = pd.Series()
            st.info("Sales data not aligned with price years.")
    else:
        avg_price_sales = pd.Series()

    # ───── Step 5: Build final tables ─────
    # Profit Table
    profit_1d = np.array(profit_clean).flatten()
    price_1d = np.array(avg_price_profit).flatten()
    ratio_profit_1d = np.round(price_1d / profit_1d, 4)

    merged_profit = pd.DataFrame(
        {
            f"{profit_label} (₹ Cr)": profit_1d,
            "Avg Stock Price (₹)": price_1d,
            f"Price / {profit_label.split()[0]} Ratio": ratio_profit_1d,
        },
        index=common_years_profit.astype(int),
    ).sort_index()

    # Sales Table
    merged_sales = pd.DataFrame()
    if not sales_clean.empty:
        sales_1d = np.array(sales_clean).flatten()
        price_sales_1d = np.array(avg_price_sales).flatten()
        ratio_sales_1d = np.round(price_sales_1d / sales_1d, 4)

        merged_sales = pd.DataFrame(
            {
                f"{sales_label} (₹ Cr)": sales_1d,
                "Avg Stock Price (₹)": price_sales_1d,
                f"Price / {sales_label.split()[0]} Ratio": ratio_sales_1d,
            },
            index=common_years_sales.astype(int),
        ).sort_index()

    # ───── CHART 1: Profit vs Price ─────
    st.subheader(f"{profit_label} vs Average Stock Price")
    chart_df_p = merged_profit[[f"{profit_label} (₹ Cr)", "Avg Stock Price (₹)"]].reset_index()
    chart_df_p.columns = ["Year", "Profit", "Price"]
    chart_long_p = chart_df_p.melt("Year", var_name="Metric", value_name="Value")

    base_p = alt.Chart(chart_long_p).encode(x=alt.X("Year:O", title="Fiscal Year (Apr–Mar)"))
    line_profit = base_p.mark_line(color="#1f77b4", strokeWidth=3).transform_filter(
        alt.datum.Metric == "Profit"
    ).encode(y=alt.Y("Value:Q", title=f"{profit_label} (₹ Cr)", scale=alt.Scale(zero=False)))
    line_price_p = base_p.mark_line(color="#ff7f0e", strokeWidth=3).transform_filter(
        alt.datum.Metric == "Price"
    ).encode(y=alt.Y("Value:Q", title="Avg Stock Price (₹)", scale=alt.Scale(zero=False)))

    chart_p = alt.layer(line_profit, line_price_p).resolve_scale(y="independent").properties(
        width=700, height=400
    )
    st.altair_chart(chart_p, use_container_width=True)

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; gap: 40px; margin-top: 10px; font-size: 14px;">
            <span><strong style="color: #1f77b4;">███</strong> {profit_label} (₹ Cr)</span>
            <span><strong style="color: #ff7f0e;">███</strong> Avg Stock Price (₹)</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ───── CHART 2: Sales vs Price (if available) ─────
    if not merged_sales.empty:
        st.subheader(f"{sales_label} vs Average Stock Price")
        chart_df_s = merged_sales[[f"{sales_label} (₹ Cr)", "Avg Stock Price (₹)"]].reset_index()
        chart_df_s.columns = ["Year", "Sales", "Price"]
        chart_long_s = chart_df_s.melt("Year", var_name="Metric", value_name="Value")

        base_s = alt.Chart(chart_long_s).encode(x=alt.X("Year:O", title="Fiscal Year (Apr–Mar)"))
        line_sales = base_s.mark_line(color="#2ca02c", strokeWidth=3).transform_filter(
            alt.datum.Metric == "Sales"
        ).encode(y=alt.Y("Value:Q", title=f"{sales_label} (₹ Cr)", scale=alt.Scale(zero=False)))
        line_price_s = base_s.mark_line(color="#ff7f0e", strokeWidth=3).transform_filter(
            alt.datum.Metric == "Price"
        ).encode(y=alt.Y("Value:Q", title="Avg Stock Price (₹)", scale=alt.Scale(zero=False)))

        chart_s = alt.layer(line_sales, line_price_s).resolve_scale(y="independent").properties(
            width=700, height=400
        )
        st.altair_chart(chart_s, use_container_width=True)

        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; gap: 40px; margin-top: 10px; font-size: 14px;">
                <span><strong style="color: #2ca02c;">███</strong> {sales_label} (₹ Cr)</span>
                <span><strong style="color: #ff7f0e;">███</strong> Avg Stock Price (₹)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Price / Sales Bar
        st.subheader(f"Valuation Multiple: Price / {sales_label.split()[0]}")
        bar_s_df = merged_sales[[f"Price / {sales_label.split()[0]} Ratio"]].reset_index()
        bar_s_df.columns = ["Year", "Ratio"]
        bar_s = alt.Chart(bar_s_df).mark_bar(color="#9467bd", size=40).encode(
            x=alt.X("Year:O"), y=alt.Y("Ratio:Q"), tooltip=["Year", alt.Tooltip("Ratio", format=".4f")]
        ).properties(width=700, height=300)
        st.altair_chart(bar_s, use_container_width=True)

    # ───── Valuation Multiple: Price / Profit ─────
    st.subheader(f"Valuation Multiple: Price / {profit_label.split()[0]}")
    bar_p_df = merged_profit[[f"Price / {profit_label.split()[0]} Ratio"]].reset_index()
    bar_p_df.columns = ["Year", "Ratio"]
    bar_p = alt.Chart(bar_p_df).mark_bar(color="#d62728", size=40).encode(
        x=alt.X("Year:O"), y=alt.Y("Ratio:Q"), tooltip=["Year", alt.Tooltip("Ratio", format=".4f")]
    ).properties(width=700, height=300)
    st.altair_chart(bar_p, use_container_width=True)

    # ───── Download CSV ─────
    combined = merged_profit.copy()
    if not merged_sales.empty:
        combined = combined.join(merged_sales.drop(columns="Avg Stock Price (₹)", errors="ignore"), how="left")
    csv = combined.to_csv().encode()
    st.download_button(
        "Download Data as CSV",
        data=csv,
        file_name=f"{ticker}_profit_sales_vs_price.csv",
        mime="text/csv",
    )

    # ───── Forecast using Profit ─────
    st.subheader(f"Forecast – Next-Year Avg Price (using {profit_label.split()[0]})")
    if len(merged_profit) >= 2:
        X = merged_profit[f"{profit_label} (₹ Cr)"].values.reshape(-1, 1)
        y = merged_profit["Avg Stock Price (₹)"].values
        model = LinearRegression().fit(X, y)
        latest_profit = merged_profit[f"{profit_label} (₹ Cr)"].iloc[-1]
        pred_price = round(float(model.predict([[latest_profit]])[0]), 2)
        b1 = round(model.coef_[0], 6)
        b0 = round(model.intercept_, 2)
        eq = f"Price = {b0} + {b1} × {profit_label.split()[0]}"
        next_fy = merged_profit.index[-1] + 1
        st.markdown(
            f"""
            **Model**: `{eq}`  
            **Latest {profit_label}** (FY {merged_profit.index[-1]}): **{latest_profit:,.0f}** ₹ Cr  
            **Forecasted Price (FY {next_fy})** → **₹{pred_price:,.2f}**  
            <small>*Use with caution. Linear model.*</small>
            """,
            unsafe_allow_html=True
        )

    # ───── Key Insights ─────
    ratio_col = f"Price / {profit_label.split()[0]} Ratio"
    avg_ratio = merged_profit[ratio_col].mean()
    latest_ratio = merged_profit[ratio_col].iloc[-1]
    change = ((latest_ratio - avg_ratio) / avg_ratio) * 100
    status = "Overvalued" if change > 20 else "Undervalued" if change < -20 else "Fairly Valued"

    st.markdown(
        f"""
        ### Key Insights ({profit_label.split()[0]} Basis)
        - **Avg Ratio**: `{avg_ratio:.4f}`
        - **Latest Ratio**: `{latest_ratio:.4f}`
        - **Change**: **{change:+.2f}%** → **{status}**
        """,
        unsafe_allow_html=True
    )

    st.caption("Data: Screener.in | Yahoo Finance | FY = Apr–Mar")