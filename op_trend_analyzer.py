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
st.title("Operating Profit & Sales vs Stock-Price Trend Analyzer")

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


def find_row(df: pd.DataFrame, name: str, threshold: float = 0.7) -> str | None:
    candidates = {
        "Operating Profit": ["Operating Profit", "OP", "EBIT", "EBITDA", "Operating Income"],
        "Sales": ["Sales", "Revenue", "Net Sales", "Total Income", "Income"]
    }[name]
    idx_lower = [i.lower().strip() for i in df.index]
    for cand in candidates:
        for idx in df.index:
            if cand.lower() in idx.lower():
                return idx
    matches = difflib.get_close_matches(name.lower(), idx_lower, n=1, cutoff=threshold)
    if matches:
        return df.index[idx_lower.index(matches[0])]
    return None


# ───────────────────────────────────────────────
# Sidebar
# ───────────────────────────────────────────────
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input(
    "Enter NSE Ticker",
    value="ITC",
    help="e.g. RELIANCE, TCS, INFY, HDFCBANK"
).strip().upper()

fy_start = st.sidebar.number_input(
    "Start Fiscal Year (Apr–Mar)",
    min_value=2000,
    max_value=datetime.date.today().year,
    value=2014,
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
        st.info("Try: RELIANCE, TCS, INFY, HDFCBANK")
        st.stop()

    # Extract Operating Profit
    op_row = find_row(pl_df, "Operating Profit")
    if not op_row:
        st.error("Could not locate 'Operating Profit' row.")
        st.stop()

    # Extract Sales
    sales_row = find_row(pl_df, "Sales")
    if not sales_row:
        st.warning("Could not locate 'Sales' row. Skipping Sales chart.")

    # ───── Step 2: Parse OP & Sales (skip TTM) ─────
    def extract_metric(row_name):
        if row_name is None:
            return pd.Series()
        series = pl_df.loc[row_name].iloc[1:]
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

    op_clean = extract_metric(op_row)
    sales_clean = extract_metric(sales_row) if sales_row else pd.Series()

    if op_clean.empty:
        st.error("No valid Operating Profit data.")
        st.stop()

    st.success(f"Fetched Operating Profit for **{len(op_clean)}** years.")

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

    # Assign FY correctly: Apr–Dec → current FY, Jan–Mar → previous FY
    price_df = price_df.copy()
    price_df["FY_Year"] = price_df.index.year
    price_df.loc[price_df.index.month <= 3, "FY_Year"] -= 1

    avg_price = price_df.groupby("FY_Year")["Close"].mean().round(2)

    # ───── Step 4: Align years (OP & Price) ─────
    common_years_op = op_clean.index.intersection(avg_price.index)
    if len(common_years_op) == 0:
        st.error("No overlapping fiscal years between OP and price.")
        st.stop()

    op_clean = op_clean.loc[common_years_op]
    avg_price = avg_price.loc[common_years_op]

    # Align Sales if available
    if not sales_clean.empty:
        common_years_sales = sales_clean.index.intersection(avg_price.index)
        if len(common_years_sales) > 0:
            sales_clean = sales_clean.loc[common_years_sales]
            avg_price_sales = avg_price.loc[common_years_sales]
        else:
            sales_clean = pd.Series()
            st.info("Sales data not aligned with price years.")

    # ───── Step 5: Build final tables ─────
    # OP Table
    op_1d = np.array(op_clean).flatten()
    price_1d = np.array(avg_price).flatten()
    ratio_op_1d = np.round(price_1d / op_1d, 4)

    merged_op = pd.DataFrame(
        {
            "Operating Profit (₹ Cr)": op_1d,
            "Avg Stock Price (₹)": price_1d,
            "Price / OP Ratio": ratio_op_1d,
        },
        index=common_years_op.astype(int),
    ).sort_index()

    # Sales Table (if available)
    if not sales_clean.empty:
        sales_1d = np.array(sales_clean).flatten()
        price_sales_1d = np.array(avg_price_sales).flatten()
        ratio_sales_1d = np.round(price_sales_1d / sales_1d, 4)

        merged_sales = pd.DataFrame(
            {
                "Sales (₹ Cr)": sales_1d,
                "Avg Stock Price (₹)": price_sales_1d,
                "Price / Sales Ratio": ratio_sales_1d,
            },
            index=common_years_sales.astype(int),
        ).sort_index()
    else:
        merged_sales = pd.DataFrame()

    # ───── DISPLAY: Dual-Axis Line Chart - OP vs Price ─────
    st.subheader("Operating Profit vs Average Stock Price")
    chart_df_op = merged_op[["Operating Profit (₹ Cr)", "Avg Stock Price (₹)"]].reset_index()
    chart_df_op.columns = ["Year", "Operating Profit (₹ Cr)", "Avg Stock Price (₹)"]
    chart_long_op = chart_df_op.melt("Year", var_name="Metric", value_name="Value")

    base_op = alt.Chart(chart_long_op).encode(x=alt.X("Year:O", title="Fiscal Year (Apr–Mar)"))
    line_op = base_op.mark_line(color="#1f77b4", strokeWidth=3).transform_filter(
        alt.datum.Metric == "Operating Profit (₹ Cr)"
    ).encode(y=alt.Y("Value:Q", title="Operating Profit (₹ Cr)", scale=alt.Scale(zero=False)))
    line_price_op = base_op.mark_line(color="#ff7f0e", strokeWidth=3).transform_filter(
        alt.datum.Metric == "Avg Stock Price (₹)"
    ).encode(y=alt.Y("Value:Q", title="Avg Stock Price (₹)", scale=alt.Scale(zero=False)))

    chart_op = alt.layer(line_op, line_price_op).resolve_scale(y="independent").properties(
        width=700, height=400
    )
    st.altair_chart(chart_op, use_container_width=True)

    st.markdown(
        """
        <div style="display: flex; justify-content: center; gap: 40px; margin-top: 10px; font-size: 14px;">
            <span><strong style="color: #1f77b4;">███</strong> Operating Profit (₹ Cr)</span>
            <span><strong style="color: #ff7f0e;">███</strong> Avg Stock Price (₹)</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ───── NEW CHART: Sales vs Price (if data exists) ─────
    if not merged_sales.empty:
        st.subheader("Sales vs Average Stock Price")
        chart_df_sales = merged_sales[["Sales (₹ Cr)", "Avg Stock Price (₹)"]].reset_index()
        chart_df_sales.columns = ["Year", "Sales (₹ Cr)", "Avg Stock Price (₹)"]
        chart_long_sales = chart_df_sales.melt("Year", var_name="Metric", value_name="Value")

        base_sales = alt.Chart(chart_long_sales).encode(x=alt.X("Year:O", title="Fiscal Year (Apr–Mar)"))
        line_sales = base_sales.mark_line(color="#2ca02c", strokeWidth=3).transform_filter(
            alt.datum.Metric == "Sales (₹ Cr)"
        ).encode(y=alt.Y("Value:Q", title="Sales (₹ Cr)", scale=alt.Scale(zero=False)))
        line_price_sales = base_sales.mark_line(color="#ff7f0e", strokeWidth=3).transform_filter(
            alt.datum.Metric == "Avg Stock Price (₹)"
        ).encode(y=alt.Y("Value:Q", title="Avg Stock Price (₹)", scale=alt.Scale(zero=False)))

        chart_sales = alt.layer(line_sales, line_price_sales).resolve_scale(y="independent").properties(
            width=700, height=400
        )
        st.altair_chart(chart_sales, use_container_width=True)

        st.markdown(
            """
            <div style="display: flex; justify-content: center; gap: 40px; margin-top: 10px; font-size: 14px;">
                <span><strong style="color: #2ca02c;">███</strong> Sales (₹ Cr)</span>
                <span><strong style="color: #ff7f0e;">███</strong> Avg Stock Price (₹)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Valuation Multiple: Price / Sales
        st.subheader("Valuation Multiple: Price / Sales")
        bar_sales_df = merged_sales[["Price / Sales Ratio"]].reset_index()
        bar_sales_df.columns = ["Year", "Ratio"]
        bar_sales = (
            alt.Chart(bar_sales_df)
            .mark_bar(color="#9467bd", size=40)
            .encode(
                x=alt.X("Year:O", title="Fiscal Year"),
                y=alt.Y("Ratio:Q", title="Price / Sales Ratio"),
                tooltip=["Year", alt.Tooltip("Ratio", format=".4f")],
            )
            .properties(width=700, height=300)
        )
        st.altair_chart(bar_sales, use_container_width=True)

    # ───── Valuation Multiple: Price / OP ─────
    st.subheader("Valuation Multiple: Price / Operating Profit")
    bar_op_df = merged_op[["Price / OP Ratio"]].reset_index()
    bar_op_df.columns = ["Year", "Ratio"]
    bar_op = (
        alt.Chart(bar_op_df)
        .mark_bar(color="#d62728", size=40)
        .encode(
            x=alt.X("Year:O", title="Fiscal Year"),
            y=alt.Y("Ratio:Q", title="Price / OP Ratio"),
            tooltip=["Year", alt.Tooltip("Ratio", format=".4f")],
        )
        .properties(width=700, height=300)
    )
    st.altair_chart(bar_op, use_container_width=True)

    # ───── Download Button (Combined) ─────
    combined_df = merged_op.copy()
    if not merged_sales.empty:
        combined_df = combined_df.join(merged_sales.drop(columns="Avg Stock Price (₹)", errors="ignore"), how="left")
    csv = combined_df.to_csv().encode()
    st.download_button(
        "Download Data as CSV",
        data=csv,
        file_name=f"{ticker}_op_sales_vs_price.csv",
        mime="text/csv",
    )

    # ───── Forecast: Next-Year Avg Price (using OP) ─────
    st.subheader("Forecast – Next-Year Average Stock Price (using OP)")
    if len(merged_op) >= 2:
        X = merged_op["Operating Profit (₹ Cr)"].values.reshape(-1, 1)
        y = merged_op["Avg Stock Price (₹)"].values
        model = LinearRegression().fit(X, y)
        latest_op = merged_op["Operating Profit (₹ Cr)"].iloc[-1]
        pred_price = float(model.predict([[latest_op]])[0])
        pred_price = round(pred_price, 2)
        b1 = round(model.coef_[0], 6)
        b0 = round(model.intercept_, 2)
        eq = f"Price = {b0} + {b1} × OP"
        next_fy = merged_op.index[-1] + 1
        st.markdown(
            f"""
            **Linear Regression Model**  
            `{eq}`  
            **Current OP** (FY {merged_op.index[-1]}): **{latest_op:,.0f}** ₹ Cr  
            **Forecasted Avg Price (FY {next_fy})** → **₹{pred_price:,.2f}**  
            <small>*Based on historical Price-OP relationship. Use with caution.*</small>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("Need at least 2 years of data for forecast.")

    # ───── Key Insights (OP-based) ─────
    avg_ratio_op = merged_op["Price / OP Ratio"].mean()
    latest_ratio_op = merged_op["Price / OP Ratio"].iloc[-1]
    change_op = ((latest_ratio_op - avg_ratio_op) / avg_ratio_op) * 100
    status_op = "Overvalued" if change_op > 20 else "Undervalued" if change_op < -20 else "Fairly Valued"

    st.markdown(
        f"""
        ### Key Insights (Price / OP)
        - **Average Ratio**: `{avg_ratio_op:.4f}`
        - **Latest Ratio (FY {merged_op.index[-1]})**: `{latest_ratio_op:.4f}`
        - **Change vs Average**: **{change_op:+.2f}%** → **{status_op}**
        """,
        unsafe_allow_html=True
    )

    if not merged_sales.empty:
        avg_ratio_sales = merged_sales["Price / Sales Ratio"].mean()
        latest_ratio_sales = merged_sales["Price / Sales Ratio"].iloc[-1]
        change_sales = ((latest_ratio_sales - avg_ratio_sales) / avg_ratio_sales) * 100
        status_sales = "Overvalued" if change_sales > 20 else "Undervalued" if change_sales < -20 else "Fairly Valued"
        st.markdown(
            f"""
            ### Key Insights (Price / Sales)
            - **Average Ratio**: `{avg_ratio_sales:.4f}`
            - **Latest Ratio (FY {merged_sales.index[-1]})**: `{latest_ratio_sales:.4f}`
            - **Change vs Average**: **{change_sales:+.2f}%** → **{status_sales}**
            """,
            unsafe_allow_html=True
        )

    st.caption("Data: Screener.in | Yahoo Finance | FY = Apr–Mar")