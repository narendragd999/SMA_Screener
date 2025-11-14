import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
import time

st.set_page_config(page_title="Add Sector from Screener", layout="wide")

st.title("📊 Add Sector / Industry from Screener.in")
st.write("Upload a file containing stock tickers. The app will fetch the Sector/Industry from Screener.in automatically.")

# ---------------------------------------------------------
# FUNCTION TO FETCH INDUSTRY
# ---------------------------------------------------------
def get_industry_from_screener(ticker: str) -> str:
    urls = [
        f"https://www.screener.in/company/{ticker}/consolidated/",
        f"https://www.screener.in/company/{ticker}/"
    ]
    industry = ""

    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        # Method 1: Look for table rows with Industry/Sector
        for th in soup.find_all("th"):
            text = th.get_text(strip=True).lower()
            if text in ["industry", "sector"]:
                td = th.find_next_sibling("td")
                if td:
                    return td.get_text(strip=True)

        # Method 2: From Peer Comparison Block
        peer = soup.find(lambda tag: tag.name in ["h2", "h3"] and "Peer comparison" in tag.get_text())
        if peer:
            link = peer.find_next("a", href=True)
            if link:
                return link.get_text(strip=True)

    return ""


# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------
uploaded = st.file_uploader("Upload Excel/CSV with stock tickers", type=["xlsx", "csv"])

if uploaded:
    # Read file
    ext = uploaded.name.split(".")[-1].lower()
    if ext == "xlsx":
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    st.write("### Preview of Uploaded File:")
    st.dataframe(df.head())

    ticker_col = st.selectbox("Select the column that contains stock tickers", df.columns)

    if st.button("🚀 Start Fetching Sectors from Screener"):
        sectors = []
        progress = st.progress(0)
        status = st.empty()

        for i, ticker in enumerate(df[ticker_col].astype(str)):
            ticker = ticker.strip().upper()

            status.write(f"Fetching Sector for **{ticker}** ({i+1}/{len(df)})")
            sector = get_industry_from_screener(ticker)
            sectors.append(sector)

            progress.progress((i+1) / len(df))
            time.sleep(1)  # slow down to avoid blocking

        df["Sector"] = sectors

        st.success("Completed! 🎉 Here is the updated data:")
        st.dataframe(df.head())

        # Download Output
        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)

        st.download_button(
            label="⬇ Download Updated Excel",
            data=output,
            file_name="tickers_with_sector.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
