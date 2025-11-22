import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
import time

st.set_page_config(page_title="Add Sector and Industry from Screener", layout="wide")
st.title("📊 Add Sector / Industry from Screener.in")
st.write("Upload a file containing stock tickers. The app will fetch both Sector and Industry from Screener.in automatically.")

# ---------------------------------------------------------
# FUNCTION TO FETCH SECTOR AND INDUSTRY
# ---------------------------------------------------------
def get_sector_industry_from_screener(ticker: str) -> tuple[str, str]:
    urls = [
        f"https://www.screener.in/company/{ticker}/consolidated/",
        f"https://www.screener.in/company/{ticker}/"
    ]
    sector = ""
    industry = ""
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        # Method 1: Look for table rows with Sector/Industry
        for th in soup.find_all("th"):
            text = th.get_text(strip=True).lower()
            td = th.find_next_sibling("td")
            if td:
                td_text = td.get_text(strip=True)
                if "sector" in text:
                    sector = td_text
                if "industry" in text:
                    industry = td_text
                if sector and industry:
                    return sector, industry
        # Method 2: From Peer Comparison Block (often gives Industry)
        peer = soup.find(lambda tag: tag.name in ["h2", "h3"] and "Peer comparison" in tag.get_text())
        if peer:
            link = peer.find_next("a", href=True)
            if link and not industry:
                industry = link.get_text(strip=True)
        # Fallback: Search for common keywords in overview
        overview = soup.find("div", class_="companyOverview") or soup.body
        if overview:
            text = overview.get_text().lower()
            if not sector:
                if "oil" in text and "gas" in text:
                    sector = "Oil & Gas"
                elif "bank" in text:
                    sector = "Financial Services"
                elif "telecom" in text or "airtel" in text:
                    sector = "Telecommunications"
                elif "it" in text:
                    sector = "Information Technology"
            if not industry:
                if "oil" in text and "chemical" in text:
                    industry = "OIL-TO-CHEMICALS"
                elif "bank" in text:
                    industry = "Banks - Private Sector"
                elif "telecom" in text:
                    industry = "Telecom - Cellular & Fixed Line"
                elif "it" in text and "services" in text:
                    industry = "IT Services & Consulting"
        # Additional fallback: Title hints
        title = soup.find("title")
        if title:
            title_text = title.get_text().lower()
            if not sector:
                if "bank" in title_text:
                    sector = "Financial Services"
                if "it services" in title_text:
                    sector = "Information Technology"
            if not industry:
                if "bank" in title_text:
                    industry = "Banks - Private Sector"
    return sector, industry

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
    if st.button("🚀 Start Fetching Sectors and Industries from Screener"):
        sectors = []
        industries = []
        progress = st.progress(0)
        status = st.empty()
        for i, ticker in enumerate(df[ticker_col].astype(str)):
            ticker = ticker.strip().upper()
            status.write(f"Fetching Sector/Industry for **{ticker}** ({i+1}/{len(df)})")
            sec, ind = get_sector_industry_from_screener(ticker)
            sectors.append(sec)
            industries.append(ind)
            progress.progress((i+1) / len(df))
            time.sleep(1)  # slow down to avoid blocking
        df["Sector"] = sectors
        df["Industry"] = industries
        st.success("Completed! 🎉 Here is the updated data:")
        st.dataframe(df.head())
        # Download Output
        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        st.download_button(
            label="⬇ Download Updated Excel",
            data=output,
            file_name="tickers_with_sector_industry.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )