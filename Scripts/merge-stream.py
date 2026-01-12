import streamlit as st
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ================= CONFIG =================
BASE_DIR = r"E:\Narendra\application\scripts"

INPUT_FOLDER = os.path.join(BASE_DIR, "data-a1")
CLEANED_FOLDER = os.path.join(BASE_DIR, "cleaned")

FINAL_OUTPUT = "MERGED_DRAFT_ROLL_CONTROL_CHART.xlsx"

ROWS_TO_DELETE = 10
TARGET_SHEET = "Draft Roll Control Chart"
USE_COLS = "B:F"
FIXED_COLUMNS = ["B", "C", "D", "E", "F"]
# =========================================

st.set_page_config(page_title="Excel A1 Merger", layout="centered")

st.title("📊 Draft Roll Control Chart – Excel Merger")

st.markdown("""
**This tool will:**
- Clean all Excel files (remove top 10 rows)
- Merge only **Draft Roll Control Chart**
- Keep valid data from **columns B–F**
""")

if not os.path.isdir(INPUT_FOLDER):
    st.error(f"❌ Input folder not found:\n{INPUT_FOLDER}")
    st.stop()

os.makedirs(CLEANED_FOLDER, exist_ok=True)

files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(".xlsx") and not f.startswith("~$")
]

if not files:
    st.warning("⚠ No Excel files found in data-a1 folder")
    st.stop()

st.success(f"✅ Found {len(files)} Excel files")

if st.button("🚀 Start Cleaning & Merge", use_container_width=True):

    progress = st.progress(0)
    status = st.empty()

    # ---------------- STEP 1 ----------------
    status.info("🔹 Step 1: Cleaning Excel files")

    for i, file in enumerate(files, start=1):
        in_path = os.path.join(INPUT_FOLDER, file)
        out_path = os.path.join(CLEANED_FOLDER, file)

        xls = pd.ExcelFile(in_path, engine="openpyxl")
        writer = pd.ExcelWriter(out_path, engine="openpyxl")

        for sheet in xls.sheet_names:
            df = pd.read_excel(
                in_path,
                sheet_name=sheet,
                header=None,
                engine="openpyxl"
            )

            df = df.iloc[ROWS_TO_DELETE:]
            df.reset_index(drop=True, inplace=True)

            df.to_excel(
                writer,
                sheet_name=sheet,
                index=False,
                header=False
            )

        writer.close()
        progress.progress(i / len(files))

    status.success("✅ Cleaning completed")

    # ---------------- STEP 2 ----------------
    status.info("🔹 Step 2: Merging Draft Roll Control Chart")

    merged_rows = []
    total_rows = 0

    cleaned_files = [
        f for f in os.listdir(CLEANED_FOLDER)
        if f.lower().endswith(".xlsx") and not f.startswith("~$")
    ]

    for file in cleaned_files:
        file_path = os.path.join(CLEANED_FOLDER, file)

        try:
            df = pd.read_excel(
                file_path,
                sheet_name=TARGET_SHEET,
                usecols=USE_COLS,
                header=None,
                engine="openpyxl"
            )
        except:
            continue

        df.columns = FIXED_COLUMNS
        df = df.dropna(how="all", subset=FIXED_COLUMNS)

        if not df.empty:
            merged_rows.append(df)
            total_rows += len(df)

    if not merged_rows:
        st.error("❌ No valid data found to merge")
        st.stop()

    final_df = pd.concat(merged_rows, ignore_index=True)
    final_df.to_excel(FINAL_OUTPUT, index=False)

    status.success("✅ Merge completed successfully")

    # ---------------- OUTPUT ----------------
    st.subheader("📈 Merge Summary")
    st.metric("Total Rows Merged", total_rows)
    st.metric("Total Files Processed", len(files))

    with open(FINAL_OUTPUT, "rb") as f:
        st.download_button(
            label="⬇ Download Final Excel",
            data=f,
            file_name=FINAL_OUTPUT,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
