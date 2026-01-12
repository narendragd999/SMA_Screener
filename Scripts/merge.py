import pandas as pd
import os
import sys
import warnings

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

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- STEP 1: CLEAN FILES ----------------
print("\n🔹 STEP 1: Cleaning Excel files (removing top 10 rows)")

if not os.path.isdir(INPUT_FOLDER):
    print("❌ Input folder not found:", INPUT_FOLDER)
    sys.exit(1)

os.makedirs(CLEANED_FOLDER, exist_ok=True)

files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(".xlsx") and not f.startswith("~$")
]

if not files:
    print("❌ No Excel files found!")
    sys.exit(1)

for file in files:
    in_path = os.path.join(INPUT_FOLDER, file)
    out_path = os.path.join(CLEANED_FOLDER, file)

    print(f"➡ Cleaning: {file}")

    xls = pd.ExcelFile(in_path, engine="openpyxl")
    writer = pd.ExcelWriter(out_path, engine="openpyxl")

    for sheet in xls.sheet_names:
        df = pd.read_excel(
            in_path,
            sheet_name=sheet,
            header=None,
            engine="openpyxl"
        )

        df = df.iloc[ROWS_TO_DELETE:]  # delete top 10 rows
        df.reset_index(drop=True, inplace=True)

        df.to_excel(
            writer,
            sheet_name=sheet,
            index=False,
            header=False
        )

    writer.close()

print("✅ Cleaning completed")

# ---------------- STEP 2: MERGE DRAFT ROLL CONTROL CHART ----------------
print("\n🔹 STEP 2: Merging Draft Roll Control Chart")

merged_rows = []
total_rows = 0

cleaned_files = [
    f for f in os.listdir(CLEANED_FOLDER)
    if f.lower().endswith(".xlsx") and not f.startswith("~$")
]

for file in cleaned_files:
    file_path = os.path.join(CLEANED_FOLDER, file)
    print(f"➡ Merging from: {file}")

    try:
        df = pd.read_excel(
            file_path,
            sheet_name=TARGET_SHEET,
            usecols=USE_COLS,
            header=None,
            engine="openpyxl"
        )
    except Exception as e:
        print("   ⚠ Sheet missing, skipped")
        continue

    df.columns = FIXED_COLUMNS

    # Keep only rows where B–F has data
    df = df.dropna(how="all", subset=FIXED_COLUMNS)

    if df.empty:
        print("   ⚠ No valid rows")
        continue

    rows = len(df)
    total_rows += rows
    print(f"   ✔ Rows added: {rows}")

    merged_rows.append(df)

# ---------------- FINAL OUTPUT ----------------
if not merged_rows:
    print("\n❌ No data merged!")
    sys.exit(1)

final_df = pd.concat(merged_rows, ignore_index=True)
final_df.to_excel(FINAL_OUTPUT, index=False)

print("\n✅ FINAL MERGE COMPLETED SUCCESSFULLY")
print("📄 Output File:", FINAL_OUTPUT)
print("📊 Total Rows Merged:", total_rows)
