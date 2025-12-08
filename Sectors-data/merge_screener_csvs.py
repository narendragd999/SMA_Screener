import pandas as pd
import glob
import csv

# Required output columns
REQUIRED_COLS = ["Sector", "Company", "Ticker", "S.No.", "CMP", "MCAP"]

INPUT_PATH = "./*.csv"
OUTPUT_FILE = "ALL_SECTORS_MERGED.csv"


def load_clean_csv(filepath):
    """
    1. Remove junk rows (rows 2–12)
    2. Find real header row (first containing CMP)
    3. Extract required columns
    4. MCAP always from column index 8 (I column in CSV)
    """

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Remove rows 1–11 (junk)
    if len(lines) > 12:
        lines = lines[0:1] + lines[12:]

    # Detect real header row
    header_index = None
    for i, line in enumerate(lines):
        if "CMP" in line:
            header_index = i
            break

    if header_index is None:
        print(f"⚠ Could not find valid header in: {filepath}")
        return pd.DataFrame()

    # Extract header
    header = [h.strip() for h in lines[header_index].split(",")]

    # Map required column names
    col_idx = {
        "Sector": header.index("Sector") if "Sector" in header else None,
        "Company": header.index("Company") if "Company" in header else None,
        "Ticker": header.index("Ticker") if "Ticker" in header else None,
        "S.No.": header.index("S.No.") if "S.No." in header else None,
        "CMP": header.index("CMP") if "CMP" in header else None,
        "MCAP": 8   # ALWAYS column I
    }

    cleaned_rows = []

    for line in lines[header_index + 1:]:
        row_raw = next(csv.reader([line]))
        row_raw = [x.strip() for x in row_raw]

        # Build cleaned output row
        row = {
            col: (row_raw[col_idx[col]] if col_idx[col] is not None and col_idx[col] < len(row_raw) else "")
            for col in REQUIRED_COLS
        }

        cleaned_rows.append(row)

    return pd.DataFrame(cleaned_rows)


# ---------------------------
# MERGE ALL FILES
# ---------------------------
files = glob.glob(INPUT_PATH)
print("Found", len(files), "files")

merged = pd.DataFrame(columns=REQUIRED_COLS)

for f in files:
    print("Processing:", f)
    df = load_clean_csv(f)
    merged = pd.concat([merged, df], ignore_index=True)

merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\n🎉 DONE — Merged CSV saved as:", OUTPUT_FILE)
print("Total rows:", len(merged))
