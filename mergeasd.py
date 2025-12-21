from pypdf import PdfMerger
import os

FOLDER = r"C:\Users\admin\Downloads\ASD-Report-12-12-2025\ASD-Report-12-12-2025"
OUTPUT = os.path.join(FOLDER, "MERGED_OUTPUT.pdf")

merger = PdfMerger()

for file in sorted(os.listdir(FOLDER)):
    if file.lower().endswith(".pdf"):
        merger.append(os.path.join(FOLDER, file))

merger.write(OUTPUT)
merger.close()

print("✅ PDFs merged successfully:", OUTPUT)
