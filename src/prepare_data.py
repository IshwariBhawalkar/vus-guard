"""Download the WHO TB mutation catalogue (2023, ODC-By) and save a slim CSV.

Source: https://github.com/GTB-tbsequencing/mutation-catalogue-2023  (Final Result Files/WHO-UCN-TB-2023.7-eng.xlsx)
Only these columns are kept: drug, gene, mutation, variant, tier, effect, FINAL CONFIDENCE GRADING, silent.
Run: python3 src/prepare_data.py
"""
import os
import subprocess

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "repo")
XLSX = os.path.join(RAW, "Final Result Files", "WHO-UCN-TB-2023.7-eng.xlsx")
OUT = os.path.join(ROOT, "data", "processed", "catalogue_slim.csv")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
if not os.path.exists(XLSX):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/GTB-tbsequencing/mutation-catalogue-2023", RAW], check=True)
# header row is row 3; column positions are fixed in the WHO sheet (105 = FINAL CONFIDENCE GRADING)
df = pd.read_excel(XLSX, sheet_name="Catalogue_master_file", header=2, usecols=[0, 1, 2, 3, 4, 5, 105, 109])
df.columns = ["drug", "gene", "mutation", "variant", "tier", "effect", "final_grade", "silent"]
assert df["final_grade"].str[0].isin(list("12345")).all(), "unexpected grade values: sheet layout changed?"
df.to_csv(OUT, index=False)
print(f"saved {len(df)} rows -> {OUT}")
