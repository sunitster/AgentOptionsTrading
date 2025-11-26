import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))
FEAT_DIR = os.path.join(BASE, "data/features")

def test_data_leakage():
    files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith(".parquet")])

    print("\n📌 DATA LEAKAGE TEST\n")

    leak_found = False

    for f in files:
        df = pd.read_parquet(os.path.join(FEAT_DIR, f))
        date = f.replace(".parquet", "")

        # 1. Check date consistency
        if not all(df["date"] == date):
            leak_found = True
            print("❌ Leakage detected in:", f)

        # 2. DTE must be >= 0
        if df["dte"].min() < 0:
            leak_found = True
            print("❌ Negative DTE in:", f)

    if not leak_found:
        print("✅ No leakage detected")

if __name__ == "__main__":
    test_data_leakage()
