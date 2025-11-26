import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))
FEAT_DIR = os.path.join(BASE, "data/features")

def breakdown():
    files = sorted(os.listdir(FEAT_DIR))
    
    null_counts = {
        "underlying": 0,
        "atm_price": 0,
        "iv_est": 0,
        "pcr": 0,
        "skew": 0,
        "dte": 0
    }
    total = len(files)

    for f in files:
        df = pd.read_parquet(os.path.join(FEAT_DIR, f))
        for col in null_counts:
            if pd.isna(df[col].iloc[0]):
                null_counts[col] += 1

    print("\n🔎 Missing Feature Breakdown:")
    for k, v in null_counts.items():
        print(f"{k:12} → {v} missing ({v/total:.2%})")

if __name__ == "__main__":
    breakdown()
