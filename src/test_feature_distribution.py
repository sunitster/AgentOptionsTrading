import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(__file__))
FEAT_DIR = os.path.join(BASE, "data/features")

def test_feature_distribution():
    issues = []

    for f in os.listdir(FEAT_DIR):
        df = pd.read_parquet(os.path.join(FEAT_DIR, f))

        for col in ["iv_est", "pcr", "skew"]:
            if df[col].isna().mean() > 0.2:
                issues.append(f"Too many NaN in {col} for {f}")

        # IV sanity
        if df["iv_est"].max() > 3:
            issues.append(f"IV > 300% in {f}")

        # PCR sanity
        if df["pcr"].max() > 10:
            issues.append(f"PCR > 10 in {f}")

    if issues:
        print("❌ Issues Found:")
        for i in issues:
            print(" -", i)
    else:
        print("✅ All distributions within expected range")

if __name__ == "__main__":
    test_feature_distribution()
