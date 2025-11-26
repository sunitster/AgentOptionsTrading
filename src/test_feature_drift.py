import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))
FEAT_DIR = os.path.join(BASE, "data/features")

def test_drift():
    files = sorted(os.listdir(FEAT_DIR))

    last = None
    for f in files:
        df = pd.read_parquet(os.path.join(FEAT_DIR, f))
        df["date"] = pd.to_datetime(df["date"].iloc[0])

        features = df.iloc[0]   # snapshot-level stats

        if last is not None:
            drift = abs(features - last)

            if drift.max() > 5 * drift.mean():   # crude rule
                print("⚠️ Drift anomaly:", f)

        last = features

if __name__ == "__main__":
    test_drift()
