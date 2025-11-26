# src/qc_report.py
import pandas as pd
import glob, os
from datetime import timedelta
from iv_utils import implied_volatility

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def qc_nifty():
    files = glob.glob(os.path.join(DATA_DIR, "nifty/*.parquet"))
    gap_count = 0
    total_missing = 0

    for f in files:
        df = pd.read_parquet(f)
        df = df.sort_index()

        # missing rows
        expected = pd.date_range(df.index.min(), df.index.max(), freq="5min", tz="Asia/Kolkata")
        missing = expected.difference(df.index)
        total_missing += len(missing)

        # interval gap check
        diffs = df.index.to_series().diff().dropna()
        gap_count += (diffs > timedelta(minutes=5)).sum()

    print("NIFTY QC:")
    print("Missing rows:", total_missing)
    print("Big gaps (>5m):", gap_count)


def qc_options():
    files = glob.glob(os.path.join(DATA_DIR, "options/*.parquet"))
    bad_iv = 0
    total = 0

    # approximate VIX (e.g. 12%-18%)
    vix_est = 0.15

    for f in files:
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            total += 1
            S = row["CLOSE"]  # spot approx
            K = row["STRIKE_PR"]
            T = max((pd.to_datetime(row["EXPIRY_DT"]) - pd.to_datetime(row["TIMESTAMP"])).days / 365, 0.001)
            mp = row["CLOSE"]
            iv = implied_volatility(S, K, T, 0.06, row["OPTION_TYP"], mp)

            if iv is not None and not (vix_est*0.9 <= iv <= vix_est*1.1):
                bad_iv += 1

    print("Options QC:")
    print("Total contracts checked:", total)
    print("IV outside ±10% range:", bad_iv)


if __name__ == "__main__":
    qc_nifty()
    qc_options()
