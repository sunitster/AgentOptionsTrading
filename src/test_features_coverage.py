import os
import pandas as pd
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(__file__))
SNAPSHOT_DIR = os.path.join(BASE, "data", "options_snapshot")
CHAIN_DIR = os.path.join(BASE, "data", "options_chain_kite")
FEAT_DIR   = os.path.join(BASE, "data", "features")

REQUIRED_COLS = [
    "date", "expiry", "strike", "option_type", "close",
    "volume", "oi", "iv", "underlying", "dte",
    "atm_strike", "atm_gap", "is_atm", "expected_range",
    "nifty_atr_14", "nifty_adx_14", "nifty_bb_bw_20",
    "ce_pe_spread", "liquid", "source_type"
]


def test_feature_coverage():

    print("\n==============================")
    print("   FEATURE COVERAGE REPORT")
    print("==============================\n")

    # ---------------------------------------------------------
    # 1. Count snapshots and feature files
    # ---------------------------------------------------------
    snapshots = set()
    if os.path.isdir(SNAPSHOT_DIR):
        snapshots.update([f.replace(".parquet","") for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".parquet")])

    if os.path.isdir(CHAIN_DIR):
        snapshots.update([f.replace(".parquet","") for f in os.listdir(CHAIN_DIR) if f.endswith(".parquet")])

    feature_files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith(".parquet")])

    print(f"Snapshots found : {len(snapshots)}")
    print(f"Features found  : {len(feature_files)}")

    # ---------------------------------------------------------
    # 2. Check missing feature days
    # ---------------------------------------------------------
    missing = []
    for s in sorted(snapshots):
        if f"{s}.parquet" not in feature_files:
            missing.append(s)

    if missing:
        print("\n⚠ Missing feature files for:")
        print(missing[:20], "...")
    else:
        print("✔ All snapshots have corresponding features")

    # ---------------------------------------------------------
    # 3. Feature validity test (column non-null ratios)
    # ---------------------------------------------------------
    col_stats = {c: [] for c in REQUIRED_COLS}

    for f in feature_files:
        df = pd.read_parquet(os.path.join(FEAT_DIR, f))
        for col in REQUIRED_COLS:
            if col in df.columns:
                col_stats[col].append(df[col].notna().mean())
            else:
                col_stats[col].append(0.0)

    # Average coverage per column
    summary = {col: sum(vals) / len(vals) for col, vals in col_stats.items()}

    print("\n--- Feature Coverage (%) ---")
    for col, cov in summary.items():
        print(f"{col:20s}: {cov*100:5.1f}%")

    # ---------------------------------------------------------
    # 4. Overall valid-row ratio
    # ---------------------------------------------------------
    file_valid_ratios = []
    for f in feature_files:
        df = pd.read_parquet(os.path.join(FEAT_DIR, f))
        file_valid_ratios.append(df.notna().mean().mean())

    overall = sum(file_valid_ratios) / len(file_valid_ratios)
    print(f"\nOverall valid ratio: {overall*100:.2f}%")

    if overall >= 0.95:
        print("✅ PASS — ≥95% valid rows\n")
    else:
        print("❌ FAIL — Below 95%\n")


if __name__ == "__main__":
    test_feature_coverage()
