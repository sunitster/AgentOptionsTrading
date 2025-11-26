# src/features.py (Regenerated, fully fixed)

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Optional

# ============================================
# PATHS
# ============================================
BASE = os.path.dirname(os.path.dirname(__file__))
SNAPSHOT_DIR = os.path.join(BASE, "data", "options_snapshot")
CHAIN_DIR = os.path.join(BASE, "data", "options_chain_kite")
NIFTY_DIR = os.path.join(BASE, "data", "nifty")
FEATURE_DIR = os.path.join(BASE, "data", "features")
os.makedirs(FEATURE_DIR, exist_ok=True)


# ============================================
# SNAPSHOT LISTING
# ============================================

def list_snapshot_files() -> list[str]:
    files = []
    for d in (SNAPSHOT_DIR, CHAIN_DIR):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".parquet"):
                files.append(os.path.join(d, f))

    byname = {os.path.basename(p): p for p in files}
    return [byname[k] for k in sorted(byname)]


def load_snapshot_for_date(date_str: str) -> pd.DataFrame:
    fname = f"{date_str}.parquet"
    p_snap = os.path.join(SNAPSHOT_DIR, fname)
    p_api = os.path.join(CHAIN_DIR, fname)

    if os.path.exists(p_snap):
        try:
            df = pd.read_parquet(p_snap)
            if len(df) > 50:
                return df
        except:
            pass

    if os.path.exists(p_api):
        return pd.read_parquet(p_api)

    if os.path.exists(p_snap):
        return pd.read_parquet(p_snap)

    raise FileNotFoundError(f"No snapshot for {date_str}")


# ============================================
# FORMAT DETECTORS
# ============================================

def is_bhavcopy_like(df: pd.DataFrame) -> bool:
    cols = set(c.upper() for c in df.columns)
    return {"INSTRUMENT", "OPTION_TYP", "STRIKE_PR"}.issubset(cols)


def is_api_like(df: pd.DataFrame) -> bool:
    cols = set(c.lower() for c in df.columns)
    return {"strike", "option_type", "expiry", "last_price"}.issubset(cols)


# ============================================
# NORMALIZERS
# ============================================

def normalize_bhavcopy(df: pd.DataFrame, date_dt: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]

    df = df[df["INSTRUMENT"].str.upper() == "OPTIDX"]

    if df.empty:
        return pd.DataFrame()

    colmap = {
        "EXPIRY_DT": "expiry",
        "STRIKE_PR": "strike",
        "OPTION_TYP": "option_type",
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "SETTLE_PR": "settle",
        "OPEN_INT": "oi",
        "CONTRACTS": "volume",
        "SYMBOL": "symbol",
        "TIMESTAMP": "timestamp",
    }

    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})

    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

    for c in ["open","high","low","close","settle","oi","volume"]:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")

    df["iv"] = np.nan
    df["option_type"] = df["option_type"].astype(str).str.upper().str.strip()

    df["date"] = date_dt
    df["source_type"] = "bhavcopy"

    keep = [
        "date","expiry","strike","option_type","open","high","low","close",
        "settle","volume","oi","iv","symbol","source_type"
    ]
    return df[keep]


def normalize_api(df: pd.DataFrame, date_dt: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()

    low = {c.lower(): c for c in df.columns}
    rename = {}

    if "strike" in low: rename[low["strike"]] = "strike"
    if "option_type" in low: rename[low["option_type"]] = "option_type"
    if "expiry" in low: rename[low["expiry"]] = "expiry"
    if "last_price" in low: rename[low["last_price"]] = "close"
    if "iv" in low: rename[low["iv"]] = "iv"
    if "oi" in low: rename[low["oi"]] = "oi"
    if "volume" in low: rename[low["volume"]] = "volume"

    df = df.rename(columns=rename)

    df["expiry"] = pd.to_datetime(df.get("expiry"), errors="coerce")
    df["strike"] = pd.to_numeric(df.get("strike"), errors="coerce")

    for c in ["close","oi","volume","iv"]:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")

    df["open"] = np.nan
    df["high"] = np.nan
    df["low"] = np.nan
    df["settle"] = np.nan

    df["date"] = date_dt
    df["source_type"] = "api"

    keep = [
        "date","expiry","strike","option_type","open","high","low","close",
        "settle","volume","oi","iv","symbol","source_type"
    ]
    df["symbol"] = "NIFTY"

    return df[keep]


def normalize_snapshot(raw, date_str):
    if raw is None or raw.empty:
        return pd.DataFrame()

    dt = pd.to_datetime(date_str)

    if is_bhavcopy_like(raw):
        return normalize_bhavcopy(raw, dt)

    if is_api_like(raw):
        return normalize_api(raw, dt)

    lowcols = {c.lower() for c in raw.columns}
    if "strike" in lowcols and "option_type" in lowcols:
        return normalize_api(raw, dt)

    return pd.DataFrame()


# ============================================
# INDEX LOADING & CACHE
# ============================================

_NIFTY_CACHE = None


def load_index_close_from_dir(index_dir, date_dt) -> Optional[float]:
    if not os.path.isdir(index_dir): return None

    files = sorted([f for f in os.listdir(index_dir) if f.endswith(".parquet")])
    if not files: return None

    latest = None
    for f in files:
        d = pd.to_datetime(f.replace(".parquet", ""))
        if d > date_dt: break

        try:
            df = pd.read_parquet(os.path.join(index_dir, f))
            for c in ["close", "Close", "CLOSE"]:
                if c in df.columns:
                    latest = float(df[c].iloc[-1])
                    break
        except:
            continue

    return latest


def load_nifty_close(date_dt):
    return load_index_close_from_dir(NIFTY_DIR, date_dt)


# ============================================
# COMPUTE NIFTY INDICATORS
# ============================================

def compute_nifty_indicators_once():
    global _NIFTY_CACHE
    if _NIFTY_CACHE is not None:
        return _NIFTY_CACHE

    if not os.path.isdir(NIFTY_DIR):
        _NIFTY_CACHE = pd.DataFrame()
        return _NIFTY_CACHE

    rows = []
    files = sorted([f for f in os.listdir(NIFTY_DIR) if f.endswith(".parquet")])

    for f in files:
        try:
            d = pd.to_datetime(f.replace(".parquet", ""))
            df = pd.read_parquet(os.path.join(NIFTY_DIR, f))

            close_col = next((c for c in ["close","Close","CLOSE"] if c in df.columns), None)
            if not close_col:
                continue

            close_val = float(df[close_col].iloc[-1])
            high_val = float(df.get("high", pd.Series([np.nan])).iloc[-1])
            low_val = float(df.get("low", pd.Series([np.nan])).iloc[-1])

            rows.append({
                "date": d,
                "close": close_val,
                "high": high_val,
                "low": low_val
            })

        except Exception as e:
            print("⚠ NIFTY parse fail:", f, e)
            continue

    df = pd.DataFrame(rows).sort_values("date")

    if df.empty:
        _NIFTY_CACHE = df
        return df

    df["return"] = df["close"].pct_change()
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    df["hl_range"] = df["high"] - df["low"]
    df["atr_5"] = df["hl_range"].rolling(5).mean()
    df["atr_14"] = df["hl_range"].rolling(14).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-9))))

    _NIFTY_CACHE = df
    return df


# ============================================
# FEATURE BUILDING
# ============================================

def get_nifty_features(date_dt):
    df = compute_nifty_indicators_once()
    row = df[df["date"] == date_dt]
    if row.empty:
        return {}

    r = row.iloc[0]
    return {
        "nifty_close": r["close"],
        "nifty_return": r["return"],
        "nifty_rsi_14": r["rsi_14"],
        "nifty_ma_5": r["ma_5"],
        "nifty_ma_20": r["ma_20"],
        "nifty_atr_5": r["atr_5"],
        "nifty_atr_14": r["atr_14"],
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = []
    date_dt = df["date"].iloc[0]

    nifty = get_nifty_features(date_dt)

    for _, row in df.iterrows():
        out.append({
            "date": row["date"],
            "expiry": row["expiry"],
            "strike": row["strike"],
            "option_type": row["option_type"],
            "close": row["close"],
            "oi": row["oi"],
            "volume": row["volume"],
            "iv": row["iv"],
            **nifty,
        })

    return pd.DataFrame(out)


# ============================================
# MAIN DRIVER FOR ONE DATE
# ============================================

def build_features_for_date(date_str: str) -> Optional[pd.DataFrame]:
    try:
        raw = load_snapshot_for_date(date_str)
    except FileNotFoundError:
        print(f"⚠ Missing snapshot → {date_str}")
        return None

    norm = normalize_snapshot(raw, date_str)
    if norm.empty:
        print(f"⚠ Normalization produced empty → {date_str}")
        return None

    df = build_features(norm)
    return df


# ============================================
# SAVE
# ============================================

def save_features(date_str: str, df: pd.DataFrame):
    if df is None or df.empty:
        return

    out = os.path.join(FEATURE_DIR, f"{date_str}.parquet")
    df.to_parquet(out, index=False)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="Build features for one date only")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild all features ignoring existing files")
    args = parser.parse_args()

    # ------------------------------------------------
    # Build for ONE DATE
    # ------------------------------------------------
    if args.date:
        print(f"📅 Building features for {args.date} ...")
        df = build_features_for_date(args.date)
        if df is not None:
            save_features(args.date, df)
            print(f"✅ Saved → {FEATURE_DIR}/{args.date}.parquet")
        else:
            print(f"❌ Failed → {args.date}")
        sys.exit(0)

    # ------------------------------------------------
    # Build for ALL DATES
    # ------------------------------------------------
    dates = [os.path.basename(p).replace(".parquet", "") for p in list_snapshot_files()]
    dates = sorted(set(dates))

    print(f"📚 Found {len(dates)} snapshot dates")

    for d in dates:
        outpath = os.path.join(FEATURE_DIR, f"{d}.parquet")
        if os.path.exists(outpath) and not args.force:
            print(f"➡️  Skipping (exists) {d}")
            continue

        print(f"⚙️  Building {d} ...")

        df = build_features_for_date(d)
        if df is None:
            print(f"❌ Failed building → {d}")
            continue

        save_features(d, df)
        print(f"✅ Saved {d}")
