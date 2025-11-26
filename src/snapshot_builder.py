# src/snapshot_builder.py
"""
Build daily option snapshots (one row per strike per day) from bhavcopy files.
Input: data/bhavcopy/YYYY-MM-DD.parquet (or .csv fallback)
Output: data/options_snapshot/YYYY-MM-DD.parquet (or csv fallback)
"""

import os
import pandas as pd
from datetime import datetime
import logging
import glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
BASE = os.path.dirname(os.path.dirname(__file__))
BHAV_DIR = os.path.join(BASE, "data", "bhavcopy")
OUT_DIR = os.path.join(BASE, "data", "options_snapshot")
os.makedirs(OUT_DIR, exist_ok=True)

def save_df_safe(df: pd.DataFrame, out_path: str):
    try:
        df.to_parquet(out_path)
        return out_path
    except Exception:
        csv_path = out_path.replace(".parquet", ".csv")
        df.to_csv(csv_path, index=False)
        return csv_path

def map_columns(df: pd.DataFrame):
    # Normalize known column names across different bhavcopy formats
    col_map = {}
    cols = [c.lower() for c in df.columns.tolist()]
    # common names
    if "instrument" in cols:
        col_map["instrument"] = [c for c in df.columns if c.lower() == "instrument"][0]
    if "symbol" in cols:
        col_map["symbol"] = [c for c in df.columns if c.lower() == "symbol"][0]
    # expiry
    for candidate in ("expiry_dt", "expiry", "expiry_date", "exp_date"):
        if candidate in cols:
            col_map["expiry"] = [c for c in df.columns if c.lower() == candidate][0]
            break
    # strike
    for candidate in ("strike_pr", "strike_price", "strike"):
        if candidate in cols:
            col_map["strike"] = [c for c in df.columns if c.lower() == candidate][0]
            break
    # option type
    for candidate in ("option_typ", "option_type", "optiontype", "option"):
        if candidate in cols:
            col_map["option_type"] = [c for c in df.columns if c.lower() == candidate][0]
            break
    # price columns
    for k in ("open", "high", "low", "close", "settle_pr", "settle"):
        if k in cols:
            col_map.setdefault(k, [c for c in df.columns if c.lower() == k][0])
    # oi and volume
    for candidate in ("open_int", "oi", "openinterest"):
        if candidate in cols:
            col_map["oi"] = [c for c in df.columns if c.lower() == candidate][0]
            break
    for candidate in ("no_of_contracts", "contracts", "traded_qty", "volume"):
        if candidate in cols:
            col_map["volume"] = [c for c in df.columns if c.lower() == candidate][0]
            break

    return col_map

def build_snapshot_for_file(path: str):
    # Accept either parquet or csv
    try:
        if path.lower().endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        logging.error(f"Failed to read {path}: {e}")
        return None

    if df.shape[0] == 0:
        logging.warning(f"Empty file {path}")
        return None

    colmap = map_columns(df)
    # Filter to option derivatives for NIFTY
    # two heuristics: INSTRUMENT == 'OPTIDX' and SYMBOL == 'NIFTY'
    sym_col = colmap.get("symbol")
    ins_col = colmap.get("instrument")

    # normalize column lookup
    def get(col):
        name = colmap.get(col)
        return df[name] if name else pd.Series([None] * len(df))

    # identify option rows
    cond = pd.Series([False] * len(df))
    if ins_col:
        cond = cond | (df[ins_col].astype(str).str.upper() == "OPTIDX")
    if sym_col:
        cond = cond | (df[sym_col].astype(str).str.upper().str.contains("NIFTY"))

    df_opts = df[cond].copy()
    if df_opts.empty:
        logging.info(f"No option rows in {os.path.basename(path)}")
        return None

    # assemble snapshot
    snapshot = pd.DataFrame()
    # date = file date
    fname_date = os.path.basename(path).replace(".parquet", "").replace(".csv", "")
    try:
        snapshot["date"] = pd.to_datetime(fname_date).date()
    except Exception:
        snapshot["date"] = pd.to_datetime(df_opts[colmap.get("expiry")] if "expiry" in colmap else datetime.today()).dt.date

    # expiry
    if "expiry" in colmap:
        snapshot["expiry"] = pd.to_datetime(df_opts[colmap["expiry"]]).dt.date
    else:
        snapshot["expiry"] = pd.NaT

    # strike
    if "strike" in colmap:
        snapshot["strike"] = pd.to_numeric(df_opts[colmap["strike"]], errors="coerce")
    else:
        snapshot["strike"] = pd.NA

    # option type (normalize to CE/PE)
    if "option_type" in colmap:
        snapshot["option_type"] = df_opts[colmap["option_type"]].astype(str).str.upper().str.replace(" ", "")
        snapshot["option_type"] = snapshot["option_type"].replace({"CE": "CE", "PE": "PE"})
    else:
        snapshot["option_type"] = pd.NA

    # prices
    for p in ("open", "high", "low", "close"):
        if p in colmap:
            snapshot[p] = pd.to_numeric(df_opts[colmap[p]], errors="coerce")
        else:
            snapshot[p] = pd.NA

    # settle
    if "settle_pr" in colmap:
        snapshot["settle"] = pd.to_numeric(df_opts[colmap["settle_pr"]], errors="coerce")
    elif "settle" in colmap:
        snapshot["settle"] = pd.to_numeric(df_opts[colmap["settle"]], errors="coerce")
    else:
        snapshot["settle"] = pd.NA

    # oi and volume
    if "oi" in colmap:
        snapshot["oi"] = pd.to_numeric(df_opts[colmap["oi"]], errors="coerce")
    else:
        snapshot["oi"] = pd.NA
    if "volume" in colmap:
        snapshot["volume"] = pd.to_numeric(df_opts[colmap["volume"]], errors="coerce")
    else:
        snapshot["volume"] = pd.NA

    # Only keep rows with CE or PE
    snapshot = snapshot[snapshot["option_type"].isin(["CE", "PE"])].reset_index(drop=True)
    if snapshot.empty:
        logging.info(f"No CE/PE rows after normalization for {path}")
        return None

    # final columns
    snapshot = snapshot[["date", "expiry", "strike", "option_type", "open", "high", "low", "close", "settle", "oi", "volume"]]
    return snapshot

def build_all_snapshots(force=False):
    files = sorted(glob.glob(os.path.join(BHAV_DIR, "*.*")))
    logging.info(f"Found {len(files)} bhav files")
    for f in files:
        date_str = os.path.basename(f).replace(".parquet", "").replace(".csv", "")
        out_path = os.path.join(OUT_DIR, f"{date_str}.parquet")
        if os.path.exists(out_path) and not force:
            logging.info(f"✔ Already built snapshot → {date_str}")
            continue
        try:
            snap = build_snapshot_for_file(f)
            if snap is not None and not snap.empty:
                saved = save_df_safe(snap, out_path)
                logging.info(f"📗 API snapshot {date_str} → {len(snap)} rows")
            else:
                logging.warning(f"⚠ No snapshot produced for {date_str}")
        except Exception as e:
            logging.exception(f"FAILED building snapshot for {date_str}: {e}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    build_all_snapshots(force=args.force)
    logging.info("Snapshot builder done.")
