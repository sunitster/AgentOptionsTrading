#!/usr/bin/env python3
# scripts/unify_options_data.py
"""
Unify option chain sources into a canonical per-date parquet under data/options_all/.

Rules:
- If a file exists in data/options_chain_kite/<date>.parquet => use it (preferred).
- Else fallback to data/options_snapshot/<date>.parquet (after normalization).
- Output canonical schema:
    date (YYYY-MM-DD), strike (float), option_type ('call'/'put'),
    expiry (YYYY-MM-DD), last_price (float), iv (float or NaN),
    volume (float), oi (float)
- Saves to data/options_all/<date>.parquet
"""

from __future__ import annotations
import os
import sys
import json
import logging
from glob import glob
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("unify_options_data")

ROOT = Path(".")
KITE_DIR = ROOT / "data" / "options_chain_kite"
SNAP_DIR = ROOT / "data" / "options_snapshot"
OUT_DIR = ROOT / "data" / "options_all"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_kite(df: pd.DataFrame) -> pd.DataFrame:
    # Expected kite columns: strike, option_type, expiry, last_price, iv, volume, oi, date
    df = df.copy()
    # canonicalize column names (lowercase)
    df.columns = [c.lower() for c in df.columns]
    # rename common variations
    mapping = {}
    for c in ("strike", "option_type", "expiry", "last_price", "iv", "volume", "oi", "date"):
        if c not in df.columns:
            # try common alternates
            if c == "oi":
                if "open_interest" in df.columns:
                    mapping["open_interest"] = "oi"
            if c == "last_price":
                if "lastprice" in df.columns:
                    mapping["lastprice"] = "last_price"
    if mapping:
        df = df.rename(columns=mapping)
    # ensure types
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["last_price"] = pd.to_numeric(df.get("last_price"), errors="coerce")
    df["iv"] = pd.to_numeric(df.get("iv"), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0)
    df["oi"] = pd.to_numeric(df.get("oi"), errors="coerce").fillna(0)
    # option_type normalize
    df["option_type"] = df["option_type"].astype(str).str.lower().replace({"ce": "call", "pe": "put"})
    # expiry & date to YYYY-MM-DD
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    # final select
    out = df[["date", "strike", "option_type", "expiry", "last_price", "iv", "volume", "oi"]].copy()
    return out


def normalize_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    # Expected snapshot columns: STRIKE_PR, OPTION_TYP, EXPIRY_DT, CLOSE, OPEN_INT, TIMESTAMP, etc.
    df = df.copy()
    # normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    # mapping
    col_map = {}
    if "strike_pr" in df.columns:
        col_map["strike_pr"] = "strike"
    if "option_typ" in df.columns:
        col_map["option_typ"] = "option_type"
    if "expiry_dt" in df.columns:
        col_map["expiry_dt"] = "expiry"
    if "close" in df.columns:
        col_map["close"] = "last_price"
    if "open_int" in df.columns:
        col_map["open_int"] = "oi"
    if "timestamp" in df.columns:
        col_map["timestamp"] = "date"
    df = df.rename(columns=col_map)
    # convert values
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["last_price"] = pd.to_numeric(df.get("last_price"), errors="coerce")
    # volume likely missing in snapshot, fill 0
    df["volume"] = pd.to_numeric(df.get("contracts"), errors="coerce").fillna(0)
    df["oi"] = pd.to_numeric(df.get("oi"), errors="coerce").fillna(0)
    df["iv"] = pd.to_numeric(df.get("iv"), errors="coerce") if "iv" in df.columns else np.nan
    df["option_type"] = df.get("option_type", "").astype(str).str.lower().replace({"ce": "call", "pe": "put"})
    df["expiry"] = pd.to_datetime(df.get("expiry"), errors="coerce").dt.strftime("%Y-%m-%d")
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.strftime("%Y-%m-%d")
    out = df[["date", "strike", "option_type", "expiry", "last_price", "iv", "volume", "oi"]].copy()
    return out


def discover_dates() -> list[str]:
    dates = set()
    # from kite files
    if KITE_DIR.exists():
        for p in KITE_DIR.glob("*.parquet"):
            name = p.stem
            try:
                # assume filename is YYYY-MM-DD
                pd.to_datetime(name)
                dates.add(name)
            except Exception:
                # try look inside file for date column
                try:
                    df = pd.read_parquet(p, columns=["date"])
                    d = pd.to_datetime(df["date"].iloc[0]).strftime("%Y-%m-%d")
                    dates.add(d)
                except Exception:
                    continue
    # from snapshot files
    if SNAP_DIR.exists():
        for p in SNAP_DIR.glob("*.parquet"):
            name = p.stem
            try:
                pd.to_datetime(name)
                dates.add(name)
            except Exception:
                # inspect file
                try:
                    df = pd.read_parquet(p, columns=["timestamp"])
                    d = pd.to_datetime(df["timestamp"].iloc[0]).strftime("%Y-%m-%d")
                    dates.add(d)
                except Exception:
                    continue
    return sorted(list(dates))


def unify_for_date(date_str: str) -> bool:
    # prefer kite
    kite_path = KITE_DIR / f"{date_str}.parquet"
    snap_path = SNAP_DIR / f"{date_str}.parquet"
    out_path = OUT_DIR / f"{date_str}.parquet"

    if out_path.exists():
        logger.info("Skipping existing %s", out_path)
        return True

    if kite_path.exists():
        try:
            df = pd.read_parquet(kite_path)
            df_norm = normalize_kite(df)
            df_norm.to_parquet(out_path, index=False)
            logger.info("Wrote unified from kite: %s", out_path)
            return True
        except Exception as e:
            logger.exception("Failed to process kite file %s: %s", kite_path, e)
            # fallback to snapshot
    if snap_path.exists():
        try:
            df = pd.read_parquet(snap_path)
            df_norm = normalize_snapshot(df)
            df_norm.to_parquet(out_path, index=False)
            logger.info("Wrote unified from snapshot: %s", out_path)
            return True
        except Exception as e:
            logger.exception("Failed to process snapshot file %s: %s", snap_path, e)
            return False

    # if neither file exists, try to find any file in kite dir with that date inside
    # (rare) — return False
    logger.warning("No source found for date %s (kite/snapshot missing)", date_str)
    return False


def main():
    dates = discover_dates()
    logger.info("Discovered %d candidate dates", len(dates))
    for d in dates:
        unify_for_date(d)
    logger.info("Done. Unified files written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
