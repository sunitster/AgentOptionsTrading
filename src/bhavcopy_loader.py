#!/usr/bin/env python3
"""
FAST NSE Option Bhavcopy Downloader (CLEAN OPTIONS ONLY)

Downloads ONLY:
    • NIFTY options (OPTIDX, CE/PE, strike>0)
    • BANKNIFTY options (OPTIDX, CE/PE, strike>0)

Futures (FUTIDX / FUTSTK) are completely removed.

Saves per-day Parquet files into: data/options_snapshot/YYYY-MM-DD.parquet

Usage:
    python src/bhavcopy_loader.py --start 2023-01-01 --end 2024-12-31 --workers 8 --force
"""

from __future__ import annotations
import argparse
import io
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

# =============================
# Paths
# =============================
BASE = os.path.dirname(os.path.dirname(__file__))
OUTDIR = os.path.join(BASE, "data", "options_snapshot")
os.makedirs(OUTDIR, exist_ok=True)


# =============================
# URL helper
# =============================
def nse_url(date: datetime) -> str:
    yyyy = date.strftime("%Y")
    mmm = date.strftime("%b").upper()
    dd = date.strftime("%d")
    fname = f"fo{dd}{mmm}{yyyy}bhav.csv.zip"
    return f"https://archives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mmm}/{fname}"


# =============================
# Utility
# =============================
def format_date(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def safe_parquet_write(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def fetch_url_bytes(session: requests.Session, url: str, timeout: float, retries: int, backoff: float, verbose=False) -> bytes:
    for attempt in range(1, retries + 2):
        try:
            if verbose:
                print(f"GET {url} (attempt {attempt})")
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            if attempt > retries:
                raise
            wait = backoff * (2 ** (attempt - 1))
            if verbose:
                print(f"Failed: {e} — retrying in {wait:.1f}s")
            time.sleep(wait)


# =============================
# Download for one date
# =============================
def download_one(
    date: datetime,
    *,
    outdir: str = OUTDIR,
    force: bool = False,
    session: Optional[requests.Session] = None,
    retries: int = 3,
    backoff: float = 1.0,
    timeout: float = 15.0,
    verbose: bool = False
) -> bool:
    date_str = format_date(date)
    out_path = os.path.join(outdir, f"{date_str}.parquet")

    if os.path.exists(out_path) and not force:
        if verbose:
            print(f"✔ Exists → {date_str}")
        return True

    url = nse_url(date)

    own_session = False
    if session is None:
        session = requests.Session()
        own_session = True

    # Download
    try:
        content = fetch_url_bytes(session, url, timeout=timeout, retries=retries, backoff=backoff, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"❌ Download failed for {date_str}: {e}")
        return False
    finally:
        if own_session:
            session.close()

    # Parse ZIP or CSV
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
        csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_files:
            if verbose:
                print(f"⚠ No CSV inside ZIP → {date_str}")
            return False
        csv_name = csv_files[0]
        with zf.open(csv_name) as fh:
            df = pd.read_csv(fh, dtype=str)
    except zipfile.BadZipFile:
        try:
            df = pd.read_csv(io.BytesIO(content), dtype=str)
        except Exception as e:
            if verbose:
                print(f"❌ Could not parse data for {date_str}: {e}")
            return False
    except Exception as e:
        if verbose:
            print(f"❌ ZIP read failure for {date_str}: {e}")
        return False

    if df.empty:
        if verbose:
            print(f"⚠ Empty bhavcopy → {date_str}")
        return False

    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]

    required = {"INSTRUMENT", "OPTION_TYP", "STRIKE_PR", "SYMBOL", "EXPIRY_DT", "CLOSE"}
    if not required.issubset(df.columns):
        if verbose:
            print(f"⚠ Missing required columns → {date_str}")
        return False

    # =============================
    # Clean & filter ONLY OPTIONS
    # =============================
    df["INSTRUMENT"] = df["INSTRUMENT"].str.upper()
    df["OPTION_TYP"] = df["OPTION_TYP"].str.upper()
    df["SYMBOL"] = df["SYMBOL"].str.upper()

    # numeric strike
    df["STRIKE_PR"] = pd.to_numeric(df["STRIKE_PR"], errors="coerce")

    mask = (
        (df["INSTRUMENT"] == "OPTIDX") &
        (df["OPTION_TYP"].isin(["CE", "PE"])) &
        (df["STRIKE_PR"] > 0) &
        (df["SYMBOL"].isin(["NIFTY", "BANKNIFTY"]))
    )

    df_opt = df[mask].copy()

    if df_opt.empty:
        if verbose:
            print(f"⚠ No OPTION rows for {date_str}")
        return False

    # Convert common columns
    df_opt["EXPIRY_DT"] = pd.to_datetime(df_opt["EXPIRY_DT"], errors="coerce", dayfirst=True)
    df_opt["TIMESTAMP"] = date_str

    # Drop unnamed junk
    df_opt = df_opt[[c for c in df_opt.columns if not c.startswith("UNNAMED")]]

    # Save
    try:
        safe_parquet_write(df_opt, out_path)
        if verbose:
            print(f"📥 Saved {date_str} → {len(df_opt)} option rows")
        return True
    except Exception as e:
        if verbose:
            print(f"❌ Failed to save parquet for {date_str}: {e}")
        return False


# =============================
# Parallel Download Range
# =============================
def download_range(start: datetime, end: datetime, workers: int = 8, **kwargs):
    print("\n=== DOWNLOAD CLEAN NSE OPTION BHAVCOPIES ===")
    print(f"Range:   {format_date(start)} → {format_date(end)}")
    print(f"Workers: {workers}")
    print("============================================\n")

    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri only
            days.append(d)
        d += timedelta(days=1)

    print(f"Total trading days: {len(days)}\n")

    results = []
    session = requests.Session()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(download_one, day, session=session, **kwargs): day
            for day in days
        }
        for fut in as_completed(futs):
            day = futs[fut]
            try:
                ok = fut.result()
                results.append((day, ok))
            except Exception as e:
                print(f"❌ Error on {format_date(day)} → {e}")
                results.append((day, False))

    session.close()

    success = sum(1 for _, ok in results if ok)
    failed = len(results) - success

    print("\n=========== SUMMARY ===========")
    print(f"Success : {success}")
    print(f"Failed  : {failed}")
    if failed > 0:
        bad = [format_date(d) for d, ok in results if not ok][:20]
        print("Failed Samples (20):", bad)
    print("================================\n")


# =============================
# CLI
# =============================
def parse_args():
    p = argparse.ArgumentParser(description="Download CLEAN NSE option bhavcopies")
    p.add_argument("--start", required=True, type=str, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, type=str, help="YYYY-MM-DD")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--backoff", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    download_range(
        start,
        end,
        workers=args.workers,
        force=args.force,
        retries=args.retries,
        backoff=args.backoff,
        timeout=args.timeout,
        verbose=args.verbose
    )
