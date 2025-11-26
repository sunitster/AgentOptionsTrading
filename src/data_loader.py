# src/data_loader.py
"""
Option B — DAILY SNAPSHOT (END OF DAY)
Saves:
 - data/nifty/YYYY-MM-DD.parquet  (daily NIFTY candle)
 - data/options_snapshot/YYYY-MM-DD.parquet (one row per strike per option)

Usage:
  python src/data_loader.py all         # run both nifty + options (recommended)
  python src/data_loader.py nifty       # only NIFTY
  python src/data_loader.py options     # only option chain snapshot
  python src/data_loader.py qc          # quick QC for today's files
"""

from __future__ import annotations
import os
import sys
import json
import time
import math
import logging
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any

import pandas as pd
import requests
import yfinance as yf

# ------- Config -------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE, "data")
NIFTY_DIR = os.path.join(DATA_DIR, "nifty")
OPT_DIR = os.path.join(DATA_DIR, "options_snapshot")

os.makedirs(NIFTY_DIR, exist_ok=True)
os.makedirs(OPT_DIR, exist_ok=True)

NSE_API_URL = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"

# Basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ------- Utilities -------
def today_str() -> str:
    return date.today().isoformat()

def ensure_ist_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Convert any DatetimeIndex (naive or tz-aware) to Asia/Kolkata.
    NEVER call tz_localize if already tz-aware.
    """
    try:
        if idx.tz is None:
            return idx.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            return idx.tz_convert("Asia/Kolkata")
    except Exception:
        # fallback: coerce to naive then treat as UTC
        try:
            naive = pd.to_datetime(idx)
            return naive.tz_localize("UTC").tz_convert("Asia/Kolkata")
        except Exception:
            raise

def save_parquet_fallback(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """
    Try to save parquet. If pyarrow/fastparquet missing, fallback to CSV with same name + .csv
    """
    try:
        df.to_parquet(path, index=index)
        logging.info(f"Saved → {path}")
    except Exception as e:
        logging.warning(f"Parquet save failed ({e}); falling back to CSV.")
        csv_path = path.rsplit(".", 1)[0] + ".csv"
        df.to_csv(csv_path, index=index)
        logging.info(f"Saved CSV fallback → {csv_path}")

# ------- NIFTY loader -------
def fetch_nifty_daily(target_date: Optional[date] = None) -> Optional[pd.DataFrame]:
    """
    Fetch a single daily candle for NIFTY using yfinance.
    Saves to data/nifty/YYYY-MM-DD.parquet

    If target_date is None -> today (useful for EOD run; you may run next morning
    to capture the closed candle).
    """
    if target_date is None:
        target_date = date.today()

    start = target_date
    end = target_date + timedelta(days=1)
    start_str = start.isoformat()
    end_str = end.isoformat()

    logging.info(f"Downloading NIFTY daily: {start_str} -> {end_str}")
    # request a 1d interval for the single day
    df = yf.download("^NSEI", start=start_str, end=end_str, interval="1d", progress=False, auto_adjust=False)

    if df.empty:
        logging.warning(f"No NIFTY daily candle returned for {start_str}")
        return None

    # Reset index to have 'Date' column
    df = df.reset_index()

    # Flatten multi-index columns if any (yfinance sometimes returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        # take first level names (Open, High, Low, Close, Volume)
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]

    # normalize column names to lower
    df.columns = [c.lower() for c in df.columns]

    # convert 'date' column to timezone-aware Asia/Kolkata if it's datetime
    if "date" in df.columns or "index" in df.columns:
        # prefer 'date' column if present
        date_col = "date" if "date" in df.columns else None
        if date_col:
            try:
                df["date"] = pd.to_datetime(df["date"])
                df["date"] = ensure_ist_index(df["date"])
            except Exception:
                # keep as naive date if conversion fails
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    else:
        # If no date column, create one using the target_date
        df["date"] = pd.Timestamp(start)

    # Ensure required columns exist
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        logging.warning(f"NIFTY daily candle missing expected columns: {missing}")

    # Save
    out_path = os.path.join(NIFTY_DIR, f"{start.isoformat()}.parquet")
    save_parquet_fallback(df, out_path, index=False)
    return df

# ------- NSE option chain snapshot loader -------
class NSESession:
    """
    Helper to fetch NSE API endpoints. NSE blocks direct API calls fairly often,
    so we perform an initial GET to the homepage to establish cookies/headers.
    """
    def __init__(self, retries: int = 3, backoff: float = 1.0):
        self.s = requests.Session()
        self.retries = retries
        self.backoff = backoff
        # common headers that mimic a browser
        self.s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        })

    def prime(self) -> None:
        """
        Hit the homepage to obtain cookies / any anti-bot tokens.
        """
        homepage = "https://www.nseindia.com"
        try:
            r = self.s.get(homepage, timeout=10)
            # ignore result; cookies will now be stored in session
            time.sleep(0.8)
        except Exception as e:
            logging.debug(f"prime() failed: {e}")

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
        """
        GET the URL with retries and backoff. Returns JSON dict or None.
        """
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.s.get(url, params=params, timeout=12)
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError:
                        logging.warning("Response JSON decode error.")
                        return None
                else:
                    logging.warning(f"NSE API returned status {resp.status_code} (attempt {attempt})")
                    last_exc = Exception(f"status {resp.status_code}")
            except Exception as e:
                logging.warning(f"NSE request failed (attempt {attempt}): {e}")
                last_exc = e
            time.sleep(self.backoff * attempt)
        logging.error(f"NSE API failed after {self.retries} attempts. Last error: {last_exc}")
        return None

def fetch_nse_option_chain_snapshot(target_date: Optional[date] = None) -> Optional[pd.DataFrame]:
    """
    Fetch option chain snapshot from NSE API and save to data/options_snapshot/YYYY-MM-DD.parquet.

    Note: NSE API returns the current live snapshot; it typically reflects the market at request time.
    For an EOD snapshot, run this after market close (or run early morning to capture the previous day's snapshot,
    but NSE does not guarantee historical snapshots are available via this endpoint).
    """
    sess = NSESession()
    sess.prime()

    logging.info("Fetching option-chain snapshot from NSE API...")
    j = sess.get_json(NSE_API_URL)
    if not j:
        logging.error("No JSON returned from NSE API.")
        return None

    # The NSE option chain JSON contains a top-level 'filtered' or 'records' field.
    # Typical structure: { "records": { "data": [ ... ], "timestamp": "..." } , ... }
    records = None
    if "records" in j:
        records = j["records"]
    else:
        # sometimes structure differs
        records = j

    data_list = None
    timestamp = None
    if isinstance(records, dict):
        # prefer 'data' inside 'records'
        data_list = records.get("data") or records.get("filtered") or []
        timestamp = records.get("timestamp") or records.get("tradeDate") or None
    else:
        data_list = []
    if not data_list:
        logging.error("Option chain JSON structure unexpected or empty 'data' field.")
        # save raw JSON for debugging
        raw_out = os.path.join(OPT_DIR, f"raw_nse_{today_str()}.json")
        with open(raw_out, "w", encoding="utf-8") as fh:
            json.dump(j, fh, indent=2)
        logging.info(f"Saved raw NSE JSON → {raw_out}")
        return None

    # Each element in data_list has 'CE' and 'PE' objects for a strike.
    # We want to produce a stacked DataFrame: one row per strike per option side (CE/PE).
    rows = []
    for rec in data_list:
        strike = rec.get("strikePrice") or rec.get("strike")
        if strike is None:
            continue
        for side in ("CE", "PE"):
            side_obj = rec.get(side)
            if not side_obj:
                # for some strikes one side may be absent
                continue
            row = {
                "date": timestamp or today_str(),
                "snapshot_ts": datetime.utcnow().isoformat(),
                "strike": float(strike),
                "option_type": side,
            }
            # pick expected numeric fields if present
            # Common fields: openInterest, changeinOpenInterest, lastPrice, highPrice, lowPrice, strikePrice, expiryDate, totalTradedVolume, underlyingValue, impliedVolatility
            # We'll map them to normalized names:
            row.update({
                "open": side_obj.get("openPrice") or side_obj.get("open") or side_obj.get("lastPrice"),
                "high": side_obj.get("highPrice") or side_obj.get("high"),
                "low": side_obj.get("lowPrice") or side_obj.get("low"),
                "close": side_obj.get("lastPrice") or side_obj.get("close") or side_obj.get("lastPrice"),
                "settle": side_obj.get("settlementPrice") or side_obj.get("lastPrice"),
                "oi": side_obj.get("openInterest") or side_obj.get("oi") or side_obj.get("openInterest"),
                "change_oi": side_obj.get("changeinOpenInterest") or side_obj.get("changeOI"),
                "volume": side_obj.get("totalTradedVolume") or side_obj.get("volume"),
                "iv": side_obj.get("impliedVolatility") or side_obj.get("iv"),
                "expiry": side_obj.get("expiryDate") or rec.get("expiryDate") or rec.get("expiry")
            })
            rows.append(row)

    if not rows:
        logging.error("After parsing, no option rows extracted.")
        return None

    df = pd.DataFrame(rows)

    # Normalize column names
    expected_cols = ["date", "snapshot_ts", "expiry", "strike", "option_type",
                     "open", "high", "low", "close", "settle", "oi", "change_oi", "volume", "iv"]
    # keep any extras but ensure order
    cols = [c for c in expected_cols if c in df.columns] + [c for c in df.columns if c not in expected_cols]
    df = df[cols]

    # cast numeric columns to floats (safe)
    for c in ["strike", "open", "high", "low", "close", "settle", "oi", "change_oi", "volume", "iv"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # convert expiry column to date if present
    if "expiry" in df.columns:
        try:
            df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
        except Exception:
            pass

    # Save
    out_path = os.path.join(OPT_DIR, f"{date.today().isoformat()}.parquet")
    save_parquet_fallback(df, out_path, index=False)
    return df

# ------- QC / Verification helpers -------
def quick_qc_nifty(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick QC for a saved nifty file path. If path None -> today file.
    Returns a dict with basic stats.
    """
    if path is None:
        path = os.path.join(NIFTY_DIR, f"{today_str()}.parquet")
    if not os.path.exists(path):
        return {"exists": False, "path": path}

    try:
        df = pd.read_parquet(path)
    except Exception:
        # try csv fallback
        if path.endswith(".parquet"):
            csv = path.rsplit(".", 1)[0] + ".csv"
            if os.path.exists(csv):
                df = pd.read_csv(csv)
            else:
                raise
        else:
            raise

    stats = {
        "exists": True,
        "path": path,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "na_count": df.isna().sum().to_dict()
    }
    return stats

def quick_qc_options(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = os.path.join(OPT_DIR, f"{today_str()}.parquet")
    if not os.path.exists(path):
        return {"exists": False, "path": path}
    try:
        df = pd.read_parquet(path)
    except Exception:
        if path.endswith(".parquet"):
            csv = path.rsplit(".", 1)[0] + ".csv"
            if os.path.exists(csv):
                df = pd.read_csv(csv)
            else:
                raise
        else:
            raise

    stats = {
        "exists": True,
        "path": path,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "na_count": df.isna().sum().to_dict(),
        "strikes_unique": int(df["strike"].nunique()) if "strike" in df.columns else None
    }
    return stats

# ------- CLI Entrypoints -------
def run_all():
    logging.info(">>> Running Option Chain + NIFTY Loader (OPTION B)")
    nifty_df = fetch_nifty_daily()
    options_df = fetch_nse_option_chain_snapshot()

    # QC summary
    logging.info("Run QC summary:")
    logging.info("NIFTY QC: %s", quick_qc_nifty())
    logging.info("Options QC: %s", quick_qc_options())

def run_nifty():
    logging.info(">>> Running NIFTY only")
    fetch_nifty_daily()
    logging.info("NIFTY QC: %s", quick_qc_nifty())

def run_options():
    logging.info(">>> Running Options only")
    fetch_nse_option_chain_snapshot()
    logging.info("Options QC: %s", quick_qc_options())

def run_qc():
    logging.info(">>> Quick QC report")
    logging.info("NIFTY QC: %s", quick_qc_nifty())
    logging.info("Options QC: %s", quick_qc_options())

# ------- Run as script -------
if __name__ == "__main__":
    action = "all"
    if len(sys.argv) >= 2:
        action = sys.argv[1].lower()

    try:
        if action == "all":
            run_all()
        elif action == "nifty":
            run_nifty()
        elif action == "options":
            run_options()
        elif action == "qc":
            run_qc()
        else:
            logging.error("Unknown action. Use one of: all, nifty, options, qc")
            sys.exit(2)
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    except Exception as e:
        logging.exception("Unhandled exception: %s", e)
        sys.exit(1)
