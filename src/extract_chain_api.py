# src/extract_chain_api.py
"""
Daily incremental extractor for NIFTY option chains (Option B modern style).

Behavior:
- For each date in the requested range it will:
    1) Try the mirror CSV endpoint (niftyportal / nsearchives) which serves CSV bhav-like files.
    2) If that fails, try fetching the NSE option-chain JSON (current snapshot). NOTE: NSE JSON does not
       provide historical snapshots by date — fallback is only useful for recent/current date.
- Normalizes into the unified columns used by your project:
    ['date','expiry','strike','option_type','open','high','low','close','settle','oi','volume']
- Saves result as parquet:
    data/options_chain_api/YYYY-MM-DD.parquet

Usage:
    python src/extract_chain_api.py
"""

from datetime import datetime, timedelta
import os
import time
import json
import logging
from typing import Optional
import requests
import pandas as pd

# -------------------------
# CONFIG
# -------------------------
BASE = os.path.dirname(os.path.dirname(__file__))
OUTDIR = os.path.join(BASE, "data", "options_chain_api")
os.makedirs(OUTDIR, exist_ok=True)

# Mirror archive (preferred)
MIRROR_TEMPLATE = "https://nsearchives.niftyportal.org/fo/{yyyy}/{mmm}/fo{dd}{mm}{yyyy}bhav.csv"

# NSE option chain API (fallback for current-day only)
NSE_OPTION_CHAIN_URL = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"

HEADERS_POOL = [
    # Multiple common user-agents helps avoid naive blocking
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
]

# polite sleep between requests (seconds)
SLEEP_BETWEEN = 5.0

# retries for each source
RETRIES = 3
BACKOFF = 3  # seconds initial backoff (exponential)

# -------------------------
# LOGGER
# -------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("extract_chain_api")


# -------------------------
# HELPERS
# -------------------------
def mirror_url_for_date(d: datetime) -> str:
    yyyy = d.strftime("%Y")
    mmm = d.strftime("%b").upper()   # e.g. JAN
    dd = d.strftime("%d")
    mm = d.strftime("%m")
    return MIRROR_TEMPLATE.format(yyyy=yyyy, mmm=mmm, dd=dd, mm=mm)


def save_parquet(df: pd.DataFrame, date: datetime):
    out = os.path.join(OUTDIR, f"{date.date()}.parquet")
    df.to_parquet(out, index=False)
    log.info("Saved %s → %d rows", out, len(df))


def normalize_bhav_df(df_raw: pd.DataFrame, date: datetime) -> pd.DataFrame:
    """
    Normalize mirror bhav CSV format (columns like INSTRUMENT, SYMBOL, OPTION_TYP, STRIKE_PR, EXPIRY_DT, OPEN, HIGH, LOW, CLOSE, SETTLE_PR, OPEN_INT, CONTRACTS, TIMESTAMP)
    into the project's unified option snapshot format:
      ['date','expiry','strike','option_type','open','high','low','close','settle','oi','volume']
    """
    df = df_raw.copy()

    # Standardize column names lower-case
    df.columns = [c.strip() for c in df.columns]

    # Keep only options rows for NIFTY and option types
    # Mirror files contain FUTIDX/FUTSTK/OPTSTK/OPTIDX etc. We want option rows:
    cond_option = df["INSTRUMENT"].str.upper().str.contains("OPT", na=False) | df.get("OPTION_TYP", "").notnull()
    df = df[cond_option].copy()

    # Map columns; many mirror files use names like OPTION_TYP, STRIKE_PR, EXPIRY_DT, OPEN, HIGH, LOW, CLOSE, SETTLE_PR, OPEN_INT, CONTRACTS
    # Create unified names with missing value handling
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
        "CONTRACTS": "volume",  # coarse mapping: contract count as 'volume' proxy (bhavcopy does not include tick-level volume)
        "TIMESTAMP": "timestamp",
    }

    new = {}
    for src, dst in colmap.items():
        if src in df.columns:
            new[dst] = df[src]
        else:
            # Try lowercase variant
            if src.lower() in df.columns:
                new[dst] = df[src.lower()]
            else:
                new[dst] = pd.NA

    out = pd.DataFrame(new)
    # Parse expiry dates if present
    if out["expiry"].notna().any():
        try:
            out["expiry"] = pd.to_datetime(out["expiry"], dayfirst=True, errors="coerce")
        except Exception:
            out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")

    # option_type normalization (CE / PE)
    out["option_type"] = out["option_type"].astype(str).str.strip().str.upper().replace({"CE": "CE", "PE": "PE", "": pd.NA})

    # set date column (the snapshot date)
    out["date"] = pd.Timestamp(date.date())

    # Reorder and cast numeric fields
    cols = ["date", "expiry", "strike", "option_type", "open", "high", "low", "close", "settle", "oi", "volume"]
    out = out.reindex(columns=cols)

    for ncol in ["strike", "open", "high", "low", "close", "settle", "oi", "volume"]:
        out[ncol] = pd.to_numeric(out[ncol], errors="coerce")

    return out


def normalize_nse_json(json_data: dict, date: datetime) -> pd.DataFrame:
    """
    Normalize NSE option chain JSON (current snapshot) into unified format.
    The JSON contains an array 'records' → 'data' with CE and PE nested per strike.
    """
    rows = []
    recs = json_data.get("records", {}).get("data", [])
    for r in recs:
        strike = r.get("strikePrice")
        expiry = r.get("expiryDate") or json_data.get("records", {}).get("expiryDates", [None])[0]
        # CE block and PE block
        ce = r.get("CE")
        pe = r.get("PE")
        if ce:
            rows.append(
                {
                    "date": pd.Timestamp(date.date()),
                    "expiry": pd.to_datetime(expiry, errors="coerce"),
                    "strike": strike,
                    "option_type": "CE",
                    "open": ce.get("open", pd.NA),
                    "high": ce.get("high", pd.NA),
                    "low": ce.get("low", pd.NA),
                    "close": ce.get("lastPrice", pd.NA),
                    "settle": ce.get("settlementPrice", pd.NA) or ce.get("close", pd.NA),
                    "oi": ce.get("openInterest", pd.NA),
                    "volume": ce.get("totalTradedVolume", pd.NA),
                }
            )
        if pe:
            rows.append(
                {
                    "date": pd.Timestamp(date.date()),
                    "expiry": pd.to_datetime(expiry, errors="coerce"),
                    "strike": strike,
                    "option_type": "PE",
                    "open": pe.get("open", pd.NA),
                    "high": pe.get("high", pd.NA),
                    "low": pe.get("low", pd.NA),
                    "close": pe.get("lastPrice", pd.NA),
                    "settle": pe.get("settlementPrice", pd.NA) or pe.get("close", pd.NA),
                    "oi": pe.get("openInterest", pd.NA),
                    "volume": pe.get("totalTradedVolume", pd.NA),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["date","expiry","strike","option_type","open","high","low","close","settle","oi","volume"])

    df = pd.DataFrame(rows)
    # cast numeric
    for c in ["strike","open","high","low","close","settle","oi","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date","expiry","strike","option_type","open","high","low","close","settle","oi","volume"]]


# -------------------------
# FETCH FUNCTIONS (sources)
# -------------------------
def fetch_from_mirror(date: datetime, session: requests.Session) -> Optional[pd.DataFrame]:
    url = mirror_url_for_date(date)
    log.info("Trying mirror for %s → %s", date.date(), url)
    for attempt in range(RETRIES):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                # parse CSV content
                df = pd.read_csv(pd.compat.StringIO(resp.text))
                if df.empty:
                    log.warning("Mirror returned empty CSV for %s", date.date())
                    return None
                # keep only rows with SYMBOL == 'NIFTY' or INSTRUMENT contains 'OPT'
                # many mirror files include many symbols; filter NIFTY derivatives
                if "SYMBOL" in df.columns:
                    df = df[df["SYMBOL"].str.upper() == "NIFTY"]
                return df
            else:
                log.warning("Mirror responded %s for %s", resp.status_code, date.date())
        except Exception as e:
            log.debug("Mirror attempt %d failed: %s", attempt + 1, e)
        time.sleep(BACKOFF * (2 ** attempt))
    return None


def fetch_from_nse_json(date: datetime, session: requests.Session) -> Optional[pd.DataFrame]:
    """
    Try to fetch option chain JSON from NSE. NOTE: this endpoint generally returns current data;
    historical dates are often not supported. Use only as fallback (best-effort).
    """
    log.info("Trying NSE JSON API for %s", date.date())
    try:
        # Get initial cookies by hitting homepage (reduces chance of 403)
        session.headers.update(HEADERS_POOL[int(time.time()) % len(HEADERS_POOL)])
        session.get("https://www.nseindia.com", timeout=10)
        # Small delay then request
        time.sleep(1.0)
        r = session.get(NSE_OPTION_CHAIN_URL, timeout=15)
        if r.status_code != 200:
            log.warning("NSE JSON status %s", r.status_code)
            return None
        data = r.json()
        df = normalize_nse_json(data, date)
        if df.empty:
            return None
        return df
    except Exception as e:
        log.debug("NSE JSON fetch failed: %s", e)
        return None


# -------------------------
# CORE: fetch-for-date (single daily incremental)
# -------------------------
def fetch_for_date(date: datetime, force: bool = False) -> bool:
    outpath = os.path.join(OUTDIR, f"{date.date()}.parquet")
    if os.path.exists(outpath) and not force:
        log.info("Exists → %s (skip)", outpath)
        return True

    session = requests.Session()
    session.headers.update(HEADERS_POOL[int(time.time()) % len(HEADERS_POOL)])

    # 1) Mirror attempt
    df_mirror = fetch_from_mirror(date, session)
    if df_mirror is not None:
        try:
            df_norm = normalize_bhav_df(df_mirror, date)
            if df_norm.empty:
                log.warning("Normalized mirror returned empty for %s", date.date())
            else:
                save_parquet(df_norm, date)
                time.sleep(SLEEP_BETWEEN)
                return True
        except Exception as e:
            log.exception("Failed to normalize/save mirror data for %s: %s", date.date(), e)

    # 2) NSE JSON fallback (best-effort; only current-day usually)
    df_json = fetch_from_nse_json(date, session)
    if df_json is not None and not df_json.empty:
        save_parquet(df_json, date)
        time.sleep(SLEEP_BETWEEN)
        return True

    log.warning("All sources failed for %s", date.date())
    return False


# -------------------------
# DRIVER: daily incremental loop
# -------------------------
def run_incremental(start: datetime, end: datetime, force: bool = False):
    cur = start
    total = 0
    success = 0
    failed_dates = []
    while cur <= end:
        # Only business days (Mon-Fri)
        if cur.weekday() < 5:
            total += 1
            ok = fetch_for_date(cur, force=force)
            if ok:
                success += 1
            else:
                failed_dates.append(cur.date())
        cur += timedelta(days=1)

    log.info("=== SUMMARY ===")
    log.info("Range: %s → %s", start.date(), end.date())
    log.info("Business days: %d", total)
    log.info("Successful: %d", success)
    log.info("Failed: %d", total - success)
    if failed_dates:
        log.info("Failed samples (first 20): %s", [str(d) for d in failed_dates[:20]])


# -------------------------
# ENTRYPOINT
# -------------------------
if __name__ == "__main__":
    # default: last 3 years incremental (safe)
    today = datetime.today()
    default_start = today - timedelta(days=365 * 3)

    print()
    log.info("Daily incremental extractor (safe mode)")
    log.info("Default start: %s", default_start.date())
    log.info("Default end  : %s", today.date())

    # Simple interactive prompt (safe) — if you want fully non-interactive, adjust below
    try:
        start_str = input(f"Start date [YYYY-MM-DD] (default {default_start.date()}): ").strip()
    except Exception:
        start_str = ""

    try:
        end_str = input(f"End date [YYYY-MM-DD] (default {today.date()}): ").strip()
    except Exception:
        end_str = ""

    if start_str:
        start = datetime.fromisoformat(start_str)
    else:
        start = default_start

    if end_str:
        end = datetime.fromisoformat(end_str)
    else:
        end = today

    force_input = input("Force redownload existing files? (y/N): ").strip().lower()
    force = force_input == "y"

    run_incremental(start, end, force=force)
