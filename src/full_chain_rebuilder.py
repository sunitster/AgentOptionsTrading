# src/full_chain_rebuilder.py
# Full Historical Rebuilder – Zerodha + Backup API
# Saves output into data/options_chain_kite/

import os
import time
import yaml
import pandas as pd
from datetime import datetime, timedelta
from kiteconnect import KiteConnect

# ----------------------------------------
# PATHS
# ----------------------------------------
BASE = os.path.dirname(os.path.dirname(__file__))
OUTDIR = os.path.join(BASE, "data", "options_chain_kite")
os.makedirs(OUTDIR, exist_ok=True)

CONFIG_PATH = os.path.join(BASE, "config", "config.yaml")


# ----------------------------------------
# LOAD CONFIG
# ----------------------------------------
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ----------------------------------------
# INIT ZERODHA SESSION
# ----------------------------------------
def get_kite():
    cfg = load_config()
    kite = KiteConnect(api_key=cfg["kite"]["api_key"])
    kite.set_access_token(cfg["kite"]["access_token"])
    return kite


# ----------------------------------------
# ZERODHA OPTION CHAIN FETCHER
# ----------------------------------------
def fetch_chain_kite(kite, tradingsymbol):
    """Fetch option chain for a given symbol (NIFTY / BANKNIFTY)."""
    try:
        oi = kite.oi(tradingsymbol)
        if not oi or "data" not in oi:
            return None

        rows = []
        for entry in oi["data"]:
            ce = entry.get("CE")
            pe = entry.get("PE")

            if ce:
                rows.append({
                    "strike": ce["strikePrice"],
                    "option_type": "CE",
                    "expiry": ce["expiry"],
                    "last_price": ce["lastPrice"],
                    "iv": ce["impliedVolatility"],
                    "volume": ce["totalTradedVolume"],
                    "oi": ce["openInterest"],
                })
            if pe:
                rows.append({
                    "strike": pe["strikePrice"],
                    "option_type": "PE",
                    "expiry": pe["expiry"],
                    "last_price": pe["lastPrice"],
                    "iv": pe["impliedVolatility"],
                    "volume": pe["totalTradedVolume"],
                    "oi": pe["openInterest"],
                })

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"⚠ Zerodha chain failed: {e}")
        return None


# ----------------------------------------
# PUBLIC API FALLBACK (working)
# ----------------------------------------
import requests

def fetch_chain_fallback(symbol="NIFTY"):
    try:
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)

        data = r.json()
        rows = []

        for item in data["records"]["data"]:
            strike = item["strikePrice"]

            if "CE" in item:
                ce = item["CE"]
                rows.append({
                    "strike": strike,
                    "option_type": "CE",
                    "expiry": ce["expiryDate"],
                    "last_price": ce.get("lastPrice", None),
                    "iv": ce.get("impliedVolatility", None),
                    "volume": ce.get("totalTradedVolume", None),
                    "oi": ce.get("openInterest", None),
                })

            if "PE" in item:
                pe = item["PE"]
                rows.append({
                    "strike": strike,
                    "option_type": "PE",
                    "expiry": pe["expiryDate"],
                    "last_price": pe.get("lastPrice", None),
                    "iv": pe.get("impliedVolatility", None),
                    "volume": pe.get("totalTradedVolume", None),
                    "oi": pe.get("openInterest", None),
                })

        return pd.DataFrame(rows)

    except Exception:
        return None


# ----------------------------------------
# FETCH FOR A SINGLE DATE
# ----------------------------------------
def fetch_for_date(date, kite, force=False):
    fname = os.path.join(OUTDIR, f"{date}.parquet")

    if os.path.exists(fname) and not force:
        print(f"✔ Exists: {date}")
        return True

    date_str = str(date)
    print(f"\n=== {date_str} ===")

    # 1️⃣ Call Zerodha first
    df = fetch_chain_kite(kite, "NIFTY")

    # 2️⃣ fallback
    if df is None or df.empty:
        print("⚠ Falling back → NSE Public API")
        df = fetch_chain_fallback("NIFTY")

    if df is None or df.empty:
        print(f"❌ Failed: {date_str}")
        return False

    df["date"] = date_str
    df.to_parquet(fname)
    print(f"📁 Saved → {fname} ({len(df)} rows)")
    return True


# ----------------------------------------
# MAIN REBUILDER
# ----------------------------------------
def run_rebuilder():
    print("🟢 Full Historical Rebuilder (KITE + fallback)")
    kite = get_kite()

    start = datetime(2024, 7, 5).date()
    end = datetime.today().date()

    cur = start

    successes = 0
    fails = 0

    while cur <= end:
        success = fetch_for_date(cur, kite)
        if success:
            successes += 1
        else:
            fails += 1

        time.sleep(0.7)  # rate-limit safety
        cur += timedelta(days=1)

    print("\n========== SUMMARY ==========")
    print(f"Total days: {successes + fails}")
    print(f"Success   : {successes}")
    print(f"Failed    : {fails}")
    print("=============================\n")


if __name__ == "__main__":
    run_rebuilder()
