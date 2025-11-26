import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.dirname(__file__))
NIFTY_DIR = os.path.join(BASE, "data", "nifty")
os.makedirs(NIFTY_DIR, exist_ok=True)

def download_nifty():
    print("Downloading NIFTY (^NSEI)...")

    ticker = yf.Ticker("^NSEI")
    hist = ticker.history(start="2010-01-01", end=datetime.today().strftime("%Y-%m-%d"))

    if hist.empty:
        print("ERROR: Could not download NIFTY data")
        return

    hist = hist.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    hist = hist.reset_index()

    for _, row in hist.iterrows():
        d = row["Date"].strftime("%Y-%m-%d")
        df = pd.DataFrame([row])
        df.to_parquet(os.path.join(NIFTY_DIR, f"{d}.parquet"), index=False)

    print("Done, saved:", len(hist), "files")

if __name__ == "__main__":
    download_nifty()
