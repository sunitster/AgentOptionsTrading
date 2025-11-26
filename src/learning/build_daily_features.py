# src/learning/build_daily_features.py
import os, glob
import pandas as pd
from datetime import timedelta

OPTION_DIR = "learning_data/option_level"
NIFTY_DIR = "data/nifty"
OUT = "learning_data/daily_features.parquet"

def load_nifty(date):
    # cheap loader: assume file named YYYY-MM-DD.parquet in data/nifty
    p = os.path.join("data/nifty", f"{date.date().isoformat()}.parquet")
    if os.path.exists(p):
        d = pd.read_parquet(p)
        return float(d['close'].iloc[0])
    return None

files = sorted(os.listdir(OPTION_DIR))
rows = []
for f in files:
    df = pd.read_parquet(os.path.join(OPTION_DIR, f))
    # compute ATM iv: find strike closest to underlying (from data/nifty)
    date = pd.to_datetime(df['date'].iloc[0])
    nifty_close = load_nifty(date)
    if nifty_close is None:
        continue
    # ATM: nearest strike rows where option_type == 'CE' or 'PE'
    df_day = df.copy()
    # compute avg_iv proxies if iv column present
    if 'iv' in df_day.columns:
        iv_atm = df_day.iloc[(df_day['strike']-nifty_close).abs().argsort()]['iv'].iloc[0]
    else:
        iv_atm = None
    # compute iv_rank over last 90 days - requires reading past files; simple approach: compute rolling using saved df later
    total_oi = df_day['oi'].sum() if 'oi' in df_day.columns else 0
    # compute skew: avg put iv - avg call iv near ATM (if iv present)
    # fallback features
    rows.append({
        'date': date.date().isoformat(),
        'nifty_close': nifty_close,
        'iv_atm': iv_atm,
        'total_oi': total_oi,
        'n_options': len(df_day),
    })
# write parquet
pd.DataFrame(rows).to_parquet(OUT, index=False)
print("wrote daily features", OUT)
