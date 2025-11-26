# src/learning/backfill_option_features.py
import os, glob
import pandas as pd
from datetime import datetime
OUT_DIR = "learning_data/option_level"
os.makedirs(OUT_DIR, exist_ok=True)

src = "data/options_snapshot"
files = sorted(os.listdir(src))
for f in files:
    path = os.path.join(src, f)
    df = pd.read_parquet(path)
    # normalize column names
    df = df.rename(columns={ 'TIMESTAMP':'date','EXPIRY_DT':'expiry','STRIKE_PR':'strike','OPTION_TYP':'option_type','CLOSE':'close','OPEN_INT':'oi' })
    df['date'] = pd.to_datetime(df['date'])
    # compute iv if missing by joining kite chain (optional)
    # keep only necessary columns
    keep = ['date','expiry','strike','option_type','close','oi','CONTRACTS','VAL_INLAKH']
    out = df[[c for c in keep if c in df.columns]].copy()
    out.to_parquet(os.path.join(OUT_DIR, f))
    print("wrote", f)
