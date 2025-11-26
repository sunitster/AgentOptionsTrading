
# -------------------------
# file: feature_store.py
# -------------------------
"""
Lightweight feature store utilities.
Responsibilities:
- append daily features + pnl to parquet/csv store
- load rolling windows for training/evaluation
"""
import os
import pandas as pd
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'learning_data')
os.makedirs(DATA_DIR, exist_ok=True)
DAILY_FEATURES_PATH = os.path.join(DATA_DIR, 'daily_features.parquet')


def append_daily_row(row: dict):
    """Append a single day's features + pnl dict to the store."""
    df = pd.DataFrame([row])
    if os.path.exists(DAILY_FEATURES_PATH):
        old = pd.read_parquet(DAILY_FEATURES_PATH)
        out = pd.concat([old, df], ignore_index=True)
    else:
        out = df
    out.to_parquet(DAILY_FEATURES_PATH, index=False)


def load_rolling_window(days: int = 90):
    """Return the last `days` rows as a DataFrame."""
    if not os.path.exists(DAILY_FEATURES_PATH):
        return pd.DataFrame()
    df = pd.read_parquet(DAILY_FEATURES_PATH)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df.tail(days).reset_index(drop=True)
    return df.tail(days).reset_index(drop=True)