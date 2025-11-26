# src/learning/backfill_labels.py

import os
import datetime as dt
import pandas as pd
from tqdm import tqdm

from src.learning.backtest_adapter import backtest_fn


OUTPUT_PATH = "learning_data/daily_labels.parquet"


def backfill_labels(
    features_path: str = "learning_data/daily_features.parquet",
    output_path: str = OUTPUT_PATH
):
    """
    Build label dataset:
        next_day_pnl = PnL from running a 1-day backtest starting the NEXT day.

    Produces:
        learning_data/daily_labels.parquet
    """

    print("=== Backfilling next_day_pnl labels ===")

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing features parquet: {features_path}")

    df = pd.read_parquet(features_path)
    df = df.sort_values("date").reset_index(drop=True)

    # Prepare output dataframe
    df["next_day_pnl"] = 0.0  # initialize clean

    # Loop through all rows and compute next-day pnl
    for i in tqdm(range(len(df) - 1)):
        today = df.loc[i, "date"]
        next_day = df.loc[i + 1, "date"]

        try:
            start = dt.datetime.strptime(next_day, "%Y-%m-%d").date()
            end = start
        except Exception:
            # already datetime type
            start = pd.to_datetime(next_day).date()
            end = start

        # empty model = baseline champion
        regime_model = {"rules": {}}

        try:
            out = backtest_fn(regime_model, start=start, end=end, weekly=True)
        except Exception as e:
            print("Backtest error:", e)
            df.at[i, "next_day_pnl"] = 0.0
            continue

        daily = out.get("daily_pnl", [])

        # extract FLOAT only
        if isinstance(daily, list) and len(daily) > 0:
            try:
                pnl_val = float(daily[0])
            except Exception:
                pnl_val = 0.0
        else:
            pnl_val = 0.0

        df.at[i, "next_day_pnl"] = pnl_val

    # Last row has no next day PnL
    df.at[len(df) - 1, "next_day_pnl"] = 0.0

    # ENFORCE CLEAN FLOAT COLUMN
    df["next_day_pnl"] = pd.to_numeric(df["next_day_pnl"], errors="coerce").fillna(0.0)

    # Finally write parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)

    print(f"Labels written → {output_path}")
    return df


# Simple manual testing
if __name__ == "__main__":
    backfill_labels()
