#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze executed trades from trades.db.

Outputs:
 - Total trades
 - Win / loss breakdown
 - Total PnL
 - Expectancy
 - Max drawdown
 - Daily performance summary
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

DB_PATH = "models/llm_trades/trades.db"


def load_trades():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT 
            ts,
            event,
            pnl,
            duration_seconds,
            ic_json,
            note
        FROM trades
        ORDER BY ts
        """,
        con,
    )
    con.close()

    # parse IC as dict
    def safe_parse(s):
        try:
            return json.loads(s) if s else {}
        except Exception:
            return {}

    df["ic"] = df["ic_json"].apply(safe_parse)
    df["timestamp"] = pd.to_datetime(df["ts"])
    return df


def compute_drawdown(pnl_series):
    """Compute max drawdown"""
    cumulative = pnl_series.cumsum()
    rolling_max = cumulative.cummax()
    dd = cumulative - rolling_max
    max_dd = dd.min()
    return max_dd


def analyze():
    df = load_trades()
    if df.empty:
        print("No trades found.")
        return

    # only EXIT events count for PnL
    exits = df[df["event"].str.contains("exit", case=False, na=False)].copy()

    if exits.empty:
        print("No exit events found — no realized PnL to analyze.")
        return

    total_trades = len(exits)
    wins = exits[exits["pnl"] > 0]
    losses = exits[exits["pnl"] < 0]

    total_pnl = exits["pnl"].sum()
    avg_win = wins["pnl"].mean() if len(wins) else 0
    avg_loss = losses["pnl"].mean() if len(losses) else 0
    win_rate = len(wins) / total_trades

    # expectancy
    expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)

    # drawdown
    max_dd = compute_drawdown(exits["pnl"])

    # daily summary
    exits["date"] = exits["timestamp"].dt.date
    daily = exits.groupby("date")["pnl"].sum()

    print("\n================ TRADE ANALYSIS ================\n")
    print(f"Total trades:       {total_trades}")
    print(f"Wins:               {len(wins)}")
    print(f"Losses:             {len(losses)}")
    print(f"Win rate:           {win_rate:.2%}")
    print(f"Total PnL:          ₹{total_pnl:.2f}")
    print(f"Avg Win:            ₹{avg_win:.2f}")
    print(f"Avg Loss:           ₹{avg_loss:.2f}")
    print(f"Expectancy:         {expectancy:.2f} per trade")
    print(f"Max Drawdown:       ₹{max_dd:.2f}")

    print("\n----- Daily PnL -----")
    for d, v in daily.items():
        print(f"{d}: ₹{v:.2f}")

    print("\n================ END REPORT ================\n")


if __name__ == "__main__":
    analyze()
