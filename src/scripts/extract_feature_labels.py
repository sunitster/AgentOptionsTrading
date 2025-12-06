#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract labeled features for ML training.

Reads:
 - models/llm_trades/records.jsonl   (entry snapshots)
 - models/llm_trades/trades.db       (exit PnL to attach labels)

Produces:
 - models/llm_trades/features.parquet

Each row (1 per entry order) includes:
 - timestamp
 - expiry
 - strikes (SP, LC, LP, SC)
 - entry_credit
 - spot
 - deltas, IV skew (if present)
 - TTE (days)
 - realized PnL label
"""

import json
import os
import pandas as pd
import sqlite3
from datetime import datetime


RECORDS_PATH = "models/llm_trades/records.jsonl"
DB_PATH = "models/llm_trades/trades.db"
OUT_PATH = "models/llm_trades/features.parquet"


def load_trades_db():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT 
            ts,
            event,
            pnl,
            ic_json,
            note
        FROM trades
        ORDER BY ts
        """,
        con,
    )
    con.close()
    df["timestamp"] = pd.to_datetime(df["ts"])
    return df


def load_records():
    rows = []
    with open(RECORDS_PATH, "r") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def find_exit_pnl(trades_df, entry_ts):
    row = trades_df[
        (trades_df["event"].str.contains("exit", case=False, na=False))
        & (trades_df["timestamp"] > entry_ts)
    ].head(1)

    if row.empty:
        return None
    return float(row.iloc[0]["pnl"])


def compute_tte(entry_ts, expiry_str):
    if not expiry_str:
        return None
    try:
        expiry = pd.to_datetime(expiry_str)
        return float((expiry - entry_ts).total_seconds() / 86400)
    except:
        return None


def extract_features():
    records = load_records()
    trades_df = load_trades_db()

    feature_rows = []

    for rec in records:
        cand = rec.get("candidate", {})
        ts_str = rec.get("time")
        if not ts_str:
            continue

        entry_ts = pd.to_datetime(ts_str)

        c = cand  # IC dict
        try:
            sp = c.get("short_put")
            sc = c.get("short_call")
            lp = c.get("long_put")
            lc = c.get("long_call")
            ec = c.get("entry_credit")
            expiry = c.get("expiry")
        except Exception:
            continue

        chain = rec.get("chain_snapshot") or []
        # extract spot
        spot = None
        if chain:
            spot = chain[0].get("spot")

        # deltas if legs_greeks available
        legs = rec.get("legs_greeks") or []
        delta_sp = delta_sc = delta_lp = delta_lc = None
        iv_sp = iv_sc = iv_lp = iv_lc = None

        for leg in legs:
            typ = leg.get("type")
            if typ == "short_put":
                delta_sp = leg.get("delta")
                iv_sp = leg.get("iv")
            elif typ == "short_call":
                delta_sc = leg.get("delta")
                iv_sc = leg.get("iv")
            elif typ == "long_put":
                delta_lp = leg.get("delta")
                iv_lp = leg.get("iv")
            elif typ == "long_call":
                delta_lc = leg.get("delta")
                iv_lc = leg.get("iv")

        # simple iv skew
        iv_skew = None
        try:
            if iv_sc is not None and iv_sp is not None:
                iv_skew = float(iv_sc) - float(iv_sp)
        except:
            pass

        tte = compute_tte(entry_ts, expiry)

        pnl = find_exit_pnl(trades_df, entry_ts)

        label_win = None
        label_sign = None
        label_bucket = None

        if pnl is not None:
            label_win = 1 if pnl > 0 else 0
            label_sign = 1 if pnl > 0 else (-1 if pnl < 0 else 0)
            label_bucket = "win" if pnl > 0 else ("lose" if pnl < 0 else "flat")

        feature_rows.append(
            {
                "timestamp": entry_ts,
                "expiry": expiry,
                "short_put": sp,
                "short_call": sc,
                "long_put": lp,
                "long_call": lc,
                "entry_credit": ec,
                "spot": spot,
                "iv_skew": iv_skew,
                "delta_sp": delta_sp,
                "delta_sc": delta_sc,
                "delta_lp": delta_lp,
                "delta_lc": delta_lc,
                "tte_days": tte,
                "realized_pnl": pnl,
                "label_win": label_win,
                "label_sign": label_sign,
                "label_bucket": label_bucket,
            }
        )

    df = pd.DataFrame(feature_rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} feature rows → {OUT_PATH}")


if __name__ == "__main__":
    extract_features()
