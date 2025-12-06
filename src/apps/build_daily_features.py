#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build daily features for ML training.

Reads:
 - models/llm_trades/records.jsonl   (entry candidate records)
 - models/llm_trades/trades.db       (exit/entry DB with realized pnl)

Produces:
 - models/llm_trades/features.parquet

This script is defensive: missing files won't raise an uncaught exception;
it will log and exit gracefully.
"""

from __future__ import annotations
import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np

# Config / paths
MONITOR_DIR = os.path.join("models", "llm_trades")
RECORDS_PATH = os.path.join(MONITOR_DIR, "records.jsonl")
TRADES_DB_PATH = os.path.join(MONITOR_DIR, "trades.db")
OUT_PARQUET = os.path.join(MONITOR_DIR, "features.parquet")
LOG = logging.getLogger("build_daily_features")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def safe_read_records(records_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(records_path):
        LOG.warning("records.jsonl not found at %s — returning empty list", records_path)
        return []
    rows = []
    with open(records_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                LOG.exception("Skipping malformed record line")
    LOG.info("Loaded %d records from %s", len(rows), records_path)
    return rows


def load_trades_db(trades_db_path: str) -> pd.DataFrame:
    if not os.path.exists(trades_db_path):
        LOG.warning("trades.db not found at %s — returning empty DataFrame", trades_db_path)
        return pd.DataFrame(columns=["ts", "event", "pnl", "ic_json", "note"])
    try:
        con = sqlite3.connect(trades_db_path)
        df = pd.read_sql("SELECT id, ts, event, pnl, ic_json, note FROM trades ORDER BY ts ASC", con)
        con.close()
        if "ts" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ts"], errors="coerce")
        else:
            df["timestamp"] = pd.NaT
        LOG.info("Loaded %d rows from trades.db", len(df))
        return df
    except Exception:
        LOG.exception("Failed to read trades.db; returning empty DataFrame")
        return pd.DataFrame(columns=["id", "ts", "event", "pnl", "ic_json", "note", "timestamp"])


def find_exit_for_entry(trades_df: pd.DataFrame, entry_time: datetime, replay_id: Optional[int] = None) -> Optional[pd.Series]:
    """
    Find the first 'exit' event after entry_time.
    If replay_id provided, prefer exit with matching replay_id in note or ic_json.
    """
    if trades_df.empty:
        return None
    # restrict to events that look like exits
    exits = trades_df[trades_df["event"].str.contains("exit", case=False, na=False)].copy()
    if exits.empty:
        return None

    # if replay_id given, try to match safest
    if replay_id is not None:
        # try to match replay_id in note or ic_json
        def match_replay(row):
            try:
                txt = ""
                if pd.notna(row.get("note")):
                    txt += str(row.get("note"))
                if pd.notna(row.get("ic_json")):
                    txt += " " + str(row.get("ic_json"))
                return str(replay_id) in txt
            except Exception:
                return False
        candidates = exits[exits.apply(match_replay, axis=1)]
        # choose first after entry_time
        candidates = candidates[candidates["timestamp"] > entry_time] if not candidates.empty else pd.DataFrame()
        if not candidates.empty:
            return candidates.iloc[0]

    # fallback: first exit with timestamp > entry_time
    try:
        candidates = exits[exits["timestamp"] > entry_time]
        if candidates.empty:
            return None
        return candidates.iloc[0]
    except Exception:
        # fallback: attempt to parse ts strings
        try:
            exits["timestamp"] = pd.to_datetime(exits["ts"], errors="coerce")
            candidates = exits[exits["timestamp"] > entry_time]
            if candidates.empty:
                return None
            return candidates.iloc[0]
        except Exception:
            return None


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
            return float(v)
        s = str(v).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def compute_iv_skew_and_deltas(record: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[List[Dict[str, Any]]]]:
    """
    Try to extract iv_skew and per-leg deltas/ivs from a record's legs_greeks or chain_snapshot.
    Returns (iv_skew, delta_sp, delta_sc, avg_delta_abs, legs_greeks)
    """
    legs = record.get("legs_greeks") or []
    iv_skew = None
    delta_sp = delta_sc = None
    deltas = []
    try:
        # legs may be a list of dicts with keys: type, delta, iv, symbol
        for lg in legs:
            try:
                t = (lg.get("type") or "").lower()
                d = _safe_float(lg.get("delta") if lg.get("delta") is not None else lg.get("delta_raw") if lg.get("delta_raw") is not None else lg.get("greek_delta"))
                iv = _safe_float(lg.get("iv"))
                deltas.append(d if d is not None else 0.0)
                if "short_put" in t or t.endswith("put") or "short_put" == t:
                    delta_sp = d
                if "short_call" in t or t.endswith("call") or "short_call" == t:
                    delta_sc = d
            except Exception:
                continue
        # iv skew attempt: call iv - put iv for short legs if present
        iv_put = iv_call = None
        for lg in legs:
            try:
                t = (lg.get("type") or "").lower()
                iv = _safe_float(lg.get("iv"))
                if "short_put" in t or t.endswith("put"):
                    iv_put = iv
                if "short_call" in t or t.endswith("call"):
                    iv_call = iv
            except Exception:
                continue
        if iv_call is not None and iv_put is not None:
            iv_skew = float(iv_call - iv_put)
        avg_delta_abs = float(np.mean([abs(x) for x in deltas])) if deltas else None
        return iv_skew, delta_sp, delta_sc, avg_delta_abs, legs
    except Exception:
        return None, None, None, None, legs


def compute_tte_days(entry_time: datetime, expiry_str: Optional[str]) -> Optional[float]:
    if expiry_str is None:
        return None
    try:
        # expiry may be "2025-12-02" or "2025-12-02T15:30:00"
        try:
            exp_dt = datetime.fromisoformat(expiry_str)
        except Exception:
            # try date-only parse
            exp_dt = datetime.fromisoformat(expiry_str.split("T")[0])
        delta = exp_dt - entry_time
        return delta.total_seconds() / 86400.0
    except Exception:
        return None


def extract_candidate_features(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a record (entry candidate), extract features for ML.
    """
    out: Dict[str, Any] = {}
    time_str = record.get("time") or record.get("timestamp")
    try:
        entry_ts = datetime.fromisoformat(time_str) if time_str else None
    except Exception:
        try:
            entry_ts = pd.to_datetime(time_str, errors="coerce").to_pydatetime() if time_str else None
        except Exception:
            entry_ts = None

    out["timestamp"] = entry_ts
    cand = record.get("candidate") or {}
    # candidate can be an object-like dict; try to pull typical fields
    def _get(k):
        try:
            if isinstance(cand, dict):
                return cand.get(k)
            # if cand is an object with attributes
            return getattr(cand, k, None)
        except Exception:
            return None

    out["expiry"] = _get("expiry") or _get("exp") or _get("expiration") or None
    out["short_put"] = _safe_float(_get("short_put") or _get("sp") or _get("short_put_strike") or _get("short_put_str"))
    out["long_put"] = _safe_float(_get("long_put") or _get("lp"))
    out["short_call"] = _safe_float(_get("short_call") or _get("sc"))
    out["long_call"] = _safe_float(_get("long_call") or _get("lc"))
    out["entry_credit"] = _safe_float(_get("entry_credit")) or 0.0
    out["lot_size"] = int(_safe_float(_get("lot_size")) or 25)
    out["size_aggressiveness"] = float(_safe_float(_get("size_aggressiveness")) or 1.0)
    out["replay_id"] = record.get("replay_id") or (record.get("candidate", {}).get("replay_id") if isinstance(record.get("candidate"), dict) else None)
    # chain snapshot to extract spot
    chain = record.get("chain_snapshot")
    spot = None
    if chain and isinstance(chain, list) and len(chain) > 0:
        first = chain[0]
        # try common keys
        for k in ("spot", "underlying", "underlying_price", "ltp", "last_price"):
            try:
                v = first.get(k)
                if v is not None:
                    spot = _safe_float(v)
                    if spot is not None:
                        break
            except Exception:
                continue
    out["spot"] = spot
    # iv/delta features
    iv_skew, delta_sp, delta_sc, avg_delta_abs, legs = compute_iv_skew_and_deltas(record)
    out["iv_skew"] = iv_skew
    out["delta_sp"] = delta_sp
    out["delta_sc"] = delta_sc
    out["avg_delta_abs"] = avg_delta_abs
    out["legs_greeks"] = legs
    # TTE
    out["tte_days"] = compute_tte_days(entry_ts if entry_ts is not None else datetime.utcnow(), out["expiry"])
    # human features
    out["candidate_raw"] = cand
    return out


def build_features_and_labels(records_path: str, trades_db_path: str, out_parquet: str) -> None:
    # load sources
    records = safe_read_records(records_path)
    trades_df = load_trades_db(trades_db_path)

    features = []
    for rec in records:
        try:
            feat = extract_candidate_features(rec)
            entry_ts = feat.get("timestamp")
            replay_id = feat.get("replay_id")
            # find exit for this entry
            exit_row = None
            if entry_ts is not None:
                exit_row = find_exit_for_entry(trades_df, entry_ts, replay_id=replay_id)
            # get realized pnl
            realized_pnl = None
            exit_reason = None
            if exit_row is not None:
                try:
                    realized_pnl = float(exit_row.get("pnl")) if exit_row.get("pnl") is not None else None
                except Exception:
                    realized_pnl = None
                exit_reason = exit_row.get("note") or exit_row.get("event")
            feat["realized_pnl"] = realized_pnl
            feat["exit_reason"] = exit_reason
            # derive labels
            if realized_pnl is None:
                feat["label_win"] = None
                feat["label_sign"] = None
            else:
                feat["label_win"] = 1 if realized_pnl > 0 else 0
                feat["label_sign"] = 1 if realized_pnl > 0 else (-1 if realized_pnl < 0 else 0)
            features.append(feat)
        except Exception:
            LOG.exception("failed to extract features for a record")

    if not features:
        LOG.warning("No features extracted — nothing to write")
        return

    # create DataFrame
    df = pd.DataFrame(features)
    # Normalize/convert columns
    # Keep columns we want for ML + meta
    keep_cols = [
        "timestamp", "expiry",
        "short_put", "long_put", "short_call", "long_call",
        "entry_credit", "lot_size", "size_aggressiveness",
        "spot", "iv_skew", "delta_sp", "delta_sc", "avg_delta_abs",
        "tte_days",
        "realized_pnl", "label_win", "label_sign",
        "replay_id", "exit_reason"
    ]
    # ensure columns exist
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None

    # cast types
    try:
        df["entry_credit"] = df["entry_credit"].astype(float).fillna(0.0)
    except Exception:
        df["entry_credit"] = df["entry_credit"].apply(lambda x: float(x) if x is not None else 0.0)

    # spot may be None; fill with 0
    df["spot"] = df["spot"].apply(lambda x: float(x) if x is not None else np.nan)

    # tte -> numeric
    df["tte_days"] = df["tte_days"].apply(lambda x: float(x) if x is not None else np.nan)

    # reorder columns
    out_df = df[keep_cols].copy()
    # add any meta we want to persist as JSON string columns (legs_greeks, candidate_raw)
    out_df["legs_greeks"] = df.get("legs_greeks").apply(lambda x: json.dumps(x, default=str) if x is not None else None)
    out_df["candidate_raw"] = df.get("candidate_raw").apply(lambda x: json.dumps(x, default=str) if x is not None else None)
    out_df["built_at"] = datetime.utcnow().isoformat()

    # ensure output directory exists
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    try:
        # write parquet
        out_df.to_parquet(out_parquet, index=False)
        LOG.info("Wrote %d feature rows to %s", len(out_df), out_parquet)
    except Exception:
        LOG.exception("Failed to write features.parquet; attempting CSV fallback")
        try:
            csv_path = out_parquet.replace(".parquet", ".csv")
            out_df.to_csv(csv_path, index=False)
            LOG.info("Wrote CSV fallback to %s", csv_path)
        except Exception:
            LOG.exception("Failed to write CSV fallback as well")


def main():
    LOG.info("Starting build_daily_features")
    build_features_and_labels(RECORDS_PATH, TRADES_DB_PATH, OUT_PARQUET)
    LOG.info("build_daily_features finished")


if __name__ == "__main__":
    main()
