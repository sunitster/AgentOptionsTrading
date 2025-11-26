# src/live/live_trade_recorder.py
"""
Live Trade Recorder + Ingest utilities

This module provides two main pieces:

1) record_trade(...) -> called by LivePaperEngine when an IC is closed (realized)
   - writes a JSONL line into models/live_trades/live_trades.jsonl
   - writes a per-trade JSON into models/live_trades/YYYYMMDD_HHMMSS-<posid>.json
   - attempts to also append to a parquet file models/live_trades/live_trades.parquet

2) ingest_live_trades(...) -> used by the training pipeline to convert the
   collected live trades into a tidy parquet file that can be consumed by
   src/learning/run_full_historical_training (or your feature store).

Design goals:
- Non-intrusive: writing files under models/live_trades only
- Robust: tolerates partial/malformed inputs
- Minimal dependencies: pandas used only if available

Usage (from LivePaperEngine):

from src.live.live_trade_recorder import record_trade

# after a trade exit has been executed and you have:
# - ic_dict: dictionary representation of the IC
# - entry_snapshot: snapshot dict at entry (can be chain_df.iloc[0].to_dict())
# - exit_snapshot: snapshot dict at exit
# - realized_pnl: float
# - exit_reason: string
# - metadata: dict (optional)

record_trade(ic_dict, entry_snapshot, exit_snapshot, realized_pnl, exit_reason, metadata)


Usage (for training ingestion):

from src.live.live_trade_recorder import ingest_live_trades
ingest_live_trades(output_parquet_path='learning_data/live_trades.parquet')

"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import uuid
import os
import logging

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

ROOT = Path("models") / "live_trades"
ROOT.mkdir(parents=True, exist_ok=True)

JSONL_PATH = ROOT / "live_trades.jsonl"
PARQUET_PATH = ROOT / "live_trades.parquet"


def _safe_dump_json(path: Path, data: Dict[str, Any]):
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, default=str, indent=2)
    except Exception:
        LOG.exception("Failed to write json to %s", path)


def _safe_append_jsonl(path: Path, data: Dict[str, Any]):
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")
    except Exception:
        LOG.exception("Failed to append jsonl to %s", path)


def record_trade(
    ic: Dict[str, Any],
    entry_snapshot: Optional[Dict[str, Any]],
    exit_snapshot: Optional[Dict[str, Any]],
    realized_pnl: float,
    exit_reason: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record a single closed IC trade to disk. Returns the saved record dict.

    Parameters
    ----------
    ic: dict
        Dictionary-like representation of the IronCondor (legs, strikes, credit, lot_size, etc.)
    entry_snapshot: dict or None
        Snapshot at entry (first row of chain_df converted to dict) - may contain 'spot', 'timestamp'
    exit_snapshot: dict or None
        Snapshot at exit - may contain 'spot', 'timestamp'
    realized_pnl: float
        Realized PnL for the whole IC (positive or negative)
    exit_reason: str
        Short code for why trade exited (TIME_EXIT, SL, TP, MANUAL, etc.)
    metadata: dict (optional)
        Any extra contextual metadata to save
    """
    now = datetime.utcnow()
    ts = now.isoformat(timespec="seconds")
    uid = f"trade-{now.strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:8]}"

    record = {
        "id": uid,
        "time_utc": ts,
        "ic": ic,
        "entry_snapshot": entry_snapshot or {},
        "exit_snapshot": exit_snapshot or {},
        "realized_pnl": float(realized_pnl),
        "exit_reason": exit_reason,
        "metadata": metadata or {},
    }

    # Write full JSON per trade
    fname = ROOT / f"{uid}.json"
    _safe_dump_json(fname, record)

    # Append to JSONL master file
    _safe_append_jsonl(JSONL_PATH, record)

    # Try to also append to parquet for fast ingestion later (if pandas available)
    try:
        import pandas as pd

        # flatten record into a single-row DataFrame
        flat = {
            "id": record["id"],
            "time_utc": record["time_utc"],
            "realized_pnl": record["realized_pnl"],
            "exit_reason": record["exit_reason"],
        }

        # extract some common fields if present
        try:
            flat["entry_spot"] = float(entry_snapshot.get("spot")) if entry_snapshot and entry_snapshot.get("spot") is not None else None
        except Exception:
            flat["entry_spot"] = None
        try:
            flat["exit_spot"] = float(exit_snapshot.get("spot")) if exit_snapshot and exit_snapshot.get("spot") is not None else None
        except Exception:
            flat["exit_spot"] = None
        # IC-level fields
        try:
            flat["entry_credit"] = float(ic.get("entry_credit")) if ic.get("entry_credit") is not None else None
        except Exception:
            flat["entry_credit"] = None
        try:
            flat["lot_size"] = int(ic.get("lot_size")) if ic.get("lot_size") is not None else None
        except Exception:
            flat["lot_size"] = None
        try:
            flat["short_put"] = float(ic.get("short_put")) if ic.get("short_put") is not None else None
            flat["short_call"] = float(ic.get("short_call")) if ic.get("short_call") is not None else None
            flat["long_put"] = float(ic.get("long_put")) if ic.get("long_put") is not None else None
            flat["long_call"] = float(ic.get("long_call")) if ic.get("long_call") is not None else None
        except Exception:
            pass

        df = pd.DataFrame([flat])
        # Append to parquet (fast). If file exists, concat; else create
        if PARQUET_PATH.exists():
            existing = pd.read_parquet(PARQUET_PATH)
            combined = pd.concat([existing, df], ignore_index=True)
            combined.to_parquet(PARQUET_PATH, index=False)
        else:
            df.to_parquet(PARQUET_PATH, index=False)
    except Exception:
        # pandas not installed or write failed - that's ok, we have JSONL
        LOG.debug("pandas not available or parquet write failed; continuing with JSONL")

    LOG.info("Recorded live trade %s realized_pnl=%.2f reason=%s", uid, realized_pnl, exit_reason)
    return record


def ingest_live_trades(output_parquet_path: str = "learning_data/live_trades.parquet") -> Path:
    """Ingest trades from models/live_trades into a single parquet file suitable for training.

    - Reads models/live_trades/live_trades.jsonl if present
    - Reads per-trade json files if present
    - Produces a tidy parquet with flattened columns
    - Returns Path to the parquet file
    """
    out_path = Path(output_parquet_path)
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required to run ingest_live_trades") from e

    rows = []
    # 1) jsonl
    if JSONL_PATH.exists():
        try:
            with JSONL_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        rows.append(obj)
                    except Exception:
                        LOG.exception("Bad jsonl line - skipping")
        except Exception:
            LOG.exception("Failed to read jsonl file")

    # 2) per-file jsons
    for p in ROOT.glob("trade-*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
                rows.append(obj)
        except Exception:
            LOG.exception("Failed to read %s", p)

    if not rows:
        LOG.info("No live trades found to ingest")
        # produce empty parquet with basic columns
        df = pd.DataFrame(columns=["id", "time_utc", "realized_pnl", "entry_spot", "exit_spot", "exit_reason"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return out_path

    # Flatten rows
    flat_rows = []
    for r in rows:
        try:
            flat = {
                "id": r.get("id"),
                "time_utc": r.get("time_utc"),
                "realized_pnl": r.get("realized_pnl"),
                "exit_reason": r.get("exit_reason"),
                # entry/exit spots if available
                "entry_spot": (r.get("entry_snapshot") or {}).get("spot"),
                "exit_spot": (r.get("exit_snapshot") or {}).get("spot"),
            }
            # pull some IC fields
            ic = r.get("ic") or {}
            flat["entry_credit"] = ic.get("entry_credit") or ic.get("total_entry_credit") or None
            flat["lot_size"] = ic.get("lot_size") or ic.get("lot") or None
            flat["short_put"] = ic.get("short_put")
            flat["short_call"] = ic.get("short_call")
            flat["long_put"] = ic.get("long_put")
            flat["long_call"] = ic.get("long_call")
            # metadata
            flat.update({f"meta_{k}": v for k, v in (r.get("metadata") or {}).items() if isinstance(v, (str, int, float, bool))})
            flat_rows.append(flat)
        except Exception:
            LOG.exception("Failed to flatten trade row")

    df = pd.DataFrame(flat_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    LOG.info("Ingested %d live trades -> %s", len(df), out_path)
    return out_path


if __name__ == "__main__":
    # quick CLI to ingest from models/live_trades into learning_data
    try:
        p = ingest_live_trades()
        print("Wrote:", p)
    except Exception as e:
        print("Failed:", e)
