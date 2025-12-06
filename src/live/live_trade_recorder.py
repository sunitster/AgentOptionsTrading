# src/live/live_trade_recorder.py
"""
Live Trade Recorder + Ingest utilities (patched, robust)

- Writes per-trade JSON + JSONL + optional parquet.
- Integrates with ML replay buffer:
    * If ReplayBufferSQLite is importable, uses its API.
    * Otherwise, falls back to direct sqlite writes into models/llm_trades/ml_replay.db
      creating a compatible `replays` table.
- Deterministic feature vector ordering and tolerant replay_id parsing.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import uuid
import os
import logging
import time as _time
import sqlite3

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

ROOT = Path("models") / "live_trades"
ROOT.mkdir(parents=True, exist_ok=True)

JSONL_PATH = ROOT / "live_trades.jsonl"
PARQUET_PATH = ROOT / "live_trades.parquet"

# ML replay DB path (match ml_pipeline default location)
ML_REPLAY_DB_PATH = Path("models") / "llm_trades" / "ml_replay.db"
ML_REPLAY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


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


def _ensure_replays_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS replays (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            ic_json TEXT,
            entry_snapshot_json TEXT,
            exit_snapshot_json TEXT,
            realized_pnl REAL,
            exit_reason TEXT,
            metadata_json TEXT,
            closed_at INTEGER
        );
        """
    )
    conn.commit()


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)
    except Exception:
        try:
            return json.dumps(str(obj))
        except Exception:
            return "\"<unserializable>\""


def _parse_replay_id(metadata: Dict[str, Any]) -> Optional[int]:
    if not isinstance(metadata, dict):
        return None
    # case-insensitive search for common keys
    keys = {k.lower(): k for k in metadata.keys()}
    for candidate in ("replay_id", "ml_replay_id", "replayid", "replayId", "id"):
        if candidate.lower() in keys:
            raw = metadata.get(keys[candidate.lower()])
            try:
                return int(raw)
            except Exception:
                try:
                    return int(float(raw))
                except Exception:
                    return None
    return None


def record_trade(
    ic: Dict[str, Any],
    entry_snapshot: Optional[Dict[str, Any]],
    exit_snapshot: Optional[Dict[str, Any]],
    realized_pnl: float,
    exit_reason: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record a single closed IC trade to disk and update ML replay buffer when possible.

    Parameters
    ----------
    ic: dict
    entry_snapshot: dict or None
    exit_snapshot: dict or None
    realized_pnl: float
    exit_reason: str
    metadata: dict (optional)
    """
    now = datetime.utcnow()
    ts = now.isoformat(timespec="seconds")
    uid = f"trade-{now.strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:8]}"

    metadata = metadata or {}

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
            try:
                existing = pd.read_parquet(PARQUET_PATH)
                combined = pd.concat([existing, df], ignore_index=True)
                combined.to_parquet(PARQUET_PATH, index=False)
            except Exception:
                # if parquet reading/appending fails just write new file
                df.to_parquet(PARQUET_PATH, index=False)
        else:
            df.to_parquet(PARQUET_PATH, index=False)
    except Exception:
        LOG.debug("pandas not available or parquet write failed; continuing with JSONL")

    # ------------------------------
    # ML replay buffer integration
    # ------------------------------
    # Behavior:
    # 1) If metadata contains 'replay_id' (or variants), update that replay entry's reward.
    # 2) Else: append a new replay row so training gets this sample immediately.
    try:
        ReplayBufferSQLite = None
        try:
            # prefer project-local class if available
            from src.ml_pipeline import ReplayBufferSQLite  # type: ignore
        except Exception:
            try:
                # fallback bare import
                from ml_pipeline import ReplayBufferSQLite  # type: ignore
            except Exception:
                ReplayBufferSQLite = None

        replay_id = _parse_replay_id(metadata)

        if ReplayBufferSQLite is not None:
            try:
                rb = ReplayBufferSQLite(path=str(ML_REPLAY_DB_PATH))
                if replay_id:
                    try:
                        rb.update_reward(replay_id, float(realized_pnl), closed_at=int(_time.time()))
                        LOG.info("Updated ML replay id=%s with reward=%.2f", replay_id, float(realized_pnl))
                    except Exception:
                        LOG.exception("Failed to update replay id=%s via ReplayBufferSQLite", replay_id)
                else:
                    # deterministic feature names -> ordered vector
                    features = {
                        "entry_credit": ic.get("entry_credit") if isinstance(ic, dict) else None,
                        "width": ic.get("width") if isinstance(ic, dict) else None,
                        "lot_size": ic.get("lot_size") if isinstance(ic, dict) else None,
                        "short_put": ic.get("short_put") if isinstance(ic, dict) else None,
                        "short_call": ic.get("short_call") if isinstance(ic, dict) else None,
                    }
                    names = list(features.keys())
                    vector = [features[n] for n in names]
                    try:
                        new_id = rb.append(entry_snapshot or {}, ic or {}, {"vector": vector, "names": names}, reward=float(realized_pnl), closed_at=int(_time.time()), meta={"source": "live_record", "origin_trade_id": uid})
                        LOG.info("Appended new ML replay id=%s from live trade record (pnl=%.2f)", new_id, float(realized_pnl))
                    except Exception:
                        LOG.exception("Failed to append new replay via ReplayBufferSQLite")
            except Exception:
                LOG.exception("ReplayBufferSQLite usage failed - falling back to direct sqlite insertion")
                ReplayBufferSQLite = None  # fall through to sqlite fallback

        # fallback: write directly into ml_replay.db if ReplayBufferSQLite unavailable
        if ReplayBufferSQLite is None:
            try:
                conn = sqlite3.connect(str(ML_REPLAY_DB_PATH), timeout=10, check_same_thread=False)
                _ensure_replays_table(conn)
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO replays (ts, ic_json, entry_snapshot_json, exit_snapshot_json, realized_pnl, exit_reason, metadata_json, closed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.utcnow().isoformat() + "Z",
                        _safe_json(ic if isinstance(ic, dict) else (ic.as_dict() if hasattr(ic, "as_dict") else str(ic))),
                        _safe_json(entry_snapshot or {}),
                        _safe_json(exit_snapshot or {}),
                        float(realized_pnl),
                        str(exit_reason),
                        _safe_json(metadata or {}),
                        int(_time.time()),
                    ),
                )
                conn.commit()
                inserted_id = cur.lastrowid
                conn.close()
                LOG.info("Inserted fallback ML replay id=%s from live trade (pnl=%.2f)", inserted_id, float(realized_pnl))
            except Exception:
                LOG.exception("Direct sqlite fallback into ml_replay.db failed (non-fatal)")

    except Exception:
        LOG.exception("ML replay buffer integration failed (non-fatal)")

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
