# src/cli/close_recovered.py
"""
CLI helper to safely close a recovered / ghost IC position.

Usage:
    python -m src.cli.close_recovered

What it does:
- Reads monitor files under models/llm_trades:
    - current_position.json
    - paper_broker_state.json
    - recovered_position.json (if present)
    - trades.db (if present)
- If there is an open/recovered position:
    - Clears open_positions in paper_broker_state.json
    - Marks current_position.json as closed
    - Appends a 'recovered_manual_close' row into trades.db with pnl=0.0
      (so we have an audit trail)
- If nothing looks open, it just prints a message and exits.

This does NOT talk to the running LivePaperEngine process. For safety, run it
with the engine stopped (or restart the engine afterward) so in-memory state
doesn't conflict with the cleaned monitor files.
"""

from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


MONITOR_DIR = Path("models") / "llm_trades"
CURRENT_POS_FILE = MONITOR_DIR / "current_position.json"
BROKER_STATE_FILE = MONITOR_DIR / "paper_broker_state.json"
RECOVERED_POS_FILE = MONITOR_DIR / "recovered_position.json"
TRADES_DB_FILE = MONITOR_DIR / "trades.db"


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _safe_write_json(path: Path, obj: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return True
    except Exception:
        return False


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        # try date-only variant
        try:
            return datetime.fromisoformat(ts.split("T")[0])
        except Exception:
            return None


def _insert_recovered_close_db(ic_snapshot: Optional[Dict[str, Any]], duration_seconds: Optional[float]) -> None:
    """
    Append a 'recovered_manual_close' event into trades.db if it exists.
    pnl is recorded as 0.0 (we're just acknowledging closure of a ghost).
    """
    try:
        if not TRADES_DB_FILE.exists():
            print("[close_recovered] trades.db not found; skipping DB entry.")
            return

        conn = sqlite3.connect(str(TRADES_DB_FILE))
        cur = conn.cursor()

        ts = datetime.utcnow().isoformat()
        event = "recovered_manual_close"
        note = "manual close of recovered/ghost position from monitor files"

        if ic_snapshot is not None:
            try:
                ic_json = json.dumps(ic_snapshot, default=str)
            except Exception:
                ic_json = json.dumps({"repr": str(ic_snapshot)}, default=str)
            symbol_summary = str(
                ic_snapshot.get("symbol")
                or ic_snapshot.get("symbol_summary")
                or ic_snapshot.get("short_put_sym")
                or ic_snapshot.get("short_call_sym")
                or "ghost_ic"
            )[:200]
        else:
            ic_json = None
            symbol_summary = "ghost_ic"

        pnl = 0.0

        cur.execute(
            """
            INSERT INTO trades (ts, event, symbol_summary, ic_json, pnl, duration_seconds, note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, event, symbol_summary, ic_json, pnl, duration_seconds, note),
        )
        conn.commit()
        conn.close()
        print("[close_recovered] DB entry written: recovered_manual_close")
    except Exception as e:
        print(f"[close_recovered] WARNING: failed to write to trades.db: {e}")


def close_recovered() -> None:
    # Load monitor files
    broker = _safe_load_json(BROKER_STATE_FILE) or {}
    current = _safe_load_json(CURRENT_POS_FILE) or {}
    recovered = _safe_load_json(RECOVERED_POS_FILE) or {}

    open_positions = broker.get("open_positions") or []

    # Detect presence of an IC snapshot
    ic_snapshot: Optional[Dict[str, Any]] = None

    # Case 1: current_position looks like an IC dict (has short_put/short_call or *_sym)
    if isinstance(current, dict) and any(
        k in current for k in ("short_put", "short_call", "short_put_sym", "short_call_sym")
    ):
        ic_snapshot = current

    # Case 2: recovered_position.json has 'ic' or 'ic_snapshot'
    if ic_snapshot is None and isinstance(recovered, dict):
        if "ic" in recovered and isinstance(recovered["ic"], dict):
            ic_snapshot = recovered["ic"]
        elif "ic_snapshot" in recovered and isinstance(recovered["ic_snapshot"], dict):
            ic_snapshot = recovered["ic_snapshot"]

    has_open_positions = bool(open_positions)
    has_ic_snapshot = ic_snapshot is not None

    if not has_open_positions and not has_ic_snapshot:
        print("[close_recovered] No recovered/ghost position detected. Nothing to close.")
        return

    print("[close_recovered] Detected recovered/ghost position.")
    print(f"  - open_positions in broker_state: {len(open_positions)}")
    print(f"  - IC snapshot present: {bool(ic_snapshot)}")

    # Estimate duration from open_ic_entry_time if present
    entry_time_iso = None
    if isinstance(current, dict):
        entry_time_iso = current.get("open_ic_entry_time") or current.get("entry_time")
    if not entry_time_iso and isinstance(recovered, dict):
        entry_time_iso = recovered.get("open_ic_entry_time") or recovered.get("entry_time")

    entry_dt = _parse_iso(entry_time_iso)
    now = datetime.utcnow()
    duration_seconds: Optional[float] = None
    if entry_dt is not None:
        duration_seconds = max(0.0, (now - entry_dt).total_seconds())

    # 1) Clear broker_state open_positions (but keep balances and history info)
    broker["open_positions"] = []
    broker["last_closed_at"] = now.isoformat()
    broker.setdefault("note", "recovered_manual_close")
    _safe_write_json(BROKER_STATE_FILE, broker)
    print(f"[close_recovered] Cleared open_positions in {BROKER_STATE_FILE.name}")

    # 2) Mark current_position as closed
    new_current: Dict[str, Any]
    if isinstance(current, dict):
        new_current = dict(current)
    else:
        new_current = {}

    new_current["open_ic_entry_time"] = None
    new_current["closed_by_close_recovered_at"] = now.isoformat()
    _safe_write_json(CURRENT_POS_FILE, new_current)
    print(f"[close_recovered] Updated {CURRENT_POS_FILE.name} to reflect closed state")

    # 3) Update recovered_position.json to mark that we've acted
    new_recovered: Dict[str, Any] = dict(recovered) if isinstance(recovered, dict) else {}
    new_recovered["closed_by_close_recovered_at"] = now.isoformat()
    new_recovered.setdefault("note", "recovered_manual_close")
    _safe_write_json(RECOVERED_POS_FILE, new_recovered)
    print(f"[close_recovered] Marked {RECOVERED_POS_FILE.name} as closed by helper")

    # 4) Append audit row to trades.db if present
    _insert_recovered_close_db(ic_snapshot=ic_snapshot, duration_seconds=duration_seconds)

    print("[close_recovered] Done. Restart your live engine so it starts from a clean state.")


if __name__ == "__main__":
    close_recovered()
