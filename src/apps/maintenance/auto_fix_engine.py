"""
AUTO-FIX ENGINE
Repairs corrupted snapshot, broker, risk, and PnL files.
Designed to run AFTER QC / anomaly detection.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
import logging

BASE = Path("models/llm_trades")
SNAP = BASE / "latest_snapshot.json"
BROKER = BASE / "paper_broker_state.json"
RISK = BASE / "risk_state.json"
PNL = BASE / "pnl_history.json"
TRADES_DB = BASE / "trades.db"

OUT_REPORT = BASE / "qc_autofix_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s auto_fix_engine: %(message)s"
)


def read_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"{path.name}: load failed: {e}")
        return None


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# =============================================================
# FIX SNAPSHOT
# =============================================================

def fix_snapshot():
    """
    RULES:
    - Must be dict with "snapshot" root OR convert array → dict wrapper
    - Every row needs: tradingsymbol, strike, expiry, last_price
    - Fill missing IV with None
    - Fill missing spot with 0
    - Remove corrupt rows
    """
    if not SNAP.exists():
        return {"status": "missing", "message": "file does not exist"}

    raw = read_json(SNAP)
    if raw is None:
        # recreate empty valid stub
        fixed = {"snapshot": [], "generated": True}
        write_json(SNAP, fixed)
        return {"status": "recreated", "message": "invalid JSON replaced"}

    # Case 1 — file is a list instead of an object
    if isinstance(raw, list):
        raw = {"snapshot": raw}

    if "snapshot" not in raw or not isinstance(raw["snapshot"], list):
        raw["snapshot"] = []

    rows = raw["snapshot"]
    fixed_rows = []

    for r in rows:
        if not isinstance(r, dict):
            continue
        if "tradingsymbol" not in r:
            continue
        if "strike" not in r:
            r["strike"] = 0
        if "expiry" not in r:
            r["expiry"] = ""
        if "last_price" not in r:
            r["last_price"] = 0.0
        if "iv" not in r:
            r["iv"] = None
        if "spot" not in r:
            r["spot"] = 0.0

        fixed_rows.append(r)

    raw["snapshot"] = fixed_rows
    write_json(SNAP, raw)

    return {
        "status": "fixed",
        "message": f"cleaned rows={len(fixed_rows)}"
    }


# =============================================================
# FIX BROKER STATE
# =============================================================

def fix_broker():
    """
    Broker state must contain:
      - open_positions: list
      - cash: float
      - pending_orders: list
    If missing: reconstruct safe defaults.
    """
    if not BROKER.exists():
        default = {
            "open_positions": [],
            "cash": 1_000_000.0,
            "pending_orders": []
        }
        write_json(BROKER, default)
        return {"status": "recreated", "message": "missing broker file recreated"}

    raw = read_json(BROKER)
    if raw is None:
        raw = {}

    if "open_positions" not in raw or not isinstance(raw["open_positions"], list):
        raw["open_positions"] = []

    if "cash" not in raw or not isinstance(raw["cash"], (int, float)):
        raw["cash"] = 1_000_000.0

    if "pending_orders" not in raw or not isinstance(raw["pending_orders"], list):
        raw["pending_orders"] = []

    write_json(BROKER, raw)
    return {"status": "fixed", "message": "broker file repaired"}


# =============================================================
# FIX RISK STATE
# =============================================================

def fix_risk():
    """
    Minimum required:
      - day (YYYY-MM-DD)
      - ic_entered_today
      - ic_closed_today
      - max_loss_hit
    """
    if not RISK.exists():
        default = {
            "day": datetime.utcnow().strftime("%Y-%m-%d"),
            "ic_entered_today": 0,
            "ic_closed_today": 0,
            "max_loss_hit": False
        }
        write_json(RISK, default)
        return {"status": "recreated", "message": "risk state missing -> recreated"}

    raw = read_json(RISK)
    if raw is None:
        raw = {}

    if "day" not in raw:
        raw["day"] = datetime.utcnow().strftime("%Y-%m-%d")

    raw.setdefault("ic_entered_today", 0)
    raw.setdefault("ic_closed_today", 0)
    raw.setdefault("max_loss_hit", False)

    write_json(RISK, raw)
    return {"status": "fixed", "message": "risk file repaired"}


# =============================================================
# FIX PNL HISTORY
# =============================================================

def fix_pnl():
    """
    PnL file must be a list of dicts. If invalid, recreate empty array.
    """
    if not PNL.exists():
        write_json(PNL, [])
        return {"status": "recreated", "message": "missing pnl history"}

    raw = read_json(PNL)
    if raw is None or not isinstance(raw, list):
        write_json(PNL, [])
        return {"status": "fixed", "message": "invalid pnl replaced with empty list"}

    # keep only valid entries
    good = []
    for p in raw:
        if (
            isinstance(p, dict)
            and "ts" in p
            and "pnl" in p
        ):
            good.append(p)

    write_json(PNL, good)
    return {"status": "fixed", "message": f"{len(good)} pnl entries retained"}


# =============================================================
# FIX TRADES.DB
# =============================================================

def fix_trades_db():
    try:
        conn = sqlite3.connect(TRADES_DB)
        cur = conn.cursor()
        # simple integrity check
        cur.execute("PRAGMA integrity_check;")
        res = cur.fetchone()
        conn.close()

        if res[0] != "ok":
            return {"status": "warning", "message": f"integrity={res[0]}"}

        return {"status": "ok", "message": "db OK"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================================================
# MAIN
# =============================================================

def run():
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "base_dir": str(BASE),
        "repairs": {
            "snapshot": fix_snapshot(),
            "broker": fix_broker(),
            "risk": fix_risk(),
            "pnl": fix_pnl(),
            "trades_db": fix_trades_db(),
        }
    }

    write_json(OUT_REPORT, report)
    logging.info(f"Auto-fix report saved -> {OUT_REPORT}")
    return report


if __name__ == "__main__":
    run()
