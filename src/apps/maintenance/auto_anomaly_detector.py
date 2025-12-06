#!/usr/bin/env python3
"""
auto_anomaly_detector.py

Cron-style detector (run-once). Reads artifacts in models/llm_trades and produces:
 - qc_anomalies_realtime.json    (detailed anomaly + metric snapshot)
 - auto_pause_trigger.json       (if immediate pause conditions met)
 - retrain_trigger.json          (if retrain conditions met)

Behaviour intentionally conservative:
 - produces informative report, does NOT modify live trading state here.
 - writes pause/retrain trigger files for an external operator or orchestrator to act on.
"""

from __future__ import annotations
import os
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
import math
import statistics

# --- CONFIG ---
BASE_DIR = Path("models") / "llm_trades"
SNAPSHOT_FNAME = "latest_snapshot.json"
PNL_FNAME = "pnl_history.json"
BROKER_FNAME = "paper_broker_state.json"
RISK_FNAME = "risk_state.json"
TRADES_DB = "trades.db"

OUT_ANOMALIES = BASE_DIR / "qc_anomalies_realtime.json"
OUT_PAUSE = BASE_DIR / "auto_pause_trigger.json"
OUT_RETRAIN = BASE_DIR / "retrain_trigger.json"

# thresholds (tune these)
THRESH = {
    "spot_missing_seconds": 30,
    "all_ltp_zero_pct": 0.95,        # 95%+ LTP zero -> bad
    "iv_zero_pct": 0.95,             # 95%+ IV zero -> bad
    "no_valid_ticks_seconds": 45,    # no fetched_at updates in this many seconds -> bad
    "db_min_rows": 1,
    "pnl_stagnant_days": 3,          # for retrain consideration (if PnL variance near zero)
    "retrain_drift_std_mult": 3.0,   # drift > mean + mult*std -> trigger retrain
}

# set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# --- utils ---
def safe_load_json(path: Path) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning("Missing file: %s", path)
        return None
    except json.JSONDecodeError as e:
        logging.warning("JSON decode error for %s: %s", path, str(e))
        return {"__json_error__": str(e)}
    except Exception as e:
        logging.exception("Failed reading JSON %s: %s", path, e)
        return {"__error__": str(e)}


def now_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    tmp.replace(path)


# --- individual checks ---
def check_snapshot(snapshot_path: Path) -> Dict[str, Any]:
    out = {
        "file": snapshot_path.name,
        "exists": snapshot_path.exists(),
        "status": "unknown",
        "metrics": {},
        "message": "",
    }
    data = safe_load_json(snapshot_path)
    if data is None:
        out["status"] = "missing"
        out["message"] = "file not found"
        return out
    if isinstance(data, dict):
        # expected shape: root is object with maybe 'snapshot' key OR the root is the snapshot list itself
        root = data
        if "snapshot" in root and isinstance(root["snapshot"], list):
            rows = root["snapshot"]
        elif isinstance(root.get("rows", None), list):
            rows = root["rows"]
        elif isinstance(root, list):
            rows = root
        else:
            # not the typical snapshot shape
            # attempt to find a list inside top-level keys
            possible = next((v for v in root.values() if isinstance(v, list)), None)
            if possible is not None:
                rows = possible
            else:
                out["status"] = "invalid"
                out["message"] = "Snapshot root is not a list/object with snapshot rows"
                return out
    elif isinstance(data, list):
        rows = data
    else:
        out["status"] = "invalid"
        out["message"] = "Unexpected JSON root type"
        return out

    n = len(rows)
    out["metrics"]["rows"] = n
    num_with_ltp = 0
    num_with_iv = 0
    num_with_spot = 0
    fetched_at_vals = []
    sample_tradings = []
    strikes = set()

    for r in rows:
        # tolerate different field names
        lp = r.get("last_price") if isinstance(r, dict) else None
        if lp is None and isinstance(r, dict):
            lp = r.get("ltp") or r.get("lastPrice") or r.get("last")
        try:
            if lp is not None and float(lp) != 0.0:
                num_with_ltp += 1
        except Exception:
            pass

        iv = r.get("iv") if isinstance(r, dict) else None
        try:
            if iv is not None and float(iv) != 0.0:
                num_with_iv += 1
        except Exception:
            pass

        spot = r.get("spot") if isinstance(r, dict) else None
        try:
            if spot is not None and float(spot) != 0.0:
                num_with_spot += 1
        except Exception:
            pass

        # fetched_at
        fa = r.get("fetched_at") if isinstance(r, dict) else None
        if isinstance(r, dict) and not fa:
            fa = r.get("fetchedAt") or r.get("fetched")
        if fa:
            try:
                # try parse as iso or epoch
                if isinstance(fa, (int, float)):
                    fetched_at_vals.append(datetime.fromtimestamp(float(fa), tz=timezone.utc))
                else:
                    fetched_at_vals.append(datetime.fromisoformat(fa))
            except Exception:
                pass

        # sample tradingsymbol
        tsym = r.get("tradingsymbol") if isinstance(r, dict) else None
        if tsym:
            sample_tradings.append(tsym)

        strike = r.get("strike") if isinstance(r, dict) else None
        if strike is not None:
            try:
                strikes.add(float(strike))
            except Exception:
                pass

    out["metrics"]["num_with_ltp"] = num_with_ltp
    out["metrics"]["num_with_iv"] = num_with_iv
    out["metrics"]["num_with_spot"] = num_with_spot
    out["metrics"]["pct_with_ltp"] = (num_with_ltp / n) if n else 0.0
    out["metrics"]["pct_with_iv"] = (num_with_iv / n) if n else 0.0
    out["metrics"]["pct_with_spot"] = (num_with_spot / n) if n else 0.0
    out["metrics"]["sample_tradings"] = sample_tradings[:10]
    out["metrics"]["sample_strikes"] = sorted(list(strikes))[:10]

    # latest fetched_at age
    if fetched_at_vals:
        latest = max(fetched_at_vals)
        age_s = (datetime.now(timezone.utc) - latest).total_seconds()
        out["metrics"]["latest_fetched_at"] = latest.isoformat()
        out["metrics"]["latest_fetched_age_s"] = age_s
    else:
        out["metrics"]["latest_fetched_at"] = None
        out["metrics"]["latest_fetched_age_s"] = None

    # decide status
    if n == 0:
        out["status"] = "invalid"
        out["message"] = "snapshot contains zero rows"
    elif out["metrics"]["pct_with_ltp"] < (1.0 - THRESH["all_ltp_zero_pct"]):
        # if less than 5% have LTP -> problematic
        out["status"] = "bad"
        out["message"] = f"Very few rows have LTP ({out['metrics']['pct_with_ltp']:.3f})"
    elif out["metrics"]["pct_with_iv"] < (1.0 - THRESH["iv_zero_pct"]):
        out["status"] = "bad"
        out["message"] = f"Very few rows have IV ({out['metrics']['pct_with_iv']:.3f})"
    else:
        out["status"] = "ok"
        out["message"] = "ok"

    return out


def check_pnl(pnl_path: Path) -> Dict[str, Any]:
    out = {
        "file": pnl_path.name,
        "exists": pnl_path.exists(),
        "status": "unknown",
        "metrics": {},
        "message": "",
    }
    data = safe_load_json(pnl_path)
    if not data:
        out["status"] = "missing" if data is None else "invalid"
        out["message"] = "missing or invalid"
        return out
    # expected list of {"timestamp": ..., "pnl": number}
    try:
        if isinstance(data, dict) and "pnl_history" in data:
            rows = data["pnl_history"]
        elif isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and all(isinstance(v, (int, float)) for v in data.values()):
            # coarse fallback
            rows = [{"pnl": v, "ts": k} for k, v in data.items()]
        else:
            rows = data.get("values") if isinstance(data, dict) else []
    except Exception:
        rows = []

    pnl_vals = []
    for r in rows:
        if isinstance(r, dict):
            try:
                pnl = r.get("pnl") or r.get("pnl_value") or r.get("value")
                if pnl is None and "y" in r:
                    pnl = r["y"]
                if pnl is not None:
                    pnl_vals.append(float(pnl))
            except Exception:
                pass

    out["metrics"]["count"] = len(pnl_vals)
    if pnl_vals:
        try:
            out["metrics"]["mean"] = statistics.mean(pnl_vals)
            out["metrics"]["stdev"] = statistics.pstdev(pnl_vals) if len(pnl_vals) > 1 else 0.0
            out["metrics"]["min"] = min(pnl_vals)
            out["metrics"]["max"] = max(pnl_vals)
        except Exception:
            pass

    if out["metrics"].get("count", 0) == 0:
        out["status"] = "invalid"
        out["message"] = "no pnl points"
    else:
        out["status"] = "ok"
        out["message"] = "ok"

    return out


def check_broker(broker_path: Path) -> Dict[str, Any]:
    out = {
        "file": broker_path.name,
        "exists": broker_path.exists(),
        "status": "unknown",
        "metrics": {},
        "message": "",
    }
    data = safe_load_json(broker_path)
    if data is None:
        out["status"] = "missing"
        out["message"] = "file missing"
        return out
    if isinstance(data, dict):
        open_positions = data.get("open_positions", None)
        cash = data.get("cash", data.get("balance"))
        out["metrics"]["has_open_positions"] = open_positions is not None
        if open_positions is not None:
            out["metrics"]["open_positions_count"] = len(open_positions) if isinstance(open_positions, list) else None
        out["metrics"]["cash"] = cash
        if open_positions is None:
            out["status"] = "invalid"
            out["message"] = "Missing key: open_positions"
        else:
            out["status"] = "ok"
            out["message"] = "ok"
    else:
        out["status"] = "invalid"
        out["message"] = "unexpected shape"
    return out


def check_risk(risk_path: Path) -> Dict[str, Any]:
    out = {
        "file": risk_path.name,
        "exists": risk_path.exists(),
        "status": "unknown",
        "metrics": {},
        "message": "",
    }
    data = safe_load_json(risk_path)
    if data is None:
        out["status"] = "missing"
        out["message"] = "file missing"
        return out
    if isinstance(data, dict):
        # we expect 'day' or 'date' key, or 'counters'
        if "day" in data or "date" in data or "counters" in data:
            out["status"] = "ok"
            out["message"] = "ok"
            out["metrics"]["keys"] = list(data.keys())[:30]
        else:
            out["status"] = "invalid"
            out["message"] = "Missing key: day"
    else:
        out["status"] = "invalid"
        out["message"] = "unexpected shape"
    return out


def check_db(db_path: Path) -> Dict[str, Any]:
    out = {
        "file": db_path.name,
        "exists": db_path.exists(),
        "status": "unknown",
        "metrics": {},
        "message": "",
    }
    if not db_path.exists():
        out["status"] = "missing"
        out["message"] = "db missing"
        return out
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        # list tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]
        out["metrics"]["tables"] = tables
        # if trades table exists, get a count and latest timestamp if present
        trade_tables = [t for t in tables if "trade" in t.lower() or "llm" in t.lower()]
        if trade_tables:
            t0 = trade_tables[0]
            cur.execute(f"SELECT COUNT(*) FROM '{t0}'")
            out["metrics"]["row_count"] = cur.fetchone()[0]
            # try timestamp column heuristics
            possible_ts_cols = ["created_at", "ts", "timestamp", "fetched_at", "updated_at", "time"]
            cur.execute(f"PRAGMA table_info('{t0}')")
            cols = [r[1] for r in cur.fetchall()]
            out["metrics"]["table_sample"] = {t0: cols[:30]}
            ts_col = next((c for c in cols if c in possible_ts_cols), None)
            if ts_col:
                try:
                    cur.execute(f"SELECT {ts_col} FROM '{t0}' ORDER BY {ts_col} DESC LIMIT 1")
                    last = cur.fetchone()
                    out["metrics"]["latest_ts_col"] = ts_col
                    out["metrics"]["latest_ts_value"] = last[0] if last else None
                except Exception:
                    pass
        else:
            out["metrics"]["row_count"] = 0
        conn.close()
        out["status"] = "ok"
        out["message"] = "ok"
    except Exception as e:
        out["status"] = "invalid"
        out["message"] = f"db error: {e}"
    return out


# --- composite logic to decide triggers ---
def decide_actions(reports: Dict[str, Any]) -> Dict[str, Any]:
    actions: Dict[str, Any] = {
        "pause": False,
        "pause_reasons": [],
        "retrain": False,
        "retrain_reasons": [],
        "warnings": [],
    }

    snap = reports.get("snapshot", {})
    broker = reports.get("broker", {})
    risk = reports.get("risk", {})
    pnl = reports.get("pnl", {})
    db = reports.get("trades_db", {})

    snap_metrics = snap.get("metrics", {}) or {}
    rows = int(snap_metrics.get("rows", 0) or 0)
    num_with_ltp = int(snap_metrics.get("num_with_ltp", 0) or 0)

    # Pause conditions
    # 1) Snapshot latest fetched_at too old or missing
    latest_age = snap_metrics.get("latest_fetched_age_s")
    if latest_age is None or (isinstance(latest_age, (int, float)) and latest_age > THRESH["spot_missing_seconds"]):
        actions["pause"] = True
        actions["pause_reasons"].append(f"latest_snapshot_fetched_age_s={latest_age}")

    # 2) Very few LTPs or very few IVs (data feed failure)
    pct_with_ltp = float(snap_metrics.get("pct_with_ltp", 0.0) or 0.0)
    pct_with_iv = float(snap_metrics.get("pct_with_iv", 0.0) or 0.0)

    # LTP failure remains hard pause
    if pct_with_ltp < (1.0 - THRESH["all_ltp_zero_pct"]):
        actions["pause"] = True
        actions["pause_reasons"].append(f"low_pct_with_ltp={pct_with_ltp:.3f}")

    # "Good LTP" definition for soft IV handling
    has_good_ltp = (rows > 0) and (num_with_ltp >= 20) and (pct_with_ltp >= 0.95)

    # IV failure:
    #  - If LTP is also bad -> hard pause
    #  - If LTP is good -> warning only (no pause)
    if pct_with_iv < (1.0 - THRESH["iv_zero_pct"]):
        if has_good_ltp:
            actions["warnings"].append(
                f"low_pct_with_iv={pct_with_iv:.3f} (LTP ok; treating as warning only)"
            )
        else:
            actions["pause"] = True
            actions["pause_reasons"].append(f"low_pct_with_iv={pct_with_iv:.3f}")

    # 3) Broker missing open_positions
    if broker.get("status") == "invalid":
        actions["pause"] = True
        actions["pause_reasons"].append("broker_state_invalid_or_missing_open_positions")

    # 4) Risk state invalid
    if risk.get("status") == "invalid":
        actions["pause"] = True
        actions["pause_reasons"].append("risk_state_invalid_or_missing_day")

    # 5) DB row count zero (no trades table)
    if db.get("metrics", {}).get("row_count", 0) < THRESH["db_min_rows"]:
        actions["pause"] = True
        actions["pause_reasons"].append("trades_db_empty_or_missing")

    # Retrain conditions (conservative)
    retrain_file = BASE_DIR / "retrain_trigger.json"
    if retrain_file.exists():
        actions["retrain"] = True
        actions["retrain_reasons"].append("existing_weekly_retrain_trigger")

    # If PnL stdev extremely low (stagnant) over recent history -> candidate retrain
    try:
        pnl_stdev = pnl.get("metrics", {}).get("stdev", None)
        if pnl_stdev is not None and pnl_stdev == 0.0 and pnl.get("metrics", {}).get("count", 0) >= 3:
            actions["retrain"] = True
            actions["retrain_reasons"].append("pnl_stdev_zero_recent")
    except Exception:
        pass

    # If snapshot corrupted repeatedly (status bad)
    if snap.get("status") in ("bad", "invalid"):
        actions["retrain_reasons"].append(f"snapshot_status_{snap.get('status')}")
        if snap.get("status") == "invalid":
            actions["retrain"] = True

    # Finalize messages
    if not actions["pause_reasons"]:
        actions.pop("pause_reasons", None)
    if not actions["retrain_reasons"]:
        actions.pop("retrain_reasons", None)
    if not actions["warnings"]:
        actions.pop("warnings", None)

    return actions


def main():
    logging.info("auto_anomaly_detector: starting check (single-run) at %s", now_ts())
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    reports: Dict[str, Any] = {}

    # snapshot
    reports["snapshot"] = check_snapshot(BASE_DIR / SNAPSHOT_FNAME)

    # pnl
    reports["pnl"] = check_pnl(BASE_DIR / PNL_FNAME)

    # broker
    reports["broker"] = check_broker(BASE_DIR / BROKER_FNAME)

    # risk
    reports["risk"] = check_risk(BASE_DIR / RISK_FNAME)

    # trades DB
    reports["trades_db"] = check_db(BASE_DIR / TRADES_DB)

    # Consolidated decisions
    actions = decide_actions(reports)

    # Compose anomalies report
    anomalies_report = {
        "timestamp": now_ts(),
        "base_dir": str(BASE_DIR),
        "reports": reports,
        "actions": actions,
    }

    # Write anomalies file
    try:
        safe_write_json(OUT_ANOMALIES, anomalies_report)
        logging.info("Wrote anomalies report -> %s", OUT_ANOMALIES)
    except Exception as e:
        logging.exception("Failed to write anomalies report: %s", e)

    # If pause action requested -> write auto_pause_trigger.json with reasons & timestamp
    if actions.get("pause"):
        pause_obj = {
            "timestamp": now_ts(),
            "reason": actions.get("pause_reasons", []),
            "source": "auto_anomaly_detector",
        }
        try:
            safe_write_json(OUT_PAUSE, pause_obj)
            logging.warning("PAUSE TRIGGER WRITTEN -> %s (reasons: %s)", OUT_PAUSE, pause_obj["reason"])
        except Exception as e:
            logging.exception("Failed to write pause trigger: %s", e)
    else:
        # if no pause, remove stale pause file to avoid accidental reuse
        try:
            if OUT_PAUSE.exists():
                OUT_PAUSE.unlink()
                logging.info("Removed stale pause trigger file: %s", OUT_PAUSE)
        except Exception:
            pass

    # If retrain action requested -> write retrain_trigger.json (idempotent)
    if actions.get("retrain"):
        retrain_obj = {
            "timestamp": now_ts(),
            "reason": actions.get("retrain_reasons", []),
            "source": "auto_anomaly_detector",
        }
        try:
            safe_write_json(OUT_RETRAIN, retrain_obj)
            logging.warning("RETRAIN TRIGGER WRITTEN -> %s (reasons: %s)", OUT_RETRAIN, retrain_obj["reason"])
        except Exception as e:
            logging.exception("Failed to write retrain trigger: %s", e)

    logging.info("auto_anomaly_detector: finished run at %s", now_ts())


if __name__ == "__main__":
    main()
