"""
daily_qc.py
Daily quality-control + self-healing for LivePaperEngine snapshots
Author: ChatGPT for Sunit Singh
Safe to drop directly into repo.
"""

import os
import json
import shutil
import datetime

BASE_DIR = os.path.join("models", "llm_trades")

FILES = {
    "snapshot":      "latest_snapshot.json",
    "pnl":           "pnl_history.json",
    "broker":        "paper_broker_state.json",
    "risk":          "risk_state.json",
    "trades_db":     "trades.db",
}

REPORT_FILE = os.path.join(BASE_DIR, "qc_report_daily.json")


# -----------------------------------------------------------
# Safe JSON loader with corruption detection
# -----------------------------------------------------------
def safe_json_load(path):
    if not os.path.exists(path):
        return None, "file_missing"

    try:
        with open(path, "r") as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"json_corrupt: {e}"
    except Exception as e:
        return None, f"load_error: {e}"


# -----------------------------------------------------------
# Auto-heal JSON corruption
# -----------------------------------------------------------
def repair_corrupt_json(path):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path + f".broken_{ts}"

    shutil.move(path, backup)

    # Minimum valid JSON to avoid dashboard crashes
    repaired = {
        "status": "reset_due_to_corruption",
        "created_at": ts,
        "data": {}
    }

    with open(path, "w") as f:
        json.dump(repaired, f, indent=2)

    return backup, repaired


# -----------------------------------------------------------
# Integrity checks for each JSON file
# -----------------------------------------------------------
def validate_snapshot(data):
    if not isinstance(data, dict):
        return False, "Snapshot root is not an object"

    # minimal keys your engine expects
    required = ["positions", "pnl", "timestamp"]
    missing = [k for k in required if k not in data]

    if missing:
        return False, f"Missing keys: {missing}"

    # Check positions sanity
    pos = data.get("positions", {})
    if not isinstance(pos, dict):
        return False, "positions must be dict"

    # Example checks (expand as strategy grows)
    if "ltp" in data and isinstance(data["ltp"], dict):
        non_zero = sum(1 for v in data["ltp"].values() if v > 0)
        if non_zero == 0:
            return False, "All LTP values are zero"

    return True, "ok"


def validate_pnl(data):
    if not isinstance(data, list):
        return False, "PNL history must be list"

    if len(data) == 0:
        return False, "PNL history empty"

    return True, "ok"


def validate_risk_state(data):
    if not isinstance(data, dict):
        return False, "Risk state must be dict"

    if "day" not in data:
        return False, "Missing key: day"

    if "counters" not in data:
        return False, "Missing key: counters"

    return True, "ok"


def validate_broker(data):
    if not isinstance(data, dict):
        return False, "Broker state must be dict"

    if "open_positions" not in data:
        return False, "Missing key: open_positions"

    return True, "ok"


# -----------------------------------------------------------
# Evaluate all files
# -----------------------------------------------------------
def run_qc():
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "base_dir": BASE_DIR,
        "checks": {},
        "summary": {
            "files_total": 0,
            "files_ok": 0,
            "files_repaired": 0,
            "files_failed": 0,
        }
    }

    for key, fname in FILES.items():
        if key == "trades_db":
            # DB existence only
            path = os.path.join(BASE_DIR, fname)
            exists = os.path.exists(path)
            report["checks"][key] = {
                "file": fname,
                "exists": exists,
                "status": "ok" if exists else "missing"
            }
            report["summary"]["files_total"] += 1
            report["summary"]["files_ok"] += int(exists)
            continue

        # JSON files
        path = os.path.join(BASE_DIR, fname)
        data, err = safe_json_load(path)

        report["summary"]["files_total"] += 1

        if err is None:
            # Validate content
            if key == "snapshot":
                ok, msg = validate_snapshot(data)
            elif key == "pnl":
                ok, msg = validate_pnl(data)
            elif key == "risk":
                ok, msg = validate_risk_state(data)
            elif key == "broker":
                ok, msg = validate_broker(data)
            else:
                ok, msg = True, "ok"

            status = "ok" if ok else "invalid"

            if ok:
                report["summary"]["files_ok"] += 1
            else:
                report["summary"]["files_failed"] += 1

            report["checks"][key] = {
                "file": fname,
                "status": status,
                "message": msg,
            }

        else:
            # Corruption or failure
            report["summary"]["files_failed"] += 1

            backup, repaired = repair_corrupt_json(path)
            report["summary"]["files_repaired"] += 1

            report["checks"][key] = {
                "file": fname,
                "status": "repaired",
                "message": err,
                "backup": backup,
            }

    # Write QC report
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== DAILY QC REPORT ===")
    print(json.dumps(report, indent=2))
    print("\nSaved:", REPORT_FILE)

    return report


if __name__ == "__main__":
    run_qc()
