"""
weekly_qc.py
Weekly quality-control + drift detection for LivePaperEngine snapshots
Place at: src/apps/maintenance/weekly_qc.py

Behavior summary:
- Loads JSON files from models/llm_trades/
- Runs deeper consistency checks than daily_qc
- Computes simple drift metrics over recent PnL history & snapshot values
- Flags anomalies (missing LTP/IV, spot==0, sudden PnL drawdown, trade frequency changes)
- Produces qc_report_weekly.json, qc_anomalies_weekly.json and retrain_trigger.json (if needed)
- Makes backups before any write and never deletes original files

Author: ChatGPT for Sunit Singh
"""
import os
import json
import shutil
import datetime
import math
from statistics import mean, stdev

BASE_DIR = os.path.join("models", "llm_trades")
SNAPSHOT_F = "latest_snapshot.json"
PNL_F = "pnl_history.json"
BROKER_F = "paper_broker_state.json"
RISK_F = "risk_state.json"

OUT_REPORT = os.path.join(BASE_DIR, "qc_report_weekly.json")
OUT_ANOMALIES = os.path.join(BASE_DIR, "qc_anomalies_weekly.json")
OUT_RETRAIN = os.path.join(BASE_DIR, "retrain_trigger.json")
BACKUP_DIR = os.path.join(BASE_DIR, "backups_weekly")

# thresholds (tunable)
MIN_LTP_NONZERO = 1             # at least 1 LTP non-zero to be healthy
SPOT_MIN = 1e-6                 # spot must be greater than this
PNL_DRAWDOWN_PCT = 0.10         # >10% drop over week triggers warning
PNL_STD_Z = 3.0                 # z-score threshold for pnl surprise
TRADE_FREQ_CHANGE_PCT = 0.5     # 50% change in weekly trade count flags anomaly

# Helper utilities -----------------------------------------------------------
def ensure_backup_dir():
    os.makedirs(BACKUP_DIR, exist_ok=True)

def safe_load_json(path):
    if not os.path.exists(path):
        return None, f"missing:{path}"
    try:
        with open(path, "r", encoding="utf8") as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"json_decode_error:{e}"
    except Exception as e:
        return None, f"load_error:{e}"

def backup_file(path, reason="weekly_qc"):
    ensure_backup_dir()
    if not os.path.exists(path):
        return None
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(path)
    dest = os.path.join(BACKUP_DIR, f"{base}.{reason}.{ts}.bak")
    shutil.copy2(path, dest)
    return dest

def write_json(path, obj):
    backup_file(path, "write_pre")
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path

# Specific validators --------------------------------------------------------
def analyze_snapshot(snapshot):
    """Return summary dict and light normalization if helpful."""
    summary = {"ok": False, "issues": [], "counts": {}, "metrics": {}}

    # If snapshot is a list (sometimes whole history dumped), wrap to avoid crash
    if isinstance(snapshot, list):
        summary["issues"].append("snapshot_root_list_wrapped")
        backup_path = backup_file(os.path.join(BASE_DIR, SNAPSHOT_F), "root_list")
        # create wrapped snapshot (do not remove original)
        wrapped = {"_wrapped_from_list": True, "entries": snapshot, "timestamp": datetime.datetime.now().isoformat()}
        write_json(os.path.join(BASE_DIR, SNAPSHOT_F), wrapped)
        snapshot = wrapped

    if not isinstance(snapshot, dict):
        summary["issues"].append("snapshot_root_not_object")
        return summary

    # check presence of core keys
    positions = snapshot.get("positions", {})
    ltp = snapshot.get("ltp") or {}
    iv = snapshot.get("iv") or {}
    spot = None
    # spot may be in snapshot["spot"] or snapshot["broker_state"]["spot"]
    if "spot" in snapshot:
        try:
            spot = float(snapshot["spot"])
        except Exception:
            spot = None
    else:
        # attempt to find any numeric spot in ltp or positions
        for v in (ltp, positions):
            if isinstance(v, dict):
                for x in v.values():
                    try:
                        if isinstance(x, (int, float)) and x > 0:
                            spot = float(x)
                            break
                    except Exception:
                        continue
            if spot:
                break

    nonzero_ltp = 0
    if isinstance(ltp, dict):
        for v in ltp.values():
            try:
                if float(v) > 0:
                    nonzero_ltp += 1
            except Exception:
                continue

    nonzero_iv = 0
    if isinstance(iv, dict):
        for v in iv.values():
            try:
                if float(v) > 0:
                    nonzero_iv += 1
            except Exception:
                continue

    summary["counts"]["ltp_nonzero"] = nonzero_ltp
    summary["counts"]["iv_nonzero"] = nonzero_iv
    summary["metrics"]["spot"] = spot

    # basic health
    if nonzero_ltp < MIN_LTP_NONZERO:
        summary["issues"].append("ltp_all_zero_or_missing")
    if nonzero_iv == 0:
        summary["issues"].append("iv_all_zero_or_missing")
    if not spot or spot <= SPOT_MIN:
        summary["issues"].append("spot_zero_or_missing")

    summary["ok"] = len(summary["issues"]) == 0
    return summary

def analyze_pnl(pnl_list):
    """
    pnl_list expected as list of dicts or numbers.
    We'll extract numeric pnl series (last N days) and compute drawdown & drift.
    """
    out = {"ok": False, "issues": [], "metrics": {}}
    if not isinstance(pnl_list, list) or len(pnl_list) == 0:
        out["issues"].append("pnl_missing_or_empty")
        return out

    # normalize to floats: support list of {"ts":..., "pnl":...} or numbers
    series = []
    for item in pnl_list:
        if isinstance(item, (int, float)):
            series.append(float(item))
        elif isinstance(item, dict):
            if "pnl" in item:
                try:
                    series.append(float(item["pnl"]))
                except Exception:
                    continue
            else:
                # try first numeric val
                for v in item.values():
                    if isinstance(v, (int, float)):
                        series.append(float(v))
                        break
    if len(series) < 2:
        out["issues"].append("pnl_series_too_short")
        return out

    # work with last 14 values as weekly-ish window (2 weeks)
    window = series[-14:]
    avg = mean(window)
    sd = stdev(window) if len(window) >= 2 else 0.0
    last = window[-1]
    prev_week_avg = mean(window[:-7]) if len(window) > 7 else None

    # drawdown from peak in window
    peak = max(window)
    drawdown = (peak - last) / (abs(peak) + 1e-9)

    out["metrics"].update({
        "window_len": len(window),
        "avg": avg,
        "std": sd,
        "last": last,
        "peak": peak,
        "drawdown_pct": drawdown
    })

    # z-score surprise
    z = (last - avg) / (sd + 1e-9)
    out["metrics"]["z"] = z

    if drawdown >= PNL_DRAWDOWN_PCT:
        out["issues"].append("pnl_drawdown_exceeds_threshold")
    if abs(z) >= PNL_STD_Z:
        out["issues"].append("pnl_zscore_anomaly")

    # weekly trade frequency (if pnl_list items include trade counts)
    trade_counts = []
    for item in pnl_list:
        if isinstance(item, dict) and "trades" in item:
            try:
                trade_counts.append(int(item["trades"]))
            except Exception:
                continue
    if trade_counts:
        # compare last 7 vs previous 7
        last7 = sum(trade_counts[-7:])
        prev7 = sum(trade_counts[-14:-7]) if len(trade_counts) >= 14 else None
        out["metrics"]["trade_count_last7"] = last7
        out["metrics"]["trade_count_prev7"] = prev7
        if prev7 and prev7 > 0:
            pct_change = abs(last7 - prev7) / prev7
            out["metrics"]["trade_count_change_pct"] = pct_change
            if pct_change >= TRADE_FREQ_CHANGE_PCT:
                out["issues"].append("trade_frequency_changed_significantly")

    out["ok"] = len(out["issues"]) == 0
    return out

def analyze_broker(broker):
    out = {"ok": False, "issues": [], "metrics": {}}
    if not isinstance(broker, dict):
        out["issues"].append("broker_not_object")
        return out
    if "open_positions" not in broker:
        out["issues"].append("missing_open_positions")
    else:
        out["metrics"]["open_positions_count"] = len(broker.get("open_positions") or [])
    out["ok"] = len(out["issues"]) == 0
    return out

def analyze_risk(risk):
    out = {"ok": False, "issues": [], "metrics": {}}
    if not isinstance(risk, dict):
        out["issues"].append("risk_not_object")
        return out
    if "day" not in risk:
        out["issues"].append("missing_key_day")
    if "counters" not in risk:
        out["issues"].append("missing_key_counters")
    out["ok"] = len(out["issues"]) == 0
    return out

# Aggregation & anomaly decision -------------------------------------------
def run_weekly_qc():
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "base_dir": BASE_DIR,
        "checks": {},
        "summary": {"files_total": 0, "files_ok": 0, "anomalies": 0},
    }
    anomalies = []

    # load files
    snapshot, s_err = safe_load_json(os.path.join(BASE_DIR, SNAPSHOT_F))
    pnl, p_err = safe_load_json(os.path.join(BASE_DIR, PNL_F))
    broker, b_err = safe_load_json(os.path.join(BASE_DIR, BROKER_F))
    risk, r_err = safe_load_json(os.path.join(BASE_DIR, RISK_F))

    # snapshot
    report["summary"]["files_total"] += 1
    if s_err:
        report["checks"]["snapshot"] = {"status": "load_error", "error": s_err}
        anomalies.append({"file": SNAPSHOT_F, "issue": s_err})
    else:
        s_summary = analyze_snapshot(snapshot)
        report["checks"]["snapshot"] = s_summary
        if not s_summary.get("ok"):
            report["summary"]["anomalies"] += 1
            anomalies.append({"file": SNAPSHOT_F, "issue": s_summary["issues"]})

    # pnl
    report["summary"]["files_total"] += 1
    if p_err:
        report["checks"]["pnl"] = {"status": "load_error", "error": p_err}
        anomalies.append({"file": PNL_F, "issue": p_err})
    else:
        p_summary = analyze_pnl(pnl)
        report["checks"]["pnl"] = p_summary
        if not p_summary.get("ok"):
            report["summary"]["anomalies"] += 1
            anomalies.append({"file": PNL_F, "issue": p_summary["issues"], "metrics": p_summary.get("metrics")})

    # broker
    report["summary"]["files_total"] += 1
    if b_err:
        report["checks"]["broker"] = {"status": "load_error", "error": b_err}
        anomalies.append({"file": BROKER_F, "issue": b_err})
    else:
        b_summary = analyze_broker(broker)
        report["checks"]["broker"] = b_summary
        if not b_summary.get("ok"):
            report["summary"]["anomalies"] += 1
            anomalies.append({"file": BROKER_F, "issue": b_summary["issues"]})

    # risk
    report["summary"]["files_total"] += 1
    if r_err:
        report["checks"]["risk"] = {"status": "load_error", "error": r_err}
        anomalies.append({"file": RISK_F, "issue": r_err})
    else:
        r_summary = analyze_risk(risk)
        report["checks"]["risk"] = r_summary
        if not r_summary.get("ok"):
            report["summary"]["anomalies"] += 1
            anomalies.append({"file": RISK_F, "issue": r_summary["issues"]})

    # Consider retrain trigger logic:
    retrain_reasons = []
    # trigger if pnl drawdown or zscore anomaly or major snapshot issues
    pnl_metrics = report["checks"].get("pnl", {}).get("metrics", {})
    if pnl_metrics:
        if pnl_metrics.get("drawdown_pct", 0) >= PNL_DRAWDOWN_PCT:
            retrain_reasons.append("pnl_drawdown")
        if abs(pnl_metrics.get("z", 0)) >= PNL_STD_Z:
            retrain_reasons.append("pnl_statistical_anomaly")

    s_issues = report["checks"].get("snapshot", {}).get("issues", [])
    if s_issues:
        # many snapshot issues -> recommend investigating data feed
        retrain_reasons.append("snapshot_issues:" + ",".join(s_issues))

    # Summary flags
    report["summary"]["files_ok"] = max(0, report["summary"]["files_total"] - report["summary"]["anomalies"])

    # Write outputs
    write_json(OUT_REPORT, report)
    write_json(OUT_ANOMALIES, anomalies)

    if retrain_reasons:
        retrain_payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "reasons": retrain_reasons,
            "suggested_action": "inspect_datafeed_and_consider_retraining_models_or_reset_params",
        }
        write_json(OUT_RETRAIN, retrain_payload)
        print("RETRAIN TRIGGERED (file written):", OUT_RETRAIN)
    else:
        # remove previous trigger if exists (safe backup)
        if os.path.exists(OUT_RETRAIN):
            backup_file(OUT_RETRAIN, "removed_no_retrain_needed")
            try:
                os.remove(OUT_RETRAIN)
            except Exception:
                pass

    print("Weekly QC report saved ->", OUT_REPORT)
    print("Weekly anomalies saved ->", OUT_ANOMALIES)
    return {"report": report, "anomalies": anomalies, "retrain": os.path.exists(OUT_RETRAIN)}

if __name__ == "__main__":
    run_weekly_qc()
