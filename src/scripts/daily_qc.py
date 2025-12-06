#!/usr/bin/env python3
"""
src/scripts/daily_qc.py

Daily QC & PnL report script for AgentOptionsTrading.

Usage:
    python -m src.scripts.daily_qc
    python -m src.scripts.daily_qc --date 2025-11-27
    python -m src.scripts.daily_qc --alert-webhook https://hooks.example.com/abcd

What it does (safe defaults):
 - Reads trades from models/llm_trades/trades.db (if present)
 - Reads pnl_history.json, latest_snapshot.json, current_position.json (if present)
 - Computes today's PnL, trades, avg entry_credit, avg width, time-in-trade stats
 - Computes simple IV / credit drift vs a stored baseline (models/llm_trades/drift_baseline.json)
 - Writes:
     - models/llm_trades/reports/daily_report_{YYYY-MM-DD}.json
     - models/llm_trades/drift/daily_drift_{YYYY-MM-DD}.json
 - Returns non-zero exit code only for serious internal errors (not for missing files)

This script is defensive and will NOT change live configs.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sqlite3
import statistics
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional

# Constants (match LivePaperEngine)
BASE_MONITOR = os.path.join("models", "llm_trades")
TRADES_DB = os.path.join(BASE_MONITOR, "trades.db")
PNL_HISTORY_FILE = os.path.join(BASE_MONITOR, "pnl_history.json")
LATEST_SNAPSHOT_FILE = os.path.join(BASE_MONITOR, "latest_snapshot.json")
CURRENT_POS_FILE = os.path.join(BASE_MONITOR, "current_position.json")
DRIFT_BASELINE_FILE = os.path.join(BASE_MONITOR, "drift_baseline.json")
REPORTS_DIR = os.path.join(BASE_MONITOR, "reports")
DRIFT_DIR = os.path.join(BASE_MONITOR, "drift")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DRIFT_DIR, exist_ok=True)
os.makedirs(BASE_MONITOR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("daily_qc")


# --------------------- helpers ---------------------
def load_trades_for_date(db_path: str, target_date: date) -> List[Dict[str, Any]]:
    """Load trades rows whose ts starts with target_date ISO (YYYY-MM-DD)."""
    if not os.path.exists(db_path):
        LOG.info("Trades DB not found at %s", db_path)
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        like_prefix = target_date.isoformat() + "%"
        cur.execute("SELECT * FROM trades WHERE ts LIKE ? ORDER BY id ASC", (like_prefix,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        LOG.exception("Failed to read trades DB")
        return []


def load_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        LOG.debug("File not found: %s", path)
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        LOG.exception("Failed to parse JSON: %s", path)
        return None


def safe_mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    try:
        return float(statistics.mean(xs))
    except Exception:
        try:
            return float(sum(xs) / len(xs))
        except Exception:
            return None


def safe_median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    try:
        return float(statistics.median(xs))
    except Exception:
        return None


def compute_drawdown(time_series: List[float]) -> Dict[str, float]:
    """
    Given a series of PnL (cumulative or balance), compute a simple max drawdown.
    If provided series is per-tick pnl, it's treated as cumulative.
    Returns dictionary {max_drawdown, peak, trough, end_balance}
    """
    if not time_series:
        return {"max_drawdown": 0.0, "peak": 0.0, "trough": 0.0, "end_balance": 0.0}
    try:
        # treat time_series as balance-like (non-decreasing not required)
        peak = time_series[0]
        max_dd = 0.0
        trough = time_series[0]
        for x in time_series:
            if x > peak:
                peak = x
            dd = peak - x
            if dd > max_dd:
                max_dd = dd
                trough = x
        return {"max_drawdown": float(max_dd), "peak": float(peak), "trough": float(trough), "end_balance": float(time_series[-1])}
    except Exception:
        LOG.exception("Failed compute_drawdown")
        return {"max_drawdown": 0.0, "peak": 0.0, "trough": 0.0, "end_balance": time_series[-1] if time_series else 0.0}


def parse_ic_json_field(ic_json_str: Optional[str], field: str) -> Optional[float]:
    """Given stored ic_json (string), try to parse and extract numeric field (entry_credit, width, etc)."""
    if not ic_json_str:
        return None
    try:
        if isinstance(ic_json_str, str):
            data = json.loads(ic_json_str)
        else:
            data = ic_json_str
        # Some as_dicts have 'entry_credit' at top level or nested
        if isinstance(data, dict):
            if field in data:
                try:
                    return float(data[field]) if data[field] is not None else None
                except Exception:
                    return None
            # try nested keys common patterns
            for k in ("meta", "metrics", "stats"):
                if k in data and isinstance(data[k], dict) and field in data[k]:
                    try:
                        return float(data[k][field])
                    except Exception:
                        return None
            # attempt width calculation from short/long legs
            if field == "width":
                try:
                    sp = data.get("short_put")
                    lp = data.get("long_put")
                    sc = data.get("short_call")
                    lc = data.get("long_call")
                    widths = []
                    if sp is not None and lp is not None:
                        widths.append(abs(float(sp) - float(lp)))
                    if sc is not None and lc is not None:
                        widths.append(abs(float(lc) - float(sc)))
                    if widths:
                        return float(sum(widths) / len(widths))
                except Exception:
                    pass
    except Exception:
        LOG.exception("parse_ic_json_field failed")
    return None


# --------------------- drift baseline helpers ---------------------
def load_baseline(path: str) -> Dict[str, Any]:
    """Load baseline JSON used to compute drift. If absent returns empty baseline."""
    data = load_json(path)
    if not data:
        # baseline structure
        return {
            "iv_mean_7d": None,
            "iv_std_7d": None,
            "credit_mean_7d": None,
            "credit_std_7d": None,
            "updated_at": None,
        }
    return data


def save_baseline(path: str, obj: Dict[str, Any]) -> None:
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception:
        LOG.exception("Failed to save baseline")


# --------------------- metrics computation ---------------------
def compute_daily_metrics(target_date: date) -> Dict[str, Any]:
    # Load trades
    trades = load_trades_for_date(TRADES_DB, target_date)
    LOG.info("Found %d trades for %s", len(trades), target_date.isoformat())

    # pnl_history timeline (may contain multiple points)
    pnl_hist = load_json(PNL_HISTORY_FILE) or []
    # attempt to extract today's pnl points (if timestamps present)
    today_points = []
    for rec in pnl_hist:
        try:
            ts = rec.get("time")
            if not ts:
                continue
            if ts.startswith(target_date.isoformat()):
                today_points.append(float(rec.get("pnl", 0.0)))
        except Exception:
            continue

    # calculate cumulative PnL for the day (if points look like balance, convert to pnl diff)
    total_pnl = None
    if today_points:
        # if points appear to be cumulative balance (non-monotonic), compute diff against earliest of day
        try:
            total_pnl = float(today_points[-1] - today_points[0])
        except Exception:
            # fallback: sum of per-tick pnls
            try:
                total_pnl = float(sum(today_points))
            except Exception:
                total_pnl = None

    # trades metrics
    entry_credits = []
    widths = []
    durations = []
    realized_pnls = []
    for t in trades:
        try:
            ic_json = t.get("ic_json") or t.get("ic_json", None) or t.get("ic_json")
            # some rows might have ic_json as a str or JSON already
            if ic_json:
                cred = parse_ic_json_field(ic_json, "entry_credit")
                if cred is not None:
                    entry_credits.append(cred)
                w = parse_ic_json_field(ic_json, "width")
                if w is not None:
                    widths.append(w)
            # duration_seconds stored on exit rows
            if t.get("duration_seconds") is not None:
                try:
                    durations.append(float(t.get("duration_seconds")))
                except Exception:
                    pass
            if t.get("event", "").lower().startswith("exit") and t.get("pnl") is not None:
                try:
                    realized_pnls.append(float(t.get("pnl")))
                except Exception:
                    pass
        except Exception:
            LOG.exception("Failed to parse trade row")

    avg_entry_credit = safe_mean(entry_credits)
    avg_width = safe_mean(widths)
    avg_duration = safe_mean(durations)
    avg_realized_pnl = safe_mean(realized_pnls)
    num_trades = len(trades)

    # drawdown using today_points (if those are balances)
    dd = compute_drawdown(today_points) if today_points else {"max_drawdown": 0.0, "peak": 0.0, "trough": 0.0, "end_balance": 0.0}

    # latest snapshot IV stats
    latest_snapshot = load_json(LATEST_SNAPSHOT_FILE)
    iv_mean = None
    iv_std = None
    iv_count = 0
    if latest_snapshot:
        try:
            # latest_snapshot is list of dicts; look for 'iv' field
            ivs = []
            for r in latest_snapshot:
                if r is None:
                    continue
                if isinstance(r, dict):
                    v = r.get("iv") or r.get("implied_vol") or r.get("ltp_iv") or r.get("iv_value")
                    if v is None and "ltp" in r and "strike" in r:
                        # naive iv not computable here -> skip
                        continue
                    try:
                        vv = float(v) if v is not None else None
                        if vv is not None:
                            ivs.append(vv)
                    except Exception:
                        continue
            if ivs:
                iv_mean = float(statistics.mean(ivs))
                iv_std = float(statistics.pstdev(ivs)) if len(ivs) > 1 else 0.0
                iv_count = len(ivs)
        except Exception:
            LOG.exception("Failed to compute IV stats from snapshot")

    report = {
        "date": target_date.isoformat(),
        "num_trades": num_trades,
        "total_pnl": total_pnl,
        "avg_entry_credit": avg_entry_credit,
        "avg_width": avg_width,
        "avg_duration_seconds": avg_duration,
        "avg_realized_pnl": avg_realized_pnl,
        "pnl_drawdown": dd,
        "iv_mean": iv_mean,
        "iv_std": iv_std,
        "iv_count": iv_count,
        "trades_sample": trades[-10:],  # last 10 raw trade rows for quick inspection
    }
    return report


# --------------------- drift detection ---------------------
def detect_drift(report: Dict[str, Any], baseline_path: str = DRIFT_BASELINE_FILE) -> Dict[str, Any]:
    """
    Compare current IV and entry credit stats vs baseline and produce drift signals.
    - If baseline missing, create it from report and return 'baseline_created' True.
    """
    baseline = load_baseline(baseline_path)
    drift = {"baseline_exists": baseline.get("updated_at") is not None, "baseline_created": False, "iv_drift_z": None, "credit_drift_z": None, "flags": []}
    # IV drift z-score (if baseline stats present)
    try:
        iv_mean = report.get("iv_mean")
        if iv_mean is not None and baseline.get("iv_mean_7d") is not None and baseline.get("iv_std_7d") not in (None, 0):
            z = (iv_mean - baseline["iv_mean_7d"]) / float(baseline["iv_std_7d"])
            drift["iv_drift_z"] = float(z)
            if abs(z) >= 1.5:
                drift["flags"].append("iv_shift_large")
        elif iv_mean is not None and not drift["baseline_exists"]:
            # create baseline from single day (will be improved by weekly job)
            baseline["iv_mean_7d"] = iv_mean
            baseline["iv_std_7d"] = report.get("iv_std", 0.0) or 0.0
    except Exception:
        LOG.exception("IV drift detection failed")

    # credit drift: compare avg_entry_credit vs baseline
    try:
        credit = report.get("avg_entry_credit")
        if credit is not None and baseline.get("credit_mean_7d") is not None and baseline.get("credit_std_7d") not in (None, 0):
            zc = (credit - baseline["credit_mean_7d"]) / float(baseline["credit_std_7d"])
            drift["credit_drift_z"] = float(zc)
            if abs(zc) >= 2.0:
                drift["flags"].append("credit_shift_large")
        elif credit is not None and not drift["baseline_exists"]:
            baseline["credit_mean_7d"] = credit
            baseline["credit_std_7d"] = 0.0
    except Exception:
        LOG.exception("Credit drift detection failed")

    # simple anomaly flags from report
    try:
        if report.get("num_trades", 0) >= 5 and (report.get("total_pnl") is not None and report["total_pnl"] < 0 and abs(report["total_pnl"]) > 0.02 * 1_000_000):
            drift["flags"].append("large_loss_many_trades")
        if report.get("avg_width") is not None and report["avg_width"] > 500:
            drift["flags"].append("unusually_wide")
    except Exception:
        LOG.exception("Post-check anomaly flags failed")

    # baseline creation if not exists
    if not drift["baseline_exists"]:
        try:
            baseline["updated_at"] = datetime.utcnow().isoformat()
            save_baseline(baseline_path, baseline)
            drift["baseline_created"] = True
            LOG.info("Drift baseline created at %s", baseline_path)
        except Exception:
            LOG.exception("Failed to create baseline")

    return drift


# --------------------- report writer ---------------------
def write_report_and_drift(report: Dict[str, Any], drift: Dict[str, Any], target_date: date) -> Dict[str, str]:
    fn_report = os.path.join(REPORTS_DIR, f"daily_report_{target_date.isoformat()}.json")
    fn_drift = os.path.join(DRIFT_DIR, f"daily_drift_{target_date.isoformat()}.json")
    try:
        with open(fn_report, "w") as f:
            json.dump(report, f, indent=2)
        with open(fn_drift, "w") as f:
            json.dump(drift, f, indent=2)
        LOG.info("Wrote report: %s and drift: %s", fn_report, fn_drift)
        return {"report": fn_report, "drift": fn_drift}
    except Exception:
        LOG.exception("Failed to write report files")
        return {}


# --------------------- optional webhook alert ---------------------
def maybe_post_webhook(webhook: Optional[str], summary_text: str) -> None:
    if not webhook:
        return
    try:
        import requests
    except Exception:
        LOG.warning("requests not available; webhook disabled")
        return
    try:
        payload = {"text": summary_text}
        resp = requests.post(webhook, json=payload, timeout=8)
        if resp.status_code >= 400:
            LOG.warning("Webhook returned status %s body=%s", resp.status_code, resp.text[:400])
    except Exception:
        LOG.exception("Webhook post failed")


# --------------------- CLI ---------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: today)")
    p.add_argument("--alert-webhook", default=None, help="Optional webhook URL to post a short summary")
    args = p.parse_args(argv)

    try:
        if args.date:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        else:
            target_date = date.today()
    except Exception:
        LOG.exception("Invalid --date")
        return 2

    # compute metrics
    try:
        report = compute_daily_metrics(target_date)
    except Exception:
        LOG.exception("Failed to compute daily metrics")
        return 3

    # detect drift
    try:
        drift = detect_drift(report)
    except Exception:
        LOG.exception("Failed to detect drift")
        drift = {"error": "drift_failed"}

    # write artifacts
    write_report_and_drift(report, drift, target_date)

    # optional webhook summary
    summary_lines = [
        f"Daily QC {target_date.isoformat()}",
        f"Trades: {report.get('num_trades')}, Total PnL: {report.get('total_pnl')}",
        f"Avg credit: {report.get('avg_entry_credit')}, Avg width: {report.get('avg_width')}",
        f"IV mean: {report.get('iv_mean')}, IV std: {report.get('iv_std')}",
        f"Drift flags: {', '.join(drift.get('flags', [])) or 'none'}",
    ]
    summary = "\n".join(summary_lines)
    LOG.info(summary)
    maybe_post_webhook(args.alert_webhook, summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
