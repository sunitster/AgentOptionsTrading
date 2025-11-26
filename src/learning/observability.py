"""
observability.py

Phase 6 — Observability & Monitoring for AgentOptionsTrading

Features:
-----------
1. Structured Logging
   - Event-level logging to JSONL files
   - Summary-level logging to CSV and JSON

2. Metrics Tracking
   - Latency tracker (start/end)
   - Slippage anomaly detector
   - Data gap detector
   - Candidate evaluation summaries

3. Alert Engine
   - Console alerts
   - File-based alert log

4. Lightweight Dashboard Utilities
   - load_run_metrics()
   - compute_summary_stats()
   - get_latest_alerts()

Usage:
------
obs = Observability(run_id="2025-11-18_1")
obs.log_event("start_run", {"features_shape": (2695842, 26)})
obs.alert("High slippage detected", level="warning")
obs.log_candidate_summary({...})
obs.end_run()
"""

from __future__ import annotations
import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional


# ------------------------------------------------------------
# Helper: ensure logging directory exists
# ------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

ALERTS_FILE = LOG_DIR / "alerts.log"


# ------------------------------------------------------------
# MAIN CLASS
# ------------------------------------------------------------
class Observability:
    def __init__(self, run_id: str, base_dir: Path = LOG_DIR):
        self.run_id = str(run_id)
        self.base_dir = base_dir

        # Event logs
        self.event_path = base_dir / f"{self.run_id}_events.jsonl"
        self.summary_path = base_dir / f"{self.run_id}_summary.csv"
        self.alerts_path = ALERTS_FILE

        # Internal buffers
        self.candidate_summaries: List[Dict[str, Any]] = []
        self.current_latency_timer = {}

        # Write header for summary CSV if first time
        if not self.summary_path.exists():
            with open(self.summary_path, "w") as f:
                f.write("timestamp,candidate_id,total_pnl,rmse,n_trades,win_rate,risk_adj_score\n")

        # Create event file
        self.event_file = open(self.event_path, "a", encoding="utf-8")

        self.log_event("obs_init", {"run_id": self.run_id})

    # ------------------------------------------------------------
    # Latency Tracking
    # ------------------------------------------------------------
    def start_latency(self, label: str):
        """Start a latency timer."""
        self.current_latency_timer[label] = time.time()

    def end_latency(self, label: str) -> float:
        """Return elapsed time and log latency event."""
        if label not in self.current_latency_timer:
            return 0.0
        elapsed = time.time() - self.current_latency_timer.pop(label)
        self.log_event("latency", {"label": label, "elapsed_sec": elapsed})
        return elapsed

    # ------------------------------------------------------------
    # Event Logging
    # ------------------------------------------------------------
    def log_event(self, event_type: str, payload: Dict[str, Any]):
        """Write a single JSONL event."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "payload": payload,
        }
        self.event_file.write(json.dumps(event) + "\n")
        self.event_file.flush()

    # ------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------
    def alert(self, message: str, level: str = "warning", meta: Optional[Dict[str, Any]] = None):
        """Send an alert — prints to console + logs to alerts file."""
        meta = meta or {}
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        msg = f"[{ts}] [{level.upper()}] {message} {json.dumps(meta)}"
        print(msg)

        with open(self.alerts_path, "a") as f:
            f.write(msg + "\n")

        # Also log as event
        self.log_event("alert", {"level": level, "message": message, "meta": meta})

    # ------------------------------------------------------------
    # Data Gap Detection
    # ------------------------------------------------------------
    def detect_data_gaps(self, df: pd.DataFrame, time_col: str = "date", tolerance_days: int = 1):
        """
        Detect missing time periods in sorted data.
        """
        if time_col not in df.columns:
            return

        ts = pd.to_datetime(df[time_col].dropna()).sort_values().unique()
        if len(ts) < 2:
            return

        gaps = []
        for i in range(1, len(ts)):
            diff = (ts[i] - ts[i-1]).days
            if diff > tolerance_days:
                gaps.append({"from": str(ts[i-1]), "to": str(ts[i]), "gap_days": diff})

        if gaps:
            self.alert("Data gaps detected", meta={"gaps": gaps})
            self.log_event("data_gaps", {"gaps": gaps})

    # ------------------------------------------------------------
    # Slippage Anomaly Detection
    # ------------------------------------------------------------
    def detect_slippage_anomaly(self, slippage_value: float, threshold_pct_of_capital: float, capital: float):
        """
        Detect if slippage is unusually large.
        """
        if slippage_value > threshold_pct_of_capital * capital:
            self.alert("Slippage anomaly", meta={"slippage": slippage_value})
            self.log_event("slippage_anomaly", {"slippage": slippage_value})

    # ------------------------------------------------------------
    # Candidate Summary Logging
    # ------------------------------------------------------------
    def log_candidate_summary(self, summary: Dict[str, Any]):
        """
        Store the summary in CSV + event log.
        Expected summary fields:
            candidate_id, total_pnl, rmse, n_trades, win_rate, risk_adj_score
        """
        self.candidate_summaries.append(summary)

        # JSONL detail event
        self.log_event("candidate_summary", summary)

        # CSV summary
        line = (
            f"{time.time()},{summary.get('candidate_id','')},{summary.get('total_pnl',0)},"
            f"{summary.get('rmse',0)},{summary.get('n_trades',0)},"
            f"{summary.get('win_rate',0)},{summary.get('risk_adj_score',0)}\n"
        )
        with open(self.summary_path, "a") as f:
            f.write(line)

    # ------------------------------------------------------------
    # End Run
    # ------------------------------------------------------------
    def end_run(self):
        """Close event file and record summary event."""
        self.log_event("run_complete", {"total_candidates": len(self.candidate_summaries)})
        self.event_file.close()

    # ------------------------------------------------------------
    # Dashboard Helpers (for CLI or Streamlit)
    # ------------------------------------------------------------
    def load_run_metrics(self) -> pd.DataFrame:
        """Load summary CSV for this run."""
        if not self.summary_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.summary_path)

    @staticmethod
    def compute_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """Return high-level metrics."""
        if df.empty:
            return {}

        return {
            "num_candidates": len(df),
            "best_pnl": df["total_pnl"].max(),
            "best_rmse": df["rmse"].min(),
            "best_risk_score": df["risk_adj_score"].max(),
            "avg_pnl": df["total_pnl"].mean(),
            "avg_rmse": df["rmse"].mean(),
        }

    @staticmethod
    def get_latest_alerts(n: int = 20) -> List[str]:
        """Return last N alerts from file."""
        if not ALERTS_FILE.exists():
            return []

        with open(ALERTS_FILE, "r") as f:
            lines = f.readlines()

        return lines[-n:]
