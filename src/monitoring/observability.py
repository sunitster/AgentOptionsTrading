"""
src/monitoring/observability.py

Lightweight observability / logging helpers for your trading stack.

Features:
- JSONL event log for trades, sessions, errors, metrics, etc.
- Helper functions to record:
    * trade events
    * session summaries
    * generic events (risk hit, circuit breaker, etc.)
- Console logging (optional) with consistent format.

Usage (example):

    from src.monitoring.observability import get_observer

    obs = get_observer()

    obs.record_event("session_start", {"symbol": "NIFTY"})
    obs.record_trade(trade_record)
    obs.record_session_summary({"n_trades": 5, "pnl": 1234.5})

This module is intentionally dependency-light and file-based.
You can later swap implementation to push to Prometheus / ELK / cloud, etc.
"""

from __future__ import annotations
import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _iso_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------
# Observability core
# ---------------------------------------------------------------------
class Observability:
    """
    Simple JSONL and console observability.

    Parameters
    ----------
    log_dir : str
        Directory where JSONL logs will be stored.
    jsonl_file : str
        Filename (within log_dir) for event logs.
    console_level : int
        Logging level for console handler (logging.INFO by default).
    enabled : bool
        If False, all operations become no-ops.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        jsonl_file: str = "events.jsonl",
        console_level: int = logging.INFO,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.log_dir = log_dir
        self.jsonl_path = os.path.join(log_dir, jsonl_file)
        self._lock = threading.Lock()

        _ensure_dir(self.log_dir)

        # Standard logger
        self.logger = logging.getLogger("Observability")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
                )
            )
            self.logger.addHandler(h)
        self.logger.setLevel(console_level)

        # Touch file so it exists
        try:
            if not os.path.exists(self.jsonl_path):
                with open(self.jsonl_path, "w", encoding="utf-8") as f:
                    f.write("")
        except Exception:
            # Don't raise on logging setup failures
            self.logger.exception("Failed to initialize JSONL log file.")

    # ------------------------------------------------------------------
    # Core JSONL writer
    # ------------------------------------------------------------------
    def _write_event(self, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            # Don't crash trading if logging fails
            self.logger.exception("Failed to write observability event.")

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def record_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        level: int = logging.INFO,
        msg: Optional[str] = None,
    ) -> None:
        """
        Generic event logger.

        Parameters
        ----------
        event_type : str
            Short name, e.g. "session_start", "risk_hit", "trade_open".
        data : dict
            Arbitrary payload (must be JSON-serializable).
        level : int
            Logging level for console.
        msg : str
            Optional human-readable message for console log.
        """
        if not self.enabled:
            return

        payload: Dict[str, Any] = {
            "ts": _iso_ts(),
            "event_type": event_type,
            "data": data or {},
        }
        self._write_event(payload)

        # Console log
        if msg is None:
            msg = f"{event_type}: {data}"
        self.logger.log(level, msg)

    def record_trade(self, trade_record: Dict[str, Any]) -> None:
        """
        Record a trade event. `trade_record` should be the dict produced by
        LivePaperTradingEngine / ExecutionEngine / agent_trade.

        This is just a convenience wrapper around record_event("trade", ...).
        """
        if not self.enabled:
            return
        self.record_event("trade", {"trade": trade_record}, level=logging.INFO)

    def record_session_summary(self, summary: Dict[str, Any]) -> None:
        """
        Record summary at end of session/day.
        E.g.: {"n_trades": 5, "pnl": 1245.0, "max_dd": -500.0}
        """
        if not self.enabled:
            return
        self.record_event("session_summary", summary, level=logging.INFO)

    def record_error(self, where: str, error: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an error event with context.
        """
        if not self.enabled:
            return
        data = {"where": where, "error": error}
        if extra:
            data["extra"] = extra
        self.record_event("error", data, level=logging.ERROR)

    def record_metric(self, name: str, value: Any, tags: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a single metric data point.

        Example:
            obs.record_metric("open_positions", 4, {"symbol": "NIFTY"})
        """
        if not self.enabled:
            return
        data = {"name": name, "value": value, "tags": tags or {}}
        self.record_event("metric", data, level=logging.DEBUG)


# ---------------------------------------------------------------------
# Singleton-style helper
# ---------------------------------------------------------------------
_global_observer: Optional[Observability] = None


def get_observer() -> Observability:
    """
    Get global Observability instance (lazy-init).

    By default:
        log_dir = "logs"
        jsonl_file = "events.jsonl"
        enabled = True

    You can override by constructing your own Observability instance
    and setting _global_observer manually if needed.
    """
    global _global_observer
    if _global_observer is None:
        _global_observer = Observability()
    return _global_observer
