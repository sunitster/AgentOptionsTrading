# engine/logger.py
import os
import json
import time
import atexit
import threading
import queue
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_TYPES = ("decisions", "ai", "risk", "orders", "trades", "pnl", "system")

def iso_ts(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.isoformat(timespec="milliseconds").replace("+00:00", "Z")

class JSONLLogger:
    """
    Thread-safe JSONL logger writing per-log-type files into daily folders:
       <root>/<YYYY-MM-DD>/<type>.jsonl

    Usage:
      JSONLLogger.configure(root="logs", app_name="agent", flush_interval=1.0)
      logger = JSONLLogger.get()
      logger.log("decisions", {"foo": "bar"})
      JSONLLogger.get().close()   # or rely on atexit
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, root: str = "logs", app_name: str = None,
                 types: tuple = DEFAULT_TYPES, flush_interval: float = 1.0,
                 queue_max: int = 10000):
        self.root = Path(root)
        self.app_name = app_name or ""
        self.types = tuple(types)
        self.flush_interval = float(flush_interval)
        self.queue_max = int(queue_max)

        self._q = queue.Queue(maxsize=self.queue_max)
        self._files: Dict[str, Path] = {}
        self._file_handles: Dict[str, Any] = {}
        self._current_date = self._today_str()
        self._stop = threading.Event()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._seq = 0
        self._lock = threading.RLock()

        # ensure directory exists
        self._ensure_dirs()
        self._writer_thread.start()
        atexit.register(self.close)

    @classmethod
    def configure(cls, root: str = "logs", app_name: str = None,
                  types: tuple = DEFAULT_TYPES, flush_interval: float = 1.0,
                  queue_max: int = 10000):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = JSONLLogger(root, app_name, types, flush_interval, queue_max)
            else:
                # allow reconfigure only for certain fields
                inst = cls._instance
                inst.root = Path(root)
                inst.app_name = app_name or inst.app_name
                inst.types = tuple(types)
                inst.flush_interval = float(flush_interval)
        return cls._instance

    @classmethod
    def get(cls):
        with cls._instance_lock:
            if cls._instance is None:
                # default configure to project logs/
                cls._instance = JSONLLogger()
            return cls._instance

    def _today_str(self) -> str:
        # date folder in local time (IST assumed by user). We'll use localtime to match folder desires.
        return datetime.now().strftime("%Y-%m-%d")

    def _ensure_dirs(self):
        date_dir = self.root / self._today_str()
        date_dir.mkdir(parents=True, exist_ok=True)
        # open file handles lazily in writer when needed

    def _rotate_if_needed(self):
        today = self._today_str()
        if today != self._current_date:
            # close existing handles and create new folder
            with self._lock:
                for h in list(self._file_handles.values()):
                    try:
                        h.flush()
                        h.close()
                    except Exception:
                        pass
                self._file_handles.clear()
                self._files.clear()
                self._current_date = today
                self._ensure_dirs()

    def _get_handle(self, typ: str):
        if typ not in self.types:
            raise ValueError(f"Unknown log type '{typ}'. Allowed: {self.types}")
        if typ in self._file_handles:
            return self._file_handles[typ]

        date_dir = self.root / self._today_str()
        date_dir.mkdir(parents=True, exist_ok=True)
        path = date_dir / f"{typ}.jsonl"
        f = open(path, "a", encoding="utf-8")
        self._files[typ] = path
        self._file_handles[typ] = f
        return f

    def log(self, typ: str, payload: Dict[str, Any]):
        """
        Enqueue an event. payload should be JSON-serializable.
        The logger will add: event_id, timestamp (ISO8601 UTC), type, seq, app_name.
        """
        if typ not in self.types:
            raise ValueError(f"Unknown log type {typ}")
        event = {
            "event_id": str(uuid.uuid4()),
            "type": typ,
            "timestamp": iso_ts(),
            "seq": self._next_seq(),
            "app": self.app_name,
            "payload": payload
        }
        try:
            self._q.put_nowait(event)
        except queue.Full:
            # Backpressure: drop oldest to make room (keeps recent events)
            try:
                _ = self._q.get_nowait()
                self._q.put_nowait(event)
                # also log a system warning event about dropped event
                sys_payload = {
                    "warning": "queue_full_event_dropped",
                    "dropped_event_type": typ,
                    "queued_max": self.queue_max
                }
                # best-effort: avoid infinite recursion
                try:
                    self._q.put_nowait({
                        "event_id": str(uuid.uuid4()),
                        "type": "system",
                        "timestamp": iso_ts(),
                        "seq": self._next_seq(),
                        "app": self.app_name,
                        "payload": sys_payload
                    })
                except Exception:
                    pass
            except Exception:
                # if cannot free space, drop silently
                pass

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _writer_loop(self):
        """
        Background writer periodically empties the queue into appropriate files.
        """
        while not self._stop.is_set():
            try:
                # block for up to flush_interval to batch writes
                item = self._q.get(timeout=self.flush_interval)
            except queue.Empty:
                # rotate check + continue
                self._rotate_if_needed()
                continue

            batch = [item]
            # drain quickly
            try:
                while True:
                    batch.append(self._q.get_nowait())
            except queue.Empty:
                pass

            self._rotate_if_needed()

            # group by type and write to appropriate files
            writes = {}
            for ev in batch:
                t = ev.get("type", "system")
                writes.setdefault(t, []).append(ev)

            for t, events in writes.items():
                try:
                    f = self._get_handle(t)
                    for ev in events:
                        # ensure compact JSON (no spaces)
                        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                    f.flush()
                except Exception as exc:
                    # fallback: print to stderr once and continue
                    print(f"[logger] failed to write {t}: {exc}")

        # flush any remaining items on stop
        self._drain_on_close()

    def _drain_on_close(self):
        try:
            items = []
            while True:
                items.append(self._q.get_nowait())
        except queue.Empty:
            pass

        if not items:
            pass
        else:
            writes = {}
            for ev in items:
                t = ev.get("type", "system")
                writes.setdefault(t, []).append(ev)
            for t, events in writes.items():
                try:
                    f = self._get_handle(t)
                    for ev in events:
                        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                    f.flush()
                except Exception:
                    pass

        # close file handles
        with self._lock:
            for h in list(self._file_handles.values()):
                try:
                    h.flush()
                    h.close()
                except Exception:
                    pass
            self._file_handles.clear()

    def close(self):
        """
        Stop the background writer and flush remaining events. Safe to call multiple times.
        """
        if not self._stop.is_set():
            self._stop.set()
            # join writer thread (with timeout)
            try:
                self._writer_thread.join(timeout=5.0)
            except Exception:
                pass

    # helper convenience wrappers (optional)
    def log_decision(self, payload: Dict[str, Any]):
        self.log("decisions", payload)

    def log_ai(self, payload: Dict[str, Any]):
        self.log("ai", payload)

    def log_risk(self, payload: Dict[str, Any]):
        self.log("risk", payload)

    def log_order(self, payload: Dict[str, Any]):
        self.log("orders", payload)

    def log_trade(self, payload: Dict[str, Any]):
        self.log("trades", payload)

    def log_pnl(self, payload: Dict[str, Any]):
        self.log("pnl", payload)

    def log_system(self, payload: Dict[str, Any]):
        self.log("system", payload)
