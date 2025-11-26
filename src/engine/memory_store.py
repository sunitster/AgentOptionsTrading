# engine/memory_store.py

from __future__ import annotations
from typing import Dict, Any

class MemoryStore:
    """
    Lightweight local engine memory.
    Tracks:
        - last width adjustment (delta)
        - last plan decisions
        - recent losses (optional for future use)
    Persists only in-memory for a single backtest or live session.
    """

    def __init__(self):
        self._memory = {
            "last_width_delta": 0,
            "history": {}  # plan_id -> { width_delta, timestamp }
        }

    # ------------------------------------------------------------
    def reset(self):
        self._memory = {
            "last_width_delta": 0,
            "history": {}
        }

    # ------------------------------------------------------------
    def record_adjustment(self, plan_id: str, width_delta: int):
        """
        Called after each final decision.
        Stores width_delta so next AI overlay can avoid repeating adjustments.
        """
        self._memory["last_width_delta"] = width_delta
        self._memory["history"][plan_id] = {
            "width_delta": width_delta
        }

    # ------------------------------------------------------------
    def get_last_width_delta(self) -> int:
        """
        Returns last adjustment width_delta (default 0).
        """
        return int(self._memory.get("last_width_delta", 0))

    # ------------------------------------------------------------
    def recall_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Returns last known data for a plan_id, if any.
        """
        return self._memory["history"].get(plan_id, {})

    # ------------------------------------------------------------
    def __repr__(self):
        return f"MemoryStore(last_width_delta={self.get_last_width_delta()})"
