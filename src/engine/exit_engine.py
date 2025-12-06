# src/engine/exit_engine.py
"""
ExitEngine — patched to include numeric kill-switch:
    - Hard numeric stop = min(max_loss_amount, max_loss_pct * account_balance)
This file is a hardened replacement of your uploaded exit_engine.py. See original for reference.
"""

from __future__ import annotations
import math
import logging
from datetime import datetime, time
from typing import List, Dict, Any, Optional

LOG = logging.getLogger("exit_engine")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO)


class ExitEngineConfig:
    """
    Configuration for ExitEngine behaviour.

    - profit_target_pct: take-profit return threshold (pnl / (credit × lot_size))
    - stop_loss_pct    : stop-loss return threshold (pnl / (credit × lot_size))
    - hard_close_time  : time-of-day based exit (15:15 etc.)
    - max_hard_loss    : legacy absolute PnL floor (e.g. -20000). Still honored.
    - disable_time_exit: if True, TIME_EXIT is never triggered (testing, paper-live)

    New fields:
    - max_loss_amount  : absolute currency value kill-switch (e.g. 25000.0)
    - max_loss_pct     : fraction of account balance to use as kill-switch (e.g. 0.02)
    - account_balance_default: fallback account balance used if engine can't read real balance
    """
    def __init__(
        self,
        profit_target_pct: float = 0.30,   # +30% on credit
        stop_loss_pct: float = -0.20,      # -20% on credit
        hard_close_time: time = time(15, 15),
        max_hard_loss: float = -20000.0,
        disable_time_exit: bool = False,
        # new parameters:
        max_loss_amount: float = 25_000.0,
        max_loss_pct: float = 0.02,
        account_balance_default: float = 1_000_000.0,
    ):
        self.profit_target_pct = float(profit_target_pct)
        self.stop_loss_pct = float(stop_loss_pct)
        self.hard_close_time = hard_close_time
        self.max_hard_loss = float(max_hard_loss)
        self.disable_time_exit = bool(disable_time_exit)

        # new fields
        self.max_loss_amount = float(max_loss_amount)
        self.max_loss_pct = float(max_loss_pct)
        self.account_balance_default = float(account_balance_default)


class ExitEngine:
    """
    Simple, self-contained ExitEngine used by LivePaperEngine.

    The engine inspects the provided `ic` (IronCondor-like object), the market
    snapshot (chain_df), current datetime `now`, and an externally-computed
    `unrealized_pnl`.

    It returns a list of action dicts. Each action dict typically contains:
        {
            "pos_id": <id>,
            "reason": <string>,
            "price": <float or None>,
            "time": <datetime>,
            "notes": <string>,
            "hard_limit": <float> (optional, for numeric-kill actions)
        }
    """
    def __init__(self, config: ExitEngineConfig):
        self.config = config

    def _read_account_balance_from_ic(self, ic) -> float:
        """
        Best-effort: attempt to read a sensible account/broker balance from the provided `ic`.
        Tries several common attribute names, otherwise returns config.account_balance_default.
        """
        if ic is None:
            return float(self.config.account_balance_default)
        for attr in ("account_balance", "balance", "broker_balance", "starting_balance", "capital"):
            try:
                val = getattr(ic, attr, None)
                if val is None:
                    # maybe as dict-like
                    if isinstance(ic, dict):
                        val = ic.get(attr)
                if val is not None:
                    try:
                        return float(val)
                    except Exception:
                        continue
            except Exception:
                continue
        return float(self.config.account_balance_default)

    def scan_and_exit(
        self,
        ic,
        chain_df,
        now: datetime,
        unrealized_pnl: float,
    ) -> List[Dict[str, Any]]:
        """
        Scan the open position `ic` against configured exit rules and return actions.

        Rules evaluated, in order of precedence:
         1) PROFIT TARGET (ret >= profit_target_pct)
         2) STOP LOSS (ret <= stop_loss_pct)
         3) LEGACY HARD LOSS (unrealized_pnl <= config.max_hard_loss)
         4) NUMERIC KILL-SWITCH (unrealized_pnl <= - numeric_kill_limit)
             where numeric_kill_limit = min(max_loss_amount, max_loss_pct * account_balance)
         5) TIME-BASED EXIT (if now >= hard_close_time)
        """
        actions: List[Dict[str, Any]] = []

        if ic is None:
            return actions

        # derive entry_credit & lot_size safely
        try:
            entry_credit = float(getattr(ic, "total_entry_credit", 0.0))
        except Exception:
            try:
                entry_credit = float(ic.get("total_entry_credit", 0.0)) if isinstance(ic, dict) else 0.0
            except Exception:
                entry_credit = 0.0

        try:
            lot_size = int(getattr(ic, "lot_size", 1))
        except Exception:
            try:
                lot_size = int(ic.get("lot_size", 1)) if isinstance(ic, dict) else 1
            except Exception:
                lot_size = 1

        # normalized return relative to credit*lot_size (guard divide by zero)
        if entry_credit != 0:
            ret = unrealized_pnl / (abs(entry_credit) * lot_size)
        else:
            ret = 0.0

        # helper to read a reasonable mark price (best-effort)
        def _current_price():
            # prefer method on ic if present
            try:
                if hasattr(ic, "current_mark_to_market"):
                    return float(ic.current_mark_to_market(chain_df))
            except Exception:
                pass
            try:
                if hasattr(ic, "mark_to_market"):
                    return float(ic.mark_to_market(chain_df))
            except Exception:
                pass
            try:
                # fallback to exit_pnl as proxy
                if hasattr(ic, "exit_pnl"):
                    return float(ic.exit_pnl(chain_df))
            except Exception:
                pass
            return None

        # 1) PROFIT TARGET
        try:
            if ret >= self.config.profit_target_pct:
                actions.append({
                    "pos_id": getattr(ic, "position_id", "IC"),
                    "reason": "PROFIT_TARGET",
                    "price": _current_price(),
                    "time": now,
                    "notes": f"Return {ret:.3f} >= profit_target_pct {self.config.profit_target_pct:.3f}",
                })
                return actions
        except Exception:
            LOG.exception("ExitEngine: profit target check failed")

        # 2) STOP LOSS (percent)
        try:
            if ret <= self.config.stop_loss_pct:
                actions.append({
                    "pos_id": getattr(ic, "position_id", "IC"),
                    "reason": "STOP_LOSS_PCT",
                    "price": _current_price(),
                    "time": now,
                    "notes": f"Return {ret:.3f} <= stop_loss_pct {self.config.stop_loss_pct:.3f}",
                })
                return actions
        except Exception:
            LOG.exception("ExitEngine: stop loss pct check failed")

        # 3) LEGACY HARD LOSS (absolute)
        try:
            if unrealized_pnl <= self.config.max_hard_loss:
                actions.append({
                    "pos_id": getattr(ic, "position_id", "IC"),
                    "reason": "HARD_LOSS",
                    "price": _current_price(),
                    "time": now,
                    "notes": f"unrealized_pnl {unrealized_pnl:.2f} <= max_hard_loss {self.config.max_hard_loss:.2f}",
                })
                return actions
        except Exception:
            LOG.exception("ExitEngine: legacy hard loss check failed")

        # 4) NUMERIC KILL-SWITCH: min(max_loss_amount, max_loss_pct * account_balance)
        try:
            account_balance = self._read_account_balance_from_ic(ic)
            numeric_limit_by_pct = float(self.config.max_loss_pct) * float(account_balance)
            numeric_kill_limit = min(float(self.config.max_loss_amount), numeric_limit_by_pct)
            # If unrealized_pnl is negative beyond the numeric limit -> force exit
            if unrealized_pnl <= -abs(numeric_kill_limit):
                actions.append({
                    "pos_id": getattr(ic, "position_id", "IC"),
                    "reason": "MAX_NUMERIC_STOP",
                    "price": _current_price(),
                    "time": now,
                    "notes": f"unrealized_pnl {unrealized_pnl:.2f} <= -numeric_kill_limit {numeric_kill_limit:.2f} (min of {self.config.max_loss_amount} and {numeric_limit_by_pct:.2f})",
                    "hard_limit": numeric_kill_limit,
                    "account_balance_used": account_balance,
                })
                return actions
        except Exception:
            LOG.exception("ExitEngine: numeric kill-switch check failed")

        # 5) TIME-BASED EXIT (hard_close_time)
        try:
            hard_close_today = False
            now_time = now.time()
            if self.config.disable_time_exit:
                hard_close_today = False
            else:
                if 9 <= now.hour < 16:  # market hours heuristic — only trigger during the day
                    hc = self.config.hard_close_time or time(23, 59)
                    if now_time >= hc:
                        hard_close_today = True
                else:
                    hard_close_today = False

            if hard_close_today:
                actions.append({
                    "pos_id": getattr(ic, "position_id", "IC"),
                    "reason": "TIME_EXIT",
                    "price": _current_price(),
                    "time": now,
                    "notes": "Hard close time reached",
                })
                return actions
        except Exception:
            LOG.exception("ExitEngine: time-based exit check failed")

        # no actions
        return actions
