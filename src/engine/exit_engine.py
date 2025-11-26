# src/engine/exit_engine.py
import math
from datetime import datetime, time
from typing import List, Dict, Any

class ExitEngineConfig:
    """
    Configuration for ExitEngine behaviour.

    - profit_target_pct: take-profit return threshold (pnl / (credit × lot_size))
    - stop_loss_pct    : stop-loss return threshold
    - hard_close_time  : time-of-day based exit (15:15 etc.)
    - max_hard_loss    : absolute PnL floor (e.g. -20000)
    - disable_time_exit: if True, TIME_EXIT is never triggered (testing, paper-live)
    """

    def __init__(
        self,
        profit_target_pct: float = 0.30,   # +30% on credit
        stop_loss_pct: float = -0.20,      # -20% on credit
        hard_close_time: time = time(15, 15),
        max_hard_loss: float = -20000.0,
        disable_time_exit: bool = False,
    ):
        self.profit_target_pct = float(profit_target_pct)
        self.stop_loss_pct = float(stop_loss_pct)
        self.hard_close_time = hard_close_time
        self.max_hard_loss = float(max_hard_loss)
        self.disable_time_exit = bool(disable_time_exit)


class ExitEngine:
    """
    Simple, self-contained ExitEngine used by LivePaperEngine.
    ...
    (rest unchanged)
    """

    def __init__(self, config: ExitEngineConfig):
        self.config = config

    def scan_and_exit(
        self,
        ic,
        chain_df,
        now: datetime,
        unrealized_pnl: float,
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []

        if ic is None:
            return actions

        entry_credit = float(getattr(ic, "total_entry_credit", 0.0))
        lot_size = int(getattr(ic, "lot_size", 1))

        if entry_credit != 0:
            ret = unrealized_pnl / (abs(entry_credit) * lot_size)
        else:
            ret = 0.0

        # PROFIT TARGET
        if ret >= self.config.profit_target_pct:
            actions.append({...})
            return actions

        # STOP LOSS
        if ret <= self.config.stop_loss_pct:
            actions.append({...})
            return actions

        # HARD LOSS
        if unrealized_pnl <= self.config.max_hard_loss:
            actions.append({...})
            return actions

        # TIME-BASED EXIT
        hard_close_today = False
        now_time = now.time()

        if self.config.disable_time_exit:
            hard_close_today = False
        else:
            if 9 <= now.hour < 16:
                hc = self.config.hard_close_time or time(23, 59)
                if now_time >= hc:
                    hard_close_today = True
            else:
                hard_close_today = False

        if hard_close_today:
            actions.append(
                {
                    "pos_id": getattr(ic, "position_id", "IC"),
                    "reason": "TIME_EXIT",
                    "price": ic.current_mark_to_market(chain_df),
                    "time": now,
                    "notes": "Hard close time reached",
                }
            )

        return actions
