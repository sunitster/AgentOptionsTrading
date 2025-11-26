"""
src/engine/risk_engine.py

Runtime risk engine used by live / paper trading.

Goals
-----
- Enforce per-trade risk cap (e.g. <= 2% of current equity).
- Enforce daily drawdown circuit breaker (e.g. stop trading if -4% on the day).
- Provide a simple API that other modules (like LivePaperEngine, dashboards)
  can use to query current risk state.

This file is deliberately self-contained and lightweight so it can be reused
by paper/live engines without pulling in the training-side risk utilities
in src.learning.risk_engine.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
import datetime as dt
import logging


log = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Configuration for risk engine."""
    max_trade_risk_pct: float = 0.02       # 2% of equity per trade
    max_daily_drawdown_pct: float = 0.04   # 4% daily drawdown


@dataclass
class RiskState:
    """Mutable state for a single trading day."""
    trade_date: Optional[dt.date] = None
    day_start_equity: float = 0.0
    day_equity: float = 0.0
    day_realized_pnl: float = 0.0
    day_unrealized_pnl: float = 0.0
    day_max_equity: float = 0.0
    day_min_equity: float = 0.0
    halted: bool = False
    halt_reason: Optional[str] = None

    # ---- compatibility aliases for LivePaperEngine ----
    @property
    def halted_for_day(self) -> bool:
        return self.halted

    @halted_for_day.setter
    def halted_for_day(self, value: bool):
        self.halted = value

    @property
    def halt_reason_for_day(self) -> Optional[str]:
        return self.halt_reason

    @halt_reason_for_day.setter
    def halt_reason_for_day(self, value: Optional[str]):
        self.halt_reason = value


    def to_dict(self) -> Dict:
        return asdict(self)


class RiskEngine:
    """
    Runtime Risk Engine.

    Typical usage from a live/paper engine:

        risk = RiskEngine(max_trade_risk_pct=0.02, max_daily_drawdown_pct=0.04)

        # On each loop:
        risk.reset_day_if_needed(today, equity)

        allowed, reason = risk.can_open_trade(equity, trade_risk_amount)
        if not allowed:
            # skip new trade, log reason
            ...

        # After PnL changes:
        risk.update_pnl(equity, realized_pnl_delta, unrealized_pnl)

    The engine maintains daily state and can signal when trading should be halted.
    """

    def __init__(
        self,
        max_trade_risk_pct: float = 0.02,
        max_daily_drawdown_pct: float = 0.04,
    ) -> None:
        self.config = RiskConfig(
            max_trade_risk_pct=max_trade_risk_pct,
            max_daily_drawdown_pct=max_daily_drawdown_pct,
        )
        self.state = RiskState()

    # ------------------------------------------------------------------
    # Day handling
    # ------------------------------------------------------------------
    def reset_day_if_needed(self, trade_date: dt.date, current_equity: float) -> None:
        """
        Reset state if a new trading day has started.
        Should be called at the top of each main loop.
        """
        if self.state.trade_date != trade_date:
            log.info(
                "RiskEngine: resetting day state for %s (prev=%s)",
                trade_date, self.state.trade_date,
            )
            self.state.trade_date = trade_date
            self.state.day_start_equity = float(current_equity)
            self.state.day_equity = float(current_equity)
            self.state.day_realized_pnl = 0.0
            self.state.day_unrealized_pnl = 0.0
            self.state.day_max_equity = float(current_equity)
            self.state.day_min_equity = float(current_equity)
            self.state.halted = False
            self.state.halt_reason = None

    # ------------------------------------------------------------------
    # Trade permission
    # ------------------------------------------------------------------
    def can_open_trade(
        self,
        current_equity: float,
        trade_risk_amount: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a new trade is allowed given:
        - per-trade risk cap
        - daily drawdown status

        Parameters
        ----------
        current_equity : float
            Current total equity (capital + PnL).
        trade_risk_amount : float
            Maximum potential loss for the trade (margin at risk / IC width * lots etc).

        Returns
        -------
        (allowed, reason)
        """
        # If already halted, no more trades
        if self.state.halted:
            return False, self.state.halt_reason or "Trading halted for the day"

        # Per-trade risk check
        max_trade_risk = current_equity * self.config.max_trade_risk_pct
        if trade_risk_amount > max_trade_risk:
            reason = (
                f"Trade risk {trade_risk_amount:.2f} exceeds "
                f"per-trade cap {max_trade_risk:.2f} "
                f"({self.config.max_trade_risk_pct*100:.1f}% of equity)"
            )
            log.warning("RiskEngine.can_open_trade blocked: %s", reason)
            return False, reason

        return True, None

    # ------------------------------------------------------------------
    # PnL updates & circuit breaker
    # ------------------------------------------------------------------
    def update_pnl(
        self,
        current_equity: float,
        realized_pnl_delta: float = 0.0,
        unrealized_pnl: float = 0.0,
    ) -> None:
        """
        Update PnL and check for daily drawdown halts.

        Parameters
        ----------
        current_equity : float
            Updated equity value.
        realized_pnl_delta : float
            Incremental realized PnL since last call.
        unrealized_pnl : float
            Current unrealized PnL (for info / analytics).
        """
        if self.state.trade_date is None:
            # If someone calls update_pnl before reset_day_if_needed
            today = dt.date.today()
            self.reset_day_if_needed(today, current_equity)

        self.state.day_equity = float(current_equity)
        self.state.day_realized_pnl += float(realized_pnl_delta)
        self.state.day_unrealized_pnl = float(unrealized_pnl)

        # Track extremes
        self.state.day_max_equity = max(self.state.day_max_equity, current_equity)
        self.state.day_min_equity = min(self.state.day_min_equity, current_equity)

        # Daily drawdown check
        if self.state.day_start_equity > 0:
            dd = (self.state.day_start_equity - self.state.day_equity) / self.state.day_start_equity
        else:
            dd = 0.0

        if not self.state.halted and dd >= self.config.max_daily_drawdown_pct:
            self.state.halted = True
            self.state.halt_reason = (
                f"Daily drawdown {dd*100:.2f}% >= "
                f"limit {self.config.max_daily_drawdown_pct*100:.2f}%"
            )
            log.error("RiskEngine: DAILY CIRCUIT BREAKER TRIGGERED: %s", self.state.halt_reason)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def is_halted(self) -> bool:
        return self.state.halted

    @property
    def halt_reason(self) -> Optional[str]:
        return self.state.halt_reason

    def get_state(self) -> Dict:
        """Return current state as a plain dict (useful for UI / logging)."""
        return self.state.to_dict()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RiskEngine(config={self.config}, state={self.state})"
