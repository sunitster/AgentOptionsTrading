"""
risk_engine.py

Phase 6 — Risk Engine for AgentOptionsTrading

Features:
- Per-trade sizing so that risk per trade <= max_risk_pct_of_capital (default 0.02 i.e. 2%)
- Daily drawdown circuit breaker (stop trading when daily loss exceeds daily_loss_limit_pct)
- Max position size caps and absolute loss caps
- Risk-adjusted scoring helper for evolution & promotion
- Simple persistence (save/load state) to remember circuit-breaker across runs

Integration:
- Instantiate PositionSizer(capital, max_risk_pct=0.02, min_size=0.01, max_size=100.0)
- Use RiskEngine(...) to enforce daily limits and check each hypothetical trade before execution
- Use risk_adjusted_score(...) to rank strategies by risk-adjusted metrics

Notes:
- This module is intentionally conservative and dependency-free (stdlib + numpy/pandas only)
- The simulator in your pipeline should call `engine.allow_trade(...)` before committing trade size,
  and then call `engine.record_trade(...)` after an executed trade (or simulated execution).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import numpy as np
import pandas as pd

STATE_PATH = Path("models") / "risk_engine_state.json"


@dataclass
class PositionSizer:
    capital: float
    max_risk_pct: float = 0.02      # max fraction of capital risked per trade
    min_size: float = 0.01          # minimum position size (in contract units)
    max_size: float = 100.0         # maximum position size
    risk_per_unit_est: float = 1.0  # estimated risk per unit (in capital units) — user should set per-instrument

    def compute_size(self, stop_distance: float = None, risk_per_unit_override: float = None) -> float:
        """ 
        Compute position size (signed magnitude) such that:
            expected_risk = size * risk_per_unit <= capital * max_risk_pct

        Parameters:
        - stop_distance: optional, not used directly here; included for API compatibility
        - risk_per_unit_override: if provided, use this estimate instead of self.risk_per_unit_est

        Returns:
            size (positive float). Caller should apply sign (long/short).
        """
        rpu = risk_per_unit_override if risk_per_unit_override is not None else self.risk_per_unit_est
        if rpu <= 0:
            # Avoid division by zero; return min size fallback
            return float(self.min_size)
        max_risk_amount = float(self.capital) * float(self.max_risk_pct)
        raw_size = max_risk_amount / float(rpu)
        # clamp to min/max size
        size = max(self.min_size, min(self.max_size, raw_size))
        return float(size)

    def update_capital(self, new_capital: float):
        self.capital = float(new_capital)


@dataclass
class RiskEngineState:
    day_start_ts: Optional[float] = None
    daily_loss: float = 0.0
    daily_profit: float = 0.0
    halted: bool = False
    last_reset_ts: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Dict[str, Any]):
        return cls(**d)


class RiskEngine:
    """
    RiskEngine enforces:
     - per-trade risk <= position_sizer.max_risk_pct
     - daily loss circuit breaker (stop trading for the rest of the day)
     - optional absolute loss caps or allowed drawdowns

    Typical usage:
        sizer = PositionSizer(capital=100000.0, max_risk_pct=0.02, risk_per_unit_est=50.0)
        engine = RiskEngine(position_sizer=sizer, daily_loss_limit_pct=0.06, capital=100000.0)
        allowed, reason = engine.allow_trade(expected_risk_amount=..., proposed_size=..., trade_meta=...)
        if allowed:
            # execute or simulate
            engine.record_trade(realized_pnl=..., size=..., trade_meta=...)
    """

    def __init__(
        self,
        position_sizer: PositionSizer,
        capital: float,
        daily_loss_limit_pct: float = 0.06,
        per_trade_abs_loss_limit: Optional[float] = None,
        enable_persistence: bool = True,
        state_path: Optional[Path] = None,
    ):
        self.position_sizer = position_sizer
        self.capital = float(capital)
        self.daily_loss_limit_pct = float(daily_loss_limit_pct)
        self.per_trade_abs_loss_limit = per_trade_abs_loss_limit
        self.enable_persistence = enable_persistence
        self.state_path = state_path or STATE_PATH
        self.state = self._load_state()
        # sync sizer capital initially
        self.position_sizer.update_capital(self.capital)

    def _load_state(self) -> RiskEngineState:
        if self.enable_persistence and self.state_path.exists():
            try:
                raw = json.load(open(self.state_path, "r"))
                return RiskEngineState.from_json(raw)
            except Exception:
                pass
        # default state: fresh day
        s = RiskEngineState(
            day_start_ts=time.time(),
            daily_loss=0.0,
            daily_profit=0.0,
            halted=False,
            last_reset_ts=time.time(),
        )
        return s

    def _persist_state(self):
        if not self.enable_persistence:
            return
        try:
            self.state.last_reset_ts = self.state.last_reset_ts or time.time()
            json.dump(self.state.to_json(), open(self.state_path, "w"), indent=2)
        except Exception:
            # don't raise on persistence failure
            pass

    def reset_daily(self, force: bool = False):
        """
        Reset daily counters (should be called at start-of-day or when a new trading day is detected).
        """
        self.state = RiskEngineState(
            day_start_ts=time.time(),
            daily_loss=0.0,
            daily_profit=0.0,
            halted=False,
            last_reset_ts=time.time(),
        )
        self._persist_state()

    def is_halted(self) -> bool:
        return bool(self.state.halted)

    def allowed_daily_remaining_loss(self) -> float:
        """
        Return how much absolute capital is left before hitting daily loss limit.
        """
        cap = float(self.capital)
        limit = cap * float(self.daily_loss_limit_pct)
        remaining = limit + self.state.daily_profit - self.state.daily_loss
        # remaining can be negative (already in loss)
        return float(remaining)

    def allow_trade(
        self,
        expected_risk_amount: Optional[float] = None,
        proposed_size: Optional[float] = None,
        trade_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Decide whether a trade should be allowed.

        - expected_risk_amount: absolute capital at risk for the trade (e.g., price move * size * risk metric)
        - proposed_size: if not None, suggested number of contracts (size) to be executed
        - trade_meta: optional dict with fields like 'direction','symbol','regime','expected_slippage'

        Returns: (allowed: bool, reason: str)
        """
        if self.is_halted():
            return False, "engine_halted_by_circuit_breaker"

        # compute expected risk amount if not provided, using sizer estimate
        if expected_risk_amount is None and proposed_size is not None:
            expected_risk_amount = proposed_size * float(self.position_sizer.risk_per_unit_est)
        if expected_risk_amount is None:
            # fallback to smallest risk unit
            expected_risk_amount = float(self.position_sizer.risk_per_unit_est) * float(
                self.position_sizer.min_size
            )

        # per-trade absolute limit
        if self.per_trade_abs_loss_limit is not None:
            per_trade_limit = float(self.per_trade_abs_loss_limit)
            if expected_risk_amount > per_trade_limit:
                return False, "per_trade_abs_limit_exceeded"

        # ensure per-trade risk pct <= configured
        max_allowed = float(self.capital) * float(self.position_sizer.max_risk_pct)
        if expected_risk_amount > max_allowed * 1.00001:
            return False, "per_trade_risk_pct_exceeded"

        # ensure not exceeding daily remaining allowed loss (be conservative)
        remaining = self.allowed_daily_remaining_loss()
        if expected_risk_amount > remaining:
            return False, "daily_loss_limit_would_be_exceeded"

        # optional extra meta checks (slippage, regime)
        if trade_meta:
            slippage = trade_meta.get("expected_slippage", None)
            if slippage is not None:
                # if slippage is unexpectedly high relative to capital, block the trade
                if slippage > 0.01 * self.capital:
                    return False, "slippage_too_high"

        return True, "allowed"

    def record_trade(self, realized_pnl: float, size: float, trade_meta: Optional[Dict[str, Any]] = None):
        """
        Record the result of a trade (realized pnl positive or negative).
        This updates daily cumulative statistics and triggers halting if limits breached.
        """
        pnl = float(realized_pnl)
        if pnl >= 0:
            self.state.daily_profit += pnl
        else:
            self.state.daily_loss += abs(pnl)

        # check daily limit breach
        max_daily_loss = float(self.capital) * float(self.daily_loss_limit_pct)
        if self.state.daily_loss > max_daily_loss:
            self.state.halted = True

        self._persist_state()

    def manual_halt(self):
        self.state.halted = True
        self._persist_state()

    def manual_unhalt(self):
        self.state.halted = False
        self.state.daily_loss = 0.0
        self.state.daily_profit = 0.0
        self.state.last_reset_ts = time.time()
        self._persist_state()

    def update_capital(self, new_capital: float):
        """Update capital (and forward to sizer)."""
        self.capital = float(new_capital)
        self.position_sizer.update_capital(float(new_capital))

    # ---------------- risk metric helpers ----------------

    @staticmethod
    def compute_max_drawdown(cum_series: pd.Series) -> float:
        """
        Compute maximum drawdown of a cumulative PnL series (absolute units).
        """
        if cum_series is None or cum_series.empty:
            return 0.0
        roll_max = cum_series.cummax()
        drawdowns = roll_max - cum_series
        return float(drawdowns.max())

    @staticmethod
    def compute_sharpe(returns: pd.Series, annualization: float = 252.0) -> float:
        if returns is None or returns.empty:
            return 0.0
        mean = returns.mean()
        std = returns.std(ddof=0)
        if std <= 0:
            return 0.0
        return float(np.sqrt(annualization) * mean / std)


def risk_adjusted_score(metrics: Dict[str, Any], risk_penalty_lambda: float = 1.0) -> float:
    """
    Produce a scalar risk-adjusted fitness score for a candidate strategy.

    metrics expected keys:
      - 'total_pnl' (absolute)
      - 'daily_pnl' (pd.Series or list-like)
      - 'n_trades' (int)
      - 'trades_df' (optional DataFrame with trade-level info)
      - 'win_rate' (optional float)

    Score formula (configurable):
      score = total_pnl - lambda * max_drawdown - gamma * volatility_penalty + bonus*win_rate

    Return:
      float score (higher is better)
    """
    pnl = float(metrics.get("total_pnl", 0.0))
    daily = metrics.get("daily_pnl", None)
    try:
        if isinstance(daily, (list, tuple, np.ndarray)):
            daily_series = pd.Series(daily)
        elif isinstance(daily, pd.Series):
            daily_series = daily
        elif hasattr(daily, "values"):
            daily_series = pd.Series(daily.values)
        else:
            daily_series = pd.Series([])
    except Exception:
        daily_series = pd.Series([])

    mdd = RiskEngine.compute_max_drawdown(daily_series.cumsum()) if not daily_series.empty else 0.0
    vol = float(daily_series.std(ddof=0)) if not daily_series.empty else 0.0
    win_rate = float(metrics.get("win_rate", metrics.get("win_rate_estimate", 0.0)))

    # tunable coefficients
    lambda_coef = float(risk_penalty_lambda)
    gamma = 0.0  # additional vol penalty (set to 0 by default)
    bonus = 0.0  # win rate bonus (use 0 in initial experiments)

    score = pnl - lambda_coef * mdd - gamma * vol + bonus * win_rate
    return float(score)


# ---------------- simple CLI helper for local testing ----------------

def example_run_demo():
    """
    Simple demo showing how to use PositionSizer + RiskEngine with simulated trades.
    """
    print("Running demo")
    sizer = PositionSizer(
        capital=100000.0,
        max_risk_pct=0.02,
        risk_per_unit_est=50.0,
        min_size=1.0,
        max_size=100.0,
    )
    engine = RiskEngine(
        position_sizer=sizer,
        capital=100000.0,
        daily_loss_limit_pct=0.06,
        enable_persistence=False,
    )

    # propose trade (risk per unit estimate 50)
    suggested_size = sizer.compute_size()
    print("Suggested size:", suggested_size)
    allow, reason = engine.allow_trade(
        expected_risk_amount=suggested_size * sizer.risk_per_unit_est,
        proposed_size=suggested_size,
    )
    print("Allow:", allow, "Reason:", reason)

    # simulate a losing trade
    engine.record_trade(realized_pnl=-1500.0, size=suggested_size)
    print("After 1 loss daily_loss:", engine.state.daily_loss, "halted?", engine.is_halted())

    # simulate further losses to trigger halt
    engine.record_trade(realized_pnl=-5000.0, size=suggested_size)
    print("After more losses daily_loss:", engine.state.daily_loss, "halted?", engine.is_halted())


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


if __name__ == "__main__":
    example_run_demo()
