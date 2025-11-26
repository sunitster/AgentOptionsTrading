# engine/pnl.py
"""
Real PnL engine for paper trading/backtests.

Provides:
 - MarketModel: deterministic mid / bid / ask pricing for a spread
 - SlippageModel: simple slippage injection
 - PnLEngine: MTM calculation, entry/exit fill simulation, expiry settlement

Design goals:
 - Deterministic and auditable (no external data)
 - Reasonably realistic: mid moves with IV percentile and underlying
 - Lightweight: integrates with existing logging routes

Assumptions (explicit):
 - plan contains keys: plan_id, width (int, points), entry_price (float, credit),
   underlying_price (float) [price at plan creation], entry_iv (float 0..100)
 - `entry_price` is credit received when the spread was sold (positive for credit)
 - All PnL numbers are per 1 lot unit (you can multiply by position size externally)

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import math

from engine.logging_routes import log_trade, log_pnl, log_system


@dataclass
class MarketModel:
    """Simple deterministic market/pricing model for spreads."""
    # spread of bid/ask around mid as fraction of mid (e.g. 0.05 -> 5%)
    spread_pct: float = 0.05

    def mid_price(self, width: int, underlying_price: float, iv_percentile: float,
                  entry_underlying_price: float = None, entry_iv: float = None) -> float:
        """
        Estimate a mid price (current cost to buy back a short spread) in currency.

        Formula (deterministic, explainable):
          base = max(1.0, width * 0.1)
          iv_factor = (iv_percentile + (entry_iv or iv_percentile)) / 100.0  # average IV effect
          move = abs(underlying_price - (entry_underlying_price or underlying_price)) * 0.01
          mid = base * (0.8 + iv_factor * 1.2) + move

        This returns the estimated value of the spread (cost to close).
        """
        base = max(1.0, width * 0.1)
        iv_factor = (float(iv_percentile) + (float(entry_iv) if entry_iv is not None else float(iv_percentile))) / 100.0
        move = abs(underlying_price - (entry_underlying_price or underlying_price)) * 0.01
        mid = base * (0.8 + iv_factor * 1.2) + move
        return float(round(mid, 4))

    def bid_ask(self, width: int, underlying_price: float, iv_percentile: float,
                entry_underlying_price: float = None, entry_iv: float = None) -> Tuple[float, float, float]:
        """
        Return (bid, mid, ask)
        """
        mid = self.mid_price(width, underlying_price, iv_percentile, entry_underlying_price, entry_iv)
        half_spread = max(0.0001, abs(mid) * self.spread_pct / 2.0)
        bid = max(0.0, mid - half_spread)
        ask = mid + half_spread
        return (round(bid, 4), round(mid, 4), round(ask, 4))


@dataclass
class SlippageModel:
    """Simple slippage model.

    - fixed_ticks: fixed currency added/subtracted as slippage
    - pct_of_mid: slippage proportional to mid price
    """
    fixed_ticks: float = 0.0
    pct_of_mid: float = 0.0

    def apply(self, quoted_price: float, side: str = "buy") -> float:
        """Apply slippage to quoted price. side: 'buy' or 'sell'.
        For buys, slippage increases price; for sells, slippage decreases price (worse for taker).
        """
        sign = 1 if side.lower() == "buy" else -1
        slip = self.fixed_ticks + abs(quoted_price) * self.pct_of_mid
        return float(round(quoted_price + sign * slip, 4))


class PnLEngine:
    """Orchestrates MTM calculation and simulated fills for paper mode."""

    def __init__(self, market_model: MarketModel = None, slippage: SlippageModel = None):
        self.market_model = market_model or MarketModel()
        self.slippage = slippage or SlippageModel()

    def compute_mtm(self, plan: Dict[str, Any], market_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute mark-to-market for a short credit spread plan.

        Returns dict:
          {
            'plan_id', 'entry_credit', 'current_mid', 'bid', 'ask', 'mtm', 'current_value'
          }
        mtm = entry_credit - current_mid  (since we sold the spread for entry_credit; to close costs current_mid)
        """
        width = int(plan.get("width", 100))
        entry_credit = float(plan.get("entry_price", 0.0))
        entry_iv = float(plan.get("entry_iv", market_row.get("ivp", 50.0)))
        entry_underlying = float(plan.get("underlying_price", market_row.get("underlying", 0.0)))

        cur_underlying = float(market_row.get("price", market_row.get("underlying", entry_underlying)))
        cur_iv = float(market_row.get("ivp", entry_iv))

        bid, mid, ask = self.market_model.bid_ask(width, cur_underlying, cur_iv, entry_underlying, entry_iv)

        current_value = mid
        mtm = entry_credit - current_value

        return {
            "plan_id": plan.get("plan_id"),
            "entry_credit": round(entry_credit, 4),
            "bid": bid,
            "mid": mid,
            "ask": ask,
            "current_value": round(current_value, 4),
            "mtm": round(mtm, 4),
            "cur_underlying": cur_underlying,
            "cur_iv": cur_iv,
        }

    def simulate_entry_fill(self, plan: Dict[str, Any], market_row: Dict[str, Any], side: str = "sell") -> Dict[str, Any]:
        """
        Simulate an entry order being filled.
        - side: 'sell' (we collect credit) or 'buy' (we pay)

        Returns fill dict with: price, qty (1), slippage, mtm_after_fill
        """
        width = int(plan.get("width", 100))
        entry_credit = float(plan.get("entry_price", 0.0))

        # use mid as quoted price for the spread
        _, mid, _ = self.market_model.bid_ask(width, market_row.get("price"), market_row.get("ivp"),
                                             plan.get("underlying_price"), plan.get("entry_iv"))

        # implied taker action: if selling credit, we get slightly worse than mid
        quoted = mid
        fill_price = self.slippage.apply(quoted, side="sell" if side=="sell" else "buy")

        # For entry, mtm immediately after fill is entry_credit - fill_price
        mtm = round(entry_credit - fill_price, 4)

        fill = {
            "plan_id": plan.get("plan_id"),
            "side": side,
            "fill_price": fill_price,
            "qty": 1,
            "slippage": round(fill_price - quoted, 4),
            "fill_ts": market_row.get("date"),
            "mtm_after_fill": mtm,
        }

        # log trade
        log_trade(trade_id=f"trade-{plan.get('plan_id')}-entry-{market_row.get('date')}",
                  order_id=f"order-{plan.get('plan_id')}-entry-{market_row.get('date')}",
                  fill=fill,
                  pnl_update={"mtm_after_fill": mtm})

        # also log pnl snapshot
        log_pnl(date=market_row.get("date"), summary={
            "plan_id": plan.get("plan_id"),
            "pnl_event": "entry_fill",
            "entry_credit": round(entry_credit,4),
            "fill_price": fill_price,
            "mtm_after_fill": mtm,
        })

        return fill

    def simulate_exit_fill(self, plan: Dict[str, Any], market_row: Dict[str, Any], side: str = "buy") -> Dict[str, Any]:
        """
        Simulate exiting the spread (buying back if we initially sold credit).
        side: 'buy' (close short) or 'sell' (close long)
        """
        width = int(plan.get("width", 100))
        entry_credit = float(plan.get("entry_price", 0.0))

        bid, mid, ask = self.market_model.bid_ask(width, market_row.get("price"), market_row.get("ivp"),
                                                plan.get("underlying_price"), plan.get("entry_iv"))
        quoted = mid
        fill_price = self.slippage.apply(quoted, side=side)

        # mtm after exit is entry_credit - fill_price (realised)
        realised = round(entry_credit - fill_price, 4)

        fill = {
            "plan_id": plan.get("plan_id"),
            "side": side,
            "fill_price": fill_price,
            "qty": 1,
            "slippage": round(fill_price - quoted, 4),
            "fill_ts": market_row.get("date"),
            "realised_pnl": realised,
        }

        log_trade(trade_id=f"trade-{plan.get('plan_id')}-exit-{market_row.get('date')}",
                  order_id=f"order-{plan.get('plan_id')}-exit-{market_row.get('date')}",
                  fill=fill,
                  pnl_update={"realised": realised})

        log_pnl(date=market_row.get("date"), summary={
            "plan_id": plan.get("plan_id"),
            "pnl_event": "exit_fill",
            "fill_price": fill_price,
            "realised": realised,
        })

        return fill

    def settle_expiry(self, plan: Dict[str, Any], underlying_at_expiry: float) -> Dict[str, Any]:
        """
        Approximate expiry settlement for a short credit spread.

        Logic (conservative approx):
          - If underlying_at_expiry lies within wings (i.e., spread keeps worthless), spread value -> 0
            realised = entry_credit
          - If underlying moves beyond wings (worst-case), spread value -> width (max loss)
            realised = entry_credit - width

        This is a simplified but auditable approximation sufficient for paper trading risk tests.
        """
        width = int(plan.get("width", 100))
        entry_credit = float(plan.get("entry_price", 0.0))
        entry_underlying = float(plan.get("underlying_price", plan.get("underlying", 0.0)))

        # Simple check: if move from entry_underlying exceeds half-width (proxy for touching wing)
        move = abs(underlying_at_expiry - entry_underlying)
        touched = move >= (width / 2.0)

        if not touched:
            realised = round(entry_credit, 4)
            status = "expired_clean"
        else:
            realised = round(entry_credit - width, 4)
            status = "expired_hit_wing"

        # log
        log_system("expiry_settlement", {"plan_id": plan.get("plan_id"), "underlying_at_expiry": underlying_at_expiry, "status": status, "realised": realised})
        log_pnl(date=plan.get("expiry_date", "expiry"), summary={
            "plan_id": plan.get("plan_id"),
            "pnl_event": status,
            "realised": realised,
        })

        return {"plan_id": plan.get("plan_id"), "realised": realised, "status": status}
