# src/trading/iron_condor_builder.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np


def _mid_price(row: pd.Series) -> float:
    """
    Helper to compute a reasonable price for an option row.
    Prefers 'ltp', else mid of 'best_bid'/'best_ask'.
    """
    if "ltp" in row and pd.notna(row["ltp"]):
        return float(row["ltp"])

    bid = float(row["best_bid"]) if "best_bid" in row and pd.notna(row["best_bid"]) else np.nan
    ask = float(row["best_ask"]) if "best_ask" in row and pd.notna(row["best_ask"]) else np.nan

    if np.isfinite(bid) and np.isfinite(ask):
        return (bid + ask) / 2.0
    if np.isfinite(bid):
        return bid
    if np.isfinite(ask):
        return ask

    return np.nan


@dataclass
class IronCondor:
    """
    Representation of an Iron Condor structure with helper methods
    required by LivePaperEngine and PaperBroker.

    NOTE:
    - Prices stored on the object are treated as *entry prices*.
    - MTM and exit PnL are computed from a fresh option chain snapshot.
    """

    symbol: str
    expiry: str

    short_put: float
    long_put: float
    short_call: float
    long_call: float

    short_put_price: float   # entry price (credit)
    long_put_price: float    # entry price (debit)
    short_call_price: float  # entry price (credit)
    long_call_price: float   # entry price (debit)

    lot_size: int = 25
    size_aggressiveness: float = 1.0

    # ---------------------------
    # Derived properties
    # ---------------------------
    @property
    def n_lots(self) -> float:
        # size_aggressiveness can be fractional; at order-build time we round.
        return max(self.size_aggressiveness, 0.0)

    @property
    def width_put(self) -> float:
        return abs(self.short_put - self.long_put)

    @property
    def width_call(self) -> float:
        return abs(self.long_call - self.short_call)

    @property
    def width(self) -> float:
        # Risk is determined by the wider side of the IC.
        return max(self.width_put, self.width_call)

    @property
    def entry_credit(self) -> float:
        """
        Net credit per unit (1 underlying) of the 4-leg structure.
        Positive for credit spreads.
        """
        return (self.short_put_price - self.long_put_price) + (
            self.short_call_price - self.long_call_price
        )

    @property
    def notional_per_side(self) -> float:
        """
        Notional exposure on the risky side per spread.
        """
        return self.width * self.lot_size * self.n_lots

    @property
    def max_profit(self) -> float:
        """
        Theoretical max profit (per spread) at expiry if all options expire worthless.
        """
        return self.entry_credit * self.lot_size * self.n_lots

    @property
    def max_loss(self) -> float:
        """
        Theoretical max loss (per spread) at expiry on one side.

        For a symmetric IC:
            max_loss_per_unit = width - entry_credit
        """
        loss_per_unit = max(self.width - self.entry_credit, 0.0)
        return loss_per_unit * self.lot_size * self.n_lots

    # ---------------------------
    # Serialization helpers
    # ---------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entry_credit"] = self.entry_credit
        d["width"] = self.width
        d["notional_per_side"] = self.notional_per_side
        d["max_profit"] = self.max_profit
        d["max_loss"] = self.max_loss
        return d

    def as_dict(self) -> Dict[str, Any]:
        return self.to_dict()

    # ---------------------------
    # Internal leg helpers
    # ---------------------------
    def _lots_int(self) -> int:
        if self.n_lots <= 0:
            return 0
        return max(1, int(round(self.n_lots)))

    def _leg_qty(self) -> int:
        """
        Underlying quantity per leg (contracts), including lot size and lots.
        """
        num_lots = self._lots_int()
        return num_lots * self.lot_size

    def _find_leg_row(
        self,
        chain_df: pd.DataFrame,
        strike_val: float,
        opt_type: str,  # 'CE' / 'PE'
    ) -> Optional[pd.Series]:
        df = chain_df.copy()
        df.columns = [c.lower() for c in df.columns]

        if "option_type" not in df.columns and "option_typ" in df.columns:
            df["option_type"] = df["option_typ"]

        if "strike" not in df.columns and "strike_pr" in df.columns:
            df["strike"] = df["strike_pr"]

        mask = (
            (df["strike"].astype(float) == float(strike_val)) &
            (df["option_type"].str.upper() == opt_type.upper())
        )
        leg = df[mask]
        if leg.empty:
            return None
        return leg.iloc[0]

    # ---------------------------
    # Orders
    # ---------------------------
    def entry_orders(self):
        """
        Build simple order dicts PaperBroker expects:
        {symbol, qty, side, price}
        """

        num_lots = self._lots_int()
        if num_lots <= 0:
            return []

        qty = num_lots * self.lot_size

        return [
            # Short put
            {
                "symbol": f"{self.symbol}_PE_{self.short_put}",
                "qty": -qty,
                "side": "SELL",
                "price": float(self.short_put_price),
            },
            # Long put
            {
                "symbol": f"{self.symbol}_PE_{self.long_put}",
                "qty": qty,
                "side": "BUY",
                "price": float(self.long_put_price),
            },
            # Short call
            {
                "symbol": f"{self.symbol}_CE_{self.short_call}",
                "qty": -qty,
                "side": "SELL",
                "price": float(self.short_call_price),
            },
            # Long call
            {
                "symbol": f"{self.symbol}_CE_{self.long_call}",
                "qty": qty,
                "side": "BUY",
                "price": float(self.long_call_price),
            },
        ]

    # ---------------------------
    # Real-life PnL computation
    # ---------------------------
    def _current_leg_price(
        self,
        chain_df: pd.DataFrame,
        strike_val: float,
        opt_type: str,
    ) -> Optional[float]:
        row = self._find_leg_row(chain_df, strike_val, opt_type)
        if row is None:
            return None
        px = _mid_price(row)
        if np.isnan(px):
            return None
        return float(px)

    def mark_to_market(self, chain_df: pd.DataFrame) -> float:
        """
        Compute *unrealized* PnL of the full IC from the current option chain.

        Uses:
            PnL = Σ [ qty_leg * (P_current_leg - entry_price_leg) ]
        where qty_leg is positive for long legs, negative for shorts.

        Returns:
            PnL in INR for the entire position (all lots, all legs).
        """
        qty = self._leg_qty()
        if qty <= 0:
            return 0.0

        # Current prices
        sp_cur = self._current_leg_price(chain_df, self.short_put, "PE")
        lp_cur = self._current_leg_price(chain_df, self.long_put, "PE")
        sc_cur = self._current_leg_price(chain_df, self.short_call, "CE")
        lc_cur = self._current_leg_price(chain_df, self.long_call, "CE")

        if any(p is None for p in (sp_cur, lp_cur, sc_cur, lc_cur)):
            # If any leg cannot be priced, better to not fake anything.
            return 0.0

        # Quantities (positive=long, negative=short)
        q_sp = -qty
        q_lp = qty
        q_sc = -qty
        q_lc = qty

        # Per-leg PnL = q * (P_now - P_entry)
        pnl_sp = q_sp * (sp_cur - self.short_put_price)
        pnl_lp = q_lp * (lp_cur - self.long_put_price)
        pnl_sc = q_sc * (sc_cur - self.short_call_price)
        pnl_lc = q_lc * (lc_cur - self.long_call_price)

        total_pnl = pnl_sp + pnl_lp + pnl_sc + pnl_lc
        return float(total_pnl)

    def exit_pnl(self, chain_df: pd.DataFrame) -> float:
        """
        Realized PnL if we CLOSE the entire IC now at current prices.

        For our simple paper broker, this is identical to mark_to_market().
        """
        return self.mark_to_market(chain_df)

    def expiry_pnl(self, spot_expiry: float) -> float:
        """
        Settlement PnL at expiry if all legs are held till expiry.

        Uses intrinsic values for each leg and the stored entry prices.
        Does not model brokerage, STT, or physical settlement nuances.
        """
        qty = self._leg_qty()
        if qty <= 0:
            return 0.0

        # Intrinsic values at expiry
        def intrinsic_call(spot: float, strike: float) -> float:
            return max(spot - strike, 0.0)

        def intrinsic_put(spot: float, strike: float) -> float:
            return max(strike - spot, 0.0)

        sp_intr = intrinsic_put(spot_expiry, self.short_put)
        lp_intr = intrinsic_put(spot_expiry, self.long_put)
        sc_intr = intrinsic_call(spot_expiry, self.short_call)
        lc_intr = intrinsic_call(spot_expiry, self.long_call)

        # Quantities (positive=long, negative=short)
        q_sp = -qty
        q_lp = qty
        q_sc = -qty
        q_lc = qty

        # Use intrinsic as "current price"
        pnl_sp = q_sp * (sp_intr - self.short_put_price)
        pnl_lp = q_lp * (lp_intr - self.long_put_price)
        pnl_sc = q_sc * (sc_intr - self.short_call_price)
        pnl_lc = q_lc * (lc_intr - self.long_call_price)

        total_pnl = pnl_sp + pnl_lp + pnl_sc + pnl_lc
        return float(total_pnl)

    def simulated_pnl(self, chain_df: Optional[pd.DataFrame] = None) -> float:
        """
        Backwards-compatible shim.

        - NEW: if chain_df is provided, returns real MTM PnL.
        - OLD: if called without data (legacy code), returns 0.0 to avoid
          lying about PnL.
        """
        if chain_df is None:
            return 0.0
        return self.mark_to_market(chain_df)


def build_iron_condor(
    chain_df: pd.DataFrame,
    symbol: str = "NIFTY",
    width: int = 150,
    lot_size: int = 25,
    size_aggressiveness: float = 1.0,
    expiry: Optional[str] = None,
    # NEW optional args for forward compatibility:
    spot: Optional[float] = None,
    lots_per_leg: int = 1,  # kept for signature compatibility; we use size_aggressiveness instead.
    **kwargs,     # swallow unused arguments safely
) -> Optional[IronCondor]:
    """
    Build a *symmetric* Iron Condor around ATM using a full option chain.

    Assumptions:
    - chain_df has columns: 'strike', 'option_type' ('CE'/'PE'), 'expiry',
      and at least one of 'ltp' or ('best_bid', 'best_ask').
    - We construct:
        short_put  = ATM - width
        long_put   = ATM - 2*width
        short_call = ATM + width
        long_call  = ATM + 2*width

    Returns:
        IronCondor or None if suitable strikes are missing.
    """
    if chain_df is None or chain_df.empty:
        return None

    df = chain_df.copy()
    df.columns = [c.lower() for c in df.columns]

    if "option_type" not in df.columns and "option_typ" in df.columns:
        df["option_type"] = df["option_typ"]

    if "strike" not in df.columns and "strike_pr" in df.columns:
        df["strike"] = df["strike_pr"]

    required_cols = {"strike", "option_type", "expiry"}
    if not required_cols.issubset(set(df.columns)):
        # Not enough info to build structure
        return None

    # Choose expiry
    if expiry is not None:
        df = df[df["expiry"] == expiry]

    if df.empty:
        return None

    # Allow engine-supplied spot to override
    if spot is None:
        if "spot" in df.columns and df["spot"].notna().any():
            spot = float(df["spot"].dropna().iloc[0])

    strikes = df["strike"].astype(float).unique()
    strikes = np.sort(strikes)

    if spot is None:
        # Fallback: approximate ATM as middle of strikes
        atm_strike = float(strikes[len(strikes) // 2])
    else:
        # Pick strike closest to spot
        idx = np.argmin(np.abs(strikes - spot))
        atm_strike = float(strikes[idx])

    # Define desired structure (symmetric IC)
    short_put = atm_strike - width
    long_put = atm_strike - 2 * width
    short_call = atm_strike + width
    long_call = atm_strike + 2 * width

    # Helper to grab the row for a specific leg
    def find_leg(strike_val: float, opt_type: str) -> Optional[pd.Series]:
        leg = df[
            (df["strike"].astype(float) == float(strike_val)) &
            (df["option_type"].str.upper() == opt_type)
        ]
        if leg.empty:
            return None
        return leg.iloc[0]

    sp_row = find_leg(short_put, "PE")
    lp_row = find_leg(long_put, "PE")
    sc_row = find_leg(short_call, "CE")
    lc_row = find_leg(long_call, "CE")

    if any(r is None for r in (sp_row, lp_row, sc_row, lc_row)):
        # Missing some strikes – can't build condor
        return None

    sp_px = _mid_price(sp_row)
    lp_px = _mid_price(lp_row)
    sc_px = _mid_price(sc_row)
    lc_px = _mid_price(lc_row)

    if any(np.isnan(x) for x in (sp_px, lp_px, sc_px, lc_px)):
        return None

    chosen_expiry = str(sp_row["expiry"])

    return IronCondor(
        symbol=symbol,
        expiry=chosen_expiry,
        short_put=float(short_put),
        long_put=float(long_put),
        short_call=float(short_call),
        long_call=float(long_call),
        short_put_price=float(sp_px),
        long_put_price=float(lp_px),
        short_call_price=float(sc_px),
        long_call_price=float(lc_px),
        lot_size=int(lot_size),
        size_aggressiveness=float(size_aggressiveness),
    )
