# src/trading/signal_generator.py
"""
SignalGenerator v2 (auto-expiry + auto-width + delta-based IC)

Design goals:
- Backwards-compatible interface: SignalGenerator(...); generate(chain_df=..., spot=...)
- Auto-expiry: choose nearest expiry >= min_days_to_expiry (prefers weekly/Thursday)
- Auto-width: expand width if chain does not contain required strikes
- Delta-based strike selection:
    - If per-row IV is available (column 'iv' or 'implied_vol'), compute Black-Scholes delta
      (requires only spot, strike, ttm days, r ~ 0.06 default).
    - Otherwise use a robust heuristic mapping from moneyness -> approximate delta.
- Heavily defensive: returns [] or None when insufficient data, never raises for missing fields.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date, timedelta
import math

import pandas as pd
import numpy as np
from scipy.stats import norm  # used for Black-Scholes delta if IV present

LOG = logging.getLogger("SignalGenerator")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


@dataclass
class ICCandidate:
    """Simple container for candidate iron condor details (serializable-friendly)."""
    symbol: str
    expiry: str
    short_put: int
    long_put: int
    short_call: int
    long_call: int
    short_put_price: float
    long_put_price: float
    short_call_price: float
    long_call_price: float
    lot_size: int
    size_aggressiveness: float
    entry_credit: float = 0.0
    width: float = 0.0
    notional_per_side: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "expiry": self.expiry,
            "short_put": self.short_put,
            "long_put": self.long_put,
            "short_call": self.short_call,
            "long_call": self.long_call,
            "short_put_price": self.short_put_price,
            "long_put_price": self.long_put_price,
            "short_call_price": self.short_call_price,
            "long_call_price": self.long_call_price,
            "lot_size": self.lot_size,
            "size_aggressiveness": self.size_aggressiveness,
            "entry_credit": self.entry_credit,
            "width": self.width,
            "notional_per_side": self.notional_per_side,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
        }


class SignalGenerator:
    def __init__(
        self,
        symbol: str = "NIFTY",
        lot_size: int = 25,
        width: int = 150,
        size_aggressiveness: float = 1.0,
        target_put_delta: float = 0.16,
        target_call_delta: float = 0.16,
        min_days_to_expiry: int = 3,
        width_multiplier: float = 1.0,
        max_width: int = 2000,
        use_bs: bool = True,
    ):
        self.symbol = symbol.upper()
        self.lot_size = int(lot_size)
        self.default_width = int(width)
        self.size_aggressiveness = float(size_aggressiveness)
        self.target_put_delta = float(target_put_delta)
        self.target_call_delta = float(target_call_delta)
        self.min_days_to_expiry = int(min_days_to_expiry)
        self.width_multiplier = float(width_multiplier)
        self.max_width = int(max_width)
        self.use_bs = bool(use_bs)

        LOG.info("SignalGenerator v2 init: %s width=%s target_put_delta=%s target_call_delta=%s",
                 self.symbol, width, self.target_put_delta, self.target_call_delta)

    # ----------------- public API -----------------
    def generate(self, chain_df: pd.DataFrame, spot: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Main entry: build candidate(s). Returns list of candidate dicts (or empty list).
        `chain_df` expected to have first row = UNDERLYING (spot), or a 'spot' column.
        """
        try:
            if chain_df is None or chain_df.empty:
                LOG.debug("SignalGenerator: empty chain_df")
                return []

            # Ensure spot
            try:
                if spot is None:
                    # try from first row's 'spot' or 'ltp'
                    first = chain_df.iloc[0]
                    spot = float(first.get("spot") or first.get("ltp") or spot)
            except Exception:
                spot = spot or 0.0

            if not spot or spot <= 0:
                LOG.warning("SignalGenerator: invalid spot=%s", spot)
                return []

            # Normalize expected columns
            df = self._normalize_chain(chain_df)

            # Choose expiry
            expiry = self._choose_expiry(df)
            if expiry is None:
                LOG.debug("SignalGenerator: no valid expiry")
                return []

            # Narrow to expiry slice
            df_e = df[df["expiry"] == expiry].copy()
            if df_e.empty:
                LOG.debug("SignalGenerator: no rows for expiry %s", expiry)
                return []

            # Determine dynamic width
            width = self._determine_width(spot)
            LOG.debug("SignalGenerator: using width=%s", width)

            # Filter strikes within width
            strikes = sorted(df_e["strike"].dropna().unique().tolist())
            strikes = [int(s) for s in strikes if abs(int(s) - int(spot)) <= width or True]  # keep all strikes but we'll pick by delta

            # compute deltas for available strikes
            strike_map = self._compute_strike_deltas(df_e, spot)

            # find candidate short strikes (closest to desired delta)
            short_put_strike = self._find_strike_for_target_delta(strike_map, side="PE", target_delta=self.target_put_delta)
            short_call_strike = self._find_strike_for_target_delta(strike_map, side="CE", target_delta=self.target_call_delta)

            if short_put_strike is None or short_call_strike is None:
                LOG.info("SignalGenerator: could not find both short strikes (put=%s call=%s)", short_put_strike, short_call_strike)
                return []

            # Determine long strikes (wings) by adding/subtracting width (ensure exist)
            # Choose wing spacing as nearest available strikes beyond the short leg
            long_put = self._choose_wing(df_e, short_put_strike, direction="down", max_width=self.max_width)
            long_call = self._choose_wing(df_e, short_call_strike, direction="up", max_width=self.max_width)

            if long_put is None or long_call is None:
                LOG.info("SignalGenerator: could not find long wings (put=%s call=%s)", long_put, long_call)
                return []

            # Obtain LTPs/prices for legs from df_e (fallback to 0 if missing)
            def price_for(sym:str, typ:str, strike:int)->float:
                row = df_e[(df_e["tradingsymbol"] == sym) | ((df_e["strike"]==strike) & (df_e["instrument_type"]==typ))]
                if not row.empty:
                    # prefer ltp, then mid of bid/ask
                    r = row.iloc[0]
                    if r.get("ltp") is not None:
                        return float(r.get("ltp"))
                    bb = r.get("best_bid")
                    ba = r.get("best_ask")
                    if bb is not None and ba is not None:
                        return (float(bb)+float(ba))/2.0
                return 0.0

            sp_short_put = int(short_put_strike)
            sp_long_put = int(long_put)
            sp_short_call = int(short_call_strike)
            sp_long_call = int(long_call)

            s_put_sym = self._symbol_for(df_e, sp_short_put, "PE")
            l_put_sym = self._symbol_for(df_e, sp_long_put, "PE")
            s_call_sym = self._symbol_for(df_e, sp_short_call, "CE")
            l_call_sym = self._symbol_for(df_e, sp_long_call, "CE")

            s_put_price = price_for(s_put_sym, "PE", sp_short_put)
            l_put_price = price_for(l_put_sym, "PE", sp_long_put)
            s_call_price = price_for(s_call_sym, "CE", sp_short_call)
            l_call_price = price_for(l_call_sym, "CE", sp_long_call)

            # Build candidate
            ic = ICCandidate(
                symbol=self.symbol,
                expiry=expiry,
                short_put=sp_short_put,
                long_put=sp_long_put,
                short_call=sp_short_call,
                long_call=sp_long_call,
                short_put_price=s_put_price,
                long_put_price=l_put_price,
                short_call_price=s_call_price,
                long_call_price=l_call_price,
                lot_size=self.lot_size,
                size_aggressiveness=self.size_aggressiveness
            )

            # compute basic stats
            try:
                credit = (ic.short_put_price + ic.short_call_price) - (ic.long_put_price + ic.long_call_price)
                ic.entry_credit = float(credit)
                ic.width = float(abs(ic.short_call - ic.long_call))  # approximate wing width
                ic.notional_per_side = ic.width * self.lot_size
                # max_profit, max_loss simple approx
                ic.max_profit = max(0.0, ic.entry_credit * self.lot_size)
                ic.max_loss = max(0.0, (ic.width * self.lot_size) - (ic.entry_credit * self.lot_size))
            except Exception:
                pass

            LOG.info("SignalGenerator: built IC candidate expiry=%s short_put=%s short_call=%s", expiry, sp_short_put, sp_short_call)
            return [ic.as_dict()]

        except Exception:
            LOG.exception("SignalGenerator.generate failed")
            return []

    # ---------------- internal helpers ----------------
    def _normalize_chain(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure expected columns exist and types normalized.
        Lower-case tolerant keys allowed.
        """
        df = df.copy()
        cols = {c.lower(): c for c in df.columns}
        # create canonical names by reading case-insensitively
        def getcol(key, default=None):
            k = key.lower()
            return cols.get(k, default)

        # standardize some names
        mapping = {}
        if getcol("tradingsymbol"):
            mapping[getcol("tradingsymbol")] = "tradingsymbol"
        if getcol("strike"):
            mapping[getcol("strike")] = "strike"
        if getcol("instrument_type"):
            mapping[getcol("instrument_type")] = "instrument_type"
        if getcol("expiry"):
            mapping[getcol("expiry")] = "expiry"
        if getcol("ltp"):
            mapping[getcol("ltp")] = "ltp"
        if getcol("best_bid"):
            mapping[getcol("best_bid")] = "best_bid"
        if getcol("best_ask"):
            mapping[getcol("best_ask")] = "best_ask"
        if getcol("open_interest"):
            mapping[getcol("open_interest")] = "open_interest"
        # implied vol optional
        if getcol("iv"):
            mapping[getcol("iv")] = "iv"
        if getcol("implied_vol"):
            mapping[getcol("implied_vol")] = "iv"

        if mapping:
            df = df.rename(columns=mapping)

        # Ensure types
        for c in ["strike"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # instrument_type unify CE/PE/UNDERLYING
        if "instrument_type" in df.columns:
            df["instrument_type"] = df["instrument_type"].astype(str).str.upper().fillna("")
        else:
            df["instrument_type"] = ""

        return df

    def _choose_expiry(self, df: pd.DataFrame) -> Optional[str]:
        """
        Choose expiry: nearest expiry >= min_days_to_expiry. Prefer weekly (Thursday).
        """
        if "expiry" not in df.columns:
            return None
        try:
            exps = [e for e in sorted(df["expiry"].dropna().unique().tolist())]
            parsed = []
            for e in exps:
                try:
                    parsed.append((e, datetime.fromisoformat(str(e)).date()))
                except Exception:
                    try:
                        parsed.append((e, datetime.strptime(str(e), "%Y-%m-%d").date()))
                    except Exception:
                        continue
            today = date.today()
            candidates = [(s, d) for (s, d) in parsed if (d - today).days >= self.min_days_to_expiry]
            if not candidates:
                return None
            # prefer Thursday weekly candidates
            th = [c for c in candidates if c[1].weekday() == 3]
            if th:
                chosen = min(th, key=lambda x: x[1])
                return chosen[0]
            # otherwise earliest candidate
            chosen = min(candidates, key=lambda x: x[1])
            return chosen[0]
        except Exception:
            return None

    def _determine_width(self, spot: float) -> int:
        """
        Auto-width: take the max of configured default and a fraction of spot,
        scaled by width_multiplier; clipped to max_width.
        """
        base = max(self.default_width, int(max(50, spot * 0.05)))
        w = int(base * self.width_multiplier)
        return min(w, self.max_width)

    def _compute_strike_deltas(self, df_e: pd.DataFrame, spot: float) -> Dict[int, Dict[str, Any]]:
        """
        Build mapping: strike -> {"CE": delta, "PE": delta, "best_ask", "best_bid", "ltp", "tradingsymbol"}
        Delta computed via:
            - If iv present and use_bs True -> compute BS delta (approx)
            - Else -> heuristic delta from moneyness
        """
        out = {}
        if df_e.empty:
            return out

        # calculate TTM fraction (in years) using expiry in df (assume same expiry)
        expiry_vals = df_e["expiry"].dropna().unique().tolist()
        ttm_days = None
        if expiry_vals:
            try:
                ex_date = datetime.fromisoformat(str(expiry_vals[0])).date()
                ttm_days = max(1, (ex_date - date.today()).days)
            except Exception:
                ttm_days = None

        iv_available = "iv" in df_e.columns and df_e["iv"].notna().any()

        # group by (strike, instrument_type)
        for _, row in df_e.iterrows():
            try:
                strike = int(row["strike"]) if row.get("strike") is not None else None
                if strike is None:
                    continue
                typ = str(row.get("instrument_type") or "").upper()
                if typ not in ("CE", "PE"):
                    continue
                ltp = row.get("ltp")
                bid = row.get("best_bid")
                ask = row.get("best_ask")
                iv = row.get("iv") if "iv" in row.index else None

                # compute delta
                d = None
                if self.use_bs and iv_available and iv is not None and ttm_days is not None and ttm_days > 0:
                    try:
                        vol = float(iv) / 100.0
                        t = float(ttm_days) / 365.0
                        d = self._bs_delta(spot=spot, strike=strike, t=t, vol=vol, option_type=typ)
                    except Exception:
                        d = None

                if d is None:
                    # fallback heuristic
                    moneyness = (strike - spot) / max(1.0, spot)
                    # For calls: delta decreases as strike rises; for puts: delta negative, but we'll use abs delta
                    # Use smooth mapping: abs_delta ≈ exp(-k*|moneyness|)
                    k = 6.0  # tuning param; 6.0 gives reasonable steepness
                    approx = math.exp(-k * abs(moneyness))
                    # Map to [0.02, 0.5] roughly
                    approx = max(0.02, min(0.5, approx * 0.5))
                    d = approx

                if strike not in out:
                    out[strike] = {"CE": None, "PE": None, "best_bid": None, "best_ask": None, "ltp": None, "tradingsymbol": None}

                out[strike][typ] = float(abs(d)) if d is not None else None
                # store a sample ltp/bid/ask/tradingsymbol (prefer CE entry)
                if out[strike]["tradingsymbol"] is None and row.get("tradingsymbol"):
                    out[strike]["tradingsymbol"] = row.get("tradingsymbol")
                out[strike]["best_bid"] = row.get("best_bid") or out[strike]["best_bid"]
                out[strike]["best_ask"] = row.get("best_ask") or out[strike]["best_ask"]
                out[strike]["ltp"] = row.get("ltp") or out[strike]["ltp"]
            except Exception:
                continue

        return out

    def _bs_delta(self, spot: float, strike: float, t: float, vol: float, option_type: str) -> float:
        """
        Black-Scholes delta (European) for calls and puts.
        r is assumed small (0.06). This is an approximation used when IV is available.
        """
        try:
            r = 0.06
            if t <= 0 or vol <= 0:
                # degenerate
                return 0.0
            d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
            if option_type.upper().startswith("C"):
                return float(norm.cdf(d1))
            else:
                # put delta = cdf(d1) - 1
                return float(norm.cdf(d1) - 1.0)
        except Exception:
            return 0.0

    def _find_strike_for_target_delta(self, strike_map: Dict[int, Dict[str, Any]], side: str, target_delta: float) -> Optional[int]:
        """
        Find the strike for which the option delta (abs) is closest to target_delta.
        If side == 'PE' we search in put deltas, else call deltas.
        """
        best = None
        best_diff = float("inf")
        for strike, info in strike_map.items():
            val = info.get(side)
            if val is None:
                continue
            diff = abs(val - target_delta)
            if diff < best_diff:
                best_diff = diff
                best = strike
        return best

    def _choose_wing(self, df_e: pd.DataFrame, short_strike: int, direction: str = "up", max_width: int = 2000) -> Optional[int]:
        """
        Choose a long wing strike given short strike.
        direction: 'up' for call side (long_call is greater strike), 'down' for put side.
        Strategy: find the nearest available strike beyond short strike where wing width <= max_width
        """
        strikes = sorted(df_e["strike"].dropna().unique().astype(int).tolist())
        if direction == "up":
            candidates = [s for s in strikes if s > short_strike and (s - short_strike) <= max_width]
            return candidates[0] if candidates else None
        else:
            candidates = [s for s in strikes if s < short_strike and (short_strike - s) <= max_width]
            return candidates[-1] if candidates else None

    def _symbol_for(self, df_e: pd.DataFrame, strike: int, typ: str) -> str:
        """
        Get a tradingsymbol for given strike and type; fallback to pattern matching.
        """
        rows = df_e[(df_e["strike"] == strike) & (df_e["instrument_type"] == typ)]
        if not rows.empty:
            return rows.iloc[0].get("tradingsymbol") or ""
        # fallback pattern: symbol + strike + typ (common pattern)
        return f"{self.symbol}{strike}{typ}"

