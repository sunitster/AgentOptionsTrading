# src/trading/signal_generator.py
"""
Robust SignalGenerator — patched shim with trading lifecycle methods.

This version preserves your existing generation logic but provides a
safe shim of ProjectIronCondor with methods:
 - entry_orders()
 - as_dict()
 - mark_to_market(chain_df)
 - exit_pnl(chain_df)
 - max_loss()
 - and computes entry_credit at init

That ensures LivePaperEngine and PaperBroker can call expected methods
even when the project's real IronCondor builder is missing.
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import datetime as dt
import pandas as pd
import numpy as np

LOG = logging.getLogger("SignalGenerator")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

# -------------------------
# Small BS helpers (only iv functions needed)
# -------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_price(option_type: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= 0:
        if str(option_type).upper().startswith("P"):
            return max(K - S, 0.0)
        return max(S - K, 0.0)
    if sigma <= 0:
        if str(option_type).upper().startswith("P"):
            return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if str(option_type).upper().startswith("P"):
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)
    else:
        price = S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return price

def bs_vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * math.sqrt(T) * _norm_pdf(d1)

def implied_volatility(option_type: str, price: float, S: float, K: float, T: float, r: float = 0.06, q: float = 0.0,
                       tol: float = 1e-6, max_iter: int = 100) -> Optional[float]:
    """Newton-Raphson implied vol. Defensive; returns None if impossible."""
    try:
        if price is None or price <= 0 or S <= 0 or K <= 0 or T <= 0:
            return None
    except Exception:
        return None
    sigma = 0.25
    for _ in range(max_iter):
        try:
            price_model = bs_price(option_type, S, K, T, r, sigma, q)
            diff = price_model - price
            if abs(diff) < tol:
                return max(1e-6, sigma)
            vega = bs_vega(S, K, T, r, sigma, q)
            if vega == 0:
                return None
            sigma = sigma - diff / vega
            # clamp
            if sigma <= 1e-8:
                sigma = 1e-8
            if sigma > 5.0:
                sigma = 5.0
        except Exception:
            return None
    return None

# -------------------------
# Try to import project IronCondor; fallback to shim if not present
# -------------------------
try:
    # Try common project import paths (allowability)
    from trading.iron_condor_builder import IronCondor as ProjectIronCondor, build_iron_condor, _mid_price  # type: ignore
    _HAS_PROJECT_BUILDER = True
    LOG.info("SignalGenerator: using project IronCondor builder.")
except Exception:
    _HAS_PROJECT_BUILDER = False
    LOG.warning("SignalGenerator: IronCondor import failed — using local shim.")

    @dataclass
    class ProjectIronCondor:
        """
        Local shim for IronCondor with lifecycle helpers required by LivePaperEngine.
        This shim is intentionally *minimal* and defensive — used only when the project
        IronCondor is not importable.
        """
        symbol: str
        expiry: Optional[str]
        short_put: float
        long_put: Optional[float]
        short_call: float
        long_call: Optional[float]
        short_put_price: float = float("nan")
        long_put_price: float = float("nan")
        short_call_price: float = float("nan")
        long_call_price: float = float("nan")
        lot_size: int = 25
        size_aggressiveness: float = 1.0

        def __post_init__(self):
            # compute sensible entry values used by risk engine if they ask
            # entry_credit = (prem received from shorts) - (prem paid for longs)
            try:
                sp = 0.0 if self.short_put_price is None or (isinstance(self.short_put_price, float) and math.isnan(self.short_put_price)) else float(self.short_put_price)
                sc = 0.0 if self.short_call_price is None or (isinstance(self.short_call_price, float) and math.isnan(self.short_call_price)) else float(self.short_call_price)
                lp = 0.0 if self.long_put_price is None or (isinstance(self.long_put_price, float) and math.isnan(self.long_put_price)) else float(self.long_put_price)
                lc = 0.0 if self.long_call_price is None or (isinstance(self.long_call_price, float) and math.isnan(self.long_call_price)) else float(self.long_call_price)
                # net premium received (per contract)
                self.entry_credit = (sp + sc) - (lp + lc)
            except Exception:
                self.entry_credit = 0.0

        def entry_orders(self):
            qty = max(1, int(round(self.size_aggressiveness))) * self.lot_size

            # -------- SAFETY PATCH FOR MISSING WINGS --------
            lp = self.long_put
            if lp is None:
                # choose sensible wing 100 points away if missing
                try:
                    lp = int(self.short_put - 100)
                except Exception:
                    lp = int(self.short_put) - 100

            lc = self.long_call
            if lc is None:
                try:
                    lc = int(self.short_call + 100)
                except Exception:
                    lc = int(self.short_call) + 100
            # ------------------------------------------------

            return [
                {"symbol": f"{self.symbol}_PE_{int(self.short_put)}", "qty": -qty, "side": "SELL", "price": float(self.short_put_price) if not (isinstance(self.short_put_price, float) and math.isnan(self.short_put_price)) else None},
                {"symbol": f"{self.symbol}_PE_{int(lp)}",            "qty": qty,  "side": "BUY",  "price": float(self.long_put_price) if not (isinstance(self.long_put_price, float) and math.isnan(self.long_put_price)) else None},
                {"symbol": f"{self.symbol}_CE_{int(self.short_call)}", "qty": -qty, "side": "SELL", "price": float(self.short_call_price) if not (isinstance(self.short_call_price, float) and math.isnan(self.short_call_price)) else None},
                {"symbol": f"{self.symbol}_CE_{int(lc)}",              "qty": qty,  "side": "BUY",  "price": float(self.long_call_price) if not (isinstance(self.long_call_price, float) and math.isnan(self.long_call_price)) else None},
            ]

        def as_dict(self) -> Dict[str, Any]:
            return {
                "symbol": self.symbol,
                "expiry": self.expiry,
                "short_put": self.short_put,
                "long_put": self.long_put,
                "short_call": self.short_call,
                "long_call": self.long_call,
                "short_put_price": None if (isinstance(self.short_put_price, float) and math.isnan(self.short_put_price)) else self.short_put_price,
                "long_put_price": None if (isinstance(self.long_put_price, float) and math.isnan(self.long_put_price)) else self.long_put_price,
                "short_call_price": None if (isinstance(self.short_call_price, float) and math.isnan(self.short_call_price)) else self.short_call_price,
                "long_call_price": None if (isinstance(self.long_call_price, float) and math.isnan(self.long_call_price)) else self.long_call_price,
                "lot_size": int(self.lot_size),
                "size_aggressiveness": float(self.size_aggressiveness),
                "entry_credit": float(getattr(self, "entry_credit", 0.0))
            }

        def _get_mid_from_chain(self, chain_df: Optional[pd.DataFrame], strike_val: float, opt_type: str) -> float:
            """Helper to get mid price from chain_df for a given strike & option type. Returns nan if not found."""
            try:
                if chain_df is None or chain_df.empty:
                    return float("nan")
                # compare strikes safely
                svals = chain_df["strike"].apply(lambda x: float(x) if (x is not None and not (isinstance(x, float) and math.isnan(x))) else np.nan)
                # But above lambda could fail for None - use safe approach below
            except Exception:
                pass

            try:
                def safe_strike(v):
                    try:
                        return float(v)
                    except Exception:
                        return np.nan
                cond = (chain_df["strike"].apply(safe_strike).fillna(np.nan) == float(strike_val)) & (chain_df["option_type"].str.upper() == opt_type.upper())
                tmp = chain_df[cond]
                if tmp.empty:
                    return float("nan")
                row = tmp.iloc[0]
                # prefer ltp, then mid of best_bid/best_ask
                if "ltp" in row and pd.notna(row["ltp"]):
                    return float(row["ltp"])
                b = row.get("best_bid")
                a = row.get("best_ask")
                if b is not None and a is not None and not (isinstance(b, float) and math.isnan(b)) and not (isinstance(a, float) and math.isnan(a)):
                    return float(b + a) / 2.0
            except Exception:
                pass
            return float("nan")

        def mark_to_market(self, chain_df: Optional[pd.DataFrame]) -> float:
            """
            Compute a simple mark-to-market (MTM) for the IC using mid prices from chain_df.
            PnL sign: positive means favorable (profit), negative means loss for the position holder.
            Calculation (per single contract):
              - short leg pnl = entry_price - current_price
              - long leg pnl  = current_price - entry_price
            Net pnl per contract multiplied by lot_size.
            """
            try:
                # current prices
                sp_cur = self._get_mid_from_chain(chain_df, self.short_put, "PE")
                sc_cur = self._get_mid_from_chain(chain_df, self.short_call, "CE")
                lp_cur = self._get_mid_from_chain(chain_df, self.long_put, "PE") if self.long_put is not None else float("nan")
                lc_cur = self._get_mid_from_chain(chain_df, self.long_call, "CE") if self.long_call is not None else float("nan")

                # entry prices (fallback to 0 if NaN to avoid exceptions)
                sp_e = 0.0 if (self.short_put_price is None or (isinstance(self.short_put_price, float) and math.isnan(self.short_put_price))) else float(self.short_put_price)
                sc_e = 0.0 if (self.short_call_price is None or (isinstance(self.short_call_price, float) and math.isnan(self.short_call_price))) else float(self.short_call_price)
                lp_e = 0.0 if (self.long_put_price is None or (isinstance(self.long_put_price, float) and math.isnan(self.long_put_price))) else float(self.long_put_price)
                lc_e = 0.0 if (self.long_call_price is None or (isinstance(self.long_call_price, float) and math.isnan(self.long_call_price))) else float(self.long_call_price)

                # compute per-contract pnl
                def leg_pnl_short(entry, current):
                    if current is None or (isinstance(current, float) and math.isnan(current)):
                        return 0.0
                    return entry - float(current)

                def leg_pnl_long(entry, current):
                    if current is None or (isinstance(current, float) and math.isnan(current)):
                        return 0.0
                    return float(current) - entry

                pnl_sp = leg_pnl_short(sp_e, sp_cur)
                pnl_sc = leg_pnl_short(sc_e, sc_cur)
                pnl_lp = leg_pnl_long(lp_e, lp_cur)
                pnl_lc = leg_pnl_long(lc_e, lc_cur)

                net_per_contract = pnl_sp + pnl_lp + pnl_sc + pnl_lc
                net_total = net_per_contract * int(self.lot_size)
                return float(net_total)
            except Exception:
                return 0.0

        def exit_pnl(self, chain_df: Optional[pd.DataFrame]) -> float:
            """
            Realize PnL by computing the same as mark_to_market and returning it.
            (PaperBroker will credit this amount.)
            """
            try:
                realized = self.mark_to_market(chain_df)
                return float(realized)
            except Exception:
                return 0.0

        def max_loss(self) -> float:
            """
            Conservative max loss estimate per contract:
            max( call width, put width ) * lot_size
            If wings missing use configured width fallback of 150.
            """
            try:
                call_width = None
                put_width = None
                if self.long_call is not None and self.short_call is not None:
                    call_width = float(self.long_call) - float(self.short_call)
                if self.long_put is not None and self.short_put is not None:
                    put_width = float(self.short_put) - float(self.long_put)
                widths = [w for w in (call_width, put_width) if w is not None and w > 0]
                if not widths:
                    # fallback to a safe default width estimate
                    default_width = 150.0
                    widths = [default_width]
                max_w = max(widths)
                return float(max_w * int(self.lot_size))
            except Exception:
                return float(150 * int(self.lot_size))


    def build_iron_condor(**kwargs):
        # Project builder not available — we'll construct with internal logic instead.
        return None

    def _mid_price(row):
        try:
            if row is None:
                return float("nan")
            if isinstance(row, (dict,)):
                # prefer ltp, then mid of best_bid/best_ask
                l = row.get("ltp")
                if l is not None and not (isinstance(l, float) and math.isnan(l)):
                    return float(l)
                b = row.get("best_bid")
                a = row.get("best_ask")
                if b is not None and a is not None:
                    return (float(b) + float(a)) / 2.0
                return float("nan")
            # pandas Series
            if "ltp" in row and not pd.isna(row["ltp"]):
                return float(row["ltp"])
            if "best_bid" in row and "best_ask" in row and not pd.isna(row["best_bid"]) and not pd.isna(row["best_ask"]):
                return (float(row["best_bid"]) + float(row["best_ask"])) / 2.0
        except Exception:
            pass
        return float("nan")

# -------------------------
# SignalGenerator class (unchanged from previous patch) but with freeze API
# -------------------------
class SignalGenerator:
    def __init__(self, width: int = 150, target_put_delta: float = 0.16, target_call_delta: float = 0.16,
                 default_r: float = 0.06, default_q: float = 0.0, **kwargs):
        self.width = int(width)
        self.target_put_delta = float(target_put_delta)
        self.target_call_delta = float(target_call_delta)
        self.default_r = float(default_r)
        self.default_q = float(default_q)

        # Common extras used by LivePaperEngine
        self.symbol = kwargs.get("symbol", None)
        self.size_aggressiveness = float(kwargs.get("size_aggressiveness", 1.0))
        LOG.info("SignalGenerator v2 init: width=%s put_delta=%s call_delta=%s symbol=%s extras=%s",
                 self.width, self.target_put_delta, self.target_call_delta, self.symbol, bool(kwargs))

        # Freeze control (minimal, in-memory)
        self._last_trade_date: Optional[dt.date] = None
        self._freeze: bool = False

    # -------------------------
    # Freeze API
    # -------------------------
    def reset_daily(self) -> None:
        """Unfreeze on a new calendar day (local system date)."""
        today = dt.date.today()
        if self._last_trade_date != today:
            # new day -> clear freeze
            self._freeze = False
            self._last_trade_date = None

    def freeze(self) -> None:
        """Set freeze flag (used after a successful candidate/entry)."""
        try:
            self._freeze = True
            self._last_trade_date = dt.date.today()
            LOG.debug("SignalGenerator: freeze set for date %s", self._last_trade_date)
        except Exception:
            self._freeze = True

    def unfreeze(self) -> None:
        """Clear freeze flag (used after position closed)."""
        try:
            self._freeze = False
            self._last_trade_date = None
            LOG.debug("SignalGenerator: unfreeze called")
        except Exception:
            self._freeze = False

    def is_frozen(self) -> bool:
        return bool(self._freeze)

    # -------------------------
    # Utilities
    # -------------------------
    def _normalize_df(self, df):
        if df is None:
            return pd.DataFrame()
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception:
                return pd.DataFrame()
        return df.copy()

    @staticmethod
    def _safe_float(x):
        try:
            if x is None:
                return None
            if isinstance(x, float):
                if math.isnan(x) or math.isinf(x):
                    return None
                return float(x)
            if isinstance(x, (int,)):
                return float(x)
            s = str(x).strip()
            if s == "":
                return None
            v = float(s)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return None

    def _time_to_expiry_years(self, expiry):
        now = dt.datetime.utcnow()
        try:
            if isinstance(expiry, (dt.date, dt.datetime)):
                ed = dt.datetime(expiry.year, expiry.month, expiry.day)
            else:
                ed = pd.to_datetime(expiry)
                ed = dt.datetime(ed.year, ed.month, ed.day)
            delta = ed - now
            secs = max(delta.total_seconds(), 0.0)
            return secs / (365.0 * 24 * 3600)
        except Exception:
            return 0.0

    def _compute_iv(self, df: pd.DataFrame, spot: float) -> pd.DataFrame:
        if df is None or df.empty or spot is None:
            return df
        df = df.copy()
        if "iv" not in df.columns:
            df["iv"] = None
        for idx, row in df.iterrows():
            try:
                if row.get("iv") is not None and not pd.isna(row.get("iv")):
                    continue
                price = self._safe_float(row.get("ltp"))
                if price is None:
                    continue
                K = self._safe_float(row.get("strike"))
                expiry = row.get("expiry")
                T = self._time_to_expiry_years(expiry)
                if T <= 0:
                    continue
                opt = (row.get("option_type") or "").upper()
                r = self._safe_float(row.get("r")) or self.default_r
                q = self._safe_float(row.get("q")) or self.default_q
                iv = implied_volatility(opt, price, float(spot), float(K), float(T), r=r, q=q)
                df.at[idx, "iv"] = iv
            except Exception:
                df.at[idx, "iv"] = None
        return df

    # -------------------------
    # Internal ATM builder (used if project builder missing)
    # -------------------------
    def _build_ic_internal_atm(self, df: pd.DataFrame, spot: float, lot_size: int = 25) -> Optional[ProjectIronCondor]:
        """
        Build best-effort symmetric iron-condor around ATM using available strikes.
        """
        try:
            if df is None or df.empty or spot is None:
                return None
            # collect valid integer strikes
            strikes = []
            for s in df["strike"].tolist():
                s_val = self._safe_float(s)
                if s_val is None:
                    continue
                # convert floats like 19500.0 -> 19500
                if float(s_val).is_integer():
                    strikes.append(int(round(s_val)))
                else:
                    strikes.append(int(round(s_val)))
            strikes = sorted(list(set(strikes)))
            if not strikes:
                return None

            # find ATM strike (nearest to spot)
            atm = min(strikes, key=lambda x: abs(x - spot))
            # candidate shorts symmetric by width
            target_short_put = atm - self.width
            target_short_call = atm + self.width

            # find nearest actual strikes for those targets
            def nearest(target):
                return min(strikes, key=lambda s: abs(s - target)) if strikes else None

            short_put = nearest(target_short_put)
            short_call = nearest(target_short_call)

            # if not found or identical, pick nearest below and above atm
            if short_put is None or short_call is None or short_put >= short_call:
                below = [s for s in strikes if s < atm]
                above = [s for s in strikes if s > atm]
                if not below or not above:
                    return None
                short_put = below[-1]
                short_call = above[0]

            # long wings: next outward available strikes
            below_all = [s for s in strikes if s < short_put]
            above_all = [s for s in strikes if s > short_call]
            long_put = below_all[-1] if below_all else None
            long_call = above_all[0] if above_all else None

            # obtain prices for legs (mid price function)
            def _get_row_by(strike_val, opt_type):
                try:
                    cond = (df["strike"].apply(lambda x: self._safe_float(x)) == float(strike_val)) & (df["option_type"].str.upper() == opt_type.upper())
                    tmp = df[cond]
                    if tmp.empty:
                        return None
                    return tmp.iloc[0]
                except Exception:
                    return None

            sp_row = _get_row_by(short_put, "PE")
            lp_row = _get_row_by(long_put, "PE") if long_put is not None else None
            sc_row = _get_row_by(short_call, "CE")
            lc_row = _get_row_by(long_call, "CE") if long_call is not None else None

            sp_px = _mid_price(sp_row) if sp_row is not None else float("nan")
            lp_px = _mid_price(lp_row) if lp_row is not None else float("nan")
            sc_px = _mid_price(sc_row) if sc_row is not None else float("nan")
            lc_px = _mid_price(lc_row) if lc_row is not None else float("nan")

            # allow some NaNs — but require both shorts to have a price
            if math.isnan(sp_px) or math.isnan(sc_px):
                return None

            expiry = None
            for r in (sp_row, sc_row, lp_row, lc_row):
                try:
                    if r is not None and "expiry" in r.index and pd.notna(r["expiry"]):
                        expiry = str(r["expiry"])
                        break
                except Exception:
                    continue

            ic = ProjectIronCondor(
                symbol=self.symbol or "NIFTY",
                expiry=expiry,
                short_put=float(short_put),
                long_put=float(long_put) if long_put is not None else None,
                short_call=float(short_call),
                long_call=float(long_call) if long_call is not None else None,
                short_put_price=float(sp_px),
                long_put_price=float(lp_px) if not math.isnan(lp_px) else float("nan"),
                short_call_price=float(sc_px),
                long_call_price=float(lc_px) if not math.isnan(lc_px) else float("nan"),
                lot_size=lot_size,
                size_aggressiveness=self.size_aggressiveness,
            )
            return ic
        except Exception:
            LOG.exception("Internal ATM builder failed")
            return None

    # -------------------------
    # Diagnostic helper
    # -------------------------
    def generate_diagnostic(self, chain_df: pd.DataFrame, spot: Optional[float] = None) -> Dict[str, Any]:
        """
        Return diagnostic summary of the chain_df (useful to paste in bug reports).
        """
        df = self._normalize_df(chain_df)
        summary = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "has_strike": "strike" in df.columns,
            "sample_strikes": [],
            "num_with_ltp": 0,
            "num_with_iv": 0,
            "num_with_delta": 0,
            "expiry_values": [],
            "spot": spot,
        }
        if "strike" in df.columns:
            strikes = []
            for s in df["strike"].tolist()[:50]:
                try:
                    strikes.append(self._safe_float(s))
                except Exception:
                    strikes.append(None)
            summary["sample_strikes"] = strikes
        for c in ("ltp", "iv", "delta"):
            if c in df.columns:
                summary[f"num_with_{c}"] = int(df[c].dropna().shape[0])
            else:
                summary[f"num_with_{c}"] = 0
        if "expiry" in df.columns:
            try:
                summary["expiry_values"] = list(pd.Series(df["expiry"].dropna().unique())[:10])
            except Exception:
                summary["expiry_values"] = []
        return summary

    # -------------------------
    # Main generator
    # -------------------------
    def generate(self, chain_df: pd.DataFrame, spot: Optional[float] = None) -> List[ProjectIronCondor]:
        # Reset daily freeze if day rolled
        try:
            self.reset_daily()
        except Exception:
            pass

        # If freeze flag is active → no more signals
        if self.is_frozen():
            LOG.debug("SignalGenerator: frozen (existing IC active). Skipping candidate generation.")
            return []

        df = self._normalize_df(chain_df)
        if df.empty:
            LOG.info("SignalGenerator: empty chain provided")
            return []

        # infer spot if not provided
        if spot is None:
            for c in ("spot", "ltp", "underlying", "last_price"):
                if c in df.columns:
                    try:
                        v = df[c].dropna().iloc[0]
                        spot = self._safe_float(v)
                        break
                    except Exception:
                        continue
        if spot is None:
            LOG.warning("SignalGenerator: no spot available; cannot generate candidate")
            LOG.info("Diagnostic: %s", self.generate_diagnostic(df, spot))
            return []

        # compute ivs (defensively) for rows missing iv
        df_iv = self._compute_iv(df, spot)

        # attempt delta-based selection when deltas exist
        # build strike -> {CE,PE} map
        try:
            strike_map = {}
            for _, row in df_iv.iterrows():
                s = self._safe_float(row.get("strike"))
                if s is None:
                    continue
                strike = int(round(s))
                if strike not in strike_map:
                    strike_map[strike] = {}
                typ = (row.get("option_type") or row.get("opt") or row.get("otype") or "").upper()
                if typ.startswith("C"):
                    strike_map[strike]["CE"] = row
                elif typ.startswith("P"):
                    strike_map[strike]["PE"] = row
        except Exception:
            strike_map = {}

        # check if any deltas present
        any_delta = False
        if "delta" in df_iv.columns:
            try:
                any_delta = df_iv["delta"].dropna().shape[0] > 0
            except Exception:
                any_delta = False

        if any_delta:
            # find strike with delta ~ target
            best_put = (None, float("inf"))
            best_call = (None, float("inf"))
            for strike, sides in strike_map.items():
                pe = sides.get("PE")
                ce = sides.get("CE")
                try:
                    if pe is not None and not pd.isna(pe.get("delta")):
                        d = abs(abs(float(pe.get("delta"))) - abs(self.target_put_delta))
                        if d < best_put[1]:
                            best_put = (strike, d)
                except Exception:
                    pass
                try:
                    if ce is not None and not pd.isna(ce.get("delta")):
                        d = abs(float(ce.get("delta")) - self.target_call_delta)
                        if d < best_call[1]:
                            best_call = (strike, d)
                except Exception:
                    pass
            sp, sc = best_put[0], best_call[0]
            if sp is not None and sc is not None:
                # perform leg selection
                long_put = sp - self.width
                long_call = sc + self.width

                # find rows
                def _get_row(s_val, typ):
                    try:
                        cond = (df_iv["strike"].apply(lambda x: self._safe_float(x)) == float(s_val)) & (df_iv["option_type"].str.upper() == typ.upper())
                        tmp = df_iv[cond]
                        if tmp.empty:
                            return None
                        return tmp.iloc[0]
                    except Exception:
                        return None

                sp_row = _get_row(sp, "PE")
                sc_row = _get_row(sc, "CE")
                lp_row = _get_row(long_put, "PE")
                lc_row = _get_row(long_call, "CE")

                sp_px = _mid_price(sp_row) if sp_row is not None else float("nan")
                sc_px = _mid_price(sc_row) if sc_row is not None else float("nan")
                lp_px = _mid_price(lp_row) if lp_row is not None else float("nan")
                lc_px = _mid_price(lc_row) if lc_row is not None else float("nan")

                if not (math.isnan(sp_px) or math.isnan(sc_px)):
                    expiry = None
                    for r in (sp_row, sc_row, lp_row, lc_row):
                        try:
                            if r is not None and "expiry" in r.index and pd.notna(r["expiry"]):
                                expiry = str(r["expiry"])
                                break
                        except Exception:
                            continue

                    ic = ProjectIronCondor(
                        symbol=self.symbol or "NIFTY",
                        expiry=expiry,
                        short_put=float(sp),
                        long_put=float(long_put) if long_put is not None else None,
                        short_call=float(sc),
                        long_call=float(long_call) if long_call is not None else None,
                        short_put_price=float(sp_px),
                        long_put_price=float(lp_px) if not math.isnan(lp_px) else float("nan"),
                        short_call_price=float(sc_px),
                        long_call_price=float(lc_px) if not math.isnan(lc_px) else float("nan"),
                        lot_size=int(df_iv.get("lot_size", 25) if isinstance(df_iv, pd.DataFrame) and "lot_size" in df_iv.columns else 25),
                        size_aggressiveness=self.size_aggressiveness,
                    )
                    LOG.info("SignalGenerator: built IronCondor via delta SP=%s SC=%s", sp, sc)
                    # Freeze after producing a valid candidate (prevents further entries until unfreeze)
                    try:
                        self.freeze()
                    except Exception:
                        pass
                    return [ic]
                else:
                    LOG.info("SignalGenerator: delta-based selection found strikes but short legs lacked price, will fallback to ATM builder")

        # Attempt project build_iron_condor if available
        try:
            if _HAS_PROJECT_BUILDER:
                ic = build_iron_condor(chain_df=df_iv, symbol=self.symbol or "NIFTY", width=self.width, lot_size=int(df_iv.get("lot_size", 25) if isinstance(df_iv, pd.DataFrame) and "lot_size" in df_iv.columns else 25), size_aggressiveness=self.size_aggressiveness, spot=spot)
                if ic is not None:
                    LOG.info("SignalGenerator: built IronCondor using project builder")
                    try:
                        self.freeze()
                    except Exception:
                        pass
                    return [ic]
        except Exception:
            LOG.exception("SignalGenerator: project build_iron_condor raised exception; falling back to internal ATM builder")

        # Final fallback: internal ATM builder (guarantees a candidate if strikes/prices permit)
        ic_internal = self._build_ic_internal_atm(df_iv, spot, lot_size=int(df_iv.get("lot_size", 25) if isinstance(df_iv, pd.DataFrame) and "lot_size" in df_iv.columns else 25))
        if ic_internal is not None:
            LOG.info("SignalGenerator: built IronCondor via internal ATM builder SP=%s SC=%s", getattr(ic_internal, "short_put", None), getattr(ic_internal, "short_call", None))
            try:
                self.freeze()
            except Exception:
                pass
            return [ic_internal]

        # Nothing worked — log detailed diagnostic and return []
        diag = self.generate_diagnostic(df_iv, spot)
        LOG.info("SignalGenerator: could not build IronCondor (delta and ATM both failed). Diagnostic: %s", diag)
        return []

# convenience default instance
_default_sig = SignalGenerator()

def generate(chain_df: pd.DataFrame, spot: Optional[float] = None):
    return _default_sig.generate(chain_df=chain_df, spot=spot)

def generate_diagnostic(chain_df: pd.DataFrame, spot: Optional[float] = None):
    return _default_sig.generate_diagnostic(chain_df, spot)

if __name__ == "__main__":
    # quick self-check/demo when running directly
    LOG.setLevel(logging.DEBUG)
    demo = pd.DataFrame([
        {"strike": 26000, "option_type": "CE", "tradingsymbol": "NIFTY26000CE", "delta": 0.18, "ltp": 120, "expiry": "2025-12-02"},
        {"strike": 26000, "option_type": "PE", "tradingsymbol": "NIFTY26000PE", "delta": -0.17, "ltp": 110, "expiry": "2025-12-02"},
        {"strike": 26100, "option_type": "CE", "tradingsymbol": "NIFTY26100CE", "delta": 0.14, "ltp": 95, "expiry": "2025-12-02"},
        {"strike": 25900, "option_type": "PE", "tradingsymbol": "NIFTY25900PE", "delta": -0.12, "ltp": 80, "expiry": "2025-12-02"},
    ])
    print("Diagnostic:", generate_diagnostic(demo, spot=26141))
    print("Candidates:", generate(demo, spot=26141))
