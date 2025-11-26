"""
SignalGenerator (live-data focused, IV-aware, builds real IronCondor objects)

- Returns IronCondor objects (so PaperBroker.open_ic works).
- Uses delta-based strikes when possible; falls back to ATM symmetric builder.
- Computes IV when missing using Black-Scholes + Newton-Raphson.
- Accepts extra kwargs (symbol, size_aggressiveness...) for compatibility.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import datetime as dt

import pandas as pd
import numpy as np

LOG = logging.getLogger("SignalGenerator")
LOG.setLevel(logging.INFO)

# -------------------------
# Black-Scholes helpers
# -------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_price(option_type: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    # T in years
    if T <= 0:
        # expired intrinsic
        if str(option_type).upper().startswith("P"):
            return max(K - S, 0.0)
        return max(S - K, 0.0)
    if sigma <= 0:
        # no vol: intrinsic in PV terms
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
    """
    Newton-Raphson on Black-Scholes for implied vol. Returns None if cannot compute.
    Defensive: returns None on invalid inputs.
    """
    try:
        if price is None or price <= 0 or S <= 0 or K <= 0 or T < 0:
            return None
    except Exception:
        return None

    sigma = 0.25
    for i in range(max_iter):
        try:
            model_price = bs_price(option_type, S, K, T, r, sigma, q)
            diff = model_price - price
            if abs(diff) < tol:
                return max(1e-6, sigma)
            vega = bs_vega(S, K, T, r, sigma, q)
            if vega == 0:
                break
            step = diff / vega
            sigma = sigma - step
            # clamp
            if sigma <= 1e-8:
                sigma = 1e-8
            if sigma > 5.0:
                sigma = 5.0
        except Exception:
            break
    return None

# -------------------------
# Use project IronCondor builder when possible
# -------------------------
try:
    # import the builder & IronCondor type from your project
    from trading.iron_condor_builder import IronCondor, build_iron_condor, _mid_price  # type: ignore
    LOG.debug("SignalGenerator: using project IronCondor builder.")
except Exception:
    # If import fails, keep a minimal shim so generator still returns an object with expected methods.
    LOG.warning("SignalGenerator: IronCondor import failed — using a local shim. "
                "This should not happen if iron_condor_builder.py exists in repo.")
    @dataclass
    class IronCondor:
        symbol: str
        expiry: str
        short_put: float = 0.0
        long_put: float = 0.0
        short_call: float = 0.0
        long_call: float = 0.0
        short_put_price: float = 0.0
        long_put_price: float = 0.0
        short_call_price: float = 0.0
        long_call_price: float = 0.0
        lot_size: int = 25
        size_aggressiveness: float = 1.0

        def entry_orders(self):
            # keep behavior compatible with paper broker expectations
            num_lots = max(1, int(round(self.size_aggressiveness)))
            qty = num_lots * self.lot_size
            return [
                {"symbol": f"{self.symbol}_PE_{self.short_put}", "qty": -qty, "side": "SELL", "price": float(self.short_put_price)},
                {"symbol": f"{self.symbol}_PE_{self.long_put}", "qty": qty, "side": "BUY", "price": float(self.long_put_price)},
                {"symbol": f"{self.symbol}_CE_{self.short_call}", "qty": -qty, "side": "SELL", "price": float(self.short_call_price)},
                {"symbol": f"{self.symbol}_CE_{self.long_call}", "qty": qty, "side": "BUY", "price": float(self.long_call_price)},
            ]

        def as_dict(self):
            return self.__dict__

        def mark_to_market(self, chain_df: pd.DataFrame) -> float:
            return 0.0

        def exit_pnl(self, chain_df: pd.DataFrame) -> float:
            return 0.0

    def build_iron_condor(*args, **kwargs):
        # best-effort: nothing to build without project file
        return None

    def _mid_price(row: pd.Series) -> float:
        try:
            if row is None or not hasattr(row, "__getitem__"):
                return float(np.nan)
            if "ltp" in row and pd.notna(row["ltp"]):
                return float(row["ltp"])
            # prefer mid of best_bid/best_ask if present
            if "best_bid" in row and "best_ask" in row and pd.notna(row["best_bid"]) and pd.notna(row["best_ask"]):
                return (float(row["best_bid"]) + float(row["best_ask"])) / 2.0
        except Exception:
            pass
        return float(np.nan)

# -------------------------
# SignalGenerator class
# -------------------------
class SignalGenerator:
    """
    Build IronCondor objects suitable for PaperBroker.open_ic().

    __init__ accepts **kwargs for backward compatibility (e.g., symbol, size_aggressiveness).
    """

    def __init__(self, width: int = 150, target_put_delta: float = 0.16, target_call_delta: float = 0.16,
                 default_r: float = 0.06, default_q: float = 0.0, **kwargs):
        self.width = int(width)
        self.target_put_delta = float(target_put_delta)
        self.target_call_delta = float(target_call_delta)
        self.default_r = float(default_r)
        self.default_q = float(default_q)

        # compatibility: collect extras commonly passed by LivePaperEngine
        self.symbol = kwargs.pop("symbol", None)
        self.size_aggressiveness = float(kwargs.pop("size_aggressiveness", 1.0))
        # optional lot_size passed via kwargs or read from chain_df
        self.extra = kwargs or {}

        LOG.info("SignalGenerator v2 init: width=%s put_delta=%s call_delta=%s symbol=%s extras=%s",
                 self.width, self.target_put_delta, self.target_call_delta, self.symbol, bool(self.extra))

    @staticmethod
    def _normalize_chain_df(chain_df: pd.DataFrame) -> pd.DataFrame:
        if chain_df is None:
            return pd.DataFrame()
        if not isinstance(chain_df, pd.DataFrame):
            try:
                return pd.DataFrame(chain_df)
            except Exception:
                return pd.DataFrame()
        return chain_df.copy()

    @staticmethod
    def _time_to_expiry(expiry_val: Any, now: Optional[dt.datetime] = None) -> float:
        if now is None:
            now = dt.datetime.utcnow()
        try:
            if isinstance(expiry_val, (dt.datetime, dt.date)):
                expiry_dt = dt.datetime(expiry_val.year, expiry_val.month, expiry_val.day)
            else:
                expiry_dt = pd.to_datetime(expiry_val)
                expiry_dt = dt.datetime(expiry_dt.year, expiry_dt.month, expiry_dt.day)
            delta = expiry_dt - now
            secs = max(delta.total_seconds(), 0.0)
            # return fraction of years
            return secs / (365.0 * 24 * 3600)
        except Exception:
            return 0.0

    def _compute_iv_for_rows(self, df: pd.DataFrame, spot: float) -> pd.DataFrame:
        """
        Compute IV column for rows that have LTP but no iv.
        Returns a copy with 'iv' column filled where computable, otherwise None.
        """
        if df is None or df.empty:
            return df
        df = df.copy()
        # ensure required columns exist
        for col in ["ltp", "strike", "expiry", "option_type", "iv"]:
            if col not in df.columns:
                df[col] = None

        ivs = []
        for idx, row in df.iterrows():
            try:
                iv_val = row.get("iv", None)
                price = row.get("ltp", None)
                if iv_val is not None and not (pd.isna(iv_val) or iv_val == ""):
                    ivs.append(float(iv_val))
                    continue
                if price is None or pd.isna(price):
                    ivs.append(None)
                    continue
                K = float(row.get("strike"))
                option_type = str(row.get("option_type") or "").upper()
                expiry = row.get("expiry")
                T = self._time_to_expiry(expiry)
                if T <= 0:
                    ivs.append(None)
                    continue
                r = float(row.get("r")) if "r" in row and not pd.isna(row.get("r")) else self.default_r
                q = float(row.get("q")) if "q" in row and not pd.isna(row.get("q")) else self.default_q
                if spot is None or spot <= 0:
                    ivs.append(None)
                    continue
                iv = implied_volatility(option_type, float(price), float(spot), K, T, r=r, q=q)
                ivs.append(iv)
            except Exception:
                LOG.debug("Failed to compute IV for row idx=%s", idx, exc_info=True)
                ivs.append(None)
        df["iv"] = ivs
        return df

    def _build_option_rows(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Build a list of dictionaries representing option rows (PE/CE).
        Each dict includes strike, option_type, tradingsymbol, ltp, iv, delta, expiry, and full row.
        """
        if df is None or df.empty:
            return []
        rows = []
        for _, r in df.iterrows():
            # accommodate Series-like or dict-like rows
            rows.append({
                "strike": r.get("strike"),
                "option_type": r.get("option_type") or r.get("opt_type") or r.get("otype"),
                "tradingsymbol": r.get("tradingsymbol"),
                "ltp": r.get("ltp"),
                "iv": r.get("iv"),
                "delta": r.get("delta") if "delta" in r.index else None,
                "expiry": r.get("expiry"),
                "row": r,
            })
        return rows

    def _find_strikes_by_delta(self, option_rows: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int]]:
        """
        Given parsed option_rows, find strike nearest to desired put and call deltas.
        Requires per-row 'delta' values to be present. If not present, returns (None,None).
        """
        if not option_rows:
            return None, None
        strike_map = {}
        for r in option_rows:
            try:
                strike = int(r.get("strike"))
            except Exception:
                continue
            otype = (r.get("option_type") or "").upper()
            strike_map.setdefault(strike, {})[otype] = r

        any_delta = any((r.get("delta") is not None and r.get("delta") != "" and not pd.isna(r.get("delta")))
                        for r in option_rows)
        if not any_delta:
            return None, None

        best_put = (None, 1e9)
        best_call = (None, 1e9)
        for strike, sides in strike_map.items():
            pe = sides.get("PE")
            ce = sides.get("CE")
            if pe and pe.get("delta") is not None:
                try:
                    dist = abs(abs(float(pe["delta"])) - self.target_put_delta)
                    if dist < best_put[1]:
                        best_put = (strike, dist)
                except Exception:
                    pass
            if ce and ce.get("delta") is not None:
                try:
                    dist = abs(float(ce["delta"]) - self.target_call_delta)
                    if dist < best_call[1]:
                        best_call = (strike, dist)
                except Exception:
                    pass
        return best_put[0], best_call[0]

    def _find_leg_row(self, df: pd.DataFrame, strike_val: float, opt_type: str) -> Optional[pd.Series]:
        """
        Return the first matching row for given strike and option type (PE/CE).
        Works case-insensitively on column names.
        """
        if df is None or df.empty:
            return None
        tmp = df.copy()
        tmp.columns = [c.lower() for c in tmp.columns]
        # map alternative names if present
        if "option_type" not in tmp.columns and "option_typ" in tmp.columns:
            tmp["option_type"] = tmp["option_typ"]
        if "strike" not in tmp.columns and "strike_pr" in tmp.columns:
            tmp["strike"] = tmp["strike_pr"]
        try:
            mask = (
                (tmp["strike"].astype(float) == float(strike_val)) &
                (tmp["option_type"].str.upper() == opt_type.upper())
            )
        except Exception:
            return None
        rows = tmp[mask]
        if rows.empty:
            return None
        return rows.iloc[0]

    def generate(self, chain_df: pd.DataFrame, spot: Optional[float] = None) -> List[IronCondor]:
        """
        Main entry. Accepts raw chain_df (live snapshot) and optional spot.
        Returns a list (0..1) of IronCondor objects ready for entry_orders() by broker.
        """
        df = self._normalize_chain_df(chain_df)
        if df.empty:
            LOG.debug("SignalGenerator: empty chain_df provided")
            return []

        # infer spot if not provided
        if spot is None:
            for c in ("spot", "underlying", "underlying_price", "last_price", "ltp"):
                if c in df.columns:
                    try:
                        val = df[c].dropna().iloc[0]
                        spot = float(val)
                        break
                    except Exception:
                        continue

        if spot is None or spot <= 0:
            LOG.warning("SignalGenerator: spot not provided or invalid; aborting generation")
            return []

        # compute IVs for rows where missing (use live LTP from chain_df)
        df_iv = self._compute_iv_for_rows(df, spot)

        option_rows = self._build_option_rows(df_iv)
        if not option_rows:
            LOG.debug("SignalGenerator: no option rows after parsing")
            return []

        # try delta-based selection first (preferred when per-row delta available)
        short_put_strike, short_call_strike = self._find_strikes_by_delta(option_rows)

        # if delta-based selection works, attempt to construct IronCondor around those shorts
        if short_put_strike is not None and short_call_strike is not None:
            # compute longs relative to shorts using width
            long_put = int(short_put_strike - self.width)
            long_call = int(short_call_strike + self.width)

            # find rows for each leg
            sp_row = self._find_leg_row(df_iv, short_put_strike, "PE")
            lp_row = self._find_leg_row(df_iv, long_put, "PE")
            sc_row = self._find_leg_row(df_iv, short_call_strike, "CE")
            lc_row = self._find_leg_row(df_iv, long_call, "CE")

            # verify we have valid rows and mid prices
            try:
                sp_px = _mid_price(sp_row) if sp_row is not None else float(np.nan)
                lp_px = _mid_price(lp_row) if lp_row is not None else float(np.nan)
                sc_px = _mid_price(sc_row) if sc_row is not None else float(np.nan)
                lc_px = _mid_price(lc_row) if lc_row is not None else float(np.nan)
            except Exception:
                sp_px = lp_px = sc_px = lc_px = float(np.nan)

            if not any(np.isnan(x) for x in (sp_px, lp_px, sc_px, lc_px)):
                # Determine expiry from any leg row (prefer short put row)
                chosen_expiry = None
                for r in (sp_row, sc_row, lp_row, lc_row):
                    if r is not None and "expiry" in r.index and pd.notna(r["expiry"]):
                        chosen_expiry = str(r["expiry"])
                        break
                chosen_expiry = chosen_expiry or (df_iv["expiry"].dropna().iloc[0] if "expiry" in df_iv.columns and df_iv["expiry"].notna().any() else None)

                ic = IronCondor(
                    symbol=self.symbol or "NIFTY",
                    expiry=chosen_expiry or "UNKNOWN",
                    short_put=float(short_put_strike),
                    long_put=float(long_put),
                    short_call=float(short_call_strike),
                    long_call=float(long_call),
                    short_put_price=float(sp_px),
                    long_put_price=float(lp_px),
                    short_call_price=float(sc_px),
                    long_call_price=float(lc_px),
                    lot_size=int(df_iv.get("lot_size", 25) if isinstance(df_iv, pd.DataFrame) and "lot_size" in df_iv.columns else 25),
                    size_aggressiveness=float(self.size_aggressiveness),
                )

                LOG.info("SignalGenerator: built IronCondor via delta strikes SP=%s SC=%s", short_put_strike, short_call_strike)
                return [ic]
            else:
                LOG.info("SignalGenerator: delta-based legs lacked prices — falling back to ATM builder")

        # Delta selection failed or legs incomplete — fallback to symmetric ATM builder
        try:
            ic = build_iron_condor(
                chain_df=df_iv,
                symbol=self.symbol or "NIFTY",
                width=self.width,
                lot_size=int(df_iv.get("lot_size", 25) if isinstance(df_iv, pd.DataFrame) and "lot_size" in df_iv.columns else 25),
                size_aggressiveness=float(self.size_aggressiveness),
                expiry=None,
                spot=spot,
            )
        except Exception:
            LOG.exception("SignalGenerator: build_iron_condor() raised an exception")
            ic = None

        if ic is None:
            LOG.info("SignalGenerator: could not build IronCondor (delta and ATM both failed)")
            return []

        LOG.info("SignalGenerator: built IronCondor (ATM) SP=%s SC=%s", getattr(ic, "short_put", None), getattr(ic, "short_call", None))
        return [ic]

# module-level convenience
_default_sig = SignalGenerator()

def generate(chain_df: pd.DataFrame, spot: Optional[float] = None):
    return _default_sig.generate(chain_df=chain_df, spot=spot)

if __name__ == "__main__":
    LOG.setLevel(logging.DEBUG)
    demo = pd.DataFrame([
        {"strike": 26000, "option_type": "CE", "tradingsymbol": "NIFTY26000CE", "delta": 0.18, "ltp": 120, "expiry": "2025-12-02"},
        {"strike": 26000, "option_type": "PE", "tradingsymbol": "NIFTY26000PE", "delta": -0.17, "ltp": 110, "expiry": "2025-12-02"},
        {"strike": 26100, "option_type": "CE", "tradingsymbol": "NIFTY26100CE", "delta": 0.14, "ltp": 95, "expiry": "2025-12-02"},
        {"strike": 25900, "option_type": "PE", "tradingsymbol": "NIFTY25900PE", "delta": -0.12, "ltp": 80, "expiry": "2025-12-02"},
    ])
    cand = generate(demo, spot=26141)
    for c in cand:
        try:
            LOG.info("Demo candidate as_dict: %s", c.as_dict())
        except Exception:
            LOG.info("Demo candidate: %s", c)
