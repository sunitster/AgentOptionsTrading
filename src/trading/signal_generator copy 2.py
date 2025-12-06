# src/trading/signal_generator.py
"""
Patched SignalGenerator (drop-in replacement) with ML scoring integrated.

Key features:
 - min_entry_credit: prevent entering ICs with tiny or zero credit (INR)
 - no_new_after: prevent generating new trades after a particular time (e.g. 15:15)
 - Always prefer real tradingsymbols from chain_df; reject candidate if essential leg symbols are missing
 - Built-in ML scoring hook (optional): loads models/llm_trades/model.pkl via deploy_scoring_hook.MLScoringEngine
 - ML score threshold fixed at 0.50 (balanced)
 - Good diagnostics and logging for why a candidate was rejected
 - Backwards compatible API: SignalGenerator.generate(chain_df, spot) -> List[IronCondor]
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import datetime as dt
import pandas as pd
import os

LOG = logging.getLogger("SignalGenerator")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------
# ML scoring: attempt import (non-fatal)
# ---------------------------------------------------------------------
ML_MODEL_PATH = os.path.join("models", "llm_trades", "model.pkl")
try:
    from src.scripts.deploy_scoring_hook import MLScoringEngine  # type: ignore
    _HAS_ML_SCORER_CLASS = True
except Exception:
    MLScoringEngine = None
    _HAS_ML_SCORER_CLASS = False

# ---------------------------------------------------------------------
# Black-Scholes helpers (kept small and robust)
# ---------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_price(option_type: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= 0:
        return max(0.0, (K - S)) if str(option_type).upper().startswith("P") else max(0.0, (S - K))
    if sigma <= 0:
        if str(option_type).upper().startswith("P"):
            return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if str(option_type).upper().startswith("P"):
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

def bs_vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * math.sqrt(T) * _norm_pdf(d1)

# ---------------------------------------------------------------------
# Try project IronCondor; fallback to local shim
# ---------------------------------------------------------------------
try:
    from trading.iron_condor_builder import IronCondor as ProjectIronCondor, build_iron_condor, _mid_price  # type: ignore
    _HAS_PROJECT_BUILDER = True
    LOG.info("SignalGenerator: using project IronCondor builder.")
except Exception:
    _HAS_PROJECT_BUILDER = False
    LOG.warning("SignalGenerator: IronCondor import failed — using local shim.")

    @dataclass
    class ProjectIronCondor:
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
        short_put_sym: Optional[str] = None
        long_put_sym: Optional[str] = None
        short_call_sym: Optional[str] = None
        long_call_sym: Optional[str] = None

        def __post_init__(self):
            try:
                sp = 0.0 if self.short_put_price is None or (isinstance(self.short_put_price, float) and math.isnan(self.short_put_price)) else float(self.short_put_price)
                sc = 0.0 if self.short_call_price is None or (isinstance(self.short_call_price, float) and math.isnan(self.short_call_price)) else float(self.short_call_price)
                lp = 0.0 if self.long_put_price is None or (isinstance(self.long_put_price, float) and math.isnan(self.long_put_price)) else float(self.long_put_price)
                lc = 0.0 if self.long_call_price is None or (isinstance(self.long_call_price, float) and math.isnan(self.long_call_price)) else float(self.long_call_price)
                self.entry_credit = (sp + sc) - (lp + lc)
            except Exception:
                self.entry_credit = 0.0

        def entry_orders(self, chain_df: Optional[pd.DataFrame] = None):
            qty = max(1, int(round(self.size_aggressiveness))) * self.lot_size
            def resolve(preferred, strike, opt):
                if preferred:
                    return preferred
                if chain_df is not None and not chain_df.empty:
                    try:
                        cond = (chain_df["strike"].astype(float) == float(strike)) & (chain_df["option_type"].str.upper() == opt.upper())
                        tmp = chain_df[cond]
                        if not tmp.empty and "tradingsymbol" in tmp.columns:
                            return str(tmp.iloc[0]["tradingsymbol"])
                    except Exception:
                        pass
                return f"{self.symbol}_{opt}_{int(round(float(strike)))}"
            lp = self.long_put if self.long_put is not None else (self.short_put - 100)
            lc = self.long_call if self.long_call is not None else (self.short_call + 100)
            return [
                {"symbol": resolve(self.short_put_sym, self.short_put, "PE"), "qty": -qty, "side": "SELL", "price": float(self.short_put_price) if not (isinstance(self.short_put_price, float) and math.isnan(self.short_put_price)) else None},
                {"symbol": resolve(self.long_put_sym, lp, "PE"), "qty": qty, "side": "BUY", "price": float(self.long_put_price) if not (isinstance(self.long_put_price, float) and math.isnan(self.long_put_price)) else None},
                {"symbol": resolve(self.short_call_sym, self.short_call, "CE"), "qty": -qty, "side": "SELL", "price": float(self.short_call_price) if not (isinstance(self.short_call_price, float) and math.isnan(self.short_call_price)) else None},
                {"symbol": resolve(self.long_call_sym, lc, "CE"), "qty": qty, "side": "BUY", "price": float(self.long_call_price) if not (isinstance(self.long_call_price, float) and math.isnan(self.long_call_price)) else None},
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
                "entry_credit": float(getattr(self, "entry_credit", 0.0)),
                "short_put_sym": self.short_put_sym,
                "long_put_sym": self.long_put_sym,
                "short_call_sym": self.short_call_sym,
                "long_call_sym": self.long_call_sym,
            }

        def _get_mid_from_chain(self, chain_df: Optional[pd.DataFrame], strike_val: float, opt_type: str) -> float:
            try:
                if chain_df is None or chain_df.empty:
                    return float("nan")
            except Exception:
                pass
            try:
                # prefer symbol match
                if opt_type.upper() == "PE":
                    pref = self.short_put_sym if float(strike_val) == float(self.short_put) else (self.long_put_sym if self.long_put is not None and float(strike_val) == float(self.long_put) else None)
                else:
                    pref = self.short_call_sym if float(strike_val) == float(self.short_call) else (self.long_call_sym if self.long_call is not None and float(strike_val) == float(self.long_call) else None)
                if pref:
                    try:
                        match = chain_df[chain_df["tradingsymbol"].astype(str) == str(pref)]
                        if not match.empty:
                            r = match.iloc[0]
                            if "ltp" in r and pd.notna(r["ltp"]):
                                return float(r["ltp"])
                            b = r.get("best_bid")
                            a = r.get("best_ask")
                            if b is not None and a is not None:
                                return float((float(b) + float(a)) / 2.0)
                    except Exception:
                        pass
                cond = (chain_df["strike"].astype(float) == float(strike_val)) & (chain_df["option_type"].str.upper() == opt_type.upper())
                tmp = chain_df[cond]
                if tmp.empty:
                    return float("nan")
                row = tmp.iloc[0]
                if "ltp" in row and pd.notna(row["ltp"]):
                    return float(row["ltp"])
                b = row.get("best_bid")
                a = row.get("best_ask")
                if b is not None and a is not None:
                    return float((float(b) + float(a)) / 2.0)
            except Exception:
                pass
            return float("nan")

        def mark_to_market(self, chain_df: Optional[pd.DataFrame]) -> float:
            try:
                sp_cur = self._get_mid_from_chain(chain_df, self.short_put, "PE")
                sc_cur = self._get_mid_from_chain(chain_df, self.short_call, "CE")
                lp_cur = self._get_mid_from_chain(chain_df, self.long_put, "PE") if self.long_put is not None else float("nan")
                lc_cur = self._get_mid_from_chain(chain_df, self.long_call, "CE") if self.long_call is not None else float("nan")
                sp_e = 0.0 if (self.short_put_price is None or (isinstance(self.short_put_price, float) and math.isnan(self.short_put_price))) else float(self.short_put_price)
                sc_e = 0.0 if (self.short_call_price is None or (isinstance(self.short_call_price, float) and math.isnan(self.short_call_price))) else float(self.short_call_price)
                lp_e = 0.0 if (self.long_put_price is None or (isinstance(self.long_put_price, float) and math.isnan(self.long_put_price))) else float(self.long_put_price)
                lc_e = 0.0 if (self.long_call_price is None or (isinstance(self.long_call_price, float) and math.isnan(self.long_call_price))) else float(self.long_call_price)
                pnl_sp = (sp_e - sp_cur) if not (isinstance(sp_cur, float) and math.isnan(sp_cur)) else 0.0
                pnl_sc = (sc_e - sc_cur) if not (isinstance(sc_cur, float) and math.isnan(sc_cur)) else 0.0
                pnl_lp = (lp_cur - lp_e) if not (isinstance(lp_cur, float) and math.isnan(lp_cur)) else 0.0
                pnl_lc = (lc_cur - lc_e) if not (isinstance(lc_cur, float) and math.isnan(lc_cur)) else 0.0
                net_per_contract = pnl_sp + pnl_lp + pnl_sc + pnl_lc
                net_total = net_per_contract * int(self.lot_size)
                return float(net_total)
            except Exception:
                return 0.0

        def exit_pnl(self, chain_df: Optional[pd.DataFrame]) -> float:
            try:
                return float(self.mark_to_market(chain_df))
            except Exception:
                return 0.0

        def max_loss(self) -> float:
            try:
                call_width = None
                put_width = None
                if self.long_call is not None and self.short_call is not None:
                    call_width = float(self.long_call) - float(self.short_call)
                if self.long_put is not None and self.short_put is not None:
                    put_width = float(self.short_put) - float(self.long_put)
                widths = [w for w in (call_width, put_width) if w is not None and w > 0]
                if not widths:
                    widths = [150.0]
                return float(max(widths) * int(self.lot_size))
            except Exception:
                return float(150 * int(self.lot_size))

# local fallback mid price helper (if project builder doesn't provide)
def _mid_price(row):
    try:
        if row is None:
            return float("nan")
        if isinstance(row, dict):
            l = row.get("ltp")
            if l is not None and not (isinstance(l, float) and math.isnan(l)):
                return float(l)
            b = row.get("best_bid")
            a = row.get("best_ask")
            if b is not None and a is not None:
                return (float(b) + float(a)) / 2.0
            return float("nan")
        if "ltp" in row and not pd.isna(row["ltp"]):
            return float(row["ltp"])
        if "best_bid" in row and "best_ask" in row and not pd.isna(row["best_bid"]) and not pd.isna(row["best_ask"]):
            return (float(row["best_bid"]) + float(row["best_ask"])) / 2.0
    except Exception:
        pass
    return float("nan")

# ---------------------------------------------------------------------
# SignalGenerator
# ---------------------------------------------------------------------
class SignalGenerator:
    def __init__(self,
                 width: int = 150,
                 target_put_delta: float = 0.16,
                 target_call_delta: float = 0.16,
                 default_r: float = 0.06,
                 default_q: float = 0.0,
                 min_entry_credit: float = 3.0,        # NEW: minimum acceptable entry credit (INR)
                 no_new_after: dt.time = dt.time(15, 15),  # NEW: do not open new trades after this time
                 ml_score_threshold: float = 0.50,     # ML threshold default (fixed at 0.50)
                 **kwargs):
        self.width = int(width)
        self.target_put_delta = float(target_put_delta)
        self.target_call_delta = float(target_call_delta)
        self.default_r = float(default_r)
        self.default_q = float(default_q)
        self.min_entry_credit = float(min_entry_credit)
        self.no_new_after = no_new_after
        self.ml_score_threshold = float(ml_score_threshold)

        self.symbol = kwargs.get("symbol", None)
        self.size_aggressiveness = float(kwargs.get("size_aggressiveness", 1.0))
        LOG.info("SignalGenerator v2 init: width=%s put_delta=%s call_delta=%s symbol=%s extras=%s min_credit=%s no_new_after=%s ml_th=%.2f",
                 self.width, self.target_put_delta, self.target_call_delta, self.symbol, bool(kwargs),
                 self.min_entry_credit, self.no_new_after, self.ml_score_threshold)

        self._last_trade_date: Optional[dt.date] = None
        self._freeze: bool = False

        # Try to load ML scorer if class available and model exists
        self.ml_scorer = None
        if _HAS_ML_SCORER_CLASS and os.path.exists(ML_MODEL_PATH):
            try:
                self.ml_scorer = MLScoringEngine(ML_MODEL_PATH)
                LOG.info("SignalGenerator: MLScoringEngine loaded from %s", ML_MODEL_PATH)
            except Exception:
                LOG.exception("SignalGenerator: failed to instantiate MLScoringEngine (scoring disabled)")

    # -----------------------
    # utilities
    # -----------------------
    def _safe_float(self, v) -> Optional[float]:
        try:
            if v is None:
                return None
            if isinstance(v, (float, int)) and not (isinstance(v, float) and math.isnan(v)):
                return float(v)
            s = str(v).strip()
            if s == "":
                return None
            return float(s)
        except Exception:
            return None

    def reset_daily(self):
        today = dt.date.today()
        if self._last_trade_date != today:
            self._last_trade_date = today
            self._freeze = False

    def freeze(self):
        self._freeze = True

    def unfreeze(self):
        self._freeze = False

    def is_frozen(self) -> bool:
        return bool(self._freeze)

    # -----------------------
    # normalization & IV
    # -----------------------
    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()
        d = df.copy()
        # ensure strike numeric
        if "strike" in d.columns:
            d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
        if "option_type" in d.columns:
            d["option_type"] = d["option_type"].astype(object).where(d["option_type"].notna(), None)
        if "tradingsymbol" not in d.columns and "symbol" in d.columns:
            d = d.rename(columns={"symbol": "tradingsymbol"})
        return d

    def _compute_iv(self, df: pd.DataFrame, spot: float) -> pd.DataFrame:
        # best-effort IV/delta; do not crash if libs absent
        d = df.copy()
        d["iv"] = None
        d["delta"] = None
        try:
            import src.iv_utils as ivu  # type: ignore
            iv_fn = None
            delta_fn = None
            for name in ("get_iv", "implied_volatility", "iv_from_price", "price_to_iv", "calc_iv"):
                if hasattr(ivu, name):
                    iv_fn = getattr(ivu, name)
                    break
            for name in ("get_delta", "delta_from_price", "black_scholes_delta"):
                if hasattr(ivu, name):
                    delta_fn = getattr(ivu, name)
                    break
            if iv_fn is None and delta_fn is None:
                return d
            for idx, row in d.iterrows():
                try:
                    ltp = row.get("ltp") or row.get("last_price") or row.get("lastPrice")
                    strike = row.get("strike")
                    expiry = row.get("expiry")
                    opt = row.get("option_type") or row.get("opt") or row.get("otype")
                    if ltp is None or spot is None or strike is None or expiry is None:
                        continue
                    iv_val = None
                    try:
                        iv_val = iv_fn(ltp, spot, float(strike), expiry)
                    except Exception:
                        try:
                            iv_val = iv_fn(price=ltp, spot=spot, strike=float(strike), expiry=expiry)
                        except Exception:
                            iv_val = None
                    if iv_val is not None:
                        d.at[idx, "iv"] = float(iv_val)
                    if delta_fn is not None:
                        out = None
                        try:
                            out = delta_fn(ltp, spot, float(strike), expiry, opt)
                        except Exception:
                            try:
                                out = delta_fn(price=ltp, spot=spot, strike=float(strike), expiry=expiry, option_type=opt)
                            except Exception:
                                out = None
                        if isinstance(out, dict):
                            d.at[idx, "delta"] = float(out.get("delta")) if out.get("delta") is not None else None
                        else:
                            try:
                                d.at[idx, "delta"] = float(out)
                            except Exception:
                                pass
                except Exception:
                    continue
            return d
        except Exception:
            # fallback: do nothing
            return d

    # -----------------------
    # Internal ATM builder (robust)
    # -----------------------
    def _build_ic_internal_atm(self, df: pd.DataFrame, spot: float, lot_size: int = 25) -> Tuple[Optional[ProjectIronCondor], Optional[str]]:
        """
        Returns (ic, reason_if_rejected). If ic is None, reason explains why.
        """
        try:
            if df is None or df.empty or spot is None:
                return None, "empty_chain_or_no_spot"
            strikes = []
            for s in df["strike"].tolist():
                s_val = self._safe_float(s)
                if s_val is None:
                    continue
                strikes.append(int(round(s_val)))
            strikes = sorted(list(set(strikes)))
            if not strikes:
                return None, "no_strikes"

            atm = min(strikes, key=lambda x: abs(x - spot))
            target_short_put = atm - self.width
            target_short_call = atm + self.width

            def nearest(target):
                return min(strikes, key=lambda s: abs(s - target)) if strikes else None

            short_put = nearest(target_short_put)
            short_call = nearest(target_short_call)

            if short_put is None or short_call is None or short_put >= short_call:
                below = [s for s in strikes if s < atm]
                above = [s for s in strikes if s > atm]
                if not below or not above:
                    return None, "cannot_find_sides"
                short_put = below[-1]
                short_call = above[0]

            below_all = [s for s in strikes if s < short_put]
            above_all = [s for s in strikes if s > short_call]
            long_put = below_all[-1] if below_all else None
            long_call = above_all[0] if above_all else None

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

            # require both short rows present and with a usable mid/ltp
            sp_px = _mid_price(sp_row) if sp_row is not None else float("nan")
            sc_px = _mid_price(sc_row) if sc_row is not None else float("nan")
            if math.isnan(sp_px) or math.isnan(sc_px):
                return None, "no_mid_on_shorts"

            # get real tradingsymbols and require they exist
            sp_sym = sp_row.get("tradingsymbol") if sp_row is not None else None
            sc_sym = sc_row.get("tradingsymbol") if sc_row is not None else None
            if not sp_sym or not sc_sym:
                # try fallback: maybe chain has symbol under different key
                return None, "missing_tradingsymbol_on_shorts"

            lp_sym = lp_row.get("tradingsymbol") if lp_row is not None else None
            lc_sym = lc_row.get("tradingsymbol") if lc_row is not None else None
            lp_px = _mid_price(lp_row) if lp_row is not None else float("nan")
            lc_px = _mid_price(lc_row) if lc_row is not None else float("nan")

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
                short_put_sym=str(sp_sym),
                long_put_sym=(str(lp_sym) if lp_sym is not None else None),
                short_call_sym=str(sc_sym),
                long_call_sym=(str(lc_sym) if lc_sym is not None else None),
            )

            # Enforce minimum entry credit
            try:
                ec = float(getattr(ic, "entry_credit", 0.0))
            except Exception:
                ec = 0.0
            if ec < self.min_entry_credit:
                return None, f"entry_credit_too_small:{ec:.2f}"

            return ic, None
        except Exception as e:
            LOG.exception("Internal ATM builder failed: %s", e)
            return None, "internal_error"

    # -----------------------
    # Diagnostic helpers
    # -----------------------
    def generate_diagnostic(self, df: pd.DataFrame, spot: Optional[float] = None) -> Dict[str, Any]:
        diag = {
            "rows": 0,
            "columns": list(df.columns) if df is not None else [],
            "has_strike": False,
            "sample_strikes": [],
            "num_with_ltp": 0,
            "num_with_iv": 0,
            "num_with_delta": 0,
            "expiry_values": [],
            "spot": spot,
        }
        if df is None or df.empty:
            return diag
        diag["rows"] = len(df)
        diag["columns"] = list(df.columns)
        if "strike" in df.columns:
            diag["has_strike"] = True
            diag["sample_strikes"] = sorted(pd.to_numeric(df["strike"], errors="coerce").dropna().unique().tolist())[:50]
        if "ltp" in df.columns:
            diag["num_with_ltp"] = int(df["ltp"].dropna().shape[0])
        if "iv" in df.columns:
            diag["num_with_iv"] = int(df["iv"].dropna().shape[0])
        if "delta" in df.columns:
            diag["num_with_delta"] = int(df["delta"].dropna().shape[0])
        if "expiry" in df.columns:
            try:
                diag["expiry_values"] = sorted(list(set([str(x) for x in df["expiry"].dropna().unique().tolist()])))
            except Exception:
                diag["expiry_values"] = []
        return diag

    # -----------------------
    # Main generate()
    # -----------------------
    def generate(self, chain_df: pd.DataFrame, spot: Optional[float] = None) -> List[ProjectIronCondor]:
        try:
            self.reset_daily()
        except Exception:
            pass
        if self.is_frozen():
            LOG.debug("SignalGenerator: frozen (existing IC active). Skipping candidate generation.")
            return []

        now = dt.datetime.utcnow()
        # If local timezone needed, LivePaperEngine can pass localized now; here we use UTC for safety.
        if self.no_new_after is not None:
            # compare only time-of-day (assume no_new_after in local/UTC consistent with engine)
            if now.time() >= self.no_new_after:
                LOG.info("SignalGenerator: current time >= no_new_after (%s). Not generating new trades.", self.no_new_after)
                return []

        df = self._normalize_df(chain_df)
        if df.empty:
            LOG.debug("SignalGenerator: empty chain provided")
            return []

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

        df_iv = self._compute_iv(df, spot)

        # Try delta-based (if available)
        strike_map: Dict[int, Dict[str, Any]] = {}
        for _, row in df_iv.iterrows():
            s = self._safe_float(row.get("strike"))
            if s is None:
                continue
            strike = int(round(s))
            if strike not in strike_map:
                strike_map[strike] = {}
            typ = (row.get("option_type") or "").upper()
            if typ.startswith("C"):
                strike_map[strike]["CE"] = row
            elif typ.startswith("P"):
                strike_map[strike]["PE"] = row

        any_delta = False
        if "delta" in df_iv.columns:
            try:
                any_delta = df_iv["delta"].dropna().shape[0] > 0
            except Exception:
                any_delta = False

        # helper: run ML scoring (if available). Returns True means "accept", False "reject", None "no decision / scorer missing"
        def _ml_accept_candidate(ic_obj: ProjectIronCondor, spot_val: Optional[float], legs_info: Optional[List[Dict]] = None) -> Optional[bool]:
            try:
                if self.ml_scorer is None:
                    return None
                # prepare candidate dict
                try:
                    cand_dict = ic_obj.as_dict() if hasattr(ic_obj, "as_dict") else (ic_obj if isinstance(ic_obj, dict) else {})
                except Exception:
                    cand_dict = ic_obj if isinstance(ic_obj, dict) else {}
                try:
                    score_info = self.ml_scorer.score_ic(cand_dict, spot=spot_val, legs=legs_info)
                except Exception:
                    LOG.exception("SignalGenerator: ML scoring failed for candidate")
                    return None
                if not isinstance(score_info, dict):
                    return None
                sc = float(score_info.get("score", 0.0))
                pwin = float(score_info.get("p_win", 0.0))
                exp_pnl = float(score_info.get("exp_pnl", 0.0))
                LOG.info("SignalGenerator: ML score for candidate: score=%.4f p_win=%.3f exp_pnl=%.2f", sc, pwin, exp_pnl)
                if sc < float(self.ml_score_threshold):
                    LOG.info("SignalGenerator: candidate rejected by ML (score %.4f < threshold %.4f)", sc, float(self.ml_score_threshold))
                    return False
                return True
            except Exception:
                return None

        # 1) Delta-based candidate
        if any_delta:
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
                long_put = sp - self.width
                long_call = sc + self.width
                # fetch rows
                def _row(s_val, typ):
                    try:
                        cond = (df_iv["strike"].apply(lambda x: self._safe_float(x)) == float(s_val)) & (df_iv["option_type"].str.upper() == typ.upper())
                        tmp = df_iv[cond]
                        return tmp.iloc[0] if not tmp.empty else None
                    except Exception:
                        return None
                sp_row = _row(sp, "PE"); sc_row = _row(sc, "CE")
                lp_row = _row(long_put, "PE"); lc_row = _row(long_call, "CE")
                sp_px = _mid_price(sp_row) if sp_row is not None else float("nan")
                sc_px = _mid_price(sc_row) if sc_row is not None else float("nan")
                lp_px = _mid_price(lp_row) if lp_row is not None else float("nan")
                lc_px = _mid_price(lc_row) if lc_row is not None else float("nan")
                if math.isnan(sp_px) or math.isnan(sc_px):
                    LOG.debug("SignalGenerator: delta candidate missing mid price on shorts; rejecting")
                else:
                    sp_sym = sp_row.get("tradingsymbol") if sp_row is not None else None
                    sc_sym = sc_row.get("tradingsymbol") if sc_row is not None else None
                    if not sp_sym or not sc_sym:
                        LOG.debug("SignalGenerator: delta candidate missing tradingsymbol on shorts; rejecting")
                    else:
                        ic = ProjectIronCondor(
                            symbol=self.symbol or "NIFTY",
                            expiry=(str(sp_row["expiry"]) if sp_row is not None and "expiry" in sp_row.index else None),
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
                            short_put_sym=str(sp_sym),
                            long_put_sym=(str(lp_row["tradingsymbol"]) if lp_row is not None and "tradingsymbol" in lp_row.index else None),
                            short_call_sym=str(sc_sym),
                            long_call_sym=(str(lc_row["tradingsymbol"]) if lc_row is not None and "tradingsymbol" in lc_row.index else None),
                        )
                        # check min_credit
                        try:
                            if float(getattr(ic, "entry_credit", 0.0)) < self.min_entry_credit:
                                LOG.info("SignalGenerator: rejected delta IC due to low entry_credit=%.2f < min=%.2f", float(getattr(ic, "entry_credit", 0.0)), self.min_entry_credit)
                            else:
                                # ML scoring check (if model loaded)
                                ml_decision = _ml_accept_candidate(ic, spot, legs_info=None)
                                if ml_decision is False:
                                    LOG.info("SignalGenerator: delta IC rejected by ML scoring.")
                                    return []
                                # if ml_decision is None or True -> accept
                                LOG.info("SignalGenerator: built IronCondor via delta SP=%s SC=%s", sp, sc)
                                return [ic]
                        except Exception:
                            LOG.exception("SignalGenerator: error checking delta IC entry credit; rejecting")

        # 2) Project builder if available
        try:
            if _HAS_PROJECT_BUILDER:
                ic = build_iron_condor(chain_df=df_iv, symbol=self.symbol or "NIFTY", width=self.width, lot_size=int(df_iv.get("lot_size", 25) if isinstance(df_iv, pd.DataFrame) and "lot_size" in df_iv.columns else 25), size_aggressiveness=self.size_aggressiveness, spot=spot)
                if ic is not None:
                    # verify real tradingsymbols exist for short legs
                    try:
                        spsym = getattr(ic, "short_put_sym", None)
                        scsym = getattr(ic, "short_call_sym", None)
                        if not spsym or not scsym:
                            LOG.debug("SignalGenerator: project builder returned candidate missing short leg symbols; rejecting")
                        else:
                            if float(getattr(ic, "entry_credit", 0.0)) < self.min_entry_credit:
                                LOG.info("SignalGenerator: rejected project IC due to low entry_credit")
                            else:
                                # ML scoring check
                                ml_decision = _ml_accept_candidate(ic, spot, legs_info=None)
                                if ml_decision is False:
                                    LOG.info("SignalGenerator: project IC rejected by ML scoring.")
                                    return []
                                LOG.info("SignalGenerator: built IronCondor using project builder")
                                return [ic]
                    except Exception:
                        LOG.exception("SignalGenerator: project IC validation failed; rejecting")
        except Exception:
            LOG.exception("SignalGenerator: project build_iron_condor raised exception; falling back to internal ATM builder")

        # 3) Internal ATM builder
        ic_internal, reason = self._build_ic_internal_atm(df_iv, spot, lot_size=int(df_iv.get("lot_size", 25) if isinstance(df_iv, pd.DataFrame) and "lot_size" in df_iv.columns else 25))
        if ic_internal is not None:
            # ML scoring check
            ml_decision = _ml_accept_candidate(ic_internal, spot, legs_info=None)
            if ml_decision is False:
                LOG.info("SignalGenerator: internal ATM IC rejected by ML scoring.")
                return []
            LOG.info("SignalGenerator: built IronCondor via internal ATM builder SP=%s SC=%s", getattr(ic_internal, "short_put", None), getattr(ic_internal, "short_call", None))
            return [ic_internal]
        else:
            LOG.debug("SignalGenerator: internal ATM builder rejected candidate (%s); diagnostic=%s", reason, self.generate_diagnostic(df_iv, spot))

        # nothing found
        return []

# convenience default instance
_default_sig = SignalGenerator()
def generate(chain_df: pd.DataFrame, spot: Optional[float] = None):
    return _default_sig.generate(chain_df=chain_df, spot=spot)
def generate_diagnostic(chain_df: pd.DataFrame, spot: Optional[float] = None):
    return _default_sig.generate_diagnostic(chain_df, spot)
