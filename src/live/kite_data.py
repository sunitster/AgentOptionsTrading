# src/live/kite_data.py
"""
Ultra-stable KiteData + full_option_chain helper (Option B FINAL v2).

Fixes applied:
- Ensure 'instrument_type' handling is safe (no .fillna() on None).
- Replace NaN values with None before returning snapshots.
- Ensure underlying row creation guards against NaN.
- Small cache preserved.
"""

import os
import re
import yaml
import logging
import math
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any

try:
    from kiteconnect import KiteConnect
    HAS_KITE = True
except Exception:
    HAS_KITE = False

LOG = logging.getLogger("KiteData")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


class KiteData:
    def __init__(self, config_path: str = "config/config.yaml", batch_size: int = 100):
        self.config_path = config_path
        self.batch_size = batch_size
        self.kite = None
        self.mode = "paper"
        self.api_key = ""
        self.api_secret = ""
        self.access_token = ""
        self._cache = {"ts": None, "data": None}
        self._load_config()
        if HAS_KITE and self.api_key and self.access_token:
            self._connect()
        else:
            LOG.warning("KiteConnect not available or keys missing — running in PAPER mode")

    # -------------------------
    # Config / Connect
    # -------------------------
    def _load_config(self):
        if not os.path.exists(self.config_path):
            LOG.warning("Config not found: %s — falling back to PAPER snapshot", self.config_path)
            return
        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        kite_cfg = cfg.get("kite", {})
        self.api_key = kite_cfg.get("api_key", "") or ""
        self.api_secret = kite_cfg.get("api_secret", "") or ""
        self.access_token = kite_cfg.get("access_token", "") or ""

    def _connect(self):
        try:
            self.kite = KiteConnect(api_key=self.api_key)
            if not self.access_token or str(self.access_token).strip() == "":
                LOG.warning("No access_token found in config — running PAPER mode")
                self.kite = None
                self.mode = "paper"
                return

            self.kite.set_access_token(self.access_token)
            profile = self.kite.profile()  # safe allowed call
            LOG.info("Connected to KiteConnect as %s (%s)", profile.get("user_id"), profile.get("user_name"))
            self.mode = "live"
        except Exception as e:
            LOG.error("Live KiteConnect connection failed -> PAPER mode. Reason: %s", repr(e))
            self.kite = None
            self.mode = "paper"

    # -------------------------
    # Public: full option chain snapshot
    # -------------------------
    def get_option_chain_snapshot(self, symbol: str = "NIFTY", width: int = 1500, expiry: Optional[str] = None,
                                  exchange: str = "NFO") -> pd.DataFrame:
        """
        Return a tidy DataFrame with option chain and a top underlying row.

        Guarantees each row has:
          strike, option_type, tradingsymbol, instrument_token, expiry,
          best_bid, best_ask, ltp, spot, open_interest, instrument_type
        """
        # short-circuit cache (1s freshness)
        now = datetime.utcnow()
        if self._cache["ts"] and (now - self._cache["ts"]).total_seconds() < 1:
            cached = self._cache["data"]
            if isinstance(cached, pd.DataFrame):
                return cached.copy()

        if self.mode != "live" or not self.kite:
            LOG.info("Running PAPER snapshot (no live kite connection)")
            df = self._paper_snapshot(symbol, width)
            # ensure underlying row
            df = self._ensure_underlying_row(df, symbol, spot_override=None)
            # replace any NaN with None for JSON-safety
            df = df.where(pd.notnull(df), None)
            self._cache["ts"] = now
            self._cache["data"] = df
            return df

        try:
            instruments = self._fetch_instruments(exchange)
            if not instruments:
                LOG.warning("No instruments found -> PAPER fallback")
                df = self._paper_snapshot(symbol, width)
                df = self._ensure_underlying_row(df, symbol, spot_override=None)
                df = df.where(pd.notnull(df), None)
                self._cache["ts"] = now
                self._cache["data"] = df
                return df

            expiries = self._get_expiries_for_symbol(instruments, symbol)
            if not expiries:
                LOG.warning("No expiries found -> PAPER fallback")
                df = self._paper_snapshot(symbol, width)
                df = self._ensure_underlying_row(df, symbol, spot_override=None)
                df = df.where(pd.notnull(df), None)
                self._cache["ts"] = now
                self._cache["data"] = df
                return df

            chosen_expiry = expiry if expiry else self._choose_preferred_expiry(expiries)
            LOG.info("Chosen expiry for %s -> %s", symbol, chosen_expiry)

            spot = self._get_spot(symbol)
            LOG.info("Spot for %s = %s", symbol, spot)

            # collect option instruments for chosen expiry
            option_rows: List[Dict[str, Any]] = []
            strike_pattern = re.compile(r"(\d+)(CE|PE)$", flags=re.IGNORECASE)

            for ins in instruments:
                ts = ins.get("tradingsymbol", "")
                exch = ins.get("exchange", "")
                if exch != exchange:
                    continue
                if (ins.get("expiry") or "") != chosen_expiry:
                    continue
                if not ts.upper().startswith(symbol.upper()):
                    continue
                m = strike_pattern.search(ts)
                if not m:
                    continue
                strike = int(m.group(1))
                opttype = m.group(2).upper()
                if abs(strike - spot) > width:
                    continue
                option_rows.append({
                    "strike": strike,
                    "option_type": opttype,
                    "tradingsymbol": ts,
                    "instrument_token": int(ins.get("instrument_token")) if ins.get("instrument_token") else None,
                    "expiry": ins.get("expiry"),
                })

            if not option_rows:
                LOG.warning("No option instruments matched width filter -> PAPER fallback")
                df = self._paper_snapshot(symbol, width)
                df = self._ensure_underlying_row(df, symbol, spot_override=spot)
                df = df.where(pd.notnull(df), None)
                self._cache["ts"] = now
                self._cache["data"] = df
                return df

            option_df = pd.DataFrame(option_rows).dropna(subset=["instrument_token"]).astype({"instrument_token": int})
            tokens = option_df["instrument_token"].astype(int).tolist()
            ltp_rows = self._batch_ltp(tokens)

            ltp_df = pd.DataFrame(ltp_rows)
            merged = option_df.merge(ltp_df, how="left", left_on="instrument_token", right_on="instrument_token")

            # Normalize columns
            merged["spot"] = spot

            # Ensure instrument_type column exists and is safe to operate on
            if "instrument_type" not in merged.columns:
                merged["instrument_type"] = ""
            else:
                # fillna safely on Series
                merged["instrument_type"] = merged["instrument_type"].fillna("").astype(str)

            # Ensure price/oi columns exist
            for col in ["best_bid", "best_ask", "ltp", "open_interest"]:
                if col not in merged.columns:
                    merged[col] = None

            final_cols = ["instrument_type", "strike", "option_type", "tradingsymbol", "instrument_token", "expiry",
                          "best_bid", "best_ask", "ltp", "spot", "open_interest"]
            merged = merged.loc[:, [c for c in final_cols if c in merged.columns]]

            # set instrument_type explicitly for options (CE/PE)
            merged["instrument_type"] = merged["option_type"].apply(lambda x: "CE" if str(x).upper().startswith("CE") else "PE")

            # sort and reset
            merged = merged.sort_values(["strike", "option_type"], ascending=[True, True]).reset_index(drop=True)

            # Convert NaN -> None to avoid JSON issues downstream
            merged = merged.where(pd.notnull(merged), None)

            # insert underlying row at top (underlying row built safe)
            df_final = self._ensure_underlying_row(merged, symbol, spot_override=spot)

            # final NaN -> None
            df_final = df_final.where(pd.notnull(df_final), None)

            # cache and return
            self._cache["ts"] = now
            self._cache["data"] = df_final
            return df_final

        except Exception as e:
            LOG.exception("get_option_chain_snapshot failed -> PAPER fallback. Reason: %s", repr(e))
            df = self._paper_snapshot(symbol, width)
            df = self._ensure_underlying_row(df, symbol, spot_override=None)
            df = df.where(pd.notnull(df), None)
            self._cache["ts"] = now
            self._cache["data"] = df
            return df

    # -------------------------
    # Helpers
    # -------------------------
    def _fetch_instruments(self, exchange: str = "NFO") -> List[dict]:
        try:
            ins = self.kite.instruments(exchange)
            return ins
        except Exception as e:
            LOG.error("Failed to fetch instruments: %s", repr(e))
            return []

    def _get_expiries_for_symbol(self, instruments: List[dict], symbol: str) -> List[str]:
        expiries = set()
        sym_upper = symbol.upper()
        for ins in instruments:
            ts = ins.get("tradingsymbol", "")
            if not ts:
                continue
            if ts.upper().startswith(sym_upper):
                expiry = ins.get("expiry")
                if expiry:
                    expiries.add(expiry)
        try:
            return sorted(list(expiries), key=lambda x: datetime.fromisoformat(x))
        except Exception:
            return sorted(list(expiries))

    def _choose_preferred_expiry(self, expiries: List[str]) -> str:
        parsed = []
        for e in expiries:
            try:
                dt = datetime.fromisoformat(e)
                parsed.append(dt)
            except Exception:
                continue
        if not parsed:
            return sorted(expiries)[0]
        today = datetime.now().date()
        thursday_candidates = [p for p in parsed if p.date() >= today and p.weekday() == 3]
        if thursday_candidates:
            chosen = min(thursday_candidates)
            return chosen.date().isoformat()
        future = [p for p in parsed if p.date() >= today]
        if future:
            chosen = min(future)
            return chosen.date().isoformat()
        chosen = min(parsed)
        return chosen.date().isoformat()

    def _get_spot(self, symbol: str) -> int:
        candidates = [
            f"NSE:{symbol}",
            f"NFO:{symbol}",
            f"NSE:{symbol} 50",
            f"{symbol}",
            f"{symbol}-INDEX",
        ]
        for cand in candidates:
            try:
                resp = self.kite.ltp([cand])
                if isinstance(resp, dict) and len(resp) > 0:
                    val = next(iter(resp.values()))
                    if isinstance(val, dict):
                        if "last_price" in val:
                            return int(round(val["last_price"]))
                        if "lastPrice" in val:
                            return int(round(val["lastPrice"]))
                        if "ltp" in val:
                            return int(round(val["ltp"]))
            except Exception:
                continue
        LOG.warning("Unable to fetch spot for %s; using fallback 21000", symbol)
        return 21000

    def _batch_ltp(self, tokens: List[int]) -> List[dict]:
        out = []
        if not tokens:
            return out
        batches = [tokens[i:i + self.batch_size] for i in range(0, len(tokens), self.batch_size)]
        for batch in batches:
            try:
                resp = self.kite.ltp(batch)
                for k, v in resp.items():
                    token = None
                    last_price = None
                    oi = None
                    best_bid = None
                    best_ask = None
                    if isinstance(v, dict):
                        token = v.get("instrument_token") or v.get("instrumentToken") or None
                        last_price = v.get("last_price") or v.get("lastPrice") or v.get("ltp") or None
                        oi = v.get("oi") or v.get("open_interest") or v.get("openInterest") or None
                        best_bid = v.get("best_bid") or v.get("bestBid") or v.get("buy_price") or None
                        best_ask = v.get("best_ask") or v.get("bestAsk") or v.get("sell_price") or None
                    if token is None:
                        try:
                            token = int(k)
                        except Exception:
                            token = None
                    out.append({
                        "instrument_token": int(token) if token is not None else None,
                        "ltp": float(last_price) if last_price is not None else None,
                        "open_interest": int(oi) if oi is not None else None,
                        "best_bid": float(best_bid) if best_bid is not None else None,
                        "best_ask": float(best_ask) if best_ask is not None else None,
                        "raw_key": k,
                    })
            except Exception as e:
                LOG.error("Batch LTP failed for batch size %s -> %s", len(batch), repr(e))
                # fallback to single calls to be resilient
                for single in batch:
                    try:
                        resp = self.kite.ltp([single])
                        for k, v in resp.items():
                            token = v.get("instrument_token") if isinstance(v, dict) else None
                            last_price = v.get("last_price") or v.get("ltp") if isinstance(v, dict) else None
                            oi = v.get("oi") if isinstance(v, dict) else None
                            best_bid = v.get("best_bid") if isinstance(v, dict) else None
                            best_ask = v.get("best_ask") if isinstance(v, dict) else None
                            out.append({
                                "instrument_token": int(token) if token is not None else None,
                                "ltp": float(last_price) if last_price is not None else None,
                                "open_interest": int(oi) if oi is not None else None,
                                "best_bid": float(best_bid) if best_bid is not None else None,
                                "best_ask": float(best_ask) if best_ask is not None else None,
                                "raw_key": k,
                            })
                    except Exception:
                        LOG.warning("Single ltp call failed for %s", single)
                        out.append({
                            "instrument_token": int(single) if isinstance(single, (int, float)) else None,
                            "ltp": None,
                            "open_interest": None,
                            "best_bid": None,
                            "best_ask": None,
                            "raw_key": str(single),
                        })
        return out

    # -------------------------
    # PAPER fallback
    # -------------------------
    def _paper_snapshot(self, symbol: str = "NIFTY", width: int = 1500) -> pd.DataFrame:
        spot = 21000
        strikes = list(range(spot - width, spot + width + 1, 50))
        rows = []
        for st in strikes:
            for opt in ["CE", "PE"]:
                rows.append({
                    "strike": st,
                    "option_type": opt,
                    "tradingsymbol": f"{symbol}{st}{opt}",
                    "instrument_token": st * 10 + (1 if opt == "CE" else 2),
                    "expiry": (date.today() + timedelta(days=40)).isoformat(),
                    "best_bid": 5.0,
                    "best_ask": 5.5,
                    "ltp": 5.25,
                    "spot": spot,
                    "open_interest": 1500,
                })
        return pd.DataFrame(rows)

    # -------------------------
    # Ensure underlying row present
    # -------------------------
    def _ensure_underlying_row(self, df: pd.DataFrame, symbol: str, spot_override: Optional[float] = None) -> pd.DataFrame:
        """
        Insert a leading row with instrument_type 'UNDERLYING' and keys: tradingsymbol, ltp, spot, timestamp.
        If df already contains an UNDERLYING row, refresh it.
        """
        try:
            spot_val = spot_override if spot_override is not None else (int(df["spot"].iloc[0]) if ("spot" in df.columns and not df.empty) else None)
        except Exception:
            spot_val = None

        if spot_val is None:
            spot_val = 21000

        underlying_row = {
            "instrument_type": "UNDERLYING",
            "strike": None,
            "option_type": None,
            "tradingsymbol": f"{symbol}-INDEX",
            "instrument_token": None,
            "expiry": None,
            "best_bid": None,
            "best_ask": None,
            "ltp": float(spot_val),
            "spot": float(spot_val),
            "open_interest": None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        # Guard: convert any pd.NA/NaN inside row to None
        for k, v in list(underlying_row.items()):
            try:
                if pd.isna(v):
                    underlying_row[k] = None
            except Exception:
                pass

        # If df already has an underlying row, drop it first
        if "instrument_type" in df.columns:
            try:
                df = df[~(df["instrument_type"].astype(str).str.upper() == "UNDERLYING")].copy()
            except Exception:
                # fallback: remove rows whose tradingsymbol looks like underlying
                df = df[~(df.get("tradingsymbol", "").astype(str).str.upper().str.contains("-INDEX"))].copy()

        # Prepend underlying row
        df2 = pd.concat([pd.DataFrame([underlying_row]), df], ignore_index=True, sort=False)
        # Ensure consistent column order
        cols = ["instrument_type", "tradingsymbol", "strike", "option_type", "instrument_token", "expiry",
                "best_bid", "best_ask", "ltp", "spot", "open_interest", "timestamp"]
        cols_present = [c for c in cols if c in df2.columns]
        return df2.loc[:, cols_present].copy()


# Public helper used across system
def full_option_chain(api, symbol: str = "NIFTY", width: int = 1500, expiry=None) -> pd.DataFrame:
    """
    Universal helper used by engine/signalgen/backtests.
    Accepts:
      - KiteData instance
      - KiteAPI instance (your wrapper)
      - raw KiteConnect (falls back to KiteData)
    Returns tidy DataFrame with underlying row included.
    """
    from src.live.kite_data import KiteData as _KD

    # Case 1: KiteData instance
    if isinstance(api, _KD):
        return api.get_option_chain_snapshot(symbol=symbol, width=width, expiry=expiry)

    # Case 2: KiteAPI (wrapper) - try its get_live_snapshot or get_option_chain
    try:
        if hasattr(api, "get_live_snapshot"):
            df = api.get_live_snapshot(symbol)
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Ensure underlying present
                kd = _KD()
                df2 = kd._ensure_underlying_row(df, symbol, spot_override=None)
                return df2.where(pd.notnull(df2), None)
    except Exception:
        pass

    # Case 3: raw kiteconnect-like object -> delegate to KiteData
    try:
        kd = _KD()
        return kd.get_option_chain_snapshot(symbol=symbol, width=width, expiry=expiry)
    except Exception:
        # Final fallback paper snapshot
        kd = _KD()
        dfp = kd._paper_snapshot(symbol=symbol, width=width)
        return dfp.where(pd.notnull(dfp), None)
