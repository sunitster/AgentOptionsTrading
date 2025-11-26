# src/live/kite_data.py
"""
Ultra-stable KiteData + full_option_chain helper (FINAL PATCHED VERSION).

Fixes:
- Live expiry scoring for correct expiry selection.
- Live chain sanitization: NO NaN/inf ever leaves this module.
- Underlying row always safe (no NaN).
- Dashboard JSON-safe output.
- No breakage to existing SignalGenerator or Broker.
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
            LOG.warning("KiteConnect unavailable or keys missing — PAPER MODE")

    # -------------------------------------------------
    # CONFIG
    # -------------------------------------------------
    def _load_config(self):
        if not os.path.exists(self.config_path):
            LOG.warning("Config not found: %s — PAPER MODE", self.config_path)
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
            if not self.access_token:
                LOG.warning("No access_token — PAPER MODE")
                self.kite = None
                self.mode = "paper"
                return

            self.kite.set_access_token(self.access_token)
            prof = self.kite.profile()
            LOG.info("Connected live as %s (%s)", prof.get("user_id"), prof.get("user_name"))
            self.mode = "live"

        except Exception as e:
            LOG.error("KiteConnect connect failed -> PAPER MODE: %s", repr(e))
            self.kite = None
            self.mode = "paper"

    # -------------------------------------------------
    # MAIN: get_option_chain_snapshot
    # -------------------------------------------------
    def get_option_chain_snapshot(
        self,
        symbol: str = "NIFTY",
        width: int = 1500,
        expiry: Optional[str] = None,
        exchange: str = "NFO"
    ) -> pd.DataFrame:

        now = datetime.utcnow()

        # Cache for <1 sec
        if self._cache["ts"] and (now - self._cache["ts"]).total_seconds() < 1:
            cached = self._cache["data"]
            if isinstance(cached, pd.DataFrame):
                return cached.copy()

        # PAPER MODE
        if self.mode != "live" or not self.kite:
            LOG.info("PAPER snapshot call")
            df = self._paper_snapshot(symbol, width)
            df = self._ensure_underlying_row(df, symbol)
            df = self._sanitize(df)
            self._cache["ts"] = now
            self._cache["data"] = df
            return df

        # LIVE MODE
        try:
            instruments = self._fetch_instruments(exchange)
            if not instruments:
                LOG.warning("No instruments — PAPER fallback")
                return self._fallback(now, symbol, width)

            expiries = self._get_expiries_for_symbol(instruments, symbol)
            if not expiries:
                LOG.warning("No expiries — PAPER fallback")
                return self._fallback(now, symbol, width)

            # --------
            # Early Spot Fetch
            # --------
            spot = self._get_spot(symbol)
            LOG.info("Spot for %s = %s", symbol, spot)

            # --------
            # EXPIRY SELECTION
            # --------
            if expiry:
                chosen_expiry = expiry
            else:
                chosen_expiry = self._choose_expiry_scored(instruments, symbol, spot, width, expiries)

            LOG.info("Chosen expiry for %s -> %s", symbol, chosen_expiry)

            # --------
            # FILTER OPTIONS
            # --------
            strike_pat = re.compile(r"(\d+)(CE|PE)$", flags=re.IGNORECASE)
            rows = []

            for ins in instruments:
                ts = ins.get("tradingsymbol", "")
                if not ts:
                    continue
                if ins.get("exchange") != exchange:
                    continue
                if ins.get("expiry") != chosen_expiry:
                    continue
                if not ts.upper().startswith(symbol.upper()):
                    continue

                m = strike_pat.search(ts)
                if not m:
                    continue
                strike = int(m.group(1))
                opt = m.group(2).upper()

                if abs(strike - spot) > width:
                    continue

                rows.append({
                    "strike": strike,
                    "option_type": opt,
                    "tradingsymbol": ts,
                    "instrument_token": int(ins.get("instrument_token") or 0),
                    "expiry": ins.get("expiry")
                })

            if not rows:
                LOG.warning("No options match width — PAPER fallback")
                return self._fallback(now, symbol, width, spot_override=spot)

            df = pd.DataFrame(rows).dropna(subset=["instrument_token"]).astype({"instrument_token": int})

            # --------
            # FETCH LTP
            # --------
            ltp_df = pd.DataFrame(self._batch_ltp(df["instrument_token"].tolist()))
            merged = df.merge(ltp_df, how="left", on="instrument_token")
            merged["spot"] = spot

            # Required columns
            required = ["best_bid", "best_ask", "ltp", "open_interest"]
            for c in required:
                if c not in merged.columns:
                    merged[c] = None

            merged["instrument_type"] = merged["option_type"].apply(
                lambda x: "CE" if str(x).upper() == "CE" else "PE"
            )

            # Order columns
            final_cols = [
                "instrument_type", "strike", "option_type", "tradingsymbol",
                "instrument_token", "expiry", "best_bid", "best_ask",
                "ltp", "spot", "open_interest"
            ]
            merged = merged.loc[:, final_cols]
            merged = merged.sort_values(["strike", "option_type"]).reset_index(drop=True)

            # Add underlying
            df_final = self._ensure_underlying_row(merged, symbol, spot_override=spot)

            # -------------------------
            # NEW PATCH: ensure strike column contains NO NaN floats before sanitization
            # This prevents `int(nan)` crashes in dashboard routes and keeps dtype safe.
            # -------------------------
            try:
                if "strike" in df_final.columns:
                    # make it object so None can be stored without upcasting to float/NaN
                    try:
                        df_final["strike"] = df_final["strike"].astype(object)
                    except Exception:
                        # if cast fails, ignore and continue
                        pass

                    def _clean_strike(x):
                        # convert NaN/inf floats -> None, cast integer-like floats to int
                        if x is None:
                            return None
                        if isinstance(x, float):
                            if math.isnan(x) or math.isinf(x):
                                return None
                            # float like 19500.0 -> int 19500
                            if x.is_integer():
                                return int(x)
                            return x
                        return x

                    df_final["strike"] = df_final["strike"].apply(_clean_strike)
            except Exception:
                LOG.debug("Strike cleaning failed (non-fatal)", exc_info=True)

            # SANITIZE: remove NaN before returning
            df_final = self._sanitize(df_final)

            self._cache["ts"] = now
            self._cache["data"] = df_final
            return df_final

        except Exception:
            LOG.exception("Live chain failed — PAPER fallback")
            return self._fallback(now, symbol, width)

    # -------------------------------------------------
    # SANITIZATION (VERY IMPORTANT)
    # -------------------------------------------------
    def _sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert NaN/inf to None for dashboard JSON safety."""
        df = df.copy()
        # Replace NaN and inf
        for c in df.columns:
            df[c] = df[c].apply(
                lambda x: None
                if (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
                else x
            )
        return df

    # -------------------------------------------------
    # FALLBACK + HELPERS
    # -------------------------------------------------
    def _fallback(self, now, symbol, width, spot_override=None):
        df = self._paper_snapshot(symbol, width)
        df = self._ensure_underlying_row(df, symbol, spot_override)
        df = self._sanitize(df)
        self._cache["ts"] = now
        self._cache["data"] = df
        return df

    def _fetch_instruments(self, exchange="NFO"):
        try:
            return self.kite.instruments(exchange)
        except Exception as e:
            LOG.error("instruments() failed: %s", repr(e))
            return []

    def _get_expiries_for_symbol(self, instruments, symbol):
        out = set()
        su = symbol.upper()
        for ins in instruments:
            ts = ins.get("tradingsymbol", "")
            if ts.upper().startswith(su) and ins.get("expiry"):
                out.add(ins["expiry"])
        try:
            return sorted(out, key=lambda x: datetime.fromisoformat(x))
        except:
            return sorted(out)

    # -------------------------------------------------
    # EXPIRY SCORING
    # -------------------------------------------------
    def _choose_expiry_scored(self, instruments, symbol, spot, width, expiries):
        strike_pat = re.compile(r"(\d+)(CE|PE)$", flags=re.IGNORECASE)
        scores: Dict[str, int] = {}

        for ins in instruments:
            ts = ins.get("tradingsymbol", "")
            exp = ins.get("expiry")
            if not ts or not exp:
                continue
            if not ts.upper().startswith(symbol.upper()):
                continue
            m = strike_pat.search(ts)
            if not m:
                continue
            strike = int(m.group(1))
            if abs(strike - spot) <= width:
                scores[exp] = scores.get(exp, 0) + 1

        if scores:
            chosen = sorted(
                scores.items(),
                key=lambda x: (-x[1], x[0])  # Best score, earliest expiry
            )[0][0]
            LOG.info("Expiry scoring selected %s (scores=%s)", chosen, scores)
            return chosen

        # fallback if scoring fails
        return self._choose_preferred_expiry(expiries)

    def _choose_preferred_expiry(self, expiries):
        parsed = []
        for e in expiries:
            try:
                parsed.append(datetime.fromisoformat(e))
            except:
                pass
        if not parsed:
            return sorted(expiries)[0]

        today = datetime.now().date()

        # Prefer Thursday >= today
        thr = [p for p in parsed if p.date() >= today and p.weekday() == 3]
        if thr:
            return min(thr).date().isoformat()

        fut = [p for p in parsed if p.date() >= today]
        if fut:
            return min(fut).date().isoformat()

        return min(parsed).date().isoformat()

    # -------------------------------------------------
    # SPOT
    # -------------------------------------------------
    def _get_spot(self, symbol):
        cands = [
            f"NSE:{symbol}",
            f"NFO:{symbol}",
            f"{symbol}",
            f"{symbol}-INDEX",
        ]
        for c in cands:
            try:
                resp = self.kite.ltp([c])
                if not resp:
                    continue
                v = next(iter(resp.values()))
                if "last_price" in v:
                    return int(round(v["last_price"]))
                if "ltp" in v:
                    return int(round(v["ltp"]))
            except:
                continue
        LOG.warning("Spot fallback for %s -> 21000", symbol)
        return 21000

    # -------------------------------------------------
    # LTP BATCH
    # -------------------------------------------------
    def _batch_ltp(self, tokens: List[int]) -> List[dict]:
        out = []
        if not tokens:
            return out

        batches = [tokens[i:i+self.batch_size] for i in range(0, len(tokens), self.batch_size)]

        for batch in batches:
            try:
                resp = self.kite.ltp(batch)
                for _, v in resp.items():
                    if not isinstance(v, dict):
                        continue
                    out.append({
                        "instrument_token": v.get("instrument_token"),
                        "ltp": v.get("last_price") or v.get("ltp"),
                        "best_bid": v.get("best_bid"),
                        "best_ask": v.get("best_ask"),
                        "open_interest": v.get("oi") or v.get("open_interest"),
                    })
            except Exception as e:
                LOG.error("batch_ltp failed: %s", repr(e))

        return out

    # -------------------------------------------------
    # PAPER SNAPSHOT
    # -------------------------------------------------
    def _paper_snapshot(self, symbol="NIFTY", width=1500):
        spot = 21000
        strikes = list(range(spot-width, spot+width+1, 50))
        rows = []
        for st in strikes:
            for opt in ["CE", "PE"]:
                rows.append({
                    "instrument_type": opt,
                    "strike": st,
                    "option_type": opt,
                    "tradingsymbol": f"{symbol}{st}{opt}",
                    "instrument_token": st * 10,
                    "expiry": (date.today()+timedelta(days=40)).isoformat(),
                    "best_bid": 5,
                    "best_ask": 5.5,
                    "ltp": 5.25,
                    "spot": spot,
                    "open_interest": 1500,
                })
        return pd.DataFrame(rows)

    # -------------------------------------------------
    # UNDERLYING ROW
    # -------------------------------------------------
    def _ensure_underlying_row(self, df, symbol, spot_override=None):
        try:
            spot_val = spot_override if spot_override is not None else df["spot"].iloc[0]
        except:
            spot_val = 21000

        und = {
            "instrument_type": "UNDERLYING",
            "tradingsymbol": f"{symbol}-INDEX",
            "strike": None,
            "option_type": None,
            "instrument_token": None,
            "expiry": None,
            "best_bid": None,
            "best_ask": None,
            "ltp": float(spot_val),
            "spot": float(spot_val),
            "open_interest": None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        # Remove any existing underlying row
        if "instrument_type" in df.columns:
            df = df[df["instrument_type"].astype(str).str.upper() != "UNDERLYING"]

        cols = [
            "instrument_type", "tradingsymbol", "strike", "option_type",
            "instrument_token", "expiry", "best_bid", "best_ask",
            "ltp", "spot", "open_interest", "timestamp"
        ]

        df2 = pd.concat([pd.DataFrame([und]), df], ignore_index=True, sort=False)
        cols_present = [c for c in cols if c in df2.columns]
        return df2.loc[:, cols_present].copy()


# -------------------------------------------------
# WRAPPER (unchanged)
# -------------------------------------------------
def full_option_chain(api, symbol="NIFTY", width=1500, expiry=None):
    from src.live.kite_data import KiteData as KD
    if isinstance(api, KD):
        return api.get_option_chain_snapshot(symbol, width, expiry)
    try:
        if hasattr(api, "get_live_snapshot"):
            df = api.get_live_snapshot(symbol)
            if isinstance(df, pd.DataFrame) and not df.empty:
                kd = KD()
                df2 = kd._ensure_underlying_row(df, symbol)
                return df2.where(pd.notnull(df2), None)
    except:
        pass
    try:
        kd = KD()
        return kd.get_option_chain_snapshot(symbol, width, expiry)
    except:
        kd = KD()
        dfp = kd._paper_snapshot(symbol, width)
        return dfp.where(pd.notnull(dfp), None)
