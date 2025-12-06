# src/live/kite_data.py
"""
Unified option chain + LTP + (optional) IV/delta computation.

This module builds a filtered option chain using kite_api.get_chain_snapshot
(which itself uses timed instruments() + cache). Then it fetches LTP for the
filtered tradingsymbols (batch first, per-symbol fallback), marks ATM, and
attempts IV/delta computation using src.iv_utils if available.

This is the patched, hardened, final version.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional
from datetime import datetime, date

import pandas as pd
import math

LOG = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

_TRAD_SYM_RE = re.compile(r"([A-Za-z0-9]+)_(PE|CE)_(\d+)$")  # fallback pattern


def _parse_entry_to_row(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    ts = entry.get("tradingsymbol") or entry.get("symbol") or entry.get("name")
    # last price detection
    ltp = None
    for k in ("last_price", "ltp", "lastPrice", "last_traded_price"):
        if k in entry and entry[k] is not None:
            try:
                ltp = float(entry[k])
                break
            except Exception:
                pass
    # strike
    strike = None
    if "strike" in entry and entry["strike"] is not None:
        try:
            strike = float(entry["strike"])
        except Exception:
            strike = None
    if strike is None and isinstance(ts, str):
        m = re.search(r"(\d{3,6})", ts)
        if m:
            try:
                strike = float(m.group(1))
            except Exception:
                strike = None
    # option type
    opt_type = None
    if isinstance(ts, str):
        m = _TRAD_SYM_RE.match(ts)
        if m:
            opt_type = m.group(2)
    if not opt_type:
        opt_type = entry.get("option_type") or entry.get("otype") or entry.get("instrument_type")
        if isinstance(opt_type, str):
            opt_type = opt_type.upper().replace("OPTION", "").strip()[:2]
    instrument_token = entry.get("instrument_token") or entry.get("token") or entry.get("instrumentToken")
    return {
        "tradingsymbol": ts,
        "strike": strike,
        "expiry": entry.get("expiry") or entry.get("expiry_date") or entry.get("expiry_dt"),
        "option_type": opt_type,
        "last_price": ltp,
        "raw": entry,
        "instrument_token": instrument_token,
    }


def _normalize_snapshot(snapshot: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if snapshot is None:
        return out
    if isinstance(snapshot, dict):
        # already keyed
        if all(isinstance(v, dict) for v in snapshot.values()):
            for k, v in snapshot.items():
                out[str(k)] = v.copy()
            return out
        if "data" in snapshot and isinstance(snapshot["data"], list):
            snapshot = snapshot["data"]
    if isinstance(snapshot, list):
        for ent in snapshot:
            if not isinstance(ent, dict):
                continue
            ts = ent.get("tradingsymbol") or ent.get("symbol") or ent.get("name")
            if ts:
                out[str(ts)] = ent.copy()
            else:
                tok = ent.get("instrument_token") or ent.get("token")
                if tok:
                    out[str(tok)] = ent.copy()
        return out
    LOG.debug("_normalize_snapshot: unexpected snapshot type %s", type(snapshot))
    return out


def _coerce_expiry(x) -> Optional[str]:
    if x is None:
        return None
    try:
        if isinstance(x, (date, datetime)):
            return x.date().isoformat() if isinstance(x, datetime) else x.isoformat()
        s = str(x)
        for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(s, fmt).date().isoformat()
            except Exception:
                pass
        # if nothing matched, return original string normalized
        return s
    except Exception:
        return None


def _choose_nearest_expiry(expiry_list: List[str]) -> Optional[str]:
    today = datetime.utcnow().date()
    best = None
    best_days = None
    for e in expiry_list:
        if not e:
            continue
        try:
            ed = datetime.fromisoformat(e).date()
        except Exception:
            try:
                ed = datetime.strptime(e, "%d-%m-%Y").date()
            except Exception:
                continue
        days = (ed - today).days
        if days < 0:
            continue
        if best_days is None or days < best_days:
            best_days = days
            best = e
    return best


def available_strikes(chain_df: pd.DataFrame) -> List[float]:
    if chain_df is None or chain_df.empty:
        return []
    strikes = pd.to_numeric(chain_df["strike"], errors="coerce").dropna().unique().tolist()
    strikes = sorted([float(x) for x in strikes])
    return strikes


def closest_strike(target: float, available: List[float]) -> Optional[float]:
    if not available:
        return None
    try:
        best = min(available, key=lambda x: abs(x - target))
        return float(best)
    except Exception:
        return None


def _filter_by_expiry_and_strikes(df: pd.DataFrame, spot: Optional[float], strike_window_count: int = 40, max_symbols: int = 150) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df_local = df.copy()
    df_local["expiry_iso"] = df_local["expiry"].apply(lambda x: _coerce_expiry(x) if x is not None else None)
    expiry_counts = df_local.groupby("expiry_iso")["strike"].nunique().fillna(0).to_dict()
    expiries = [k for k in expiry_counts.keys() if k is not None]
    selected_expiry = _choose_nearest_expiry(expiries) if expiries else None
    LOG.debug("kite_data._filter: expiries_found=%d selected=%s", len(expiries), selected_expiry)
    if selected_expiry:
        df_filtered = df_local[df_local["expiry_iso"] == selected_expiry].copy()
        LOG.debug("kite_data._filter: rows for selected expiry=%d", len(df_filtered))
    else:
        df_filtered = df_local.copy()
        LOG.debug("kite_data._filter: no expiry selected, using all rows=%d", len(df_filtered))
    if spot is None or (isinstance(spot, float) and math.isnan(spot)):
        LOG.debug("kite_data._filter: spot missing; returning top %d rows", min(len(df_filtered), max_symbols))
        return df_filtered.head(max_symbols)
    strikes = available_strikes(df_filtered)
    if not strikes:
        LOG.debug("kite_data._filter: no numeric strikes available; returning top %d rows", min(len(df_filtered), max_symbols))
        return df_filtered.head(max_symbols)
    strikes_sorted = sorted(strikes)
    import bisect
    idx = bisect.bisect_left(strikes_sorted, float(spot))
    half = max(1, int(strike_window_count // 2))
    lo = max(0, idx - half)
    hi = min(len(strikes_sorted), idx + half)
    selected_strikes = set(strikes_sorted[lo:hi])
    LOG.debug("kite_data._filter: strikes_total=%d selected=%d (window=%d)", len(strikes_sorted), len(selected_strikes), strike_window_count)
    selected_ints = set(int(s) for s in selected_strikes)
    df_window = df_filtered[df_filtered["strike"].apply(lambda x: (False if pd.isna(x) else int(float(x)) in selected_ints))].copy()
    if df_window.empty:
        LOG.debug("kite_data._filter: window resulted in no rows; returning top %d rows of expiry", min(len(df_filtered), max_symbols))
        return df_filtered.head(max_symbols)
    if len(df_window) > max_symbols:
        df_window = df_window.head(max_symbols).copy()
        LOG.debug("kite_data._filter: limited df_window to max_symbols=%d", max_symbols)
    return df_window


def _attempt_iv_and_delta_computation(df: pd.DataFrame, spot: Optional[float]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        import src.iv_utils as ivu  # type: ignore
    except Exception:
        try:
            import iv_utils as ivu  # type: ignore
        except Exception:
            LOG.debug("_attempt_iv_and_delta_computation: iv_utils not available; skipping IV/delta")
            df["iv"] = None
            df["delta"] = None
            return df

    iv_fn = None
    delta_fn = None
    for name in ("implied_volatility", "iv_from_price", "price_to_iv", "calc_iv", "get_iv"):
        if hasattr(ivu, name):
            iv_fn = getattr(ivu, name)
            break
    for name in ("delta_from_price", "black_scholes_delta", "get_delta", "get_greeks", "greeks"):
        if hasattr(ivu, name):
            delta_fn = getattr(ivu, name)
            break

    df["iv"] = None
    df["delta"] = None

    for idx, row in df.iterrows():
        ltp = row.get("last_price")
        strike = row.get("strike")
        expiry = row.get("expiry")
        otype = row.get("option_type")
        try:
            iv_val = None
            delta_val = None
            if iv_fn and ltp is not None and spot is not None and strike is not None and expiry is not None:
                try:
                    iv_val = iv_fn(ltp, spot, float(strike), expiry)
                except TypeError:
                    try:
                        iv_val = iv_fn(price=ltp, spot=spot, strike=float(strike), expiry=expiry)
                    except Exception:
                        try:
                            iv_val = iv_fn(ltp, float(strike), spot, expiry)
                        except Exception:
                            iv_val = None
                except Exception:
                    iv_val = None
            if delta_fn and ltp is not None and spot is not None and strike is not None and expiry is not None:
                try:
                    out = None
                    try:
                        out = delta_fn(ltp, spot, float(strike), expiry, otype)
                    except TypeError:
                        try:
                            out = delta_fn(price=ltp, spot=spot, strike=float(strike), expiry=expiry, option_type=otype)
                        except Exception:
                            try:
                                out = delta_fn(spot, float(strike), expiry, otype)
                            except Exception:
                                out = None
                    if isinstance(out, dict):
                        if "delta" in out:
                            delta_val = out.get("delta")
                        elif "greeks" in out and isinstance(out["greeks"], dict):
                            delta_val = out["greeks"].get("delta")
                    else:
                        delta_val = out
                except Exception:
                    delta_val = None
            if iv_val is not None:
                try:
                    df.at[idx, "iv"] = float(iv_val)
                except Exception:
                    df.at[idx, "iv"] = None
            if delta_val is not None:
                try:
                    df.at[idx, "delta"] = float(delta_val)
                except Exception:
                    df.at[idx, "delta"] = None
        except Exception:
            df.at[idx, "iv"] = None
            df.at[idx, "delta"] = None

    return df


def _populate_ltp_for_df(df: pd.DataFrame, kite_api, timeout_seconds: float = 5.0) -> pd.DataFrame:
    """
    Try batch quote for df.tradingsymbols; fallback to per-symbol kite_api.quote().
    Populates df['last_price'] (and a convenience alias df['ltp']).

    This version translates the DataFrame's tradingsymbol -> quoteable symbol
    using kite_api.to_zerodha_symbol(...) (if available). Then it performs a
    batch quote (if supported), otherwise per-symbol safe fallbacks. Results
    are mapped back to original tradingsymbols so the rest of the system
    can continue to work unchanged.
    """
    if df is None or df.empty:
        return df

    original_syms = [str(x) for x in df["tradingsymbol"].tolist() if x]
    if not original_syms:
        return df

    # Map original -> quote symbol (use translator if available)
    orig_to_quote: Dict[str, str] = {}
    quote_to_orig: Dict[str, str] = {}
    for orig in original_syms:
        try:
            if hasattr(kite_api, "to_zerodha_symbol"):
                qsym = kite_api.to_zerodha_symbol(orig)
            else:
                qsym = orig
        except Exception:
            qsym = orig
        orig_to_quote[orig] = qsym
        # keep last mapping if duplicate quote symbol encountered
        quote_to_orig[qsym] = orig

    # Unique quote symbols preserving order
    seen = set()
    quote_symbols: List[str] = []
    for v in orig_to_quote.values():
        if v not in seen:
            seen.add(v)
            quote_symbols.append(v)

    quotes: Dict[str, Any] = {}

    # 1) Try kite_api.quote(quote_symbols) if it supports batch
    try:
        qresp = None
        try:
            qresp = kite_api.quote(quote_symbols)
        except TypeError:
            qresp = None
        except Exception:
            qresp = None

        if isinstance(qresp, dict) and qresp:
            quotes = qresp
        elif isinstance(qresp, list) and qresp:
            for i, q in enumerate(qresp):
                if i < len(quote_symbols) and isinstance(q, dict):
                    quotes[quote_symbols[i]] = q
    except Exception:
        quotes = {}

    # 2) Try low-level client batch (if kite_api exposes it)
    if not quotes:
        try:
            if hasattr(kite_api, "_client") and hasattr(kite_api, "_call_with_timeout"):
                def _call_batch():
                    try:
                        return kite_api._client.quote(quote_symbols)
                    except Exception:
                        return None
                try:
                    qresp = kite_api._call_with_timeout(_call_batch, timeout_seconds)
                    if isinstance(qresp, dict):
                        quotes = qresp
                except Exception:
                    quotes = {}
        except Exception:
            quotes = {}

    # 3) Fallback to per-quote-symbol safe single quote
    if not quotes:
        for qsym in quote_symbols:
            try:
                q = kite_api.quote(qsym)
                if q:
                    # normalize
                    if isinstance(q, dict) and qsym in q and isinstance(q[qsym], dict):
                        quotes[qsym] = q[qsym]
                    else:
                        quotes[qsym] = q
                else:
                    quotes[qsym] = {}
            except Exception:
                # try original if translator produced something odd
                try:
                    orig_try = quote_to_orig.get(qsym, qsym)
                    q2 = kite_api.quote(orig_try)
                    quotes[qsym] = q2 if q2 else {}
                except Exception:
                    quotes[qsym] = {}

    # Helper to extract last price from returned quote structure
    def _quote_last_price(q):
        if q is None:
            return None
        if isinstance(q, (int, float)):
            try:
                return float(q)
            except Exception:
                return None
        if isinstance(q, dict):
            for k in ("last_price", "ltp", "lastPrice"):
                if k in q and q[k] is not None:
                    try:
                        return float(q[k])
                    except Exception:
                        continue
            # nested
            for _, v in q.items():
                if isinstance(v, dict):
                    for k in ("last_price", "ltp", "lastPrice"):
                        if k in v and v[k] is not None:
                            try:
                                return float(v[k])
                            except Exception:
                                continue
        return None

    # Map back to original tradingsymbols
    ltp_map: Dict[str, Optional[float]] = {}
    for qsym, qobj in quotes.items():
        try:
            lp = _quote_last_price(qobj)
            orig = quote_to_orig.get(qsym, qsym)
            ltp_map[orig] = lp
        except Exception:
            ltp_map[quote_to_orig.get(qsym, qsym)] = None

    # For any original symbols not present in ltp_map, set None
    for orig in original_syms:
        if orig not in ltp_map:
            ltp_map[orig] = None

    df["last_price"] = df["tradingsymbol"].map(lambda x: ltp_map.get(str(x)))
    df["ltp"] = df["last_price"]
    return df


def full_option_chain(kite_api, symbol_root: str, strike_window_count: int = 40, max_symbols: int = 150) -> pd.DataFrame:
    """
    Fetch and normalize the option chain snapshot for `symbol_root` using kite_api.
    Returns a pandas.DataFrame filtered to nearest expiry and a strike window around spot.
    Also fetches LTP for the filtered tradingsymbols, marks ATM, and attempts IV/delta.
    """
    start_ts = time.time()
    try:
        try:
            chain_snapshot = kite_api.get_chain_snapshot(symbol_root)
            LOG.debug("full_option_chain: get_chain_snapshot returned type=%s", type(chain_snapshot).__name__)
        except Exception as e:
            LOG.exception("full_option_chain: kite_api.get_chain_snapshot raised exception for %s: %s", symbol_root, e)
            chain_snapshot = {}

        if not chain_snapshot:
            elapsed = time.time() - start_ts
            LOG.warning("full_option_chain: no chain_snapshot for %s (len=%s) after %.2fs — returning empty DataFrame",
                        symbol_root, (len(chain_snapshot) if chain_snapshot is not None and hasattr(chain_snapshot, '__len__') else "None"), elapsed)
            fetched_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            spot = None
            try:
                spot = kite_api.get_spot(symbol_root)
            except Exception:
                spot = None
            df = pd.DataFrame(columns=["tradingsymbol", "strike", "expiry", "option_type", "last_price", "raw", "instrument_token", "spot", "fetched_at"])
            df["spot"] = float(spot) if spot is not None else None
            df["fetched_at"] = fetched_at
            return df

    except Exception as e:
        LOG.exception("full_option_chain: unexpected error while fetching chain for %s: %s", symbol_root, e)
        fetched_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        spot = None
        try:
            spot = kite_api.get_spot(symbol_root)
        except Exception:
            spot = None
        df = pd.DataFrame(columns=["tradingsymbol", "strike", "expiry", "option_type", "last_price", "raw", "instrument_token", "spot", "fetched_at"])
        df["spot"] = float(spot) if spot is not None else None
        df["fetched_at"] = fetched_at
        return df

    norm = _normalize_snapshot(chain_snapshot)
    rows: List[Dict[str, Any]] = []
    for ts, entry in norm.items():
        r = _parse_entry_to_row(entry)
        if r is None:
            continue
        if r.get("tradingsymbol") in (None, ""):
            r["tradingsymbol"] = ts
        rows.append(r)

    df = pd.DataFrame(rows)
    fetched_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    spot = None
    try:
        spot = kite_api.get_spot(symbol_root)
    except Exception as e:
        LOG.warning("full_option_chain: kite_api.get_spot failed for %s: %s", symbol_root, e)
        spot = None

    if df.empty:
        if isinstance(chain_snapshot, dict) and chain_snapshot:
            alt_rows = []
            for k, v in chain_snapshot.items():
                r = _parse_entry_to_row(v if isinstance(v, dict) else {"tradingsymbol": k})
                if r:
                    if r.get("tradingsymbol") in (None, ""):
                        r["tradingsymbol"] = k
                    alt_rows.append(r)
            if alt_rows:
                df = pd.DataFrame(alt_rows)

    if not df.empty:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["option_type"] = df["option_type"].astype(object).where(df["option_type"].notna(), None)
        df["option_type"] = df["option_type"].apply(lambda x: (str(x).upper().replace(" ", "") if x is not None else None))
        df["last_price"] = pd.to_numeric(df["last_price"], errors="coerce")
        df["expiry"] = df["expiry"].apply(lambda x: _coerce_expiry(x) if x is not None else None)
        df["raw_summary"] = df["raw"].apply(lambda x: {k: x.get(k) for k in list(x.keys())[:6]} if isinstance(x, dict) else x)
        df["spot"] = float(spot) if spot is not None else None
        df["fetched_at"] = fetched_at

        # filter aggressively
        df_filtered = _filter_by_expiry_and_strikes(df, spot=spot, strike_window_count=strike_window_count, max_symbols=max_symbols)

        # fetch LTP for filtered tradingsymbols (batch w/ fallback)
        df_filtered = _populate_ltp_for_df(df_filtered, kite_api, timeout_seconds=5.0)

        # ATM detection
        try:
            strikes_list = available_strikes(df_filtered)
            atm_strike = None
            if spot is not None and strikes_list:
                atm_strike = closest_strike(float(spot), strikes_list)
            df_filtered["is_atm"] = df_filtered["strike"].apply(lambda s: (False if pd.isna(s) or atm_strike is None else int(float(s)) == int(float(atm_strike))))
            df_filtered["atm_distance"] = df_filtered["strike"].apply(lambda s: (None if pd.isna(s) or atm_strike is None else abs(float(s) - float(atm_strike))))
        except Exception:
            df_filtered["is_atm"] = False
            df_filtered["atm_distance"] = None

        # attempt IV/delta computation
        df_filtered = _attempt_iv_and_delta_computation(df_filtered, spot=spot)

        # sort sensibly
        try:
            df_filtered["_expiry_key"] = df_filtered["expiry"].apply(lambda x: x or "")
            df_filtered = df_filtered.sort_values(by=["_expiry_key", "strike", "option_type"], ascending=[True, True, True])
            df_filtered = df_filtered.drop(columns=["_expiry_key"])
        except Exception:
            pass

        LOG.debug("full_option_chain: finished for %s rows=%d spot=%s elapsed=%.2fs",
                  symbol_root, len(df_filtered), df_filtered["spot"].iloc[0] if (not df_filtered.empty and "spot" in df_filtered.columns) else None, time.time() - start_ts)

        # ensure columns presence expected by rest of code
        for c in ("tradingsymbol", "expiry", "strike", "option_type", "last_price", "ltp", "iv", "delta", "is_atm", "atm_distance", "spot", "fetched_at", "raw_summary"):
            if c not in df_filtered.columns:
                df_filtered[c] = None

        return df_filtered

    # empty case
    empty_df = pd.DataFrame(columns=["tradingsymbol", "strike", "expiry", "option_type", "last_price", "raw", "instrument_token", "spot", "fetched_at"])
    empty_df["spot"] = float(spot) if spot is not None else None
    empty_df["fetched_at"] = fetched_at
    LOG.debug("full_option_chain: finished for %s rows=0 spot=%s elapsed=%.2fs", symbol_root, empty_df["spot"].iloc[0] if empty_df is not None else None, time.time() - start_ts)
    return empty_df
