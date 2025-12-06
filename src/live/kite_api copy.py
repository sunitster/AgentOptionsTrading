"""
src/live/kite_api.py

Defensive, backward-compatible Kite API wrapper.

- Loads config from project-root/config/config.yaml by default.
- Exposes: get_spot, quote, get_option_chain, get_chain_snapshot, cfg property.
- Uses fixtures (_test_quotes, _test_chain) when live client is unavailable.
- Masks secrets in logs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

LOG = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


class KiteAPIError(RuntimeError):
    pass


class KiteAPI:
    def __init__(self, config_path: Optional[str | Path] = None, mode: str = "live"):
        self.mode = (mode or "live").lower()
        # default config path -> project-root/config/config.yaml
        if config_path:
            self.config_path = Path(config_path).resolve()
        else:
            # __file__ -> src/live/kite_api.py
            # parents[2] -> project-root
            self.config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

        self._cfg: Dict[str, Any] = {}
        self._client = None
        self._kite_class = None
        self.user_id = None
        self.allow_spot_fallback = False

        self._load_config()
        self._init_client()

    # -------------------------
    # Config loader
    # -------------------------
    def _load_config(self) -> None:
        if not self.config_path.exists():
            LOG.error("KiteAPI loading config from: %s", str(self.config_path))
            LOG.error("KiteAPI: config not found at %s", str(self.config_path))
            self._cfg = {}
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            # mask tokens for logs
            if isinstance(cfg, dict):
                if cfg.get("access_token"):
                    cfg["access_token_masked"] = ("*" * 6) + str(cfg.get("access_token"))[-4:]
                if cfg.get("api_key"):
                    cfg["api_key_masked"] = ("*" * 6) + str(cfg.get("api_key"))[-4:]
            self._cfg = cfg
            self.user_id = cfg.get("user_id")
            self.allow_spot_fallback = bool(cfg.get("allow_spot_fallback", False))
            LOG.info("KiteAPI loading config from: %s", str(self.config_path))
        except Exception as e:
            LOG.exception("KiteAPI: failed to read/parse config: %s", e)
            self._cfg = {}

    # -------------------------
    # Client init (deferred network calls)
    # -------------------------
    def _init_client(self) -> None:
        # Try to import KiteConnect if available
        try:
            from kiteconnect import KiteConnect  # type: ignore
            self._kite_class = KiteConnect
            LOG.debug("KiteAPI: kiteconnect available")
        except Exception:
            LOG.info("KiteAPI: kiteconnect not available; running in PAPER/offline mode")
            self._kite_class = None

        api_key = self._cfg.get("api_key")
        access_token = self._cfg.get("access_token")

        if self._kite_class and self.mode == "live":
            if not api_key or not access_token:
                LOG.warning("KiteAPI: api_key or access_token missing; cannot initialize LIVE client")
                self._client = None
                return
            try:
                client = self._kite_class(api_key=api_key)
                client.set_access_token(access_token)
                self._client = client
                LOG.info("KiteAPI initialized (LIVE). api_key=%s user_id=%s", self._cfg.get("api_key_masked"), self.user_id)
                return
            except Exception as e:
                LOG.exception("KiteAPI: failed to initialize LIVE client: %s", e)
                self._client = None
                return

        # PAPER / offline
        LOG.info("KiteAPI initialized (shim). mode=%s user_id=%s", self.mode, self.user_id)
        self._client = None

    # -------------------------
    # Public config property (user expected this)
    # -------------------------
    @property
    def cfg(self) -> Dict[str, Any]:
        return self._cfg

    # -------------------------
    # quote / spot helpers
    # -------------------------
    def quote(self, tradingsymbol: str) -> Optional[Dict[str, Any]]:
        """
        Return quote dict for tradingsymbol. Prefers live client, then fixtures.
        Returns None if quote unavailable.
        """
        # Live client path
        if self._client:
            try:
                # KiteConnect.quote accepts a string or list in different versions
                try:
                    resp = self._client.quote(tradingsymbol)
                except TypeError:
                    resp = self._client.ltp([tradingsymbol])
                if isinstance(resp, dict):
                    # resp might be nested under the symbol
                    if tradingsymbol in resp and isinstance(resp[tradingsymbol], dict):
                        return resp[tradingsymbol]
                    # direct fields
                    return resp
            except Exception as e:
                LOG.debug("KiteAPI.quote live fetch failed for %s: %s", tradingsymbol, e)

        # fixture fallback
        fixtures = self._cfg.get("_test_quotes") or {}
        for prefix, v in fixtures.items():
            # if tradingsymbol starts with fixture key, return it
            if str(tradingsymbol).startswith(str(prefix)):
                if isinstance(v, dict):
                    return v
                else:
                    return {"last_price": v}
        return None

    def get_spot(self, symbol: str) -> Optional[float]:
        """
        Return numeric spot price for symbol (e.g., 'NIFTY').
        Returns None if unavailable and fallback disabled.
        """
        # Try live client first
        if self._client:
            try:
                # try common query forms
                candidates = [symbol, f"NSE:{symbol}", f"NFO:{symbol}"]
                for cand in candidates:
                    try:
                        q = self.quote(cand)
                        if q and ("last_price" in q or "ltp" in q):
                            val = q.get("last_price") or q.get("ltp")
                            return float(val)
                    except Exception:
                        continue
                raise RuntimeError("no usable quote from client")
            except Exception as e:
                LOG.warning("KiteAPI.get_spot: LIVE fetch failed for %s: %s", symbol, e)

        # fixture _test_quotes
        fixtures = self._cfg.get("_test_quotes") or {}
        if fixtures and symbol in fixtures:
            v = fixtures[symbol]
            if isinstance(v, dict) and "last_price" in v:
                try:
                    return float(v["last_price"])
                except Exception:
                    pass
            if isinstance(v, (int, float)):
                return float(v)

        # fallback behavior
        if self.allow_spot_fallback:
            fallback_value = self._cfg.get("spot_fallback_value")
            if fallback_value is not None:
                try:
                    return float(fallback_value)
                except Exception:
                    pass
            LOG.warning("KiteAPI.get_spot: allow_spot_fallback enabled but no fallback value; returning 0.0")
            return 0.0

        return None

    # -------------------------
    # Option chain helpers
    # -------------------------
    def get_option_chain(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """
        Return list of normalized option entries for the symbol (LIVE) or fixture (PAPER).
        Each entry is a dict with keys like 'tradingsymbol','strike','option_type','expiry','ltp'.
        """
        # LIVE client path (attempt, but keep defensive)
        if self._client:
            try:
                # Try to fetch instruments for exchange 'NFO' if available
                try:
                    instruments = self._client.instruments("NFO")
                except Exception:
                    # some versions expect no args
                    instruments = self._client.instruments()

                # Filter instruments matching the symbol name and option type
                rows = []
                for inst in instruments:
                    # instrument dict shapes vary; try common keys
                    name = inst.get("name") or inst.get("tradingsymbol") or inst.get("symbol")
                    if not name:
                        continue
                    # match by provided symbol string (e.g., NIFTY)
                    if str(name).upper().startswith(str(symbol).upper()):
                        rows.append(inst)

                # Enrich rows with quotes
                if rows:
                    ts = [r.get("tradingsymbol") or r.get("tradingsymbol", "") for r in rows]
                    # fetch quotes in batches
                    quotes = {}
                    try:
                        qresp = self._client.quote(ts)
                        if isinstance(qresp, dict):
                            quotes = qresp
                    except Exception:
                        quotes = {}

                    out = []
                    for r in rows:
                        tsym = r.get("tradingsymbol") or r.get("symbol") or r.get("name")
                        q = quotes.get(tsym, {}) if isinstance(quotes, dict) else {}
                        item = {
                            "tradingsymbol": tsym,
                            "strike": r.get("strike"),
                            "option_type": r.get("instrument_type") or r.get("segment") or r.get("option_type"),
                            "expiry": r.get("expiry"),
                            "ltp": q.get("last_price") or q.get("ltp") or None,
                            "raw": r,
                        }
                        out.append(item)
                    return out
            except Exception as e:
                LOG.debug("KiteAPI.get_option_chain: live fetch failed: %s", e)

        # fixture path
        fixtures = self._cfg.get("_test_chain") or {}
        if fixtures and symbol in fixtures:
            # fixture may be list or dict
            ch = fixtures[symbol]
            if isinstance(ch, list):
                return ch
            if isinstance(ch, dict):
                # convert dict values to list
                try:
                    return list(ch.values())
                except Exception:
                    return None
        return None

    def get_chain_snapshot(self, symbol_root: str) -> Dict[str, Dict[str, Any]]:
        """
        Normalize option chain into a dict keyed by tradingsymbol.
        This is what src.live.kite_data.full_option_chain expects.
        """
        # Prefer test fixture (makes offline tests deterministic)
        fixtures = self._cfg.get("_test_chain") or {}
        if fixtures and symbol_root in fixtures:
            ch = fixtures[symbol_root]
            if isinstance(ch, dict):
                return {str(k): v for k, v in ch.items()}
            if isinstance(ch, list):
                out = {}
                for ent in ch:
                    if isinstance(ent, dict):
                        ts = ent.get("tradingsymbol") or ent.get("symbol")
                        if ts:
                            out[str(ts)] = ent
                return out

        # Live client path: try get_option_chain then normalize
        oc = self.get_option_chain(symbol_root)
        if oc:
            out = {}
            for ent in oc:
                if not isinstance(ent, dict):
                    continue
                ts = ent.get("tradingsymbol") or ent.get("symbol")
                key = str(ts) if ts else None
                if not key:
                    # fallback: try to create key from option_type+strike
                    try:
                        strike = ent.get("strike")
                        otype = ent.get("option_type") or ent.get("instrument_type") or ""
                        key = f"{symbol_root}_{str(otype)}_{int(float(strike))}" if strike is not None else None
                    except Exception:
                        key = None
                if key:
                    out[key] = ent
            return out

        # Nothing available -> return empty dict
        return {}

    # ---------------------------------------------------------------------
    # Convenience representation
    # ---------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<KiteAPI mode={self.mode} user={self.user_id or 'unknown'} cfg_keys={list(self._cfg.keys())}>"

# small CLI test if run directly
if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--spot", default=None)
    p.add_argument("--chain", default=None)
    args = p.parse_args()
    api = KiteAPI(config_path=args.config if args.config else None, mode="live")
    if args.spot:
        print("spot:", api.get_spot(args.spot))
    if args.chain:
        print(json.dumps(api.get_chain_snapshot(args.chain), indent=2))
