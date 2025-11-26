import os
import yaml
import traceback
from typing import Any, Dict, List
from kiteconnect import KiteConnect
import logging

LOG = logging.getLogger(__name__)

# -------------------------------------------------------------
# ALWAYS load config/config.yaml using ABSOLUTE PATH (SAFE)
# -------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")

def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CONFIG NOT FOUND: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


class KiteAPI:
    def __init__(self, mode: str = "live"):
        self.mode = mode.lower()

        cfg = load_config()
        kite_cfg = cfg.get("kite", {})

        self.api_key = kite_cfg.get("api_key", "").strip()
        self.api_secret = kite_cfg.get("api_secret", "").strip()
        self.access_token = kite_cfg.get("access_token", "").strip()
        self.symbol = kite_cfg.get("symbol", "NIFTY").strip()
        self.exchange = kite_cfg.get("exchange", "NFO").strip()

        LOG.info(f"KiteAPI loading config from: {CONFIG_PATH}")
        LOG.info(f"Using api_key={self.api_key}, access_token={self.access_token[:6]}******")

        if self.mode == "live":
            self._init_live()
        elif self.mode == "paper":
            self._init_paper()
        else:
            raise ValueError(f"Invalid mode '{mode}', use 'live' or 'paper'")

    # ---------------------------------------------------------
    # STRICT LIVE INIT — no fallback
    # ---------------------------------------------------------
    def _init_live(self):
        if not self.api_key:
            raise RuntimeError("api_key missing in config/config.yaml")
        if not self.access_token:
            raise RuntimeError("access_token missing in config/config.yaml")

        self.kite = KiteConnect(api_key=self.api_key)
        self.kite.set_access_token(self.access_token)

        try:
            profile = self.kite.profile()
            LOG.info("LIVE API Verified — Logged into Kite.")
            LOG.info(f"User ID: {profile.get('user_id')}")
        except Exception as e:
            LOG.error("LIVE INIT FAILED!")
            LOG.error(traceback.format_exc())
            raise RuntimeError("🚨 LIVE INIT FAILED — Invalid access_token")

        LOG.info("KiteAPI initialized in LIVE mode")

    # ---------------------------------------------------------
    # PAPER MODE
    # ---------------------------------------------------------
    def _init_paper(self):
        self.kite = None
        LOG.info("KiteAPI initialized in PAPER mode (simulated)")

    # ---------------------------------------------------------
    # FETCH LIVE INSTRUMENTS (Index options)
    # ---------------------------------------------------------
    def get_option_chain(self, symbol: str) -> List[Dict[str, Any]]:
        if self.mode != "live":
            raise RuntimeError("get_option_chain() called in PAPER mode")

        try:
            instruments = self.kite.instruments(self.exchange)
        except Exception as e:
            raise RuntimeError(f"Failed fetching instruments: {e}")

        chain = [
            inst for inst in instruments
            if inst["tradingsymbol"].startswith(symbol)
            and inst["instrument_type"] in ("CE", "PE")
        ]

        return chain

    # ---------------------------------------------------------
    # FETCH LIVE LTP SNAPSHOT
    # ---------------------------------------------------------
    def get_live_snapshot(self, symbol: str):
        if self.mode != "live":
            raise RuntimeError("get_live_snapshot() called in PAPER mode")

        chain = self.get_option_chain(symbol)
        tokens = [inst["instrument_token"] for inst in chain if inst["instrument_token"]]

        if not tokens:
            raise RuntimeError("No instrument tokens found for LIVE snapshot")

        quotes = self.kite.ltp(tokens)

        enriched = []
        for inst in chain:
            token = inst["instrument_token"]
            q = quotes.get(token, {})
            inst2 = inst.copy()
            inst2["ltp"] = q.get("last_price") or q.get("ltp") or 0.0
            enriched.append(inst2)

        return enriched
