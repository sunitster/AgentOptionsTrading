# src/trading/signal_generator.py

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd

from src.live.kite_api import KiteAPI
from src.live import kite_data  # for full_option_chain if defined
from src.trading.iron_condor_builder import build_iron_condor, IronCondor


log = logging.getLogger(__name__)


class SignalGenerator:
    """
    Simple live / paper signal generator for Iron Condors.

    Responsibilities:
    - Optionally pull option chain snapshot (via kite_data.full_option_chain or KiteAPI.get_live_snapshot)
      when the engine does NOT provide one.
    - Build 1 Iron Condor candidate using build_iron_condor().
    - Return it as a list so the engine can iterate over candidates.

    DESIGN (Option A):
    - Preferred: engine passes (chain_df, spot) into generate().
    - Backwards compatible: if chain_df is None, we fetch it internally.
    """

    def __init__(
        self,
        symbol: str = "NIFTY",
        lot_size: int = 25,
        width: int = 150,
        kite_api: Optional[KiteAPI] = None,
        size_aggressiveness: float = 1.0,
    ) -> None:
        self.symbol = symbol
        self.lot_size = lot_size
        self.width = width
        self.size_aggressiveness = size_aggressiveness

        # Keep API available for legacy usage / fallback fetches
        self.api = kite_api or KiteAPI(mode="auto")

    # ------------------------------------------------------------------
    # Core internal helper
    # ------------------------------------------------------------------
    def _get_chain_df(self) -> Optional[pd.DataFrame]:
        """
        Try to fetch a reasonably full option chain.

        Priority:
        1) src.live.kite_data.full_option_chain(api, symbol)
        2) fallback to api.get_live_snapshot(symbol)
        """
        chain_df: Optional[pd.DataFrame] = None

        full_chain_fn = getattr(kite_data, "full_option_chain", None)
        if callable(full_chain_fn):
            try:
                chain_df = full_chain_fn(self.api, symbol=self.symbol)
                if chain_df is not None and not chain_df.empty:
                    return chain_df
            except Exception as e:
                log.warning(f"full_option_chain() failed, falling back to snapshot: {e!r}")

        # Fallback: use basic snapshot (may be synthetic or partial)
        try:
            snap = self.api.get_live_snapshot(self.symbol)
            if snap is not None and not snap.empty:
                return snap
        except Exception as e:
            log.error(f"get_live_snapshot() failed: {e!r}")

        return None

    def _build_candidates(
        self,
        chain_df: Optional[pd.DataFrame] = None,
        spot: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> List[IronCondor]:
        """
        Build one Iron Condor candidate from the latest chain.

        Preferred: engine passes chain_df (and spot if needed by future logic).
        Backwards compatible: if chain_df is None, we fetch it internally.
        """
        now = now or datetime.utcnow()

        # If engine didn't pass a snapshot, fetch it here (legacy behavior)
        if chain_df is None:
            chain_df = self._get_chain_df()

        if chain_df is None or chain_df.empty:
            log.warning("SignalGenerator: no chain data available, returning no candidates.")
            return []

        ic = build_iron_condor(
            chain_df=chain_df,
            symbol=self.symbol,
            width=self.width,
            lot_size=self.lot_size,
            size_aggressiveness=self.size_aggressiveness,
        )

        if ic is None:
            log.info("SignalGenerator: build_iron_condor returned None (not enough strikes).")
            return []

        log.info("SignalGenerator: built IC candidate: %s", asdict(ic))
        return [ic]

    # ------------------------------------------------------------------
    # Public APIs (flexible + backwards compatible)
    # ------------------------------------------------------------------
    def generate(self, *args, **kwargs) -> List[IronCondor]:
        """
        Flexible public function to generate IC candidates.

        Supported call patterns:

        NEW (engine-driven, Option A):
            generate(chain_df, spot)
            generate(chain_df, spot, now=dt)
            generate(chain_df=..., spot=..., now=...)

        OLD (legacy):
            generate()
            generate(now=dt)
            generate(datetime_obj)   # treated as 'now'
        """
        chain_df: Optional[pd.DataFrame] = kwargs.get("chain_df")
        spot: Optional[float] = kwargs.get("spot")
        now: Optional[datetime] = kwargs.get("now")

        if args:
            # If first positional arg looks like a DataFrame, treat as chain_df
            if isinstance(args[0], pd.DataFrame):
                chain_df = args[0]
                if len(args) > 1:
                    spot = args[1]
                if len(args) > 2:
                    now = args[2]
            else:
                # Backwards compatible: generate(now) positional
                now = args[0]

        return self._build_candidates(chain_df=chain_df, spot=spot, now=now)

    def generate_candidates(self, *args, **kwargs) -> List[IronCondor]:
        """
        Backwards-compatible alias.

        Some earlier engines may call .generate_candidates(), so we keep this.
        """
        return self.generate(*args, **kwargs)

    def maybe_generate(self, *args, **kwargs) -> List[IronCondor]:
        """
        Another alias — in case the engine calls .maybe_generate().
        """
        return self.generate(*args, **kwargs)


# Convenience function if you prefer functional style somewhere else
def generate_iron_condor_candidates(
    symbol: str = "NIFTY",
    lot_size: int = 25,
    width: int = 150,
    kite_api: Optional[KiteAPI] = None,
    size_aggressiveness: float = 1.0,
    now: Optional[datetime] = None,
    chain_df: Optional[pd.DataFrame] = None,
    spot: Optional[float] = None,
) -> List[IronCondor]:
    """
    Helper for one-shot candidate generation.

    Backwards compatible:
      - Existing code that only passes (symbol, lot_size, width, kite_api, size_aggressiveness, now)
        continues to work.

    New usage:
      - Pass chain_df and spot when you already have a snapshot.
    """
    sg = SignalGenerator(
        symbol=symbol,
        lot_size=lot_size,
        width=width,
        kite_api=kite_api,
        size_aggressiveness=size_aggressiveness,
    )
    return sg.generate(chain_df=chain_df, spot=spot, now=now)
