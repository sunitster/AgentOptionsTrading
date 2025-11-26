# src/execution/live_ic_runner.py
"""
Runner that ties Kite API snapshot -> SignalGenerator -> ExecutionEngine (paper)
and logs results to models/live_paper/ic_run_{date}.json
"""

import os
import json
import datetime
import logging
from pathlib import Path

import yaml
from src.live.kite_api import KiteAPI
from src.trading.signal_generator import SignalGenerator
from src.execution.execution_engine import ExecutionEngine  # existing
from src.utils.io import ensure_dir_exists  # optional; or use os.makedirs

log = logging.getLogger(__name__)

# config
ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / 'config' / 'config.yaml'
LOG_DIR = ROOT / 'models' / 'live_paper' / 'iron_condors'


def read_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


class LiveICRunner:
    def __init__(self, api_key: str = None, access_token: str = None, exchange: str = 'NFO'):
        cfg = read_config()
        kite_cfg = cfg.get('kite', {}) if cfg else {}
        self.api_key = api_key or kite_cfg.get('api_key')
        self.access_token = access_token or kite_cfg.get('access_token')
        self.exchange = exchange or kite_cfg.get('exchange', 'NFO')

        if not self.api_key or not self.access_token:
            raise ValueError("kite api_key and access_token required in config or constructor")

        self.kite = KiteAPI(api_key=self.api_key, access_token=self.access_token)
        self.signaller = SignalGenerator(width=150, lot_size=1, lots_per_leg=15)
        # paper execution engine instance
        self.engine = ExecutionEngine(starting_capital=1_000_000, mode='paper') if 'mode' in ExecutionEngine.__init__.__code__.co_varnames else ExecutionEngine(starting_capital=1_000_000)
        ensure_dir_exists(LOG_DIR)

    def run_once(self, forced_strategy: dict = None):
        snapshot = self.kite.get_live_snapshot(underlying=self.kite.kite_instrument if hasattr(self.kite, 'kite_instrument') else 'BANKNIFTY')
        if snapshot is None or snapshot.empty:
            log.warning("No snapshot returned from Kite")
            return {'note': 'no snapshot'}

        signals = self.signaller.generate(snapshot, forced_strategy=forced_strategy)

        # convert signals to DataFrame expected by ExecutionEngine.run_signals (if engine expects DataFrame)
        # Here we'll attempt to pass signals directly if engine.run_signals supports list/dict input.
        try:
            trades, summary = self.engine.run_signals(signals)
        except TypeError:
            # fallback: if engine expects DataFrame, create a simple DataFrame with leg info
            rows = []
            for s in signals:
                if s['action'] != 'ENTER':
                    continue
                for leg in s['ic']['legs']:
                    rows.append({
                        'action': 'PLACE',
                        'tradingsymbol': leg['tradingsymbol'],
                        'side': leg['side'],
                        'lots': leg['lots'],
                        'price': leg['price'],
                        'instrument_token': leg.get('instrument_token', None),
                        'meta': s['ic']['meta']
                    })
            import pandas as pd
            df = pd.DataFrame(rows)
            trades, summary = self.engine.run_signals(df)

        # log results
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        fname = LOG_DIR / f"ic_run_{ts}.json"
        with open(fname, 'w') as f:
            json.dump({'timestamp': ts, 'signals': signals, 'summary': summary}, f, indent=2, default=str)

        return summary
