# src/execution/live_paper_runner.py

import pandas as pd
from src.live.kite_api import KiteAPI
from src.trading.signal_generator import SignalGenerator
from src.execution.execution_engine import ExecutionEngine

class LivePaperRunner:
    def __init__(self, api_key, access_token):
        self.api = KiteAPI(api_key, access_token)
        self.signal_generator = SignalGenerator()
        self.engine = ExecutionEngine(mode="paper", starting_capital=10_00_000)

    def run_once(self):
        snapshot = self.api.get_live_snapshot("BANKNIFTY")
        if snapshot.empty:
            return {"error": "No data fetched"}

        signals = self.signal_generator.generate(snapshot)

        trades, summary = self.engine.run_signals(signals)

        return summary
