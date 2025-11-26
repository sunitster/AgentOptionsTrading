import time
from datetime import datetime
from src.live.kite_data import KiteData
from src.live.paper_broker import PaperBroker
from src.learning.strategy_adapter import StrategyAdapter
from src.learning.risk_engine import RiskEngine
from src.execution.execution_engine import ExecutionEngine

class LiveRunner:
    def __init__(self, api_key, access_token, strategy_params):
        self.data = KiteData(api_key, access_token)
        self.broker = PaperBroker()
        self.strategy = StrategyAdapter(strategy_params)
        self.risk = RiskEngine()
        self.exec_engine = ExecutionEngine()
        self.last_run_minute = None

    def loop(self):
        print("🚀 Starting LIVE PAPER TRADING")
        while True:
            now = datetime.now()

            if now.minute != self.last_run_minute:  # run once per minute
                self.last_run_minute = now.minute
                self.run_once()

            time.sleep(1)

    def run_once(self):
        # 1. Fetch live OHLC & OC
        chain = self.data.fetch_option_chain("BANKNIFTY")
        spot = self.data.fetch_ltp(260105)  # BANKNIFTY token
        
        # 2. Prepare feature row
        feature_row = self.strategy.prepare_live_features(chain, spot)
        
        # 3. Generate signals
        signal = self.strategy.generate_signal(feature_row)

        # 4. Risk checks
        if not self.risk.allow_trade(self.broker.capital):
            print("⛔ Trade blocked by risk engine")
            return

        # 5. Execute strikes
        trades = self.exec_engine.execute(signal, chain, self.broker)
        print("LIVE TRADES:", trades)
