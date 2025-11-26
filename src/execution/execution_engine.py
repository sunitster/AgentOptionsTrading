import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import List

from src.live.kite_api import KiteAPI
from src.trading.signal_generator import SignalGenerator

LOG = logging.getLogger("ExecutionEngine")


class ExecutionEngine:
    """
    Simple execution engine for PAPER trading using live Kite option chain snapshots.

    - run_paper_day() will:
        1. fetch chain via KiteAPI.get_option_chain()
        2. generate candidates via SignalGenerator.generate()
        3. validate candidate and place paper orders via kite.place_order()
        4. track PnL as synthetic (entry_premium * lot * quantity)
    """

    def __init__(self, starting_capital=1_000_000, lot_size=25, ic_width=150, cfg_path="config/config.yaml"):
        self.log = LOG
        self.kite = KiteAPI(cfg_path, mode="auto")
        self.capital = float(starting_capital)
        self.lot_size = int(lot_size)
        self.ic_width = int(ic_width)
        self.pnl = 0.0
        self.trades = []
        self.signal_gen = SignalGenerator(ic_width=self.ic_width, lot_size=self.lot_size)
        self.results_dir = "models/llm_trades"
        os.makedirs(self.results_dir, exist_ok=True)

    def _place_paper_ic(self, cand: dict):
        """
        For IC we place 4 legs as paper orders and compute a synthetic order_id.
        """
        qty = cand.get("lot_size", self.lot_size)
        # each leg quantity = lot size
        legs = [
            {"tradingsymbol": cand["short_put"], "option_type": "PE", "side": "SELL", "qty": qty, "price": cand["short_put_p"]},
            {"tradingsymbol": cand["long_put"], "option_type": "PE", "side": "BUY", "qty": qty, "price": cand["long_put_p"]},
            {"tradingsymbol": cand["short_call"], "option_type": "CE", "side": "SELL", "qty": qty, "price": cand["short_call_p"]},
            {"tradingsymbol": cand["long_call"], "option_type": "CE", "side": "BUY", "qty": qty, "price": cand["long_call_p"]},
        ]
        order_ids = []
        for leg in legs:
            # Build tradingsymbol text (we'll use the actual tradingsymbol lookup later if needed)
            ts = f"{leg['option_type']}{leg['qty']}"  # placeholder for clarity
            order = {
                "tradingsymbol": ts,
                "transaction_type": "SELL" if leg["side"] == "SELL" else "BUY",
                "quantity": leg["qty"],
                "price": leg["price"],
                "exchange": self.kite.exchange if hasattr(self.kite, "exchange") else "NFO",
            }
            resp = self.kite.place_order(order)
            order_ids.append(resp.get("order_id"))
        # synthetic pnl: credit * qty * lot
        credit = cand.get("entry_premium", 0.0)
        pnl = credit * qty * 1.0  # premium * lots; adjust multiplier if needed
        trade = {
            "time": datetime.utcnow().isoformat(),
            "candidate": cand,
            "order_ids": order_ids,
            "pnl": pnl,
            "status": "closed" if self.kite.mode == "paper" else "open",
        }
        return trade

    def run_paper_day(self, symbol="NIFTY", max_trades=2):
        """
        Execute a simple daily paper trading routine:
        - fetch option chain
        - generate candidates
        - execute top validated candidate(s) up to max_trades
        - store trades and update capital/pnl
        """
        chain = self.kite.get_option_chain(symbol=symbol, max_strikes=60)
        if chain is None or chain.empty:
            self.log.warning("No option chain available, aborting paper day.")
            return {"note": "no simulation feed found", "feed_path": None, "starting_capital": self.capital}

        candidates = self.signal_gen.generate(chain, max_candidates=6)
        executed = []
        for cand in candidates[:max_trades]:
            if not self.signal_gen.validate(cand, chain):
                self.log.info(f"Candidate failed validation: {cand}")
                continue
            trade = self._place_paper_ic(cand)
            executed.append(trade)
            self.trades.append(trade)
            self.pnl += trade["pnl"]
            self.capital += trade["pnl"]

        # summary
        summary = {
            "note": "paper-day-run",
            "feed_path": None,
            "starting_capital": float(self.capital - self.pnl),
            "ending_capital": float(self.capital),
            "total_pnl": float(self.pnl),
            "trade_count": len(executed),
            "win_rate": None,
            "max_drawdown": None,
            "trades": executed,
        }
        # persist today's trades to file
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_path = os.path.join(self.results_dir, f"paper_trades_{ts}.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, default=str, indent=2)
        self.log.info("Paper trading day complete, summary saved: %s", out_path)
        return summary
