# Full corrected live_engine.py

import logging
import os
import pickle
from typing import Iterable, Dict, Any
from uuid import uuid4

from src.deploy.model_registry import ModelRegistry
from src.learning.observability import Observability
from src.execution.execution_controller import ExecutionController
from src.execution.execution_engine import ExecutionEngine
from src.learning.risk_engine import RiskEngine

logger = logging.getLogger("live_engine")
logger.setLevel(os.environ.get("LIVE_ENGINE_LOGLEVEL", "INFO"))

# observability
obs = Observability(run_id=str(uuid4()))

# global registry
_registry = ModelRegistry()


def load_model(name: str):
    entry = _registry.latest(name)
    return entry


class LiveEngine:
    def __init__(self, capital: float = 100000, model_name: str = "lightgbm_champion"):
        self.capital = capital
        self.model_name = model_name

        self.model_meta = load_model(model_name)
        if not self.model_meta:
            raise RuntimeError(f"No model found for {model_name}")

        path = self.model_meta.get("path") or self.model_meta.get("artifact_path")
        if not path:
            raise RuntimeError(f"Model entry missing path: {self.model_meta}")

        self.model = self._load_model(path)

        # execution
        self.exec_engine = ExecutionEngine()
        self.controller = ExecutionController(self.exec_engine)

        # risk
        self.risk = RiskEngine(
            starting_capital=capital,
            max_risk_per_trade=0.02,
            max_daily_drawdown=0.1,
        )

        obs.event("live_engine.init", {"model": self.model_meta})

    def _load_model(self, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict_row(self, row: Dict[str, Any]):
        try:
            vals = list(row.values())
            pred = self.model.predict([vals])
            return float(pred[0])
        except Exception:
            if callable(self.model):
                return float(self.model(row))
            raise

    def handle_stream(self, rows: Iterable[Dict[str, Any]]):
        out = []

        for r in rows:
            try:
                p = self.predict_row(r)
                obs.event("live_engine.prediction", {"pred": p, "row_ts": r.get("date")})

                decisions = self.controller.generate_trade_instructions_from_prediction(
                    r, p, capital=self.capital
                )

                approved = []
                for d in decisions:
                    ok = self.risk.can_open_trade(risk_amount=d.get("notional", 0))
                    if not ok:
                        obs.warning("live_engine.trade_rejected_by_risk", {"decision": d})
                        continue

                    res = self.exec_engine.place_order_paper(d)
                    self.risk.update_after_trade(res.get("pnl", 0))

                    obs.event("live_engine.executed", {"decision": d, "result": res})
                    approved.append(res)

                out.extend(approved)

            except Exception as e:
                obs.error("live_engine.error", {"error": str(e)})
                logger.exception("error in live loop")

        return out
