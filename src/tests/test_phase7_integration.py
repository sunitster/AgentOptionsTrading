"""
Phase 7 Integration Test
------------------------

Verifies the FULL pipeline works:

    StrategyAdapter -> BacktestReport -> RiskEngine ->
    ExecutionEngine -> Scoring

This test uses synthetic data (very small) so it runs fast.
"""

import pandas as pd
import numpy as np

from src.learning.strategy_adapter import StrategyAdapter
from src.learning.backtest_report import BacktestReport
from src.execution.execution_engine import ExecutionEngine
from src.execution.execution_controller import ExecutionController
from src.learning.scoring import Scoring
from src.learning.risk_engine import RiskEngine


def test_phase7_integration():
    # -----------------------------
    # 1. Create synthetic features
    # -----------------------------
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "iv": [0.22, 0.18, 0.25, 0.20, 0.19],
        "dte": [25, 21, 23, 20, 19],
        "moneyness": [1.00, 0.98, 1.02, 1.01, 0.99],
        "close": [100, 102, 101, 103, 104],
        "target": [0.3, -0.1, 0.4, -0.2, 0.1],  # small PnL targets
    })

    # -----------------------------
    # 2. Use a simple strategy
    # -----------------------------
    strat = {
        "direction": "bear",
        "min_iv": 0.15,
        "max_iv": 0.30,
        "size_aggressiveness": 1.0,
        "dte_min": 15,
        "dte_max": 30,
    }

    adapter = StrategyAdapter()
    features, filters = adapter.convert(strat, df)

    # Basic correctness
    assert isinstance(filters, dict)
    assert "iv_filter" in filters

    # -----------------------------------------
    # 3. Run backtest (PnL calculation)
    # -----------------------------------------
    backtest = BacktestReport(df, features=features, filters=filters)
    report = backtest.run()

    assert "total_pnl" in report
    assert isinstance(report["total_pnl"], (int, float))

    # -----------------------------------------
    # 4. Execution preparation
    # -----------------------------------------
    exec_engine = ExecutionEngine()
    controller = ExecutionController(exec_engine)

    decisions = controller.generate_trade_instructions(
        features,
        filters,
        capital=100000
    )

    assert isinstance(decisions, list)
    assert len(decisions) > 0
    assert "size" in decisions[0]

    # -----------------------------------------
    # 5. Risk Engine check
    # -----------------------------------------
    risk = RiskEngine(max_daily_loss_pct=0.02, max_position_pct=0.02)
    evaluated = risk.apply(decisions)

    assert isinstance(evaluated, list)
    assert len(evaluated) == len(decisions)
    assert "approved" in evaluated[0]

    # -----------------------------------------
    # 6. Scoring: evaluate final strategy winner
    # -----------------------------------------
    score = Scoring.compare(
        challenger_pnl=report["total_pnl"],
        champion_pnl=50000  # synthetic champion
    )

    assert "is_winner" in score

    # -----------------------------------------
    # If we reached here, Phase 7 works
    # -----------------------------------------
    print("\n[Phase 7 Integration Test] SUCCESS — Full pipeline works.\n")
