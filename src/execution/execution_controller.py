"""
execution_controller.py
Phase 7 — Execution Orchestration Layer

This module:
- Accepts a strategy specification (dict from LLM or numeric optimizer)
- Generates trades using the strategy_adapter
- Executes them via ExecutionEngine
- Applies risk rules (risk_engine)
- Generates performance reports (BacktestReport)
- Returns clean structured outputs to training pipeline
"""

import logging
from datetime import datetime

from ..learning.strategy_adapter import StrategyAdapter
from ..execution.execution_engine import ExecutionEngine
from ..learning.backtest_report import BacktestReport
from ..learning.risk_engine import RiskEngine

log = logging.getLogger(__name__)


class ExecutionController:
    """
    High-level controller that:
    1) Builds trades from strategy
    2) Executes trades in ExecutionEngine
    3) Enforces risk rules
    4) Produces a BacktestReport
    """

    def __init__(self, capital=1_000_000, max_daily_dd=0.05, per_trade_risk=0.02):
        self.capital = capital

        # Create submodules
        self.strategy_adapter = StrategyAdapter()
        self.exec_engine = ExecutionEngine(starting_capital=capital)
        self.risk_engine = RiskEngine(
            max_daily_drawdown=max_daily_dd,
            per_trade_risk=per_trade_risk,
            starting_capital=capital,
        )

    def run_backtest(self, strategy_dict, df, features, bounds, sim_kwargs):
        """
        Primary entry point called by run_full_historical_training.py.

        Parameters
        ----------
        strategy_dict : dict
            Final interpreted strategy (from LLM or numeric challenger)
        df : pd.DataFrame
            Validation dataset
        features : list[str]
            Feature list
        bounds : dict
            Hard bounds (iv, dte, moneyness)
        sim_kwargs : dict
            parameters passed through the chain

        Returns
        -------
        report : BacktestReport
        """
        start = datetime.now()
        strat_label = f"{strategy_dict.get('direction')}-{start.strftime('%Y%m%d-%H%M%S')}"

        log.info(f"[ExecutionController] Starting backtest for {strat_label}")

        # ===================================================================
        # 1) Convert LLM strategy into structured trade signals
        # ===================================================================
        log.info("[ExecutionController] Generating trade signals...")
        trades = self.strategy_adapter.generate_trades(
            df=df,
            strategy=strategy_dict,
            features=features,
            bounds=bounds,
            **sim_kwargs,
        )

        if len(trades) == 0:
            log.warning("[ExecutionController] Strategy generated 0 trades.")
            return BacktestReport.empty(label=strat_label)

        # ===================================================================
        # 2) Risk pre-check — position sizing + risk rejection
        # ===================================================================
        log.info(
            f"[ExecutionController] Applying per-trade risk limit "
            f"({self.risk_engine.per_trade_risk * 100:.2f}%)..."
        )
        trades = self.risk_engine.apply_pretrade_risk(trades)

        if len(trades) == 0:
            log.warning("[ExecutionController] All trades rejected by risk engine (pretrade).")
            return BacktestReport.empty(label=strat_label)

        # ===================================================================
        # 3) Execute trades inside market simulator
        # ===================================================================
        log.info("[ExecutionController] Executing trades...")
        fills = self.exec_engine.execute_trades(trades)

        # ===================================================================
        # 4) Daily risk controls — circuit breakers
        # ===================================================================
        log.info("[ExecutionController] Applying daily risk circuit breakers...")
        fills = self.risk_engine.apply_posttrade_limits(fills)

        # ===================================================================
        # 5) Generate backtest report
        # ===================================================================
        log.info("[ExecutionController] Generating backtest report...")

        report = BacktestReport.from_fills(
            fills=fills,
            starting_capital=self.capital,
            label=strat_label,
        )

        dur = datetime.now() - start
        log.info(
            f"[ExecutionController] Backtest complete for {strat_label} "
            f"PnL={report.total_pnl:.2f} "
            f"Duration={dur.total_seconds():.2f}s"
        )

        return report


# ------------------------------------------------------------------------------
# Utility function used by run_full_historical_training.py
# ------------------------------------------------------------------------------
def run_single_strategy(strategy_dict, df, features, bounds, sim_kwargs):
    """
    One-shot backtest runner used in candidate evaluation.
    """
    controller = ExecutionController()
    return controller.run_backtest(strategy_dict, df, features, bounds, sim_kwargs)
