# backtest_report.py
# Phase 7 — Backtest Analytics & Performance Metrics
# Works with ExecutionEngine + simulate_strategy results

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


# -------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------

def safe_div(a, b):
    return a / b if b not in (0, None) else 0.0


def max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    dd = (equity_curve - roll_max) / roll_max
    return dd.min()  # negative value


def calculate_sharpe(return_series: pd.Series) -> float:
    if return_series.std() == 0:
        return 0.0
    return np.sqrt(252) * return_series.mean() / return_series.std()


def calculate_sortino(return_series: pd.Series) -> float:
    downside = return_series[return_series < 0]
    if downside.std() == 0:
        return 0.0
    return np.sqrt(252) * return_series.mean() / downside.std()


def calculate_calmar(equity_curve: pd.Series) -> float:
    mdd = abs(max_drawdown(equity_curve))
    return safe_div((equity_curve.iloc[-1] / equity_curve.iloc[0] - 1), mdd)


# -------------------------------------------------------------
# Structured results container
# -------------------------------------------------------------
@dataclass
class BacktestReport:
    total_pnl: float
    num_trades: int
    hit_ratio: float
    profit_factor: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    daily_returns: pd.Series
    equity_curve: pd.Series
    trades: pd.DataFrame
    metadata: Dict[str, Any]


# -------------------------------------------------------------
# Main API called by simulator
# -------------------------------------------------------------
def generate_backtest_report(
    trades: pd.DataFrame,
    daily_pnl: pd.Series,
    metadata: Optional[Dict[str, Any]] = None
) -> BacktestReport:

    """
    trades DataFrame must contain:
        ['entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl', 'symbol', 'side']

    daily_pnl is indexed by date.
    """

    metadata = metadata or {}

    # Equity curve
    equity_curve = daily_pnl.cumsum()

    # Daily return series
    capital = metadata.get("initial_capital", 1_000_000)
    daily_returns = daily_pnl / capital

    # Win/loss stats
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]

    hit_ratio = safe_div(len(wins), max(1, len(trades)))

    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())

    profit_factor = safe_div(gross_profit, gross_loss)

    # Ratios
    sharpe = calculate_sharpe(daily_returns)
    sortino = calculate_sortino(daily_returns)
    calmar = calculate_calmar(equity_curve)
    mdd = max_drawdown(equity_curve)

    report = BacktestReport(
        total_pnl=daily_pnl.sum(),
        num_trades=len(trades),
        hit_ratio=hit_ratio,
        profit_factor=profit_factor,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=mdd,
        daily_returns=daily_returns,
        equity_curve=equity_curve,
        trades=trades,
        metadata=metadata,
    )

    return report


# -------------------------------------------------------------
# Pretty printing for logs
# -------------------------------------------------------------
def format_report(report: BacktestReport) -> str:
    return (
        f"Total PnL: {report.total_pnl:,.2f}\n"
        f"Trades: {report.num_trades}\n"
        f"Hit Ratio: {report.hit_ratio:.2%}\n"
        f"Profit Factor: {report.profit_factor:.2f}\n"
        f"Sharpe: {report.sharpe:.2f}\n"
        f"Sortino: {report.sortino:.2f}\n"
        f"Calmar: {report.calmar:.2f}\n"
        f"Max Drawdown: {report.max_drawdown:.2%}\n"
    )
