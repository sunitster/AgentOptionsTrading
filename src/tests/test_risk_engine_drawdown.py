# src/tests/test_risk_engine_drawdown.py
import pytest
from src.learning.risk_engine import RiskEngine

def test_daily_reset():
    engine = RiskEngine(
        starting_capital=100000,
        max_risk_per_trade=0.02,
        max_daily_drawdown=0.05
    )

    engine.update_after_trade(pnl=-2000)
    assert engine.daily_loss == 2000

    engine.new_day_reset()
    assert engine.daily_loss == 0
    assert engine.halted is False
    assert engine.day_starting_equity == engine.equity

def test_daily_drawdown_trigger():
    engine = RiskEngine(
        starting_capital=100000,
        max_risk_per_trade=0.02,
        max_daily_drawdown=0.10
    )

    # A loss large enough to exceed 10% of starting capital
    engine.update_after_trade(pnl=-15000)

    assert engine.halted is True
    assert engine.equity == 85000
