# src/tests/test_risk_engine_limits.py
import pytest
from src.learning.risk_engine import RiskEngine

def test_equity_updates_on_profit():
    engine = RiskEngine(
        starting_capital=100000,
        max_risk_per_trade=0.02,
        max_daily_drawdown=0.05
    )

    engine.update_after_trade(pnl=+500)
    assert engine.equity == 100500
    assert engine.daily_loss == 0

def test_equity_updates_on_loss():
    engine = RiskEngine(
        starting_capital=100000,
        max_risk_per_trade=0.02,
        max_daily_drawdown=0.05
    )

    engine.update_after_trade(pnl=-800)
    assert engine.equity == 99200
    assert engine.daily_loss == 800

def test_trade_after_exceeding_daily_loss_is_blocked():
    engine = RiskEngine(
        starting_capital=100000,
        max_risk_per_trade=0.02,
        max_daily_drawdown=0.05
    )

    # Max daily loss = 5000  
    engine.update_after_trade(pnl=-6000)

    assert engine.halted is True
    assert engine.can_open_trade(500) is False
