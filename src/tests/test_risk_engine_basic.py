# src/tests/test_risk_engine_basic.py
import pytest
from src.learning.risk_engine import RiskEngine

def test_initial_state():
    engine = RiskEngine(
        starting_capital=100000,
        max_risk_per_trade=0.02,
        max_daily_drawdown=0.05
    )

    assert engine.equity == 100000
    assert engine.day_starting_equity == 100000
    assert engine.daily_loss == 0
    assert engine.halted is False


def test_can_open_trade_under_risk_limit():
    engine = RiskEngine(
        starting_capital=100000,
        max_risk_per_trade=0.02,
        max_daily_drawdown=0.05
    )

    # Trade risking 1% (allowed)
    allowed = engine.can_open_trade(risk_amount=1000)
    assert allowed is True

def test_rejects_trade_exceeding_risk_limit():
    engine = RiskEngine(
        starting_capital=50000,
        max_risk_per_trade=0.02,
        max_daily_drawdown=0.05
    )

    # Max risk = 1000; this one risks 2000 → reject
    allowed = engine.can_open_trade(risk_amount=2000)
    assert allowed is False
