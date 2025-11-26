# -------------------------
# file: model_evaluator.py
# -------------------------
"""
Model evaluator: runs backtests on candidate models and computes metrics.
This scaffold expects your backtest function to be importable:
from src.apps.iron_condor_v4 import run_backtest_with_regime_model
which should accept a `regime_model` (dict) and return metrics dict with keys: sharpe, max_dd, pnl_series, etc.
If you don't have that function, adapt evaluate() to call your backtest API.
"""
import numpy as np
import pandas as pd
from typing import Dict


def evaluate(regime_model: Dict, backtest_fn) -> Dict:
    """Run backtest_fn(regime_model) -> metrics dict.
    backtest_fn must return a dict with keys 'daily_pnl' (pd.Series or list) at minimum.
    """
    res = backtest_fn(regime_model)
    daily = res.get('daily_pnl')
    if isinstance(daily, (list, tuple)):
        daily = pd.Series(daily)
    elif isinstance(daily, pd.Series):
        pass
    else:
        raise ValueError('backtest_fn must return daily_pnl as list/Series')

    metrics = {
        'sharpe': _sharpe(daily),
        'max_dd': _max_drawdown(daily),
        'total_pnl': float(daily.sum()),
        'daily_count': int(len(daily))
    }
    return {**res, **metrics}


def _sharpe(series: pd.Series, days_per_year: int = 252):
    if series.empty:
        return 0.0
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma == 0:
        return 0.0
    return float(mu / sigma) * (days_per_year ** 0.5)


def _max_drawdown(series: pd.Series):
    if series.empty:
        return 0.0
    cumulative = series.cumsum()
    peak = cumulative.cummax()
    dd = (cumulative - peak)
    return float(dd.min())