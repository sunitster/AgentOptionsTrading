# src/learning/backtest_adapter.py

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import datetime as dt
from src.apps.iron_condor_v4 import run_backtest_with_regime_model


def backtest_fn(regime_model: dict = None,
                start: dt.date = None,
                end: dt.date = None,
                weekly: bool = True):
    """
    Phase-5 compliant backtest wrapper.

    - regime_model: JSON dict of rule model
    - start/end:   date range for testing
    - weekly:      whether to create weekly plans (default True)

    Returns:
        {
            "daily_pnl": [...float...],
            "dates": [...ISO date strings...],
            "summary": {
                "total_pnl": float,
                "sharpe": float,
                "max_dd": float,
                "winning_days_pct": float
            }
        }
    """

    if start is None or end is None:
        raise ValueError("backtest_fn requires start and end date.")

    try:
        res = run_backtest_with_regime_model(
            regime_model,
            start=start,
            end=end,
            weekly=weekly
        )
    except Exception as e:
        print("ERROR from run_backtest_with_regime_model:", e)
        return {
            "daily_pnl": [],
            "dates": [],
            "summary": {"total_pnl": 0, "sharpe": 0, "max_dd": 0, "winning_days_pct": 0}
        }

    daily = res.get("daily_pnl", [])
    dates = res.get("dates", [])

    if len(daily) > 1:
        import numpy as np
        pnl = np.array(daily)
        total = float(pnl.sum())
        sharpe = float((pnl.mean() / (pnl.std() + 1e-9)) * (252 ** 0.5))

        curve = pnl.cumsum()
        high = np.maximum.accumulate(curve)
        dd = high - curve
        max_dd = float(dd.max())

        win_pct = float((pnl > 0).mean())
    else:
        sharpe = 0
        total = 0
        max_dd = 0
        win_pct = 0

    summary = {
        "total_pnl": total,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "winning_days_pct": win_pct
    }

    return {
        "daily_pnl": daily,
        "dates": dates,
        "summary": summary
    }


if __name__ == "__main__":
    print("Testing backtest_adapter…")
    rm = {"rules": {}}
    out = backtest_fn(
        rm,
        start=dt.date(2025, 1, 1),
        end=dt.date(2025, 1, 10)
    )
    print(out["summary"])
