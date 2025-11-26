"""
scoring.py
Unified performance scoring for challenger vs champion.

This module:
- Computes quality scores from BacktestReport objects
- Normalizes pnl, risk, drawdown, sharpe into a single score
- Determines whether challenger should replace champion
"""

import numpy as np


class ScoringEngine:
    """
    Unified scoring model for comparing strategies.

    We compute:
        - total_pnl
        - sharpe_ratio
        - max_drawdown
        - win_rate
        - stability_score (returns autocorrelation)
        - risk_adjusted_pnl

    Then combine into:
        final_score = weighted sum
    """

    def __init__(self,
                 w_pnl=0.40,
                 w_sharpe=0.30,
                 w_drawdown=0.20,
                 w_stability=0.10,
                 dd_penalty=1.5,
                 min_trade_threshold=20):
        self.w_pnl = w_pnl
        self.w_sharpe = w_sharpe
        self.w_drawdown = w_drawdown
        self.w_stability = w_stability

        self.dd_penalty = dd_penalty
        self.min_trade_threshold = min_trade_threshold

    # ----------------------------------------------------------------------
    # Scoring helpers
    # ----------------------------------------------------------------------
    def _norm(self, x):
        """Normalize component scores to avoid explosion."""
        if x is None or np.isnan(x):
            return 0.0
        return float(np.tanh(x / 10000))  # prevents pnl from exploding

    def _neg_norm(self, x):
        """Normalize where lower is better (drawdown)."""
        if x is None or np.isnan(x):
            return 0.0
        return float(1 - np.tanh(x / 5000))

    def compute_score(self, report):
        """
        Computes a unified score from BacktestReport.

        Parameters
        ----------
        report : BacktestReport

        Returns
        -------
        float : score
        """

        pnl = report.total_pnl
        sharpe = report.sharpe
        mdd = report.max_drawdown
        stability = report.stability

        # Minimum trades requirement
        if report.num_trades < self.min_trade_threshold:
            return -9999  # reject extremely low sample strategies

        score = (
            self.w_pnl * self._norm(pnl) +
            self.w_sharpe * sharpe +
            self.w_drawdown * self._neg_norm(mdd * self.dd_penalty) +
            self.w_stability * stability
        )

        return float(score)

    # ----------------------------------------------------------------------
    # Tournament logic: choose champion
    # ----------------------------------------------------------------------
    def challenger_beats_champion(self, challenger_report, champion_report):
        """
        Determine if challenger should replace champion.

        We require:
        - Challenger's score > Champion score * 1.01 (1% improvement)
        - Challenger drawdown ≤ Champion drawdown * 1.25 (risk cap)
        - Challenger must have ≥ minimum trades
        """

        challenger_score = self.compute_score(challenger_report)
        champion_score = self.compute_score(champion_report)

        score_ok = challenger_score > champion_score * 1.01
        dd_ok = challenger_report.max_drawdown <= champion_report.max_drawdown * 1.25
        trades_ok = challenger_report.num_trades >= self.min_trade_threshold

        return score_ok and dd_ok and trades_ok


# =====================================================================
# Scoring class (thin wrapper for Phase-7 integration tests)
# =====================================================================

class Scoring:
    """
    Unified scoring interface expected by Phase-7 tests.
    Wraps the existing functional scoring API.
    """

    def __init__(self):
        pass

    def score(self, challenger_pnl, champion_pnl, challenger_rmse=None, champion_rmse=None):
        """
        Returns a dict with comparison results.
        """
        return {
            "challenger_pnl": challenger_pnl,
            "champion_pnl": champion_pnl,
            "challenger_rmse": challenger_rmse,
            "champion_rmse": champion_rmse,
            "is_better": challenger_pnl > champion_pnl and (
                challenger_rmse is None or challenger_rmse <= champion_rmse
            ),
            "improvement": challenger_pnl - champion_pnl,
        }
