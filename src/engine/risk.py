# engine/risk.py

"""
risk.py
--------
Contains deterministic, rule-based risk detection logic for the trading engine.

Responsibilities:
- Convert plan + thresholds → individual risk signals.
- Combine signals → risk score (0+).
- Maintain clean and testable risk evaluation logic.
"""

from typing import Dict, Any


# ----------------------------------------------------------
# 1. Compute individual risk signals
# ----------------------------------------------------------

def compute_risk_signals(plan: Dict[str, Any], thresholds: Dict[str, float]) -> Dict[str, int]:
    """
    Returns dictionary of individual binary risk signals.
    Each signal = 1 (risk) or 0 (no risk).
    
    Expected threshold keys:
        ivp_reduce_threshold
        ivp_increase_threshold
        proximity_threshold

    Expected plan keys:
        iv_percentile
        underlying
        underlying_price
        recent_pnl
    """

    ivp = float(plan.get("iv_percentile", 50))
    underlying = float(plan.get("underlying", 0))
    price = float(plan.get("underlying_price", underlying))
    recent_pnl = float(plan.get("recent_pnl", 0))

    t_ivp_reduce = float(thresholds.get("ivp_reduce_threshold", 70))
    t_ivp_increase = float(thresholds.get("ivp_increase_threshold", 20))
    t_proximity = float(thresholds.get("proximity_threshold", 100))

    # ------------------------------------------
    # Signal 1: High IV (reduce width risk)
    # ------------------------------------------
    high_iv_risk = 1 if ivp >= t_ivp_reduce else 0

    # ------------------------------------------
    # Signal 2: Low IV (increase width scenario)
    # ------------------------------------------
    low_iv_risk = 1 if ivp <= t_ivp_increase else 0

    # ------------------------------------------
    # Signal 3: Proximity (ATM = HIGH risk)
    # ------------------------------------------
    proximity = abs(underlying - price)
    proximity_risk = 1 if proximity <= t_proximity else 0

    # ------------------------------------------
    # Signal 4: Recent loss
    # ------------------------------------------
    recent_loss_risk = 1 if recent_pnl < 0 else 0

    return {
        "high_iv": high_iv_risk,
        "low_iv": low_iv_risk,
        "proximity": proximity_risk,
        "recent_loss": recent_loss_risk
    }


# ----------------------------------------------------------
# 2. Combine signals → risk score
# ----------------------------------------------------------

def compute_risk_score(risk_signals: Dict[str, int]) -> int:
    """
    Computes a composite risk score as the sum of all individual risk signals.
    
    Example:
        {'high_iv':1, 'low_iv':0, 'proximity':1, 'recent_loss':0} → risk_score = 2

    This score is used for:
        - Deciding when to call AI
        - Guiding adjustments
        - Logging risk severity
    """
    return sum(int(v) for v in risk_signals.values())


# ----------------------------------------------------------
# 3. Optional (future): verbose diagnostics
# ----------------------------------------------------------

def risk_debug_info(plan: Dict[str, Any], risk_signals: Dict[str, int]) -> Dict[str, Any]:
    """
    Returns structured risk diagnostic data for logs or transparency.
    Not required by engine.py but helpful for debugging.
    """
    return {
        "iv_percentile": plan.get("iv_percentile"),
        "underlying": plan.get("underlying"),
        "underlying_price": plan.get("underlying_price"),
        "recent_pnl": plan.get("recent_pnl"),
        "risk_signals": risk_signals,
        "risk_score": compute_risk_score(risk_signals)
    }
