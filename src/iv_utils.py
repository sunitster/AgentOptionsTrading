# src/iv_utils.py
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm

# -------------------------
# Black-Scholes Price
# -------------------------
def bs_price(S, K, T, r, sigma, option_type):
    if T <= 0:
        return max(0, (S-K if option_type=="CE" else K-S))

    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    if option_type == "CE":
        return S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
    else:
        return K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# -------------------------
# Implied Volatility
# -------------------------
def implied_volatility(S, K, T, r, option_type, market_price, max_iter=100):
    sigma = 0.2  # initial guess
    for i in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        vega = (S * norm.pdf((log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))) 
                * sqrt(T))

        if vega < 1e-6:
            return np.nan

        diff = price - market_price
        sigma -= diff / vega

        if abs(diff) < 1e-6:
            return sigma
    return np.nan

# -------------------------
# Helpers for live chain IV / delta (used by src.live.kite_data)
# -------------------------
from datetime import datetime, date

def _year_fraction_from_expiry(expiry, today=None):
    """
    Convert an expiry (str or date) to year fraction for Black-Scholes.
    Very forgiving: handles ISO 'YYYY-MM-DD' and a few common formats.
    """
    if today is None:
        today = date.today()

    if expiry is None:
        return 0.0

    # already a date/datetime
    if isinstance(expiry, datetime):
        ed = expiry.date()
    elif isinstance(expiry, date):
        ed = expiry
    else:
        s = str(expiry)
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y", "%Y/%m/%d", "%d/%m/%Y"):
            try:
                ed = datetime.strptime(s, fmt).date()
                break
            except Exception:
                ed = None
        if ed is None:
            # fallback: assume expiry string is already ISO-ish and try that
            try:
                ed = datetime.fromisoformat(s).date()
            except Exception:
                return 0.0

    days = max((ed - today).days, 0)
    if days <= 0:
        return 0.0
    return days / 365.0


def iv_from_price(price, spot, strike, expiry, option_type="CE", r=0.0):
    """
    Wrapper so src.live.kite_data can call a simple IV-from-price function.

    Parameters match what kite_data._attempt_iv_and_delta_computation passes:
      price  -> option LTP
      spot   -> underlying spot
      strike -> strike
      expiry -> expiry (string or date)
      option_type -> 'CE' or 'PE'
      r      -> risk-free rate (default 0)
    """
    T = _year_fraction_from_expiry(expiry)
    if T <= 0:
        return np.nan
    try:
        return implied_volatility(spot, float(strike), T, float(r), str(option_type).upper(), float(price))
    except Exception:
        return np.nan


def black_scholes_delta(spot, strike, expiry, option_type="CE", r=0.0, price=None):
    """
    Simple Black–Scholes delta, using iv_from_price if sigma not given.
    Used by kite_data if present.
    """
    T = _year_fraction_from_expiry(expiry)
    if T <= 0:
        return np.nan

    # get sigma from price if needed
    sigma = None
    try:
        if price is not None:
            sigma = iv_from_price(price, spot, strike, expiry, option_type=option_type, r=r)
    except Exception:
        sigma = None

    try:
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            sigma = 0.2  # conservative fallback
        d1 = (log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        if str(option_type).upper().startswith("C"):
            return norm.cdf(d1)
        else:
            return -norm.cdf(-d1)
    except Exception:
        return np.nan
