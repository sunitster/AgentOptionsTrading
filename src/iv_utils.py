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
