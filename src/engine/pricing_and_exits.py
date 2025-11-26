import math
import random
from dataclasses import dataclass, field
from typing import List
from datetime import datetime, timedelta

# -------------------------
# Black-Scholes & Greeks
# -------------------------
def norm_pdf(x):
    return math.exp(-0.5*x*x) / math.sqrt(2*math.pi)

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_price_call(S, K, r, q, sigma, t):
    if t <= 0 or sigma <= 0:
        return max(0.0, S*math.exp(-q*t) - K*math.exp(-r*t))
    d1 = (math.log(S/K) + (r - q + 0.5 * sigma*sigma) * t) / (sigma*math.sqrt(t))
    d2 = d1 - sigma*math.sqrt(t)
    return S*math.exp(-q*t)*norm_cdf(d1) - K*math.exp(-r*t)*norm_cdf(d2)

def bs_price_put(S, K, r, q, sigma, t):
    # put-call parity
    return bs_price_call(S,K,r,q,sigma,t) - S*math.exp(-q*t) + K*math.exp(-r*t)

def bs_greeks(S,K,r,q,sigma,t, kind='call'):
    if t <= 0 or sigma <= 0:
        return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
    d1 = (math.log(S/K) + (r - q + 0.5*sigma*sigma)*t) / (sigma*math.sqrt(t))
    d2 = d1 - sigma*math.sqrt(t)
    pdf_d1 = norm_pdf(d1)
    delta = math.exp(-q*t) * norm_cdf(d1) if kind=='call' else math.exp(-q*t)*(norm_cdf(d1)-1)
    gamma = math.exp(-q*t) * pdf_d1 / (S*sigma*math.sqrt(t))
    vega = S * math.exp(-q*t) * pdf_d1 * math.sqrt(t) / 100.0   # per 1 vol point = 1%
    theta = (-S*sigma*math.exp(-q*t)*pdf_d1/(2*math.sqrt(t)) - r*K*math.exp(-r*t)*norm_cdf(d2) + q*S*math.exp(-q*t)*norm_cdf(d1))
    if kind=='put':
        theta = (-S*sigma*math.exp(-q*t)*pdf_d1/(2*math.sqrt(t)) + r*K*math.exp(-r*t)*norm_cdf(-d2) - q*S*math.exp(-q*t)*norm_cdf(-d1))
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}

# -------------------------
# IV path & smile
# -------------------------
def update_iv(current_iv, base_iv, mean_reversion, daily_vol, spike_prob, spike_size):
    # Ornstein-Uhlenbeck style simple step
    reversion = mean_reversion * (base_iv - current_iv)
    shock = random.gauss(0, daily_vol)
    if random.random() < spike_prob:
        shock += random.choice([-1,1]) * spike_size
    next_iv = max(0.01, current_iv + reversion + shock)
    return next_iv

def strike_iv(base_iv, strike, spot, slope=0.4):
    # simple linear skew: slope is fraction difference per unit moneyness
    moneyness = (strike - spot) / spot
    return max(0.01, base_iv * (1 + slope * moneyness))

# -------------------------
# Price a spread (list of legs)
# -------------------------
@dataclass
class Leg:
    kind: str       # 'call' or 'put'
    strike: float
    qty: int        # positive = long, negative = short
    expiry: datetime

@dataclass
class Trade:
    legs: List[Leg]
    entry_date: datetime
    entry_iv: float
    credit: float
    highest_profit: float = field(default=-9e9)
    open: bool = True
    days_held: int = 0

def price_leg(leg: Leg, spot, r, q, iv, today):
    t = max(0.0, (leg.expiry - today).days / 365.0)
    if leg.kind == 'call':
        return bs_price_call(spot, leg.strike, r, q, iv, t)
    else:
        return bs_price_put(spot, leg.strike, r, q, iv, t)

def price_trade_mid(trade: Trade, spot, r, q, iv_surface_fn, today):
    total = 0.0
    for leg in trade.legs:
        iv = iv_surface_fn(leg.strike, spot)
        mid = price_leg(leg, spot, r, q, iv, today)
        total += leg.qty * mid
    return total

# -------------------------
# Exit logic (SL, TP, trailing, min_hold, iv suppression)
# -------------------------
def check_exit(trade: Trade, current_mtm, params, current_iv, today):
    # profit measured as credit - mtm_for_position (credit received positive)
    # If credited (short premium), a positive profit is credit - mtm; adapt sign to your convention.
    profit = trade.credit - current_mtm
    trade.days_held += 1

    # update highest profit
    trade.highest_profit = max(trade.highest_profit, profit)

    # Min hold days (prevent same-day exit)
    if trade.days_held <= params['min_hold_days']:
        return False, 'min_hold_days'

    # TP
    if profit >= params['tp_pct'] * trade.credit:
        return True, 'tp'

    # Trailing logic
    if profit >= params['tp_pct'] * trade.credit * params['trail_start_pct']:
        # start trailing
        if profit <= trade.highest_profit * (1.0 - params['trail_pct']):
            return True, 'trailing_stop'

    # SL suppressed if IV spike above threshold
    if current_iv > trade.entry_iv * params['iv_exit_suppression']:
        # skip SL for today
        return False, 'iv_suppressed'

    # SL check (loss relative to credit)
    if profit <= -params['sl_multiplier'] * trade.credit:
        return True, 'sl'

    # otherwise keep
    return False, 'keep'
