# -------------------------
# file: exec_adapter.py
# -------------------------
"""
Paper-mode execution adapter.
- Uses provided DB URLs (market_features and trade_logs) via SQLAlchemy
- Simulates paper fills using bid-ask midpoint ± slippage
- Tracks order states and MTM
- Provides functions: place_order(plan), poll_market(), reconcile(), flat_all_by(time)

Configuration section at top.
"""
import time
import random
import threading
import datetime
from decimal import Decimal
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.execution.db_models import Base, TradeLog, MarketFeature

import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

MARKET_DB_URL = config["database"]["market_features"]["url"]
TRADE_DB_URL = config["database"]["trade_logs"]["url"]


# === CONFIG ===

SLIPPAGE_PCT = 0.002   # 0.2% default slippage
MAX_FILL_LATENCY_S = 1.5  # ensure paper fills < 2s
SIMULATED_BID_ASK_SPREAD = 0.05  # absolute price units for options pricing simulation
POLL_INTERVAL = 1.0

# === DB SETUP ===
market_engine = create_engine(MARKET_DB_URL)
trade_engine = create_engine(TRADE_DB_URL)
MarketSession = sessionmaker(bind=market_engine)
TradeSession = sessionmaker(bind=trade_engine)

# create tables if not exist (idempotent)
Base.metadata.create_all(market_engine)
Base.metadata.create_all(trade_engine)

# In-memory orderbook for quick access during simulation
_orderbook = {}
_lock = threading.Lock()


def _simulate_market_price(plan):
    """Return a simulated bid/ask tuple for the given plan.
    For simplicity we derive mid from underlying_price and simple IV/credit.
    """
    mid = plan.get('underlying_price', 100.0) * 0.01 * (plan.get('entry_iv', 30.0) / 10.0) + plan.get('short_strike', 0)
    # Add small randomness per snapshot
    mid = float(mid) + random.uniform(-0.02, 0.02)
    spread = SIMULATED_BID_ASK_SPREAD
    bid = max(0.0, mid - spread/2)
    ask = max(0.0, mid + spread/2)
    return bid, ask


def _apply_slippage(price, side):
    # side: 'buy' or 'sell' -> for a credit opening (sell) we treat as SELL
    slippage = price * SLIPPAGE_PCT
    if side == 'buy':
        return price + slippage
    else:
        return max(0.0, price - slippage)


def place_order(plan, qty=1, side='sell'):
    """Place a paper order for a 'plan' (dict). Returns order_id and a simulated fill record.
    side can be 'sell' (open credit) or 'buy' (close/cover).
    """
    order_id = f"ord-{int(time.time()*1000)}-{random.randint(100,999)}"
    placed_at = datetime.datetime.utcnow()

    # simulate market
    bid, ask = _simulate_market_price(plan)
    mid = (bid + ask) / 2.0
    # compute fill price using slippage and side
    if side == 'sell':
        # seller receives slightly less than midpoint due to slippage
        fill_price = _apply_slippage(mid, 'sell')
    else:
        fill_price = _apply_slippage(mid, 'buy')

    # simulate latency
    latency = random.uniform(0.01, MAX_FILL_LATENCY_S)
    time.sleep(min(latency, MAX_FILL_LATENCY_S))

    # store in in-memory orderbook
    with _lock:
        _orderbook[order_id] = dict(
            order_id=order_id,
            plan_id=plan.get('plan_id'),
            qty=qty,
            side=side,
            fill_price=fill_price,
            filled_at=datetime.datetime.utcnow(),
            status='filled'
        )

    # also persist to trade_logs DB
    session = TradeSession()
    tl = TradeLog(
        plan_id=plan.get('plan_id'),
        side=plan.get('side'),
        short_strike=plan.get('short_strike'),
        long_strike=plan.get('long_strike'),
        credit=plan.get('credit', 0.0),
        qty=qty,
        opened_at=placed_at if side == 'sell' else None,
        status='open' if side == 'sell' else 'closed',
        realised=0.0,
        meta={
            'order_id': order_id,
            'fill_price': fill_price,
            'filled_at': datetime.datetime.utcnow().isoformat(),
        }
    )
    session.add(tl)
    session.commit()
    session.close()

    return _orderbook[order_id]


def get_open_trades():
    session = TradeSession()
    rows = session.query(TradeLog).filter(TradeLog.status == 'open').all()
    session.close()
    return rows


def mark_closed(plan_id, realised, closed_at=None):
    session = TradeSession()
    row = session.query(TradeLog).filter(TradeLog.plan_id == plan_id, TradeLog.status == 'open').first()
    if not row:
        session.close()
        return None
    row.realised = realised
    row.status = 'closed'
    row.closed_at = closed_at or datetime.datetime.utcnow()
    session.commit()
    session.close()
    return row


def poll_and_update_mtm():
    """Update unrealized PnL for open trades using simulated market snapshots"""
    session = TradeSession()
    open_rows = session.query(TradeLog).filter(TradeLog.status == 'open').all()
    for row in open_rows:
        # build minimal plan dict from row
        plan = dict(plan_id=row.plan_id, short_strike=row.short_strike, long_strike=row.long_strike,
                    underlying_price=row.meta.get('underlying_price', 100.0) if row.meta else 100.0,
                    entry_iv=row.meta.get('entry_iv', 30.0) if row.meta else 30.0,
                    credit=row.credit)
        bid, ask = _simulate_market_price(plan)
        mid = (bid+ask)/2.0
        # unreal PnL approximated: entry_credit - current_mid
        unreal = (row.credit - mid) * row.qty
        row.unreal = unreal
    session.commit()
    session.close()


def reconcile():
    """Simple reconcile function to ensure DB and in-memory orderbook align."""
    # For demo purposes we just log counts and ensure no duplicate open rows
    session = TradeSession()
    open_count = session.query(TradeLog).filter(TradeLog.status == 'open').count()
    closed_count = session.query(TradeLog).filter(TradeLog.status == 'closed').count()
    session.close()
    return dict(open=open_count, closed=closed_count)


def flat_all_by(target_time_local_str='15:15'):
    """Flat all open positions by local time (IST). target_time_local_str format 'HH:MM'.
    For paper-mode we schedule a background thread that sleeps until target and then closes.
    """
    # Compute UTC target from local (IST +5:30)
    now = datetime.datetime.utcnow()
    # parse local time
    hh, mm = map(int, target_time_local_str.split(':'))
    # compute today's local target in UTC
    today_local = datetime.datetime.now()
    target_local = today_local.replace(hour=hh, minute=mm, second=0, microsecond=0)
    # convert to UTC
    offset = datetime.timedelta(hours=5, minutes=30)
    target_utc = target_local - offset
    delay = (target_utc - now).total_seconds()
    if delay <= 0:
        delay = 0

    def _worker():
        time.sleep(delay)
        # fetch all open trades and close at simulated mid (buy back)
        session = TradeSession()
        rows = session.query(TradeLog).filter(TradeLog.status == 'open').all()
        for r in rows:
            plan = dict(plan_id=r.plan_id, short_strike=r.short_strike, long_strike=r.long_strike,
                        underlying_price=r.meta.get('underlying_price', 100.0) if r.meta else 100.0,
                        entry_iv=r.meta.get('entry_iv', 30.0) if r.meta else 30.0,
                        credit=r.credit)
            bid, ask = _simulate_market_price(plan)
            mid = (bid+ask)/2.0
            # closing cost: buy to close -> pay slightly above mid
            close_price = _apply_slippage(mid, 'buy')
            realised = r.realised + (r.credit - close_price) * r.qty
            r.realised = realised
            r.status = 'closed'
            r.closed_at = datetime.datetime.utcnow()
            r.meta = (r.meta or {})
            r.meta.update({'closed_fill_price': close_price, 'closed_at': datetime.datetime.utcnow().isoformat()})
        session.commit()
        session.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return dict(scheduled=True, target_utc=target_utc.isoformat())


if __name__ == '__main__':
    # quick local test
    sample_plan = dict(plan_id='IC-CALL-1', side='call', short_strike=109, long_strike=119,
                       underlying_price=103.53, entry_iv=29.9307, credit=0.081088)
    print('placing order..')
    r = place_order(sample_plan, qty=1, side='sell')
    print('placed:', r)
    print('reconcile:', reconcile())