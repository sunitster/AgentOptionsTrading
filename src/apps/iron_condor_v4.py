#!/usr/bin/env python3
"""
iron_condor_v4.py — Fully integrated Phase 3/4/5 file (silent-enabled)

Features:
- Backtest engine (market snapshots, signal generation, MTM, exits)
- run_backtest_with_regime_model(regime_model, ..., silent=True) to suppress logging (used by label/backfill jobs)
- Graceful handling of extra kwargs so older callers that pass unexpected arguments (e.g. 'silent') won't break
- Simple fallback PnL engine for reproducible behavior without market connectivity
- LightGBM challenger trainer (optional), fallback model writer
- Small artifact saving for daily PnL plot when matplotlib is available

Usage examples:
    python -m src.apps.iron_condor_v4 --backtest --weekly --start 2025-01-01 --end 2025-01-10

API:
    from src.apps.iron_condor_v4 import run_backtest_with_regime_model
    res = run_backtest_with_regime_model(regime_model, start=date(2025,1,1), end=date(2025,1,10), silent=True)
"""
from __future__ import annotations
import argparse
import logging
import sys
import datetime as dt
import random
import os
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Third-party ML libs (optional)
try:
    import lightgbm as lgb  # type: ignore
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore
    except Exception:
        RandomForestRegressor = None  # type: ignore

try:
    import matplotlib  # type: ignore
    matplotlib.use('Agg')  # headless
    import matplotlib.pyplot as plt  # type: ignore
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

# Logging
logger = logging.getLogger("iron_condor_v4")
stream_h = logging.StreamHandler(sys.stdout)
stream_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(stream_h)
logger.setLevel(logging.INFO)

# Ensure artifact paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) if __file__ else os.getcwd()
ARTIFACTS = os.path.join(ROOT, 'artifacts')
MODELS_DIR = os.path.join(ROOT, 'models')
os.makedirs(ARTIFACTS, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------- helpers --------------------
def coerce_datetime(x: Any) -> Optional[dt.datetime]:
    if x is None:
        return None
    if isinstance(x, dt.datetime):
        return x
    if isinstance(x, dt.date):
        return dt.datetime.combine(x, dt.time())
    try:
        return dt.datetime.fromisoformat(str(x))
    except Exception:
        try:
            import dateutil.parser as dup  # type: ignore
            return dup.parse(str(x))
        except Exception:
            return None

def date_only(x: Any) -> Optional[dt.date]:
    d = coerce_datetime(x)
    return d.date() if d else None

# -------------------- PnL engine (fallback) --------------------
class FallbackPnLEngine:
    def __init__(self, rng_seed: Optional[int] = None):
        self._rng = random.Random(rng_seed)

    def simulate_entry_fill(self, plan: Dict[str, Any], snapshot: Dict[str, Any]):
        # entry price heuristic: width * 0.0005 * spot
        price = round(max(0.01, 0.0005 * int(plan.get('width', 10)) * float(snapshot.get('price', 100.0))), 6)
        plan['entry_price'] = price
        return {'fill_price': price}

    def compute_mtm(self, plan: Dict[str, Any], snapshot: Dict[str, Any]) -> float:
        # simple mark model: negative entry_price with small drift/noise
        entry = float(plan.get('entry_price', 0.0))
        # decay as expiry approaches
        if plan.get('expiry_date'):
            days_left = max(1, (coerce_datetime(plan.get('expiry_date')) - coerce_datetime(snapshot.get('date'))).days if coerce_datetime(plan.get('expiry_date')) and coerce_datetime(snapshot.get('date')) else 5)
        else:
            days_left = 5
        decay = 0.5 * (1.0 - 1.0 / days_left)
        noise = self._rng.gauss(0, max(1e-6, abs(entry) * 0.02))
        mark = -entry * (1 - decay) + noise
        return float(mark)

    def close_position(self, plan: Dict[str, Any], price: float) -> float:
        realised = float(plan.get('entry_price', 0.0)) - float(price)
        return float(realised)

    def settle_expiry(self, plan: Dict[str, Any], spot: float) -> float:
        # analytic vertical payoff (credit - intrinsic)
        entry_credit = float(plan.get('entry_price', 0.0))
        side = plan.get('side', 'call')
        short_k = float(plan.get('short_strike', 0.0))
        long_k = float(plan.get('long_strike', 0.0))
        if side == 'call':
            value = max(0.0, spot - short_k) - max(0.0, spot - long_k)
        else:
            value = max(0.0, short_k - spot) - max(0.0, long_k - spot)
        realised = entry_credit - float(value)
        return float(realised)

# -------------------- Exit engine --------------------
@dataclass
class ExitRules:
    profit_pct: float = 0.5
    loss_mult: float = 1.2
    min_hold_days: int = 2

class ExitEngine:
    def __init__(self, cfg: ExitRules):
        self.cfg = cfg

    def should_exit(self, plan: Dict[str, Any], current_mark: float, as_of_date: dt.date) -> Optional[Dict[str, Any]]:
        entry = float(plan.get('entry_price', 0.0))
        profit = entry - current_mark
        tp = self.cfg.profit_pct * entry
        sl = -self.cfg.loss_mult * entry
        opened_dt = coerce_datetime(plan.get('opened') or plan.get('opened_at') or plan.get('entry_date') or plan.get('opened_date'))
        if opened_dt and (as_of_date - opened_dt.date()).days < self.cfg.min_hold_days:
            return None
        if profit >= tp:
            return {'reason': 'tp', 'price': current_mark}
        if profit <= sl:
            return {'reason': 'sl', 'price': current_mark}
        return None

# -------------------- Market & signal generation --------------------
def load_market_data(symbol: str, start: dt.date, end: dt.date) -> List[Dict[str, Any]]:
    days = (end - start).days + 1
    snaps = []
    for i in range(days):
        d = start + dt.timedelta(days=i)
        # skip weekends
        if d.weekday() >= 5:
            continue
        spot = 100.0 + 5.0 * math.sin(i / 7.0) + 0.05 * i
        iv = 25.0 + 5.0 * math.cos(i / 30.0)
        snaps.append({'date': d, 'price': round(spot, 2), 'iv': round(iv, 4)})
    return snaps

def generate_iron_condor_signals(market_snapshots: List[Dict[str, Any]], width: int = 10) -> List[Dict[str, Any]]:
    signals = []
    for s in market_snapshots:
        d = s['date']
        # example: open on Mondays
        if d.weekday() == 0:
            spot = s['price']
            expiry = d + dt.timedelta(days=4)
            short_put = round(spot - width / 2)
            long_put = short_put - width
            short_call = round(spot + width / 2)
            long_call = short_call + width
            ic = {'expiry': expiry, 'short_call': short_call, 'long_call': long_call, 'short_put': short_put, 'long_put': long_put}
            signals.append({'date': d, 'type': 'open_ic', 'ic': ic, 'spot': spot, 'iv': s.get('iv', 25.0)})
        # mtm snapshot each trading day
        signals.append({'date': d, 'type': 'mtm', 'spot': s['price'], 'iv': s.get('iv', 25.0)})
    return signals

# -------------------- Regime application --------------------
import copy as _copy

def _apply_regime_to_plan(plan: Dict[str, Any], regime_model: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if regime_model is None:
        return plan
    p = _copy.deepcopy(plan)
    rules = regime_model.get('rules', {}) if isinstance(regime_model, dict) else {}

    # should_trade rules
    st = rules.get('should_trade', {})
    iv_max = st.get('iv_max')
    entry_iv = p.get('entry_iv', 0.0)
    try:
        entry_iv = float(entry_iv)
    except Exception:
        entry_iv = 0.0
    if iv_max is not None:
        try:
            if entry_iv > float(iv_max):
                return None
        except Exception:
            pass

    # side rules (simple mapping)
    sr = rules.get('side_rules', {})
    iv_delta = p.get('iv_delta', 0.0)
    try:
        iv_delta = float(iv_delta)
    except Exception:
        iv_delta = 0.0
    if iv_delta > 0 and sr.get('when_iv_rising'):
        p['side'] = 'put' if 'put' in sr.get('when_iv_rising', '') else 'call'
    elif iv_delta < 0 and sr.get('when_iv_falling'):
        p['side'] = 'put' if 'put' in sr.get('when_iv_falling', '') else 'call'

    # strike/width overrides
    s_rules = rules.get('strike_rules', {})
    width = s_rules.get('width')
    width_high_iv = s_rules.get('width_high_iv')
    if width is not None:
        try:
            p['width'] = int(width)
        except Exception:
            pass
    if width_high_iv is not None and iv_max is not None:
        try:
            if entry_iv >= float(iv_max) * 0.9:
                p['width'] = int(width_high_iv)
        except Exception:
            pass
    if 'short_strike' in p and 'width' in p:
        try:
            p['long_strike'] = p['short_strike'] + int(p['width'])
        except Exception:
            pass

    # sizing rules (very simple)
    size_rules = rules.get('size_rules', {})
    low_vol = size_rules.get('low_vol')
    high_vol = size_rules.get('high_vol')
    if low_vol is not None or high_vol is not None:
        try:
            if iv_max is not None and entry_iv <= float(iv_max):
                p['qty'] = int(low_vol or p.get('qty', 1))
            else:
                p['qty'] = int(high_vol or p.get('qty', 1))
        except Exception:
            pass

    # exit rules override
    exit_rules = rules.get('exit_rules', {})
    if exit_rules:
        p['exit_rules'] = exit_rules

    return p

# -------------------- Core backtest (silent-enabled) --------------------
def run_backtest_with_regime_model(
        regime_model: Optional[Dict[str, Any]] = None,
        symbol: str = 'UNDERLYING',
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
        weekly: bool = True,
        silent: bool = False,
        **kwargs
) -> Dict[str, Any]:
    """
    Primary backtest entrypoint.

    Parameters
    ----------
    regime_model : Optional[Dict]
        JSON-like dict containing trading/regime rules.
    symbol : str
        Underlying symbol (ignored by fallback market generator).
    start, end : date
        Backtest date range.
    weekly : bool
        Whether to build weekly plans (affects signal generation).
    silent : bool
        If True, suppresses INFO logging (used by batch label generation/backfiller).
    **kwargs :
        Accept unknown extra args for backwards compatibility (they are ignored).

    Returns
    -------
    Dict with keys:
        - daily_pnl: list of floats
        - daily: list of per-day dicts (date, spot, realised, unreal, plans)
        - trades_remaining: list of open plans at end
        - meta: metadata dict
    """
    # support callers that pass start/end as strings
    if isinstance(start, str):
        try:
            start = dt.datetime.fromisoformat(start).date()
        except Exception:
            start = None
    if isinstance(end, str):
        try:
            end = dt.datetime.fromisoformat(end).date()
        except Exception:
            end = None

    start = start or (dt.date.today() - dt.timedelta(days=30))
    end = end or dt.date.today()

    # Silence logger if requested
    prev_level = logger.level
    if silent:
        logger.setLevel(logging.WARNING)

    try:
        snaps = load_market_data(symbol, start, end)
        signals = generate_iron_condor_signals(snaps, width=10 if not regime_model else int(regime_model.get('default_width', 10)))

        pnl_engine = FallbackPnLEngine(rng_seed=42)
        exit_engine = ExitEngine(ExitRules(profit_pct=0.5, loss_mult=1.2, min_hold_days=2))

        active_plans: List[Dict[str, Any]] = []
        daily_results: List[Dict[str, Any]] = []

        plan_counter = 0

        for sig in signals:
            if sig['type'] == 'open_ic':
                ic = sig['ic']
                spot = sig['spot']
                iv = sig.get('iv', 25.0)

                for side in ['call', 'put']:
                    plan_counter += 1
                    plan = {
                        'plan_id': f'IC-{side.upper()}-{plan_counter}',
                        'side': side,
                        'width': abs(ic[f'long_{side}'] - ic[f'short_{side}']) if ic else 10,
                        'short_strike': ic[f'short_{side}'],
                        'long_strike': ic[f'long_{side}'],
                        'underlying_price': spot,
                        'entry_iv': iv,
                        'expiry_date': ic['expiry'],
                        'iv_delta': 0.0,
                        'qty': 1
                    }

                    # Apply regime; support regime_model being None
                    p2 = _apply_regime_to_plan(plan, regime_model) if regime_model is not None else plan
                    if p2 is None:
                        if not silent:
                            logger.info('Regime skipped plan %s', plan['plan_id'])
                        continue
                    plan = p2

                    # Fill entry
                    try:
                        pnl_engine.simulate_entry_fill(plan, {'date': sig['date'], 'price': spot, 'iv': iv})
                    except Exception:
                        # ensure entry_price exists even if simulation fails
                        plan.setdefault('entry_price', 0.01)
                    plan['opened'] = sig['date']
                    plan['entry_price'] = float(plan.get('entry_price', 0.0))

                    active_plans.append(plan)
                    if not silent:
                        logger.info('Opened %s credit=%.6f qty=%d', plan['plan_id'], plan['entry_price'], int(plan.get('qty', 1)))

            elif sig['type'] == 'mtm':
                date = sig['date']
                spot = sig['spot']
                iv = sig.get('iv', 25.0)
                daily_realised = 0.0
                daily_unreal = 0.0
                to_remove = []
                snapshot_plans = []

                for plan in list(active_plans):
                    mark = pnl_engine.compute_mtm(plan, {'date': date, 'price': spot, 'iv': iv})
                    entry = float(plan.get('entry_price', 0.0))
                    unreal = entry - float(mark)
                    snapshot_plans.append({'plan_id': plan['plan_id'], 'mark': mark, 'unreal': unreal})

                    # plan-specific exit rules override if present
                    exit_cfg = ExitRules()
                    if 'exit_rules' in plan:
                        er = plan['exit_rules']
                        exit_cfg = ExitRules(profit_pct=er.get('profit_pct', 0.5), loss_mult=er.get('loss_mult', 1.2), min_hold_days=er.get('min_hold_days', 2))
                    decision = ExitEngine(exit_cfg).should_exit(plan, mark, date)
                    if decision:
                        realised = pnl_engine.close_position(plan, decision['price'])
                        if not silent:
                            logger.info('Closed %s reason=%s realised=%.6f', plan['plan_id'], decision['reason'], realised)
                        daily_realised += float(realised)
                        to_remove.append(plan)

                    daily_unreal += unreal

                # expiry settlement
                for plan in list(active_plans):
                    if plan in to_remove:
                        continue
                    exp_dt = coerce_datetime(plan.get('expiry_date'))
                    if exp_dt and date >= exp_dt.date():
                        settled = pnl_engine.settle_expiry(plan, spot)
                        if not silent:
                            logger.info('Settled %s at expiry -> realised=%.6f', plan['plan_id'], settled)
                        daily_realised += float(settled)
                        to_remove.append(plan)

                # remove closed/settled
                active_plans = [p for p in active_plans if p not in to_remove]

                daily_results.append({'date': date, 'spot': spot, 'realised': round(daily_realised, 6), 'unreal': round(daily_unreal, 6), 'plans': snapshot_plans})

        # compute daily_pnl series
        daily_pnl = [d['realised'] + d['unreal'] for d in daily_results]

        # Save artifact plot if possible
        if MPL_AVAILABLE and daily_pnl:
            try:
                fig, ax = plt.subplots()
                ax.plot([d['date'] for d in daily_results], daily_pnl, marker='o')
                ax.set_title('Daily PnL')
                ax.set_xlabel('Date')
                ax.set_ylabel('PnL')
                fig.autofmt_xdate()
                outp = os.path.join(ARTIFACTS, f'daily_pnl_{start}_{end}.png')
                fig.savefig(outp)
                plt.close(fig)
                if not silent:
                    logger.info('Saved PnL plot to %s', outp)
            except Exception as e:
                logger.debug('Failed to save PnL plot: %s', e)

        meta = {'open_count': len(active_plans), 'plans_open': [p.get('plan_id') for p in active_plans]}
        result = {'daily_pnl': daily_pnl, 'daily': daily_results, 'trades_remaining': active_plans, 'meta': meta}
        return result
    finally:
        # restore logger level
        if silent:
            logger.setLevel(prev_level)

# -------------------- Backtest adapter for Phase5 --------------------
def backtest_fn(regime_model: Optional[Dict] = None,
                start: Optional[dt.date] = None,
                end: Optional[dt.date] = None,
                weekly: bool = True):
    """
    Phase-5 compliant backtest wrapper.

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

    # call with silent=True for batch label generation/backfilling
    res = run_backtest_with_regime_model(regime_model, start=start, end=end, weekly=weekly, silent=True)

    daily = res.get('daily_pnl', [])
    dates = [d.get('date').isoformat() if isinstance(d.get('date'), dt.date) else str(d.get('date')) for d in res.get('daily', [])]

    # Calculate metrics
    if len(daily) > 1:
        import numpy as np
        pnl = np.array(daily, dtype=float)
        total = float(pnl.sum())
        sharpe = float((pnl.mean() / (pnl.std() + 1e-9)) * (252 ** 0.5))
        curve = pnl.cumsum()
        high = np.maximum.accumulate(curve)
        dd = high - curve
        max_dd = float(dd.max())
        win_pct = float((pnl > 0).mean())
    else:
        sharpe = 0.0
        total = 0.0
        max_dd = 0.0
        win_pct = 0.0

    summary = {
        "total_pnl": total,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "winning_days_pct": win_pct
    }

    return {"daily_pnl": daily, "dates": dates, "summary": summary}

# -------------------- LightGBM challenger trainer --------------------
def train_lightgbm_challenger(feature_df, target_series, model_path: Optional[str] = None):
    """Train a simple LightGBM regressor that predicts next-day pnl.
    feature_df: pd.DataFrame
    target_series: pd.Series
    """
    import pandas as pd  # local import to avoid heavy deps at module import time
    if LGB_AVAILABLE:
        dtrain = lgb.Dataset(feature_df.values, label=target_series.values)
        params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1}
        bst = lgb.train(params, dtrain, num_boost_round=100)
        model_path = model_path or os.path.join(MODELS_DIR, 'lgb_challenger.txt')
        bst.save_model(model_path)
        return {'type': 'lightgbm', 'path': model_path}
    else:
        # fallback: save a trivial JSON that maps mean pnl
        model_path = model_path or os.path.join(MODELS_DIR, 'fallback_challenger.json')
        model = {'type': 'fallback', 'mean': float(target_series.mean())}
        with open(model_path, 'w') as f:
            json.dump(model, f)
        return {'type': 'fallback', 'path': model_path}

# -------------------- Utility: convert LLM JSON candidate to numeric features --------------------
def llm_candidate_to_numeric(candidate: Dict) -> Dict:
    out = {}
    rules = candidate.get('rules', {})
    st = rules.get('should_trade', {})
    out['iv_max'] = float(st.get('iv_max', 0))
    sr = rules.get('side_rules', {})
    out['when_iv_rising_put'] = 1 if 'put' in sr.get('when_iv_rising', '') else 0
    out['when_iv_falling_put'] = 1 if 'put' in sr.get('when_iv_falling', '') else 0
    s_rules = rules.get('strike_rules', {})
    out['width'] = float(s_rules.get('width', 0))
    out['width_high_iv'] = float(s_rules.get('width_high_iv', 0))
    size = rules.get('size_rules', {})
    out['low_vol_size'] = float(size.get('low_vol', 0))
    out['high_vol_size'] = float(size.get('high_vol', 0))
    exit_r = rules.get('exit_rules', {})
    out['profit_pct'] = float(exit_r.get('profit_pct', 0.5))
    out['loss_mult'] = float(exit_r.get('loss_mult', 1.2))
    return out

# -------------------- CLI & helpers --------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog='iron_condor_v4')
    p.add_argument('--backtest', action='store_true')
    p.add_argument('--weekly', action='store_true')
    p.add_argument('--start', required=False, type=str)
    p.add_argument('--end', required=False, type=str)
    p.add_argument('--regime_model', required=False, type=str)
    p.add_argument('--silent', action='store_true', help='Run backtest with suppressed logging')
    return p.parse_args(argv)

def iso_to_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, '%Y-%m-%d').date()

def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    start = iso_to_date(args.start) if args.start else dt.date.today() - dt.timedelta(days=30)
    end = iso_to_date(args.end) if args.end else dt.date.today()
    regime = None
    if args.regime_model:
        with open(args.regime_model, 'r') as f:
            regime = json.load(f)
    if args.backtest:
        res = run_backtest_with_regime_model(regime, 'UNDERLYING', start, end, weekly=args.weekly, silent=args.silent)
        print('Done. Summary:', res.get('meta', {}))

if __name__ == '__main__':
    main()
