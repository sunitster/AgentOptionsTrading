# trade_identifier/iron_condor.py
"""
Balanced v3.1 — Full Advanced RAG + Config (config.yaml) Implementation
- Reads `config.yaml` at repo root for database, rag, ollama, and strategy settings
- Full weekly backtest + RAG retriever using configured DBs
- Structured logging, Ollama connector, safe fallbacks

Config expected (example provided by user):


Run examples:
  python trade_identifier/iron_condor.py --backtest --weekly --ai --start 2025-01-01 --end 2025-06-30 --verbose

Note: This file intentionally keeps AI decisions advisory only. Use environment variables or config.yaml to override credentials.
"""
from __future__ import annotations
import os
import sys
import hashlib
import math
import json
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from typing import List, Dict, Optional, Any, Tuple

import pandas as pd
import numpy as np
import requests
import yaml
from sqlalchemy import create_engine, text

# -------------------------
# Load config.yaml (repo root)
# -------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, 'config.yaml')
if not os.path.exists(CONFIG_PATH):
    print('⚠ config.yaml not found at', CONFIG_PATH, file=sys.stderr)
    # continue with reasonable defaults
    CONFIG = {}
else:
    with open(CONFIG_PATH, 'r') as fh:
        CONFIG = yaml.safe_load(fh) or {}

# Convenience getters with defaults
def cfg_get(path: str, default=None):
    parts = path.split('.')
    cur = CONFIG
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

# -------------------------
# Config values
# -------------------------
FEATURE_DIR = os.path.join(ROOT, 'data', 'features')
NIFTY_DIR = os.path.join(ROOT, 'data', 'nifty')
LOT_SIZE = int(cfg_get('strategy.lot_size', 25))
AI_CONFIDENCE_THRESHOLD = float(cfg_get('strategy.ai_confidence_threshold', 0.75))
ENTRY_WEEKDAYS = tuple(cfg_get('strategy.weekly_entry_days', [3,4]))

OLLAMA_BASE = cfg_get('ollama.base_url', 'http://localhost:11434/v1')
OLLAMA_MODEL = cfg_get('ollama.model', 'llama3.2')
AI_TIMEOUT = float(cfg_get('ollama.timeout', 6.0))

RAG_TOP_K = int(cfg_get('rag.top_k', 6))
RAG_MIN_IV = float(cfg_get('rag.min_iv', 0))
RAG_MIN_OI = float(cfg_get('rag.min_oi', 0))

DB_FEATURES_URL = cfg_get('database.market_features.url', None)
DB_TRADES_URL = cfg_get('database.trade_logs.url', None)

PER_LEG_SLIP = float(cfg_get('strategy.per_leg_slip', 0.5)) if cfg_get('strategy.per_leg_slip', None) is not None else 0.5
COMMISSION_PER_LEG = float(cfg_get('strategy.commission_per_leg', 15.0)) if cfg_get('strategy.commission_per_leg', None) is not None else 15.0
RISK_FREE_RATE = float(cfg_get('strategy.risk_free_rate', 0.06))

# -------------------------
# Logging setup
# -------------------------
LOG_DIR = os.path.join(ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'iron_condor_v3_1.log')
logger = logging.getLogger('iron_condor_v3_1')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

# -------------------------
# Data classes
# -------------------------
@dataclass
class Leg:
    timestamp: pd.Timestamp
    symbol: str
    expiry: pd.Timestamp
    strike: float
    option_type: str
    quantity: int
    price: float
    order_id: Optional[str] = None
    source: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    def is_call(self) -> bool: return str(self.option_type).upper().startswith('C')
    def is_put(self) -> bool: return str(self.option_type).upper().startswith('P')

@dataclass
class StrategyTrade:
    trade_id: str
    strategy: str
    entry_time: pd.Timestamp
    expiry: pd.Timestamp
    legs: List[Leg] = field(default_factory=list)
    status: str = 'OPEN'
    exit_time: Optional[pd.Timestamp] = None
    pnl: Optional[float] = None
    credit: Optional[float] = None
    max_loss: Optional[float] = None
    breakevens: Optional[Tuple[Optional[float], Optional[float]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['entry_time'] = d['entry_time'].isoformat()
        d['expiry'] = d['expiry'].isoformat()
        d['exit_time'] = d['exit_time'].isoformat() if d['exit_time'] is not None else None
        d['legs'] = [{k:(v.isoformat() if isinstance(v,pd.Timestamp) else v) for k,v in {
            'timestamp':l.timestamp,'symbol':l.symbol,'expiry':l.expiry,'strike':l.strike,'option_type':l.option_type,'quantity':l.quantity,'price':l.price,'source':l.source}.items()} for l in self.legs]
        return d

# -------------------------
# Math / BS helpers
# -------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option: str = 'call') -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if (option=='call' and S>K) else ( -1.0 if option=='put' and K>S else 0.0)
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    return _norm_cdf(d1) if option=='call' else (_norm_cdf(d1)-1.0)

# -------------------------
# Utilities
# -------------------------

def _gen_trade_id(prefix: str, entry_time: pd.Timestamp, expiry: pd.Timestamp, legs: List[Leg]) -> str:
    legs_sig = '|'.join(sorted([f"{int(l.strike)}:{l.option_type}:{l.quantity}:{l.price}" for l in legs]))
    payload = f"{prefix}|{entry_time.isoformat()}|{expiry.isoformat()}|{legs_sig}"
    h = hashlib.sha1(payload.encode()).hexdigest()[:12]
    return f"{prefix}_{entry_time.strftime('%Y%m%dT%H%M%S')}_{h}"


def _compute_metrics_from_legs(legs: List[Leg], lot: int = LOT_SIZE) -> Dict[str, Any]:
    if not legs: return {'credit':0.0,'max_loss':0.0,'call_width':0.0,'put_width':0.0,'breakevens':(None,None)}
    calls=[l for l in legs if l.is_call()]; puts=[l for l in legs if l.is_put()]
    net_premium = sum([-l.quantity * l.price for l in legs])
    credit_amt = float(np.nan_to_num(net_premium * lot, nan=0.0))
    call_width = (max([c.strike for c in calls]) - min([c.strike for c in calls])) if calls else 0.0
    put_width = (max([p.strike for p in puts]) - min([p.strike for p in puts])) if puts else 0.0
    max_width = max(call_width, put_width)
    max_loss_amt = float(np.nan_to_num(max_width * lot - credit_amt, nan=0.0))
    short_put = next((p for p in puts if p.quantity < 0), None)
    short_call = next((c for c in calls if c.quantity < 0), None)
    breakeven_put = short_put.strike - net_premium if short_put is not None else None
    breakeven_call = short_call.strike + net_premium if short_call is not None else None
    return {'credit':credit_amt,'max_loss':max_loss_amt,'call_width':call_width,'put_width':put_width,'breakevens':(breakeven_put,breakeven_call)}

# -------------------------
# Feature loaders
# -------------------------

def list_feature_files() -> List[str]:
    if not os.path.isdir(FEATURE_DIR): return []
    return sorted([os.path.join(FEATURE_DIR,f) for f in os.listdir(FEATURE_DIR) if f.endswith('.parquet')])


def load_features_for_date(date_str: str) -> Optional[pd.DataFrame]:
    p = os.path.join(FEATURE_DIR, f"{date_str}.parquet")
    if os.path.exists(p):
        try:
            return pd.read_parquet(p)
        except Exception as e:
            logger.warning('parquet load fail %s: %s', p, e)
    return None

# -------------------------
# DB engine helpers
# -------------------------

def get_db_engine(url: Optional[str] = None):
    url = url or DB_FEATURES_URL
    if not url:
        return None
    try:
        eng = create_engine(url, pool_pre_ping=True)
        with eng.connect() as conn:
            conn.execute(text('SELECT 1'))
        return eng
    except Exception as e:
        logger.warning('DB connect failed for %s: %s', url, e)
        return None

# -------------------------
# RAG retriever (advanced): uses market_features DB then falls back to local files
# -------------------------

def rag_retrieve_for_plan(plan: StrategyTrade, engine=None, top_k: int = RAG_TOP_K) -> List[Dict[str,Any]]:
    docs = []
    engine = engine or get_db_engine()
    date_str = pd.to_datetime(plan.entry_time).date().isoformat()
    if engine is not None:
        try:
            q = text("""
                SELECT id, date, title, content, iv, oi
                FROM feature_docs
                WHERE date <= :d
                ORDER BY date DESC, COALESCE(iv,0) DESC, COALESCE(oi,0) DESC
                LIMIT :k
            """)
            with engine.connect() as conn:
                res = conn.execute(q, {'d': date_str, 'k': top_k})
                for row in res:
                    try:
                        docs.append({'id': row['id'], 'date': str(row['date']), 'title': row['title'], 'content': row['content'], 'iv': float(row['iv']) if row['iv'] is not None else None, 'oi': float(row['oi']) if row['oi'] is not None else None})
                    except Exception:
                        continue
        except Exception as e:
            logger.warning('RAG DB query failed: %s', e)
            engine = None

    if not docs:
        # fallback to local features
        try:
            files = list_feature_files()
            if not files:
                return []
            names = [os.path.basename(f).replace('.parquet','') for f in files]
            cand = None
            for n,f in zip(names, files):
                if n <= date_str:
                    cand = f
            if cand is None:
                cand = files[-1]
            df = pd.read_parquet(cand)
            if 'iv' in df.columns and not df['iv'].dropna().empty:
                top = df.sort_values('iv', ascending=False).head(top_k)
            elif 'oi' in df.columns and not df['oi'].dropna().empty:
                top = df.sort_values('oi', ascending=False).head(top_k)
            else:
                top = df.head(top_k)
            for _,r in top.iterrows():
                docs.append({'id': None, 'date': str(r['date']), 'title': f"Strike {r['strike']} {r['option_type']}", 'content': json.dumps(r.dropna().to_dict(), default=str), 'iv': float(r.get('iv')) if 'iv' in r and not pd.isna(r.get('iv')) else None, 'oi': float(r.get('oi')) if 'oi' in r and not pd.isna(r.get('oi')) else None})
        except Exception as e:
            logger.warning('RAG fallback failed: %s', e)
    # filter by min thresholds
    filtered = []
    for d in docs:
        ivv = d.get('iv') if d.get('iv') is not None else 0
        oiv = d.get('oi') if d.get('oi') is not None else 0
        if ivv >= RAG_MIN_IV and oiv >= RAG_MIN_OI:
            filtered.append(d)
    return filtered[:top_k]

# -------------------------
# Ollama connector
# -------------------------

def ollama_request(prompt: str, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE, timeout: float = AI_TIMEOUT) -> Optional[Dict[str,Any]]:
    try:
        url = f"{base_url}/chat/completions"
        payload = {
            'model': model,
            'messages': [{'role':'user','content':prompt}],
            'max_tokens': 512,
            'temperature': 0.0
        }
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code != 200:
            logger.warning('ollama status %s body %s', r.status_code, r.text[:500])
            return None
        data = r.json()
        text = None
        if 'choices' in data and len(data['choices'])>0:
            text = data['choices'][0].get('message',{}).get('content') or data['choices'][0].get('text')
        if not text:
            return None
        parsed = json.loads(text.strip())
        return parsed
    except Exception as e:
        logger.warning('ollama_request failed: %s', e)
        return None

# -------------------------
# Prompt / AI logic
# -------------------------

def build_rag_prompt(plan: StrategyTrade, context_docs: List[Dict[str,Any]], extra: Dict[str,Any]=None) -> str:
    p_json = json.dumps(plan.to_dict())
    docs_text = ''
    for d in context_docs[:RAG_TOP_K]:
        docs_text += f"- {d.get('date')}: {d.get('title')} | iv={d.get('iv')} oi={d.get('oi')}\n  {d.get('content')[:500]}\n"
    extra_text = json.dumps(extra) if extra else ''
    prompt = {
        'task':'advise_weekly_plan_with_context',
        'plan': plan.to_dict(),
        'context_docs': docs_text,
        'extra': extra_text,
        'response_schema': {'confidence':'number','suggestion':'string','explanation':'string'}
    }
    return json.dumps(prompt)


def ai_decision_with_rag(plan: StrategyTrade, engine=None) -> Dict[str,Any]:
    engine = engine or get_db_engine()
    docs = rag_retrieve_for_plan(plan, engine=engine, top_k=RAG_TOP_K)
    prompt = build_rag_prompt(plan, docs, extra={'note':'Respond only with JSON: {"confidence":0..1, "suggestion":"keep|switch_to_fly|tighten_legs|skip", "explanation":string}'})
    parsed = ollama_request(prompt)
    if parsed is None:
        logger.info('AI unavailable — fallback to keep')
        return {'confidence':0.0,'suggestion':'keep','explanation':'fallback','docs_used_count':len(docs)}
    try:
        confidence = float(parsed.get('confidence',0.0))
    except Exception:
        confidence = 0.0
    suggestion = str(parsed.get('suggestion','keep'))
    explanation = str(parsed.get('explanation',''))
    return {'confidence':confidence,'suggestion':suggestion,'explanation':explanation,'docs_used_count':len(docs)}

# -------------------------
# Weekly chooser and plan builder (BS-based) — reuse earlier patterns
# -------------------------

def next_weekly_expiry(from_date: pd.Timestamp) -> pd.Timestamp:
    d = pd.to_datetime(from_date).normalize()
    for add in range(1, 15):
        cand = d + pd.Timedelta(days=add)
        if cand.weekday() in ENTRY_WEEKDAYS:
            return pd.to_datetime(cand)
    return d + pd.Timedelta(days=7)


def choose_strikes_weekly(df_chain: pd.DataFrame, entry_date: pd.Timestamp, expiry: pd.Timestamp, target_delta_min=0.15, target_delta_max=0.25, r=RISK_FREE_RATE) -> Optional[Dict[str,Any]]:
    if df_chain is None or df_chain.empty: return None
    underlying = float(df_chain['nifty_close'].iloc[0]) if 'nifty_close' in df_chain.columns and not pd.isna(df_chain['nifty_close'].iloc[0]) else float(df_chain['close'].dropna().mean())
    days = max((pd.to_datetime(expiry).normalize() - pd.to_datetime(entry_date).normalize()).days, 1)
    T = days / 365.0
    sigma = float(df_chain['iv'].dropna().mean())/100.0 if 'iv' in df_chain.columns and df_chain['iv'].dropna().size>0 else 0.22
    calls = df_chain[df_chain['option_type'].str.startswith('C')].copy()
    puts = df_chain[df_chain['option_type'].str.startswith('P')].copy()
    if calls.empty or puts.empty: return None
    calls['bs_delta'] = calls['strike'].apply(lambda K: bs_delta(underlying, float(K), T, r, sigma, 'call'))
    puts['bs_delta'] = puts['strike'].apply(lambda K: abs(bs_delta(underlying, float(K), T, r, sigma, 'put')))
    target_mid = (target_delta_min + target_delta_max) / 2.0
    calls['delta_diff'] = calls['bs_delta'].apply(lambda d: abs(d - target_mid))
    puts['delta_diff'] = puts['bs_delta'].apply(lambda d: abs(d - target_mid))
    short_calls = calls[(calls['bs_delta'] >= target_delta_min) & (calls['bs_delta'] <= target_delta_max)].sort_values('delta_diff')
    short_puts = puts[(puts['bs_delta'] >= target_delta_min) & (puts['bs_delta'] <= target_delta_max)].sort_values('delta_diff')
    if short_calls.empty or short_puts.empty:
        short_calls = calls.sort_values('delta_diff').head(5)
        short_puts = puts.sort_values('delta_diff').head(5)
    sc = short_calls.iloc[0]
    sp = short_puts.iloc[0]
    possible_lc = calls[calls['strike'] > sc['strike']].sort_values('strike')
    possible_lp = puts[puts['strike'] < sp['strike']].sort_values('strike', ascending=False)
    if possible_lc.empty or possible_lp.empty: return None
    lc = possible_lc.iloc[0]; lp = possible_lp.iloc[0]
    return {
        'short_call': {'strike': float(sc['strike']), 'price': float(sc.get('close', 0.0)), 'bs_delta': float(sc['bs_delta'])},
        'long_call':  {'strike': float(lc['strike']), 'price': float(lc.get('close', 0.0)), 'bs_delta': float(bs_delta(underlying, float(lc['strike']), T, r, sigma, 'call'))},
        'short_put':  {'strike': float(sp['strike']), 'price': float(sp.get('close', 0.0)), 'bs_delta': float(sp['bs_delta'])},
        'long_put':   {'strike': float(lp['strike']), 'price': float(lp.get('close', 0.0)), 'bs_delta': float(bs_delta(underlying, float(lp['strike']), T, r, sigma, 'put'))},
        'underlying': underlying, 'days': days, 'sigma': sigma
    }


def build_weekly_trade_plan(legs_dict: Dict[str,Any], entry_date: pd.Timestamp, expiry: pd.Timestamp, lot: int = LOT_SIZE, slip: float = PER_LEG_SLIP, commission: float = COMMISSION_PER_LEG) -> Optional[StrategyTrade]:
    if legs_dict is None: return None
    now = pd.to_datetime(entry_date)
    lts = []
    lts.append(Leg(now,'NIFTY',expiry,legs_dict['short_call']['strike'],'CE',-1,legs_dict['short_call']['price'],source='plan'))
    lts.append(Leg(now,'NIFTY',expiry,legs_dict['long_call']['strike'],'CE',+1,legs_dict['long_call']['price'],source='plan'))
    lts.append(Leg(now,'NIFTY',expiry,legs_dict['short_put']['strike'],'PE',-1,legs_dict['short_put']['price'],source='plan'))
    lts.append(Leg(now,'NIFTY',expiry,legs_dict['long_put']['strike'],'PE',+1,legs_dict['long_put']['price'],source='plan'))
    metrics = _compute_metrics_from_legs(lts, lot=lot)
    credit = metrics['credit']
    width = max(metrics['call_width'], metrics['put_width'])
    if width <= 0: return None
    total_slip = slip * len(lts) * lot
    total_comm = commission * len(lts)
    net_credit = credit - total_slip - total_comm
    # acceptance: require net_credit >= 0.20 * width * lot
    if net_credit < 0.20 * width * lot: return None
    trade_id = _gen_trade_id('WPLAN', now, expiry, lts)
    st = StrategyTrade(trade_id, 'WEEKLY_IRON_CONDOR', now, expiry, lts, status='PLAN')
    st.credit = net_credit; st.max_loss = metrics['max_loss']; st.breakevens = metrics['breakevens']
    st.meta.update({'width': width, 'lot': lot, 'sigma': legs_dict.get('sigma'), 'raw_credit': credit, 'slippage': total_slip, 'commission': total_comm, 'accepted': True})
    return st

# -------------------------
# Underlying loader
# -------------------------

def load_nifty_close_for_date(date_dt: pd.Timestamp) -> Optional[float]:
    if not os.path.isdir(NIFTY_DIR): return None
    fname = f"{pd.to_datetime(date_dt).date()}.parquet"
    p = os.path.join(NIFTY_DIR, fname)
    if os.path.exists(p):
        try:
            df = pd.read_parquet(p);
            col = next((c for c in ['close','Close','CLOSE'] if c in df.columns), None)
            if col: return float(df[col].iloc[-1])
        except Exception as e:
            logger.warning('nifty load fail %s: %s', p, e)
            return None
    files = sorted([f for f in os.listdir(NIFTY_DIR) if f.endswith('.parquet')])
    for f in files:
        try:
            d = pd.to_datetime(f.replace('.parquet',''))
            if d.date() == pd.to_datetime(date_dt).date():
                df = pd.read_parquet(os.path.join(NIFTY_DIR, f)); col = next((c for c in ['close','Close','CLOSE'] if c in df.columns), None)
                if col: return float(df[col].iloc[-1])
        except Exception:
            continue
    return None

# -------------------------
# IV table helpers for regime tests
# -------------------------

def compute_daily_mean_iv_table() -> pd.DataFrame:
    files = list_feature_files()
    rows = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if df.empty: continue
            date = pd.to_datetime(df['date'].iloc[0])
            mean_iv = float(df['iv'].dropna().mean()) if 'iv' in df.columns else np.nan
            rows.append({'date': date, 'mean_iv': mean_iv})
        except Exception:
            continue
    out = pd.DataFrame(rows).sort_values('date')
    if out.empty: return out
    out['iv_pct_rank'] = out['mean_iv'].rank(pct=True)
    return out


def iv_percentile_for_date(date_dt: pd.Timestamp, cache: Optional[pd.DataFrame] = None) -> Optional[float]:
    if cache is None: cache = compute_daily_mean_iv_table()
    if cache.empty: return None
    row = cache[cache['date'] == pd.to_datetime(date_dt)]
    if row.empty:
        row = cache[cache['date'] <= pd.to_datetime(date_dt)].tail(1)
        if row.empty: return None
    return float(row['iv_pct_rank'].iloc[0]) * 100.0

# -------------------------
# Runner: weekly backtest + RAG + Ollama
# -------------------------

def run_weekly_with_ai_and_rag(start_date: Optional[str] = None, end_date: Optional[str] = None, max_trades: int = 500, verbose: bool = False):
    logger.info('Starting run_weekly_with_ai_and_rag start=%s end=%s max=%s', start_date, end_date, max_trades)
    engine = get_db_engine(DB_FEATURES_URL)
    files = list_feature_files()
    if not files:
        logger.error('No feature files found in %s', FEATURE_DIR); return {}
    dates = sorted([os.path.basename(f).replace('.parquet','') for f in files])
    if start_date: dates = [d for d in dates if d >= start_date]
    if end_date: dates = [d for d in dates if d <= end_date]
    executed = []
    skipped = []
    iv_table = compute_daily_mean_iv_table()
    for d in dates:
        if len(executed) >= max_trades: break
        entry_dt = pd.to_datetime(d)
        if entry_dt.weekday() not in ENTRY_WEEKDAYS: continue
        df = load_features_for_date(d)
        if df is None or df.empty: logger.debug('no features for %s', d); continue
        iv_pct = iv_percentile_for_date(entry_dt, cache=iv_table)
        nifty_close = float(df['nifty_close'].iloc[0]) if 'nifty_close' in df.columns and not pd.isna(df['nifty_close'].iloc[0]) else None
        atr14 = float(df['nifty_atr_14'].iloc[0]) if 'nifty_atr_14' in df.columns and not pd.isna(df['nifty_atr_14'].iloc[0]) else None
        regime_range = (atr14 / max(nifty_close,1)) < 0.015 if (nifty_close and atr14 is not None) else False
        if iv_pct is None or not (20 <= iv_pct <= 60 and regime_range): logger.debug('skip %s iv_pct=%s regime=%s', d, iv_pct, regime_range); continue
        expiry = next_weekly_expiry(entry_dt)
        legs_dict = choose_strikes_weekly(df, entry_dt, expiry)
        if legs_dict is None: logger.debug('choose_strikes failed %s', d); continue
        plan = build_weekly_trade_plan(legs_dict, entry_dt, expiry)
        if plan is None: logger.debug('plan rejected %s', d); continue
        ai_resp = ai_decision_with_rag(plan, engine=engine)
        logger.info('AI decision %s -> %s', plan.trade_id, ai_resp)
        applied_plan = plan
        if ai_resp['suggestion'] == 'switch_to_fly' and ai_resp['confidence'] >= AI_CONFIDENCE_THRESHOLD:
            df_chain = df
            sc = plan.legs[0].strike; sp = plan.legs[2].strike
            calls = df_chain[df_chain['option_type'].str.startswith('C')]
            puts = df_chain[df_chain['option_type'].str.startswith('P')]
            possible_lc = calls[calls['strike']>sc].sort_values('strike')
            possible_lp = puts[puts['strike']<sp].sort_values('strike', ascending=False)
            if not possible_lc.empty and not possible_lp.empty:
                new_lc = possible_lc.iloc[0]; new_lp = possible_lp.iloc[0]
                fly_legs = []
                now = plan.entry_time
                fly_legs.append(Leg(now,'NIFTY',plan.expiry,sc,'CE',-1,plan.legs[0].price,source='ai'))
                fly_legs.append(Leg(now,'NIFTY',plan.expiry,float(new_lc['strike']),'CE',+1,float(new_lc.get('close',0.0)),source='ai'))
                fly_legs.append(Leg(now,'NIFTY',plan.expiry,sp,'PE',-1,plan.legs[2].price,source='ai'))
                fly_legs.append(Leg(now,'NIFTY',plan.expiry,float(new_lp['strike']),'PE',+1,float(new_lp.get('close',0.0)),source='ai'))
                fly_metrics = _compute_metrics_from_legs(fly_legs)
                if fly_metrics['credit'] > 0 and fly_metrics['max_loss'] <= plan.max_loss * 1.5:
                    new_trade = StrategyTrade(_gen_trade_id('AFLY', plan.entry_time, plan.expiry, fly_legs), 'IRON_FLY', plan.entry_time, plan.expiry, fly_legs, status='PLAN')
                    new_trade.credit = fly_metrics['credit']; new_trade.max_loss = fly_metrics['max_loss']
                    new_trade.meta.update({'from_ai':True,'ai_confidence':ai_resp['confidence'],'ai_explanation':ai_resp['explanation']})
                    applied_plan = new_trade
                    logger.info('Applied AI-converted fly %s', new_trade.trade_id)
        underlying_at_expiry = load_nifty_close_for_date(applied_plan.expiry)
        if underlying_at_expiry is None:
            logger.warning('no underlying at expiry for %s', applied_plan.trade_id); skipped.append({'date':d,'trade':applied_plan.trade_id}); continue
        pnl = 0.0
        for leg in applied_plan.legs:
            intrinsic = (underlying_at_expiry - leg.strike) if leg.is_call() else (leg.strike - underlying_at_expiry)
            intrinsic = max(intrinsic, 0.0)
            pnl += -leg.quantity * intrinsic * LOT_SIZE
        total_pnl = applied_plan.credit + pnl
        applied_plan.pnl = total_pnl; applied_plan.exit_time = applied_plan.expiry; applied_plan.status='CLOSED'
        executed.append(applied_plan)
        logger.info('Executed %s pnl=%.2f', applied_plan.trade_id, applied_plan.pnl)
    logger.info('Run complete executed=%s skipped=%s', len(executed), len(skipped))
    if executed:
        wins = [t for t in executed if t.pnl>0]; win_pct = len(wins)/len(executed)*100
        logger.info('Executed trades=%d winpct=%.2f', len(executed), win_pct)
    return {'executed': executed, 'skipped': skipped}

# -------------------------
# CLI
# -------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--weekly', action='store_true')
    parser.add_argument('--ai', action='store_true')
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--max', type=int, default=500)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.backtest and args.weekly and args.ai:
        run_weekly_with_ai_and_rag(start_date=args.start, end_date=args.end, max_trades=args.max, verbose=args.verbose)
    else:
        logger.info('Balanced v3.1 (config.yaml) loaded. Use --backtest --weekly --ai to run with RAG + Ollama.')
