#!/usr/bin/env python3
"""
scripts/build_all_features.py (with Black-Scholes implied vol fill)

- Loads unified option-chain files from data/options_all/*.parquet
- Loads NIFTY OHLC files from data/nifty/*.parquet
- Normalizes timezone issues (tz-aware -> tz-naive)
- Computes NIFTY time-series indicators (ATR, RSI, vol20, trend slope)
- Computes per-date option-chain aggregates
- Computes per-option features (moneyness, DTE, vol/oi ratios)
- Computes Black-Scholes implied vol for ALL rows where possible (r = 0.06)
- Writes per-date enriched files to data/features_enriched/<date>.parquet
- Writes concatenated learning file to learning_data/daily_features_enriched.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np
from typing import Dict
from tqdm import tqdm
import math
import time

# -------------------- Config --------------------
ROOT = Path('.')
OPTIONS_ALL = ROOT / 'data' / 'options_all'
NIFTY_DIR = ROOT / 'data' / 'nifty'
OUT_DIR = ROOT / 'data' / 'features_enriched'
OUT_DIR.mkdir(parents=True, exist_ok=True)
LEARNING_OUT = ROOT / 'learning_data' / 'daily_features_enriched.parquet'
LEARNING_OUT.parent.mkdir(parents=True, exist_ok=True)

# Black-Scholes interest rate
RISK_FREE_RATE = 0.06  # user choice: 6%

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('build_all_features')

# -------------------- Date/time helpers --------------------

def make_tz_naive(series_or_scalar):
    """
    Convert tz-aware or tz-naive inputs to tz-naive pandas datetime64[ns].
    Accepts a Series-like or scalar. Returns Series for sequence input, Timestamp for scalar input.
    """
    is_scalar = not hasattr(series_or_scalar, '__iter__') or isinstance(series_or_scalar, (str, bytes))
    if is_scalar:
        try:
            s = pd.to_datetime(series_or_scalar, errors='coerce')
        except Exception:
            return pd.NaT
        if getattr(s, 'tzinfo', None) is not None:
            try:
                return s.tz_convert('UTC').tz_localize(None)
            except Exception:
                try:
                    return pd.to_datetime(s.tz_localize(None))
                except Exception:
                    return pd.NaT
        return pd.to_datetime(s)
    s = pd.to_datetime(pd.Series(series_or_scalar), errors='coerce')
    try:
        sample = s.iloc[0]
        if hasattr(sample, 'tzinfo') and sample.tzinfo is not None:
            return s.dt.tz_convert('UTC').dt.tz_localize(None)
    except Exception:
        pass
    try:
        return s.dt.tz_localize(None)
    except Exception:
        try:
            return s.astype('datetime64[ns]')
        except Exception:
            return pd.to_datetime(s).astype('datetime64[ns]')

# -------------------- NIFTY loader & indicators --------------------

def load_nifty_df() -> pd.DataFrame:
    files = sorted(glob(str(NIFTY_DIR / '*.parquet')))
    rows = []
    for f in files:
        try:
            d = pd.read_parquet(f)
            if 'Date' in d.columns:
                d = d.rename(columns={'Date': 'date'})
            if 'date' not in d.columns:
                date_from_name = Path(f).stem
                d['date'] = date_from_name
            r = d.iloc[0].to_dict()
            rows.append(r)
        except Exception as e:
            logger.warning('Skipping nifty file %s: %s', f, e)
    if not rows:
        raise RuntimeError('No NIFTY files found under data/nifty')
    nifty = pd.DataFrame(rows)
    nifty['date'] = make_tz_naive(nifty.get('date'))
    nifty = nifty.sort_values('date').reset_index(drop=True)
    nifty.columns = [c.lower() for c in nifty.columns]
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c not in nifty.columns:
            nifty[c] = np.nan
    return nifty

def compute_nifty_indicators(nifty: pd.DataFrame) -> pd.DataFrame:
    df = nifty.copy().sort_values('date').reset_index(drop=True)
    df['ret_1'] = df['close'].pct_change().fillna(0.0)
    df['tr'] = (df['high'] - df['low']).abs()
    df['atr_14'] = df['tr'].rolling(window=14, min_periods=1).mean()
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_14'] = df['rsi_14'].fillna(50.0)
    df['vol_20'] = df['ret_1'].rolling(window=20, min_periods=1).std() * np.sqrt(252)
    df['log_close'] = np.log(df['close'].replace(0, np.nan)).ffill().fillna(0.0)
    slopes = []
    w = 21
    for i in range(len(df)):
        start = max(0, i - w + 1)
        y = df['log_close'].iloc[start:i+1].values
        if len(y) < 2:
            slopes.append(0.0)
        else:
            try:
                slope = np.polyfit(np.arange(len(y)), y, 1)[0]
            except Exception:
                slope = 0.0
            slopes.append(float(slope))
    df['trend_slope_21'] = slopes
    return df

# -------------------- Option-chain aggregates --------------------

def aggregate_option_chain(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {
            'total_oi': 0.0,
            'weighted_avg_iv': np.nan,
            'call_minus_put_iv': np.nan,
            'avg_last_price': np.nan,
        }
    d = df.copy()
    for col in ['iv', 'oi', 'last_price']:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors='coerce')
    d['oi'] = d.get('oi', pd.Series(0, index=d.index)).fillna(0)
    total_oi = d['oi'].sum() if 'oi' in d.columns else 0.0
    if total_oi > 0 and 'iv' in d.columns:
        weighted_iv = (d['iv'].fillna(0) * d['oi']).sum() / max(1e-9, total_oi)
    else:
        weighted_iv = d['iv'].mean() if 'iv' in d.columns else np.nan
    call_minus_put_iv = np.nan
    if 'option_type' in d.columns and 'iv' in d.columns:
        calls = d[d['option_type'].str.lower() == 'call']
        puts = d[d['option_type'].str.lower() == 'put']
        call_iv = calls['iv'].mean() if not calls.empty else np.nan
        put_iv = puts['iv'].mean() if not puts.empty else np.nan
        if not (pd.isna(call_iv) or pd.isna(put_iv)):
            call_minus_put_iv = float(call_iv - put_iv)
    avg_last = d['last_price'].mean() if 'last_price' in d.columns else np.nan
    return {
        'total_oi': float(total_oi),
        'weighted_avg_iv': float(weighted_iv) if not pd.isna(weighted_iv) else np.nan,
        'call_minus_put_iv': float(call_minus_put_iv) if not pd.isna(call_minus_put_iv) else np.nan,
        'avg_last_price': float(avg_last) if not pd.isna(avg_last) else np.nan,
    }

# -------------------- Black-Scholes & IV solver --------------------

def bs_price(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes price for European call/put.
    S: spot
    K: strike
    T: time to expiry in years (float)
    r: annual risk-free rate (decimal)
    sigma: vol (annual)
    """
    if T <= 0:
        # option is at expiry; price = intrinsic
        if option_type == 'call':
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    if sigma <= 0:
        # approximate: discounted intrinsic?
        if option_type == 'call':
            return max(S - K * math.exp(-r * T), 0.0)
        else:
            return max(K * math.exp(-r * T) - S, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
    # normal CDF via erf
    if option_type == 'call':
        price = S * nd1 - K * math.exp(-r * T) * nd2
    else:
        # put-call parity could be used but use direct formula
        price = K * math.exp(-r * T) * (1 - nd2) - S * (1 - nd1)
    return max(price, 0.0)

def implied_vol_from_price(S, K, T, r, price_market, option_type='call', tol=1e-6, max_iter=60):
    """
    Solve for implied vol given market price using a robust bisection + secant-ish method.
    Returns sigma or np.nan if solve fails.
    """
    # quick sanity checks
    if price_market <= 0 or S <= 0 or K <= 0 or T < 0:
        return np.nan

    # intrinsic bounds
    if option_type == 'call':
        intrinsic = max(S - K * math.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * math.exp(-r * T) - S, 0.0)
    # If market price below intrinsic - impossible
    if price_market < intrinsic - 1e-8:
        return np.nan

    # Upper bound: use a high vol
    low_sigma = 1e-8
    high_sigma = 5.0  # 500% vol upper bound
    try:
        low_price = bs_price(S, K, T, r, low_sigma, option_type)
        high_price = bs_price(S, K, T, r, high_sigma, option_type)
    except Exception:
        return np.nan

    # If market price outside [low_price, high_price], no solution
    if not (low_price - 1e-12 <= price_market <= high_price + 1e-12):
        # sometimes low_price > price_market due to numerical; handle gracefully
        if price_market > high_price:
            # maybe price > high_price due to dividends etc. return nan
            return np.nan

    # Bisection
    for i in range(max_iter):
        mid = 0.5 * (low_sigma + high_sigma)
        mid_price = bs_price(S, K, T, r, mid, option_type)
        # compare
        if abs(mid_price - price_market) < tol:
            return mid
        # decide side
        if mid_price > price_market:
            high_sigma = mid
        else:
            low_sigma = mid
    # fallback: return midpoint
    sigma = 0.5 * (low_sigma + high_sigma)
    return sigma

# -------------------- Per-option features (with IV computation) --------------------

def per_option_features(option_df: pd.DataFrame, nifty_row: pd.Series, ocagg: Dict[str, float]) -> pd.DataFrame:
    df = option_df.copy()
    df.columns = [c.lower() for c in df.columns]

    # normalize numeric columns
    strike_col = 'strike' if 'strike' in df.columns else ('strike_pr' if 'strike_pr' in df.columns else None)
    if strike_col is None:
        df['strike'] = pd.NA
    else:
        df['strike'] = pd.to_numeric(df.get(strike_col), errors='coerce')
    df['last_price'] = pd.to_numeric(df.get('last_price', df.get('close', np.nan)), errors='coerce').fillna(0.0)
    df['iv'] = pd.to_numeric(df.get('iv', np.nan), errors='coerce')
    df['oi'] = pd.to_numeric(df.get('oi', df.get('open_int', 0)), errors='coerce').fillna(0)
    df['volume'] = pd.to_numeric(df.get('volume', df.get('contracts', 0)), errors='coerce').fillna(0)
    df['option_type'] = df.get('option_type', df.get('option_typ', 'call')).astype(str).str.lower().replace({'ce': 'call', 'pe': 'put'})

    # dates normalized
    df['date'] = make_tz_naive(df.get('date', pd.NaT))
    df['expiry'] = make_tz_naive(df.get('expiry', df.get('expiry_dt', pd.NaT)))

    underlying_close = float(nifty_row.get('close', np.nan))

    # moneyness / relative strike
    df['moneyness'] = underlying_close / df['strike']
    df['rel_strike'] = (df['strike'] - underlying_close) / underlying_close

    # DTE
    df['dte'] = (df['expiry'] - df['date']).dt.days.clip(lower=0)

    # is_itm
    df['is_itm'] = (((df['option_type'] == 'call') & (df['strike'] <= underlying_close)) |
                    ((df['option_type'] == 'put') & (df['strike'] >= underlying_close))).astype(int)

    # vol/oi ratio
    df['vol_oi_ratio'] = df['volume'] / (df['oi'].replace(0, np.nan))
    df['vol_oi_ratio'] = df['vol_oi_ratio'].fillna(0.0)

    # merge aggregates and nifty indicators
    df['total_oi'] = ocagg.get('total_oi', np.nan)
    df['weighted_avg_iv'] = ocagg.get('weighted_avg_iv', np.nan)
    df['call_minus_put_iv'] = ocagg.get('call_minus_put_iv', np.nan)
    df['avg_last_price'] = ocagg.get('avg_last_price', np.nan)

    df['nifty_close'] = nifty_row.get('close', np.nan)
    df['nifty_ret_1'] = nifty_row.get('ret_1', np.nan)
    df['nifty_atr_14'] = nifty_row.get('atr_14', np.nan)
    df['nifty_rsi_14'] = nifty_row.get('rsi_14', np.nan)
    df['nifty_vol_20'] = nifty_row.get('vol_20', np.nan)
    df['nifty_trend_slope_21'] = nifty_row.get('trend_slope_21', np.nan)

    # Compute implied vol for ALL rows where possible (user requested)
    n = len(df)
    iv_bs = np.full(n, np.nan, dtype=float)
    cache = {}  # simple dict cache
    successes = 0
    failures = 0
    start_time = time.time()

    for idx, row in df.iterrows():
        S = float(underlying_close) if not pd.isna(underlying_close) else np.nan
        K = float(row['strike']) if not pd.isna(row['strike']) else np.nan
        price_mkt = float(row['last_price']) if not pd.isna(row['last_price']) else 0.0
        dte_days = int(row['dte']) if not pd.isna(row['dte']) else 0
        T = max(dte_days / 365.0, 0.0)
        otype = str(row['option_type']) if not pd.isna(row['option_type']) else 'call'

        # sanity: price must be > 0. and T > 0
        if price_mkt <= 0.0 or K <= 0 or S <= 0 or T <= 0:
            iv_bs[idx] = np.nan
            failures += 1
            continue

        # caching key: round values to reasonable precision to increase cache hits
        key = (round(S, 2), round(K, 2), round(T, 5), round(price_mkt, 2), otype)
        if key in cache:
            iv_val = cache[key]
            iv_bs[idx] = iv_val
            successes += 1 if not np.isnan(iv_val) else 0
            continue

        try:
            iv_val = implied_vol_from_price(S, K, T, RISK_FREE_RATE, price_mkt, option_type=otype, tol=1e-6, max_iter=60)
            if iv_val is None or np.isnan(iv_val):
                failures += 1
                iv_bs[idx] = np.nan
                cache[key] = np.nan
            else:
                iv_bs[idx] = float(iv_val)
                cache[key] = float(iv_val)
                successes += 1
        except Exception:
            failures += 1
            iv_bs[idx] = np.nan
            cache[key] = np.nan

    elapsed = time.time() - start_time
    logger.info("IV compute: rows=%d, successes=%d, failures=%d, time=%.1fs", n, successes, failures, elapsed)

    df['iv_bs'] = iv_bs
    # since user requested all rows be computed, use bs IV as iv_imputed
    # Keep original iv for reference
    df['iv_imputed'] = df['iv_bs']

    # derived
    df['atr_by_price'] = df['nifty_atr_14'] / df['nifty_close'].replace(0, np.nan)

    # clean numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df

# -------------------- Main --------------------

def main():
    logger.info('Loading NIFTY series...')
    nifty = load_nifty_df()
    nifty = compute_nifty_indicators(nifty)
    nifty = nifty.set_index('date', drop=False)

    files = sorted(glob(str(OPTIONS_ALL / '*.parquet')))
    if not files:
        logger.error('No unified option files found at %s. Run unify script first.', OPTIONS_ALL)
        return

    all_out = []
    logger.info('Processing %d dates from %s', len(files), OPTIONS_ALL)
    for f in tqdm(files, desc='dates'):
        date_str = Path(f).stem
        try:
            option_df = pd.read_parquet(f)
            if option_df.empty:
                continue
            option_df.columns = [c.lower() for c in option_df.columns]

            # normalize dates -> tz-naive datetimes (vectorized)
            option_df['date'] = make_tz_naive(option_df.get('date', date_str))
            option_df['expiry'] = make_tz_naive(option_df.get('expiry', option_df.get('expiry_dt', pd.NaT)))

            # dts used for lookup (tz-naive)
            dts = make_tz_naive(date_str)
            if pd.isna(dts):
                logger.warning('Could not parse date from filename %s; skipping', date_str)
                continue

            mask = nifty['date'] <= dts
            if not mask.any():
                logger.warning('No nifty row <= %s found; skipping %s', dts, date_str)
                continue
            nifty_row = nifty.loc[mask].iloc[-1]

            ocagg = aggregate_option_chain(option_df)

            enriched = per_option_features(option_df, nifty_row, ocagg)

            out_path = OUT_DIR / f"{date_str}.parquet"
            enriched.to_parquet(out_path, index=False)
            all_out.append(enriched)

        except Exception as e:
            logger.exception('Failed processing %s: %s', f, e)
            continue

    if all_out:
        df_all = pd.concat(all_out, ignore_index=True, sort=False)
        df_all.to_parquet(LEARNING_OUT, index=False)
        logger.info('Wrote concatenated learning data to %s', LEARNING_OUT)
    else:
        logger.warning('No enriched files produced.')

if __name__ == '__main__':
    main()
