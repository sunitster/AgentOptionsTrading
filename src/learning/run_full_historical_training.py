#!/usr/bin/env python3
"""
run_full_historical_training.py — FULL UPGRADE (Phase2 + Phase3 + Phase6)

Key improvements:
- Parallel-safe Observability (workers receive a lightweight WorkerObservability)
- Risk Engine enforcement + per-trade checks + daily drawdown circuit breaker
- Expanded LLM strategy generation: exploration, crossover, mutation, diversity, memory
- Joblib-based parallel evaluation (safe, low-memory)
- Additional observability events (latency, slippage, data gaps, candidate summaries)
- Keeps previous functionality intact (LightGBM training, champion promotion, exports)

Drop-in replacement for your pipeline; configure constants below as needed.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Optional sklearn clustering (used for diversity parents)
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    KMeans = None
    SKLEARN_AVAILABLE = False

# local modules (must exist in src/learning/)
from src.learning.strategy_adapter import apply_strategy_adapter
from src.learning.observability import Observability
from src.learning.risk_engine import PositionSizer, RiskEngine, risk_adjusted_score

# ----------------------------
# Configuration & constants
# ----------------------------
ROOT = Path('.').resolve()
FEATURES_PATH = ROOT / 'learning_data' / 'daily_features_enriched.parquet'
LABELS_PATH = ROOT / 'learning_data' / 'daily_labels.parquet'
OUT_MODEL_DIR = ROOT / 'models'
OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHAMPION_MODEL_PATH = OUT_MODEL_DIR / 'lightgbm_champion.pkl'
CHAMPION_METRICS_JSON = OUT_MODEL_DIR / 'champion_metrics.json'
FEATURES_JSON = OUT_MODEL_DIR / 'feature_list.json'
PAST_WINNERS_PATH = OUT_MODEL_DIR / 'past_winners.json'
LLM_RESULTS_DIR = OUT_MODEL_DIR / 'llm_results'
LLM_TRADES_DIR = OUT_MODEL_DIR / 'llm_trades'
LLM_RESULTS_DIR.mkdir(exist_ok=True)
LLM_TRADES_DIR.mkdir(exist_ok=True)

LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_full_historical_training")

# LightGBM defaults
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42,
}

TARGET_COL = "next_day_pnl"
TIME_COL = "date"
VALIDATION_DAYS = 90

# Parallel & search settings
CPU_PARALLEL_JOBS = max(1, min(8, (joblib.cpu_count() or 1) - 1))
N_ROUNDS = 4
CANDIDATES_PER_ROUND = 6
EXPLORATION_TEMP_BASE = 0.25
EXPLORATION_TEMP_INC = 0.20
EXPLORATION_TEMP_MAX = 0.75
STAGNATION_RATIO = 0.75
SMART_RESTART_ROUNDS = 2
MIN_PARAM_DISTANCE = 0.15
LEADERBOARD_SIZE = 30
PAST_WINNERS_MAX = 80
PAST_WINNERS_ROLLING = 12

# Risk defaults
DEFAULT_STARTING_CAPITAL = 100000.0
DEFAULT_MAX_RISK_PCT = 0.02
DEFAULT_DAILY_LOSS_PCT = 0.06
SLIPPAGE_ANOMALY_THRESHOLD_PCT = 0.005  # 0.5% of capital

# fallback strategies (templates)
LLM_FALLBACK_TEMPLATES = [
    {"direction": "bear", "min_iv": 0.10, "max_iv": 0.50, "dte_min": 5, "dte_max": 60, "size_aggressiveness": 1.0},
    {"direction": "neutral", "min_iv": 0.05, "max_iv": 0.40, "dte_min": 7, "dte_max": 120, "size_aggressiveness": 0.8},
    {"direction": "bull", "min_iv": 0.08, "max_iv": 0.45, "dte_min": 7, "dte_max": 45, "size_aggressiveness": 1.2},
    {"direction": "neutral", "min_iv": 0.12, "max_iv": 0.60, "dte_min": 14, "dte_max": 60, "size_aggressiveness": 1.0},
]

# ----------------------------
# Utility helpers
# ----------------------------

def safe_load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_parquet(path)
    return df

def _json_default(obj):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_json(date_format='iso')
    if hasattr(obj, "tolist"):
        try:
            return list(obj)
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    raise TypeError(f"Type {type(obj)} not serializable")

# ----------------------------
# Observability: worker-safe wrapper
# ----------------------------
class WorkerObservability:
    """
    Lightweight, stateless observability for worker processes.
    Methods mirror those used in simulate_trading; they must be picklable.
    They do minimal in-process logging and return quickly.
    """
    def __init__(self):
        # no open files or complex state
        self._start_times = {}

    def start_latency(self, label: str):
        self._start_times[label] = time.time()

    def end_latency(self, label: str) -> float:
        if label not in self._start_times:
            return 0.0
        elapsed = time.time() - self._start_times.pop(label)
        # do not write files here
        return elapsed

    def log_event(self, event_type: str, payload: dict):
        # no-op or lightweight print for debug in workers
        return

    def detect_slippage_anomaly(self, slippage_value: float, threshold_pct_of_capital: float, capital: float):
        # stateless check; do nothing here
        return

    def log_candidate_summary(self, summary: dict):
        # workers don't write summary files; main process will
        return

# ----------------------------
# Feature selection & dataset prep
# ----------------------------
def auto_select_features(df: pd.DataFrame) -> List[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric if c != TARGET_COL]
    final = []
    for c in features:
        nunique = df[c].nunique(dropna=True)
        if nunique <= 1: continue
        if nunique > 0.99 * len(df): continue
        final.append(c)
    logger.info("Auto-selected %d numeric features", len(final))
    return final

def time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
    max_date = df[TIME_COL].max()
    cutoff = max_date - pd.Timedelta(days=VALIDATION_DAYS)
    train = df[df[TIME_COL] <= cutoff].reset_index(drop=True)
    val = df[df[TIME_COL] > cutoff].reset_index(drop=True)
    logger.info("Time split: train=%s val=%s cutoff=%s max=%s", train.shape, val.shape, cutoff.date(), max_date.date())
    return train, val

def prepare_datasets(df: pd.DataFrame, features: List[str]):
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    logger.info("After dropping NaN target rows shape=%s", df.shape)
    train_df, val_df = time_split(df)
    train_df = train_df.dropna(subset=features, how='all').reset_index(drop=True)
    val_df = val_df.dropna(subset=features, how='all').reset_index(drop=True)
    logger.info("After dropping all-NaN feature rows -> train=%s val=%s", train_df.shape, val_df.shape)
    for f in features:
        med = pd.concat([train_df[f], val_df[f]]).median()
        train_df[f] = train_df[f].fillna(med)
        val_df[f] = val_df[f].fillna(med)
    return train_df, val_df, features

# ----------------------------
# LightGBM training helpers
# ----------------------------
def train_lightgbm(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str]):
    X_train = train_df[feature_cols]; y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols]; y_val = val_df[TARGET_COL]
    ds_train = lgb.Dataset(X_train, label=y_train)
    ds_val = lgb.Dataset(X_val, label=y_val)
    callbacks = [lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]
    model = lgb.train(params=LGB_PARAMS, train_set=ds_train, valid_sets=[ds_train, ds_val],
                      valid_names=["train", "val"], num_boost_round=2000, callbacks=callbacks)
    return model

def build_numeric_challenger(train_df: pd.DataFrame, val_df: pd.DataFrame, features: List[str]):
    model = train_lightgbm(train_df, val_df, features)
    val_pred = model.predict(val_df[features], num_iteration=getattr(model, "best_iteration", None))
    val_true = val_df[TARGET_COL]
    rmse = float(np.sqrt(np.mean((val_pred - val_true) ** 2)))
    return model, rmse

# ----------------------------
# Simulation (worker-friendly and main-process aware)
# ----------------------------
def simulate_trading(model, df: pd.DataFrame, features: List[str], *,
                     position_sizing: str = "proportional",
                     fixed_size: float = 1.0,
                     size_scale: float = 1.0,
                     max_size: float = 100.0,
                     slippage_per_unit: float = 0.0,
                     commission_per_unit: float = 0.0,
                     skip_zero_pred: bool = True,
                     prediction_threshold: float = 0.0,
                     risk_engine: Optional[RiskEngine] = None,
                     observability = None) -> Dict[str, Any]:
    """
    Simulate trading. 'observability' can be WorkerObservability in workers, Observability in main.
    'risk_engine' should be picklable (it is), but internal position_sizer may contain state.
    Keep this function pure-ish for easy parallelism.
    """
    df = df.reset_index(drop=True).copy()
    X = df[features]
    preds = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
    df["_pred"] = preds
    # assign simple regimes if missing
    if "regime" not in df.columns or df["regime"].isna().all():
        if "iv_bs" in df.columns:
            try:
                df["regime"] = pd.qcut(df["iv_bs"].rank(method='first'), 3, labels=["low_iv", "mid_iv", "high_iv"])
            except Exception:
                df["regime"] = "all"
        else:
            df["regime"] = "all"

    trades = []
    skipped = 0
    # iterate row-wise — keep it simple and deterministic
    for idx, row in df.iterrows():
        pred = float(row["_pred"])
        # skip small predictions
        if skip_zero_pred and abs(pred) <= prediction_threshold:
            trades.append({"idx": idx, "pnl": 0.0, "size": 0.0, "regime": row.get("regime", "all"), "date": row.get(TIME_COL)})
            continue
        sign = 1 if pred > 0 else -1
        if position_sizing == "fixed":
            size = sign * abs(fixed_size)
        else:
            size = sign * min(max_size, abs(pred) * size_scale)

        realized = float(row[TARGET_COL]) if TARGET_COL in row else 0.0
        gross = realized * size
        cost = (slippage_per_unit + commission_per_unit) * abs(size)
        pnl = gross - cost

        expected_risk_amount = abs(size)  # simplistic; RiskEngine will interpret with its sizer

        # check with risk engine (if present)
        allowed = True
        reason = None
        if risk_engine is not None:
            try:
                allowed, reason = risk_engine.allow_trade(expected_risk_amount=expected_risk_amount,
                                                          proposed_size=abs(size),
                                                          trade_meta={"regime": row.get("regime"), "date": row.get(TIME_COL)})
            except Exception:
                # on any risk check failure, prefer skipping trade (conservative)
                allowed = False
                reason = "risk_engine_error"
        if not allowed:
            skipped += 1
            if observability:
                try:
                    observability.log_event("trade_skipped_by_risk", {"idx": idx, "reason": reason})
                except Exception:
                    pass
            trades.append({"idx": idx, "pnl": 0.0, "size": 0.0, "regime": row.get("regime", "all"), "date": row.get(TIME_COL)})
            continue

        # detect slippage anomaly via observability (best-effort)
        if observability:
            try:
                observability.detect_slippage_anomaly(cost, SLIPPAGE_ANOMALY_THRESHOLD_PCT, (risk_engine.capital if risk_engine else DEFAULT_STARTING_CAPITAL))
            except Exception:
                pass

        # record trade in risk engine (best-effort)
        if risk_engine is not None:
            try:
                risk_engine.record_trade(realized_pnl=pnl, size=size, trade_meta={"date": row.get(TIME_COL), "regime": row.get("regime")})
            except Exception:
                pass

        trades.append({"idx": idx, "pnl": float(pnl), "size": float(size), "regime": row.get("regime", "all"), "date": row.get(TIME_COL)})

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty and "date" in trades_df.columns and trades_df["date"].notna().any():
        daily = trades_df.groupby("date")["pnl"].sum().sort_index()
    else:
        daily = trades_df.groupby(trades_df.index)["pnl"].sum()

    total = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
    n_trades = int((trades_df["size"] != 0).sum()) if not trades_df.empty else 0
    # log the simulation end event (if main observability supplied this will persist)
    if observability:
        try:
            observability.log_event("simulate_trading_complete", {"total_pnl": total, "n_trades": n_trades, "skipped": skipped})
        except Exception:
            pass

    return {"total_pnl": total, "n_trades": n_trades, "trades_df": trades_df, "daily_pnl": daily}

# ----------------------------
# Simple strategy repair/clamp/mutate/crossover
# ----------------------------
def _repair_strategy(raw: dict) -> dict:
    s = {}
    for k, v in (raw or {}).items():
        if isinstance(v, str):
            vs = v.strip()
            if vs.endswith('%'):
                try:
                    s[k] = float(vs.strip('%')) / 100.0
                    continue
                except Exception:
                    pass
            try:
                if vs.replace('.', '', 1).lstrip('-').isdigit():
                    s[k] = float(vs) if '.' in vs else int(vs)
                    continue
            except Exception:
                pass
            s[k] = v
        else:
            s[k] = v
    # set defaults
    s.setdefault('direction', 'neutral')
    try:
        s['direction'] = str(s['direction']).lower()
        if s['direction'] not in ('bull', 'bear', 'neutral'):
            s['direction'] = 'neutral'
    except Exception:
        s['direction'] = 'neutral'
    for k in ('min_iv', 'max_iv', 'size_aggressiveness'):
        if k in s:
            try:
                s[k] = float(s[k])
            except Exception:
                s[k] = 0.0
    for k in ('dte_min', 'dte_max'):
        if k in s:
            try:
                s[k] = int(s[k])
            except Exception:
                s[k] = 0
    if 'min_iv' not in s: s['min_iv'] = 0.0
    if 'max_iv' not in s: s['max_iv'] = 1.0
    return s

def _clamp_to_bounds(candidate: dict, bounds: dict) -> dict:
    c = deepcopy(candidate)
    if 'min_iv' in c:
        c['min_iv'] = max(bounds.get('iv_min', 0.0), min(bounds.get('iv_max', 1.0), float(c['min_iv'])))
    if 'max_iv' in c:
        c['max_iv'] = max(bounds.get('iv_min', 0.0), min(bounds.get('iv_max', 1.0), float(c['max_iv'])))
    if 'min_iv' in c and 'max_iv' in c and c['min_iv'] > c['max_iv']:
        c['min_iv'], c['max_iv'] = c['max_iv'], c['min_iv']
    if 'dte_min' in c:
        c['dte_min'] = int(max(bounds.get('dte_min', 0), min(bounds.get('dte_max', 9999), int(c['dte_min']))))
    if 'dte_max' in c:
        c['dte_max'] = int(max(bounds.get('dte_min', 0), min(bounds.get('dte_max', 9999), int(c['dte_max']))))
    if 'dte_min' in c and 'dte_max' in c and c['dte_min'] > c['dte_max']:
        c['dte_min'], c['dte_max'] = c['dte_max'], c['dte_min']
    return c

def _mutate_strategy(base: dict, exploration_scale: float = 0.2, bounds: Optional[dict] = None) -> dict:
    s = deepcopy(base)
    bounds = bounds or {}
    for key in ('min_iv', 'max_iv', 'size_aggressiveness'):
        if key in s:
            try:
                scale = 1.0 + np.random.uniform(-exploration_scale, exploration_scale)
                s[key] = float(s[key]) * scale
            except Exception:
                s[key] = float(s.get(key, 0.0))
            if key in ('min_iv', 'max_iv') and bounds:
                s[key] = max(bounds.get('iv_min', 0.0), min(bounds.get('iv_max', 1.0), s[key]))
    for key in ('dte_min', 'dte_max'):
        if key in s:
            try:
                delta = int(np.random.uniform(-int(5 * exploration_scale * 10 + 1), int(5 * exploration_scale * 10 + 1)))
                s[key] = int(s.get(key, 0)) + delta
            except Exception:
                s[key] = int(s.get(key, 0))
            if bounds:
                s[key] = max(bounds.get('dte_min', 0), min(bounds.get('dte_max', 9999), s[key]))
    if random.random() < 0.05:
        s['direction'] = random.choice(['bull', 'bear', 'neutral'])
    s = _repair_strategy(s)
    s = _clamp_to_bounds(s, bounds or {})
    return s

def _crossover(a: dict, b: dict, bounds: Optional[dict] = None) -> dict:
    child = {}
    child['direction'] = random.choice([a.get('direction', 'neutral'), b.get('direction', 'neutral')])
    for k in ('min_iv', 'max_iv', 'size_aggressiveness'):
        va = a.get(k); vb = b.get(k)
        vals = [v for v in (va, vb) if v is not None]
        if vals:
            avg = float(sum(vals) / len(vals))
            noise = np.random.normal(0, 0.05 * (abs(avg) + 1e-6))
            child[k] = float(max(0.0, avg + noise))
    for k in ('dte_min', 'dte_max'):
        va = a.get(k); vb = b.get(k)
        present = [v for v in (va, vb) if v is not None]
        if present:
            avg = int(sum(present) / len(present))
            delta = int(np.random.randint(-5, 6))
            child[k] = max(0, avg + delta)
    # allowed_regimes union
    ar = []
    if a.get('allowed_regimes'): ar.extend(a.get('allowed_regimes', []))
    if b.get('allowed_regimes'): ar.extend(b.get('allowed_regimes', []))
    if ar:
        child['allowed_regimes'] = list(dict.fromkeys(ar))
    child = _repair_strategy(child)
    child = _clamp_to_bounds(child, bounds or {})
    return child

# ----------------------------
# Strategy generation (LLM + memory + fallback)
# ----------------------------
def generate_llm_challengers(n: int = 3, exploration_temp: float = 0.2, past_winners: Optional[List[dict]] = None, bounds: Optional[dict] = None, prefer_memory: bool = True) -> List[dict]:
    """
    Attempt to generate candidate strategies. Prefer mutated past winners first (if prefer_memory),
    then fallback to templates and random seeds. This function is synchronous and simple.
    """
    candidates = []
    past_winners = past_winners or []
    bounds = bounds or {}

    # Prefer memory — mutate top winners
    if prefer_memory and past_winners:
        for w in past_winners:
            if len(candidates) >= n:
                break
            strat = w.get('strategy') if isinstance(w, dict) and 'strategy' in w else w
            if not isinstance(strat, dict):
                continue
            mutated = _mutate_strategy(strat, exploration_scale=exploration_temp, bounds=bounds)
            candidates.append(mutated)

    # Try to call an LLM (ollama) if available — keep safe: timeout + robust parsing
    try:
        # we won't fail pipeline if ollama isn't installed — fallback to templates below
        import subprocess
        prompt = "Produce a JSON array of 3 candidate option strategy objects. Each must include: direction(bull|bear|neutral), min_iv, max_iv, dte_min, dte_max. Optional: size_aggressiveness, allowed_regimes."
        proc = subprocess.run(["ollama", "run", "llama3.2", "--no-stream", "--prompt", prompt], capture_output=True, check=False, timeout=18)
        raw = proc.stdout.decode("utf-8", errors="ignore").strip()
        if raw:
            start = raw.find('['); end = raw.rfind(']')
            if start != -1 and end != -1 and end > start:
                block = raw[start:end+1]
                parsed = json.loads(block)
                for p in parsed:
                    if len(candidates) >= n:
                        break
                    repaired = _repair_strategy(p)
                    mutated = _mutate_strategy(repaired, exploration_scale=exploration_temp, bounds=bounds)
                    candidates.append(mutated)
    except Exception:
        # no-op: LLM generation is optional
        pass

    # Fill with fallback templates mutated
    for t in LLM_FALLBACK_TEMPLATES:
        if len(candidates) >= n: break
        mutated = _mutate_strategy(t, exploration_scale=exploration_temp, bounds=bounds)
        candidates.append(mutated)

    # Deduplicate preserving order
    out = []
    seen = set()
    for c in candidates:
        key = json.dumps(c, sort_keys=True)
        if key in seen: continue
        seen.add(key)
        out.append(c)
        if len(out) >= n: break

    logger.info("generate_llm_challengers -> returning %d candidates (explore_temp=%.3f)", len(out), exploration_temp)
    return out

# ----------------------------
# Diversity & clustering utilities
# ----------------------------
def _strategy_vector(candidate: dict, bounds: dict) -> np.ndarray:
    iv_min, iv_max = bounds.get('iv_min', 0.0), bounds.get('iv_max', 1.0)
    dte_min, dte_max = bounds.get('dte_min', 0), bounds.get('dte_max', 999)
    vec = []
    for k in ('min_iv', 'max_iv'):
        v = float(candidate.get(k, iv_min))
        vec.append((v - iv_min) / max(1e-9, iv_max - iv_min))
    for k in ('dte_min', 'dte_max'):
        v = float(candidate.get(k, dte_min))
        vec.append((v - dte_min) / max(1e-9, dte_max - dte_min))
    v = float(candidate.get('size_aggressiveness', 1.0))
    vec.append((v - 0.2) / (2.0 - 0.2))
    dir_map = {'bull': 0.0, 'neutral': 0.5, 'bear': 1.0}
    vec.append(dir_map.get(candidate.get('direction', 'neutral'), 0.5))
    return np.array(vec, dtype=float)

def param_distance(a: dict, b: dict, bounds: dict) -> float:
    va = _strategy_vector(a, bounds)
    vb = _strategy_vector(b, bounds)
    return float(np.linalg.norm(va - vb))

def diversity_filter(candidates: List[dict], memory: List[dict], bounds: dict, min_distance: float = MIN_PARAM_DISTANCE) -> List[dict]:
    kept = []
    mem_strats = []
    for m in memory:
        try:
            s = m.get('strategy', m) if isinstance(m, dict) else m
            if isinstance(s, dict):
                mem_strats.append(s)
        except Exception:
            pass
    for c in candidates:
        too_close = False
        for mem in mem_strats:
            if param_distance(c, mem, bounds) < min_distance:
                too_close = True; break
        if too_close: continue
        for k in kept:
            if param_distance(c, k, bounds) < min_distance:
                too_close = True; break
        if too_close: continue
        kept.append(c)
    logger.info("Diversity filter: kept %d / %d candidates", len(kept), len(candidates))
    return kept

def cluster_parents(past_list: List[dict], n_clusters: int = 3, bounds: Optional[dict] = None):
    if not past_list: return []
    bounds = bounds or {}
    X, recs = [], []
    for r in past_list:
        try:
            s = r.get('strategy', r) if isinstance(r, dict) else r
            vec = _strategy_vector(s, bounds)
            X.append(vec); recs.append(r)
        except Exception:
            continue
    if len(X) <= 1:
        return [[r] for r in recs]
    X = np.stack(X)
    k = min(n_clusters, len(recs))
    if SKLEARN_AVAILABLE and k >= 2:
        try:
            km = KMeans(n_clusters=k, random_state=42).fit(X)
            clusters = [[] for _ in range(k)]
            for idx, label in enumerate(km.labels_):
                clusters[label].append(recs[idx])
            return clusters
        except Exception:
            pass
    clusters = [[] for _ in range(k)]
    for i, r in enumerate(recs):
        clusters[i % k].append(r)
    return clusters

# ----------------------------
# Champion load / compare / promote
# ----------------------------
def load_champion(path: Path):
    if not path.exists():
        return None, None
    try:
        model = joblib.load(path)
        # attempt to load metrics
        metrics = None
        if Path(CHAMPION_METRICS_JSON).exists():
            try:
                metrics = json.load(open(CHAMPION_METRICS_JSON, 'r'))
            except Exception:
                metrics = None
        return model, metrics
    except Exception:
        logger.exception("Failed to load champion model")
        return None, None

def compare_and_promote(champion_model, champion_metrics, challenger_model, challenger_metrics):
    try:
        champ_pnl = champion_metrics.get('total_pnl') if champion_metrics else float('nan')
    except Exception:
        champ_pnl = float('nan')
    chall_pnl = challenger_metrics.get('total_pnl', float('-inf'))
    logger.info("Champion total_pnl=%s Challenger total_pnl=%s", champ_pnl, chall_pnl)
    promote = False
    if champion_metrics is None or champ_pnl is None or math.isnan(champ_pnl):
        promote = True
    elif chall_pnl >= champ_pnl - 1e-6:
        promote = True
    if promote:
        try:
            joblib.dump(challenger_model, CHAMPION_MODEL_PATH)
            with open(CHAMPION_METRICS_JSON, 'w') as f:
                json.dump(challenger_metrics, f, default=_json_default, indent=2)
            logger.info("Promoted challenger to champion and wrote metrics")
        except Exception:
            logger.exception("Failed to persist champion")
        return True
    return False

# ----------------------------
# Past winners memory
# ----------------------------
def load_past_winners(limit: int = PAST_WINNERS_MAX) -> List[dict]:
    try:
        if Path(PAST_WINNERS_PATH).exists():
            arr = json.load(open(PAST_WINNERS_PATH, 'r'))
            return arr[:limit]
    except Exception:
        pass
    return []

def update_past_winners(strategy: dict, pnl: float, meta: dict, max_len: int = PAST_WINNERS_MAX, rolling_len: int = PAST_WINNERS_ROLLING):
    arr = load_past_winners(max_len)
    key = None
    try:
        key = json.dumps(strategy, sort_keys=True)
    except Exception:
        key = str(strategy)
    found = False
    for rec in arr:
        try:
            existing_key = json.dumps(rec.get('strategy', rec), sort_keys=True)
        except Exception:
            existing_key = str(rec.get('strategy', rec))
        if existing_key == key:
            hist = rec.get('pnl_history', [])
            hist.insert(0, float(pnl))
            hist = hist[:rolling_len]
            rec['pnl_history'] = hist
            rec['avg_pnl'] = float(np.mean(hist)) if hist else 0.0
            rec['stability'] = float(np.std(hist)) if len(hist) > 1 else 0.0
            rec['meta'] = meta
            rec['last_seen'] = time.time()
            found = True
            break
    if not found:
        new = {"strategy": strategy, "pnl_history": [float(pnl)], "avg_pnl": float(pnl), "stability": 0.0, "meta": meta, "first_seen": time.time(), "last_seen": time.time()}
        arr.insert(0, new)
    # dedupe & trim
    uniq = []
    seen = set()
    for r in arr:
        try:
            k = json.dumps(r.get('strategy', r), sort_keys=True)
        except Exception:
            k = str(r)
        if k in seen: continue
        seen.add(k); uniq.append(r)
        if len(uniq) >= max_len: break
    try:
        json.dump(uniq, open(PAST_WINNERS_PATH, 'w'), indent=2)
    except Exception:
        logger.debug("Failed to persist past winners memory")

# ----------------------------
# Candidate evaluation (executed in worker processes)
# ----------------------------
def evaluate_candidate_worker(candidate: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, base_features: List[str], bounds: dict, sim_kwargs: dict) -> Optional[dict]:
    """
    Worker-safe evaluation wrapper.
    IMPORTANT: sim_kwargs should NOT contain main-process Observability instance.
    Workers will create WorkerObservability() and pass to simulate_trading.
    """
    try:
        # repair and clamp candidate
        cand = _repair_strategy(candidate)
        cand = _clamp_to_bounds(cand, bounds or {})

        # Build per-strategy dataset using strategy_adapter
        strat_train, strat_val, strat_features, strat_meta = apply_strategy_adapter(train_df, val_df, strategy_json=cand, base_features=base_features, debug=False)
        if len(strat_train) < 3000 or len(strat_val) < 200:
            # too small sample — skip
            return None

        # train local model
        model, rmse = build_numeric_challenger(strat_train, strat_val, strat_features)

        # Use WorkerObservability inside worker
        worker_obs = WorkerObservability()
        worker_sim_kwargs = dict(sim_kwargs)
        worker_sim_kwargs['observability'] = worker_obs

        metrics = simulate_trading(model, strat_val, strat_features, **worker_sim_kwargs)

        result = {
            "strategy": cand,
            "meta": strat_meta,
            "rmse": float(rmse),
            "metrics": {"total_pnl": float(metrics.get("total_pnl", 0.0)), "n_trades": int(metrics.get("n_trades", 0)), "daily_pnl": metrics.get("daily_pnl"), "trades_df": metrics.get("trades_df")}
        }
        return result
    except Exception as e:
        # Return None on errors but log a bit
        logger.debug("Worker evaluation error: %s", e)
        return None

# ----------------------------
# Hard bounds computation
# ----------------------------
def compute_hard_bounds(train_df: pd.DataFrame) -> dict:
    bounds = {}
    if 'iv_bs' in train_df.columns:
        try:
            bounds['iv_min'] = float(train_df['iv_bs'].quantile(0.01))
            bounds['iv_max'] = float(train_df['iv_bs'].quantile(0.99))
        except Exception:
            bounds['iv_min'], bounds['iv_max'] = 0.0, 1.0
    else:
        bounds['iv_min'], bounds['iv_max'] = 0.0, 1.0
    if 'dte' in train_df.columns:
        try:
            bounds['dte_min'] = int(train_df['dte'].quantile(0.01))
            bounds['dte_max'] = int(train_df['dte'].quantile(0.99))
        except Exception:
            bounds['dte_min'], bounds['dte_max'] = 0, 9999
    else:
        bounds['dte_min'], bounds['dte_max'] = 0, 9999
    if 'moneyness' in train_df.columns:
        try:
            bounds['moneyness_min'] = float(train_df['moneyness'].quantile(0.01))
            bounds['moneyness_max'] = float(train_df['moneyness'].quantile(0.99))
        except Exception:
            bounds['moneyness_min'], bounds['moneyness_max'] = -10.0, 10.0
    else:
        bounds['moneyness_min'], bounds['moneyness_max'] = -10.0, 10.0
    logger.info("Computed hard bounds: %s", bounds)
    return bounds

# ----------------------------
# Main orchestration
# ----------------------------
def main():
    random.seed(42)
    np.random.seed(42)

    run_id = f"run_{int(time.time())}"
    obs = Observability(run_id=run_id)
    obs.log_event("main_start", {"run_id": run_id})
    obs.start_latency("total_run")

    # initialize risk engine & sizer (persist disabled by default)
    starting_capital = float(DEFAULT_STARTING_CAPITAL)
    sizer = PositionSizer(capital=starting_capital, max_risk_pct=DEFAULT_MAX_RISK_PCT, risk_per_unit_est=1.0, min_size=1.0, max_size=1000.0)
    engine = RiskEngine(position_sizer=sizer, capital=starting_capital, daily_loss_limit_pct=DEFAULT_DAILY_LOSS_PCT, enable_persistence=False)

    # load features + labels (if needed)
    obs.start_latency("load_features")
    df = safe_load_parquet(FEATURES_PATH)
    obs.end_latency("load_features")
    obs.log_event("features_loaded", {"shape": df.shape})

    if TARGET_COL not in df.columns:
        # --- FIX: Ensure both sides have datetime64 dtype before merge ---
        labels = pd.read_parquet(LABELS_PATH)

        # Normalize date types
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
        labels[TIME_COL] = pd.to_datetime(labels[TIME_COL], errors='coerce')

        df = df.merge(labels, on=[TIME_COL], how="left")
        obs.log_event("labels_merged", {"shape": df.shape})


    # detect gaps
    try:
        obs.detect_data_gaps(df, time_col=TIME_COL, tolerance_days=1)
    except Exception:
        pass

    # features
    features = auto_select_features(df)
    # prefer some domain features at front if present
    for pref in ['iv_bs', 'iv_imputed', 'moneyness', 'dte', 'atr_by_price']:
        if pref in df.columns and pref not in features:
            features.insert(0, pref)

    train_df, val_df, features = prepare_datasets(df, features)
    bounds = compute_hard_bounds(train_df)

    # simulation kwargs (do NOT include the main Observability instance)
    sim_kwargs = {
        "position_sizing": "proportional",
        "fixed_size": 1.0,
        "size_scale": 5.0,
        "max_size": 100.0,
        "slippage_per_unit": 0.0,
        "commission_per_unit": 0.0,
        "skip_zero_pred": True,
        "prediction_threshold": 0.0,
        "risk_engine": engine,
        # intentionally exclude 'observability' to keep workers picklable
    }

    # Numeric challenger training & simulation (main process)
    obs.start_latency("numeric_train")
    numeric_model, numeric_rmse = build_numeric_challenger(train_df, val_df, features)
    obs.end_latency("numeric_train")
    obs.log_event("numeric_trained", {"rmse": numeric_rmse})

    obs.start_latency("numeric_sim")
    numeric_metrics = simulate_trading(numeric_model, val_df, features, **{**sim_kwargs, "observability": obs})
    obs.end_latency("numeric_sim")
    obs.log_event("numeric_simulation", {"total_pnl": numeric_metrics.get("total_pnl", 0.0), "n_trades": numeric_metrics.get("n_trades", 0)})

    obs.log_candidate_summary({
        "candidate_id": "numeric_challenger",
        "total_pnl": float(numeric_metrics.get("total_pnl", 0.0)),
        "rmse": float(numeric_rmse),
        "n_trades": int(numeric_metrics.get("n_trades", 0)),
        "win_rate": float(numeric_metrics.get("trades_df", pd.DataFrame()).query("pnl>0").shape[0] / max(1, int(numeric_metrics.get("n_trades", 0)))) if numeric_metrics.get("n_trades", 0) > 0 else 0.0,
        "risk_adj_score": float(risk_adjusted_score({"total_pnl": numeric_metrics.get("total_pnl", 0.0), "daily_pnl": numeric_metrics.get("daily_pnl", [])}, risk_penalty_lambda=1.0))
    })

    # load champion if available and simulate for baseline
    champ_model, champ_metrics = load_champion(CHAMPION_MODEL_PATH)
    champion_metrics = None
    if champ_model is not None:
        try:
            obs.start_latency("champion_sim")
            champion_metrics = simulate_trading(champ_model, val_df, features, **{**sim_kwargs, "observability": obs})
            obs.end_latency("champion_sim")
            obs.log_event("champion_loaded", {"total_pnl": champion_metrics.get("total_pnl", 0.0)})
            logger.info("Loaded champion total_pnl=%s", champion_metrics.get("total_pnl", 0.0))
        except Exception:
            champion_metrics = None

    # Compare numeric challenger vs champion
    promoted = compare_and_promote(champ_model, champion_metrics, numeric_model, numeric_metrics)
    if promoted:
        logger.info("Numeric challenger promoted to champion")
        champ_model = numeric_model
        champion_metrics = numeric_metrics

    # prepare memory & leaderboard
    past_winners = load_past_winners(limit=20)
    leaderboard: List[dict] = []
    total_evaluated = 0
    best_llm_overall = None

    exploration_temp = EXPLORATION_TEMP_BASE
    stagnation_counter = 0

    # LLM evolution rounds
    for round_idx in range(N_ROUNDS):
        obs.log_event("llm_round_start", {"round": round_idx + 1, "explore_temp": exploration_temp})
        logger.info("LLM generation round %d/%d (explore_temp=%.3f)", round_idx+1, N_ROUNDS, exploration_temp)

        raw_candidates = generate_llm_challengers(n=CANDIDATES_PER_ROUND, exploration_temp=exploration_temp, past_winners=past_winners, bounds=bounds, prefer_memory=True)
        candidates = diversity_filter(raw_candidates, past_winners, bounds, min_distance=MIN_PARAM_DISTANCE)

        # crossover children produced from cluster representatives
        crossover_children = []
        clusters = cluster_parents(leaderboard or past_winners, n_clusters=3, bounds=bounds)
        reps = []
        for cl in clusters:
            if not cl: continue
            top = sorted(cl, key=lambda x: x.get('pnl', x.get('avg_pnl', 0.0)), reverse=True)[0]
            reps.append(top)
        for a, b in zip(reps, reps[1:]):
            sa = a.get('strategy', a)
            sb = b.get('strategy', b)
            if isinstance(sa, dict) and isinstance(sb, dict):
                child = _crossover(sa, sb, bounds=bounds)
                child = _mutate_strategy(child, exploration_scale=exploration_temp, bounds=bounds)
                crossover_children.append(child)
            if len(crossover_children) >= 3:
                break

        candidates_to_eval = candidates + crossover_children
        candidates_to_eval = diversity_filter(candidates_to_eval, past_winners + [i.get('strategy') for i in leaderboard], bounds, min_distance=MIN_PARAM_DISTANCE)

        if not candidates_to_eval:
            # widen exploration and re-generate
            exploration_temp = min(EXPLORATION_TEMP_MAX, exploration_temp + EXPLORATION_TEMP_INC)
            logger.warning("No candidates after filtering — increased exploration_temp to %.3f", exploration_temp)
            raw_candidates = generate_llm_challengers(n=CANDIDATES_PER_ROUND, exploration_temp=exploration_temp, past_winners=past_winners, bounds=bounds, prefer_memory=False)
            candidates_to_eval = diversity_filter(raw_candidates, past_winners, bounds, min_distance=MIN_PARAM_DISTANCE)

        # run evaluations in parallel (workers get WorkerObservability via evaluate_candidate_worker)
        obs.start_latency("parallel_eval")
        try:
            # keep sim_kwargs picklable: don't pass the main obs object
            worker_sim_kwargs = dict(sim_kwargs)
            # worker function will create WorkerObservability internally
            results = Parallel(n_jobs=CPU_PARALLEL_JOBS, verbose=0)(
                delayed(evaluate_candidate_worker)(cand, train_df, val_df, features, bounds, worker_sim_kwargs) for cand in candidates_to_eval
            )
        except Exception as e:
            logger.exception("Parallel evaluation failed: %s", e)
            results = []
        obs.end_latency("parallel_eval")

        round_best_pnl = float('-inf')
        round_best_rec = None

        for res in results:
            if not res:
                continue
            total_evaluated += 1
            cand = res['strategy']
            pnl = float(res['metrics'].get('total_pnl', 0.0))
            rmse = float(res.get('rmse', 0.0))
            n_trades = int(res['metrics'].get('n_trades', 0))
            logger.info("Candidate PnL=%.6f rmse=%.6f", pnl, rmse)

            # persist candidate result
            rid = f"r{round_idx+1}_{int(time.time()*1000)}_{random.randint(0,9999)}"
            try:
                json.dump({"strategy": cand, "meta": res.get('meta', {}), "rmse": rmse, "pnl": pnl, "n_trades": n_trades, "timestamp": time.time()},
                          open(LLM_RESULTS_DIR / f"{rid}.json", "w"), indent=2, default=_json_default)
            except Exception:
                logger.debug("Failed to persist candidate json")

            # try persist trades if present
            try:
                trades_df = res['metrics'].get('trades_df', pd.DataFrame())
                if not trades_df.empty:
                    trades_df.to_parquet(LLM_TRADES_DIR / f"{rid}.parquet")
            except Exception:
                logger.debug("Failed to persist trades parquet")

            # log summary to main observability (main process does the logging/persistence)
            try:
                obs.log_candidate_summary({
                    "candidate_id": rid,
                    "total_pnl": pnl,
                    "rmse": rmse,
                    "n_trades": n_trades,
                    "win_rate": float(trades_df.query("pnl>0").shape[0]) / max(1, n_trades) if n_trades > 0 else 0.0,
                    "risk_adj_score": float(risk_adjusted_score({"total_pnl": pnl, "daily_pnl": res['metrics'].get('daily_pnl', [])}, risk_penalty_lambda=1.0))
                })
            except Exception:
                logger.debug("Failed to write candidate summary to observability")

            # on promising candidate, perform synchronous retrain+simulate (safe, main process)
            try:
                champ_pnl = champion_metrics.get('total_pnl') if champion_metrics else float('-inf')
                if pnl >= champ_pnl:
                    logger.info("Candidate outperforms champion -> synchronous retrain + simulate")
                    # full training with apply_strategy_adapter
                    strat_train, strat_val, strat_features, strat_meta = apply_strategy_adapter(train_df, val_df, strategy_json=cand, base_features=features, debug=False)
                    if len(strat_train) >= 3000 and len(strat_val) >= 200:
                        strat_model, _ = build_numeric_challenger(strat_train, strat_val, strat_features)
                        strat_metrics = simulate_trading(strat_model, val_df, features, **{**sim_kwargs, "observability": obs})
                        promoted2 = compare_and_promote(champ_model, champion_metrics, strat_model, strat_metrics)
                        if promoted2:
                            champ_model = strat_model
                            champion_metrics = strat_metrics
                            try:
                                json.dump(cand, open(OUT_MODEL_DIR / 'champion_strategy.json', 'w'), indent=2, default=_json_default)
                            except Exception:
                                pass
            except Exception:
                logger.exception("Synchronous retrain/simulate failed")

            # update leaderboard and memory
            leaderboard.append({"strategy": cand, "pnl": pnl, "meta": res.get('meta', {})})
            leaderboard = sorted(leaderboard, key=lambda x: x["pnl"], reverse=True)[:LEADERBOARD_SIZE]
            try:
                update_past_winners(cand, pnl, res.get('meta', {}))
            except Exception:
                logger.debug("Failed to update past winners memory")

            # track bests
            if pnl > round_best_pnl:
                round_best_pnl = pnl
                round_best_rec = res
            if best_llm_overall is None or pnl > (best_llm_overall.get('metrics', {}).get('total_pnl', float('-inf'))):
                best_llm_overall = {"strategy": cand, "metrics": res['metrics'], "meta": res.get('meta', {})}

        # stagnation detection
        champ_pnl_val = champion_metrics.get('total_pnl') if champion_metrics else float('-inf')
        if round_best_pnl <= STAGNATION_RATIO * champ_pnl_val:
            stagnation_counter += 1
            logger.info("Round best pnl %.2f is <= %.2f * champ (%.2f) -> stagnation_counter=%d", round_best_pnl, STAGNATION_RATIO, champ_pnl_val, stagnation_counter)
        else:
            stagnation_counter = 0

        if stagnation_counter >= SMART_RESTART_ROUNDS:
            old_temp = exploration_temp
            exploration_temp = min(EXPLORATION_TEMP_MAX, exploration_temp + EXPLORATION_TEMP_INC)
            logger.warning("Stagnation detected — increased exploration_temp %.3f -> %.3f", old_temp, exploration_temp)
            # inject random seeds into past winners to increase diversity
            for i in range(3):
                rand_template = random.choice(LLM_FALLBACK_TEMPLATES)
                child = _mutate_strategy(rand_template, exploration_scale=exploration_temp, bounds=bounds)
                try:
                    update_past_winners(child, 0.0, {"injected": True})
                except Exception:
                    pass
            stagnation_counter = 0

    # end rounds

    if best_llm_overall:
        obs.log_event("best_llm_overall", {"pnl": float(best_llm_overall["metrics"]["total_pnl"]), "strategy": best_llm_overall["strategy"]})
        logger.info("Best LLM overall: pnl=%.4f", best_llm_overall["metrics"]["total_pnl"])

    # save features list
    try:
        with open(FEATURES_JSON, 'w') as f:
            json.dump(features, f, default=_json_default, indent=2)
        logger.info("Saved feature list to %s", FEATURES_JSON)
    except Exception:
        logger.warning("Failed saving feature list")

    obs.log_event("main_complete", {"total_evaluated": total_evaluated})
    obs.end_latency("total_run")
    obs.end_run()
    logger.info("run_full_historical_training complete")

if __name__ == "__main__":
    main()
