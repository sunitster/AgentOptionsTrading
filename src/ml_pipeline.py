# src/ml_pipeline.py
"""
ML pipeline: LLM-assisted candidate proposer + scorer ensemble + bandit selector + replay buffer.

Key:
- OllamaProposer: accepts either an `ollama.Client` (preferred) or an OpenAI-like client.
- FeatureEngineer: simple numeric features for candidates.
- ScorerEnsemble: ensemble of regressors (LightGBM if available, sklearn fallback).
- ThompsonSelector: picks candidate using Thompson sampling.
- ReplayBufferSQLite: persistent replay buffer for training.
- run_selection_once: glue function to propose, score, select, risk-check, append to replay, and execute via callback.

Drop into `src/ml_pipeline.py` and import as:
from src.ml_pipeline import run_selection_once, ReplayBufferSQLite, ScorerEnsemble, FeatureEngineer
"""

from __future__ import annotations
import json
import os
import sqlite3
import time
import traceback
from typing import Callable, Dict, List, Any, Tuple
import numpy as np
import math
import logging

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# try ML libs
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

try:
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# We'll accept either a real ollama.Client or an OpenAI-like client.
# We deliberately don't import ollama here to keep this file portable;
# the caller passes an instantiated client object.

####################################################################
# Replay buffer (SQLite) - persistent, simple
####################################################################
class ReplayBufferSQLite:
    def __init__(self, path: str = "ml_replay.db"):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                state_json TEXT,
                action_json TEXT,
                features_json TEXT,
                reward REAL,
                closed_at INTEGER,
                meta_json TEXT
            )
            """
        )
        self._conn.commit()

    def append(self, state: dict, action: dict, features: dict, reward: float = None, closed_at: int = None, meta: dict = None) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO trades (timestamp, state_json, action_json, features_json, reward, closed_at, meta_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                int(time.time()),
                json.dumps(state, default=str),
                json.dumps(action, default=str),
                json.dumps(features, default=str),
                None if reward is None else float(reward),
                None if closed_at is None else int(closed_at),
                None if meta is None else json.dumps(meta, default=str),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_reward(self, trade_id: int, reward: float, closed_at: int = None) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE trades SET reward = ?, closed_at = ? WHERE id = ?",
            (float(reward), None if closed_at is None else int(closed_at), int(trade_id)),
        )
        self._conn.commit()

    def sample_recent(self, limit: int = 1000) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT id, timestamp, state_json, action_json, features_json, reward, closed_at, meta_json FROM trades ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r[0],
                "timestamp": r[1],
                "state": json.loads(r[2]) if r[2] else None,
                "action": json.loads(r[3]) if r[3] else None,
                "features": json.loads(r[4]) if r[4] else None,
                "reward": r[5],
                "closed_at": r[6],
                "meta": json.loads(r[7]) if r[7] else None,
            })
        return result

####################################################################
# Ollama / LLM candidate proposer (robust)
####################################################################
class OllamaProposer:
    def __init__(self, client: Any, model_name: str = "llama3.2"):
        """
        client: either an `ollama.Client` instance (preferred) or an OpenAI-like client object.
        model_name: the model to ask for (e.g., "llama3.2").
        """
        self.client = client
        self.model_name = model_name

    def _build_messages(self, snapshot: dict, n_candidates: int = 5) -> List[Dict[str, str]]:
        system = {
            "role": "system",
            "content": (
                "You are a derivatives strategist. Given an options chain snapshot (symbol, spot, expiry, "
                "ATM IV, iv_skew, available strikes and recent context), output a JSON array of candidate iron condors. "
                "Each candidate must be a JSON object with keys: short_put, long_put, short_call, long_call, width, "
                "entry_credit, suggested_lots, rationale (short). Output only valid JSON (array). Up to n_candidates."
            ),
        }
        user = {"role": "user", "content": json.dumps({"snapshot": snapshot, "n_candidates": n_candidates}, default=str)}
        return [system, user]

    def _call_ollama_client(self, messages: List[Dict[str, str]], timeout_s: int = 10) -> str:
        """
        Call an ollama.Client instance and return the content string.
        """
        # The expected shape: client.chat(model=..., messages=[...]) -> dict with 'message': {'content': ...}
        try:
            # If client is an ollama.Client, it should have a .chat(...) method
            if hasattr(self.client, "chat") and callable(getattr(self.client, "chat")):
                # Ollama python client expects messages as list of dicts with role/content
                # Some versions accept either direct messages or list-of-dicts; we pass the built messages.
                resp = self.client.chat(model=self.model_name, messages=messages)
                # resp may be dict-like; try to extract message.content
                if isinstance(resp, dict):
                    # Common formats:
                    if "message" in resp and isinstance(resp["message"], dict) and "content" in resp["message"]:
                        return resp["message"]["content"]
                    # Some clients may give a string directly
                    if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                        ch = resp["choices"][0]
                        if isinstance(ch, dict):
                            return ch.get("message", {}).get("content") or ch.get("text") or str(resp)
                # fallback to string representation
                return str(resp)
        except Exception:
            # bubble up for the caller to try other methods
            raise

    def _call_openai_like(self, messages: List[Dict[str, str]], timeout_s: int = 10) -> str:
        """
        Call an OpenAI-like client (with chat.completions.create) and return the content string.
        """
        try:
            # Attempt common OpenAI-like path
            if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions") and hasattr(self.client.chat.completions, "create"):
                resp = self.client.chat.completions.create(model=self.model_name, messages=messages)
                # Try to extract content in OpenAI wrapper shape
                try:
                    return resp.choices[0].message.content
                except Exception:
                    return str(resp)
            # fallback: try direct completions.create
            if hasattr(self.client, "completions") and hasattr(self.client.completions, "create"):
                resp = self.client.completions.create(model=self.model_name, messages=messages)
                try:
                    return resp.choices[0].message.content
                except Exception:
                    return str(resp)
        except Exception:
            raise

    def _http_fallback(self, messages: List[Dict[str, str]], timeout_s: int = 10) -> str:
        """
        Direct HTTP fallback to Ollama REST API endpoint: {OLLAMA_URL}/api/chat
        """
        try:
            import requests
            host = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
            url = host + "/api/chat"
            payload = {"model": self.model_name, "messages": messages}
            r = requests.post(url, json=payload, timeout=timeout_s)
            r.raise_for_status()
            text = r.text.strip()
            # Ollama sometimes streams NDJSON; take last JSON object if NDJSON
            if "\n" in text:
                last = text.strip().splitlines()[-1]
                try:
                    jr = json.loads(last)
                except Exception:
                    jr = r.json()
            else:
                jr = r.json()
            # extract message content
            if isinstance(jr, dict):
                if "message" in jr and isinstance(jr["message"], dict) and "content" in jr["message"]:
                    return jr["message"]["content"]
                if "choices" in jr and isinstance(jr["choices"], list) and jr["choices"]:
                    ch = jr["choices"][0]
                    return ch.get("message", {}).get("content") or ch.get("text") or json.dumps(jr)
            return json.dumps(jr)
        except Exception:
            raise

    def propose(self, snapshot: dict, n_candidates: int = 5, timeout_s: int = 10) -> List[dict]:
        messages = self._build_messages(snapshot, n_candidates=n_candidates)

        content = None
        # Try several client shapes in order: ollama.Client -> OpenAI-like -> HTTP fallback
        try:
            content = self._call_ollama_client(messages, timeout_s=timeout_s)
        except Exception:
            try:
                content = self._call_openai_like(messages, timeout_s=timeout_s)
            except Exception:
                try:
                    content = self._http_fallback(messages, timeout_s=timeout_s)
                except Exception as e:
                    # final fallback: log and return empty
                    print("OllamaProposer: all client calls failed:", e)
                    traceback.print_exc()
                    return []

        # Normalize content to string
        if content is None:
            return []

        if not isinstance(content, str):
            content = str(content)

        # Try to parse returned content as JSON
        try:
            cand = json.loads(content)
            # Accept either list or dict with "candidates"
            if isinstance(cand, list):
                return cand
            if isinstance(cand, dict):
                if "candidates" in cand and isinstance(cand["candidates"], list):
                    return cand["candidates"]
                # Some LLMs return an object describing candidates; wrap it
                return [cand]
        except Exception:
            # heuristics: extract first JSON array substring
            try:
                text = content
                start = text.find('[')
                end = text.rfind(']')
                if start != -1 and end != -1 and end > start:
                    maybe = text[start:end+1]
                    cand = json.loads(maybe)
                    if isinstance(cand, list):
                        return cand
            except Exception:
                pass

        # Last attempt: if content looks like a single JSON object, try to parse inline braces
        try:
            text = content.strip()
            if text.startswith("{") and text.endswith("}"):
                cand = json.loads(text)
                return [cand]
        except Exception:
            pass

        # No structured candidates found
        return []

####################################################################
# Feature engineering (simple numeric features)
####################################################################
class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.scaler_fitted = False

    def candidate_to_vector(self, candidate: dict, snapshot: dict) -> Tuple[np.ndarray, List[str]]:
        spot = float(snapshot.get("spot", 0.0) or 0.0)
        atm_iv = float(snapshot.get("atm_iv", 0.0) or 0.0)
        iv_skew = float(snapshot.get("iv_skew", 0.0) or 0.0)
        tte_days = float(snapshot.get("tte_days", snapshot.get("time_to_expiry_days", 0.0)) or 0.0)

        def strike_dist(s):
            try:
                return (float(s) - spot) / (spot + 1e-9)
            except Exception:
                return 0.0

        sp = float(candidate.get("short_put", 0.0) or 0.0)
        lp = float(candidate.get("long_put", 0.0) or 0.0)
        sc = float(candidate.get("short_call", 0.0) or 0.0)
        lc = float(candidate.get("long_call", 0.0) or 0.0)
        entry_credit = float(candidate.get("entry_credit", 0.0) or 0.0)
        width = float(candidate.get("width", abs(sc - sp) if sp and sc else candidate.get("width", 0)) or 0.0)
        lots = float(candidate.get("suggested_lots", 1.0) or 1.0)

        features = [
            spot,
            atm_iv,
            iv_skew,
            tte_days,
            strike_dist(sp),
            strike_dist(lp),
            strike_dist(sc),
            strike_dist(lc),
            entry_credit,
            width,
            lots,
            entry_credit / (width + 1e-9),
            (abs(sp - sc)) / (spot + 1e-9),
        ]
        names = [
            "spot", "atm_iv", "iv_skew", "tte_days",
            "short_put_rel", "long_put_rel", "short_call_rel", "long_call_rel",
            "entry_credit", "width", "lots", "credit_width_ratio", "overall_width_rel"
        ]
        x = np.array(features, dtype=float).reshape(1, -1)
        if self.scaler is not None and self.scaler_fitted:
            x = self.scaler.transform(x)
        return x.flatten(), names

    def fit_scaler_from_features(self, X: np.ndarray):
        if self.scaler is not None:
            self.scaler.fit(X)
            self.scaler_fitted = True

####################################################################
# Scorer ensemble (predict expected reward + uncertainty)
####################################################################
class ScorerEnsemble:
    def __init__(self, n_models: int = 5):
        self.n_models = n_models
        self.models = []
        self.is_lgb = False
        self._build_models()
        self.trained = False
        self.feature_dim = None

    def _build_models(self):
        if LGB_AVAILABLE:
            self.is_lgb = True
            for i in range(self.n_models):
                model = lgb.LGBMRegressor(n_estimators=100, random_state=42 + i, verbosity=-1)
                self.models.append(model)
        elif SKLEARN_AVAILABLE:
            for i in range(self.n_models):
                model = SGDRegressor(max_iter=1000, tol=1e-3)
                self.models.append(model)
        else:
            raise RuntimeError("No regression backend available. Install lightgbm or scikit-learn.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.feature_dim = X.shape[1]
        for i, m in enumerate(self.models):
            idx = np.random.choice(len(X), size=len(X), replace=True)
            Xi = X[idx]
            yi = y[idx]
            try:
                if self.is_lgb:
                    m.fit(Xi, yi)
                else:
                    # partial_fit requires an initial call with classes in some regressors; SGDRegressor supports direct partial_fit
                    for _ in range(3):
                        m.partial_fit(Xi, yi)
            except Exception:
                traceback.print_exc()
        self.trained = True

    def incremental_update(self, X: np.ndarray, y: np.ndarray):
        for m in self.models:
            try:
                if self.is_lgb:
                    m.fit(X, y, init_model=m)
                else:
                    m.partial_fit(X, y)
            except Exception:
                traceback.print_exc()

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = []
        for m in self.models:
            try:
                p = m.predict(X)
                preds.append(p)
            except Exception:
                preds.append(np.zeros(X.shape[0]))
        preds = np.array(preds)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0) + 1e-6
        return mean, std

####################################################################
# Thompson selector (use ensemble to sample)
####################################################################
class ThompsonSelector:
    def __init__(self, scorer: ScorerEnsemble):
        self.scorer = scorer

    def pick(self, X_candidates: np.ndarray) -> int:
        means, stds = self.scorer.predict_with_uncertainty(X_candidates)
        samples = np.random.normal(loc=means, scale=stds)
        chosen = int(np.argmax(samples))
        return chosen

####################################################################
# Glue: run selection once
####################################################################
def _signature_from_candidate(candidate: dict) -> str:
    """
    Build same signature as feedback loop:
    symbol_expiry_shortput_longput_shortcall_longcall
    """
    try:
        sym = candidate.get("symbol") or candidate.get("underlying") or "UNK"
        expiry = candidate.get("expiry") or candidate.get("exp") or ""
        sp = candidate.get("short_put") or candidate.get("shortPut") or candidate.get("short_put_strike") or 0
        lp = candidate.get("long_put") or candidate.get("longPut") or candidate.get("long_put_strike") or 0
        sc = candidate.get("short_call") or candidate.get("shortCall") or candidate.get("short_call_strike") or 0
        lc = candidate.get("long_call") or candidate.get("longCall") or candidate.get("long_call_strike") or 0
        # coerce to ints if possible
        try:
            spv = int(float(sp))
            lpv = int(float(lp))
            scv = int(float(sc))
            lcv = int(float(lc))
        except Exception:
            spv = sp
            lpv = lp
            scv = sc
            lcv = lc
        return f"{sym}_{expiry}_{spv}_{lpv}_{scv}_{lcv}"
    except Exception:
        return str(candidate)


def run_selection_once(
    snapshot: dict,
    ollama_client: Any,
    execute_trade_callback: Callable[[dict], dict],
    risk_check_callback: Callable[[dict], bool],
    replay_db_path: str = "ml_replay.db",
    ensemble: ScorerEnsemble = None,
    fe: FeatureEngineer = None,
    proposer: OllamaProposer = None,
    selector: ThompsonSelector = None,
    n_candidates: int = 5
) -> dict:
    """
    Propose candidates, score, select, append to replay buffer, risk-check, and execute.

    Guarantees:
      - The replay buffer entry id (replay_id) is returned and injected into the trade_payload
        under trade_payload['meta']['replay_id'] (and 'ml_replay_id' / 'replayId' variants).
      - The execute_trade_callback return value is augmented with 'replay_id' where possible.

    New behaviour:
      - Consults models/llm_trades/model_weights.json (if present) and applies signature weights by
        scaling the predicted mean before Thompson sampling:
            sample ~ Normal(loc = mean * weight, scale = std)
      - Unknown signatures use a small epsilon default weight.
    """
    # components
    if fe is None:
        fe = FeatureEngineer()
    if proposer is None:
        proposer = OllamaProposer(ollama_client, model_name=os.getenv("ML_MODEL_NAME", "llama3.2"))
    if ensemble is None:
        ensemble = ScorerEnsemble(n_models=5)
    if selector is None:
        selector = ThompsonSelector(ensemble)
    rb = ReplayBufferSQLite(path=replay_db_path)

    # propose
    candidates = proposer.propose(snapshot, n_candidates=n_candidates)
    if not candidates:
        return {"status": "no_candidates", "candidates": []}

    # features
    X_list = []
    features_list = []
    signatures = []
    for c in candidates:
        try:
            x, names = fe.candidate_to_vector(c, snapshot)
            X_list.append(x)
            features_list.append({"names": names, "vector": list(x)})
            signatures.append(_signature_from_candidate(c))
        except Exception:
            traceback.print_exc()
            continue

    if not X_list:
        return {"status": "no_valid_features", "candidates": candidates}

    X = np.vstack(X_list)

    # warm-start ensemble from replay buffer if untrained
    if not ensemble.trained:
        recent = rb.sample_recent(limit=2000)
        X_hist = []
        y_hist = []
        for r in recent:
            feat = r.get("features")
            if feat and r.get("reward") is not None:
                try:
                    vec = np.array(feat.get("vector") or feat)
                    if len(vec) == X.shape[1]:
                        X_hist.append(vec)
                        y_hist.append(float(r["reward"]))
                except Exception:
                    continue
        if len(X_hist) >= 50:
            X_hist = np.vstack(X_hist)
            y_hist = np.array(y_hist)
            try:
                fe.fit_scaler_from_features(X_hist)
                if fe.scaler is not None and fe.scaler_fitted:
                    X_hist = fe.scaler.transform(X_hist)
                ensemble.fit(X_hist, y_hist)
            except Exception:
                traceback.print_exc()

    # scale if needed
    if fe.scaler is not None and fe.scaler_fitted:
        try:
            X = fe.scaler.transform(X)
        except Exception:
            pass

    # ---------------------------
    # Load model weights (optional)
    # ---------------------------
    weights_path = os.path.join("models", "llm_trades", "model_weights.json")
    weights = {}
    try:
        if os.path.exists(weights_path):
            with open(weights_path, "r") as wf:
                weights = json.load(wf) or {}
                # ensure keys are strings
                weights = {str(k): float(v) for k, v in weights.items()}
                LOG.info("Loaded %d model weights from %s", len(weights), weights_path)
    except Exception:
        LOG.exception("Failed to load model_weights.json - continuing with defaults")
        weights = {}

    # default small epsilon for unseen candidates (keeps exploration alive)
    epsilon_weight = 1e-2

    # ---------------------------
    # If ensemble is not trained, fallback
    # ---------------------------
    chosen_idx = None
    chosen_candidate = None
    chosen_features = None

    try:
        if ensemble.trained:
            # compute means and stds
            means, stds = ensemble.predict_with_uncertainty(X)
            # apply weights by signature
            weighted_means = []
            for i, sig in enumerate(signatures):
                w = weights.get(sig)
                if w is None:
                    # attempt to match with cast signature types (int vs str)
                    w = weights.get(str(sig))
                if w is None:
                    w = epsilon_weight
                # ensure non-negative weight
                try:
                    w = max(0.0, float(w))
                except Exception:
                    w = epsilon_weight
                weighted_means.append(means[i] * w)

            weighted_means = np.array(weighted_means)
            # sample using uncertainty but scale mean by weight
            samples = np.random.normal(loc=weighted_means, scale=stds)
            chosen_idx = int(np.argmax(samples))
            chosen_candidate = candidates[chosen_idx]
            chosen_features = features_list[chosen_idx]
            LOG.info("ML selection: idx=%d sig=%s mean=%.4f std=%.4f weight=%.4f weighted_mean=%.4f", chosen_idx, signatures[chosen_idx], float(means[chosen_idx]), float(stds[chosen_idx]), float(weights.get(signatures[chosen_idx], epsilon_weight)), float(weighted_means[chosen_idx]))
        else:
            # ensemble not trained: fallback to lightweight heuristic using weights only
            # if weights present, pick highest-weight signature; otherwise pick random candidate
            if weights:
                # pick candidate with max weight among signatures (fall back to epsilon)
                weight_vals = [weights.get(sig, weights.get(str(sig), epsilon_weight)) for sig in signatures]
                chosen_idx = int(np.argmax(np.array(weight_vals)))
                chosen_candidate = candidates[chosen_idx]
                chosen_features = features_list[chosen_idx]
                LOG.info("Fallback weight-only selection: idx=%d sig=%s weight=%.4f", chosen_idx, signatures[chosen_idx], float(weight_vals[chosen_idx]))
            else:
                # random fallback
                chosen_idx = int(np.random.randint(0, len(candidates)))
                chosen_candidate = candidates[chosen_idx]
                chosen_features = features_list[chosen_idx]
                LOG.info("Fallback random selection (no ensemble, no weights): idx=%d", chosen_idx)
    except Exception:
        traceback.print_exc()
        # final fallback: pick first candidate
        try:
            chosen_idx = 0
            chosen_candidate = candidates[chosen_idx]
            chosen_features = features_list[chosen_idx]
        except Exception:
            return {"status": "selection_failed", "candidates": candidates}

    # risk check
    trade_payload = {
        "candidate": chosen_candidate,
        "features": chosen_features,
        "snapshot_meta": {"spot": snapshot.get("spot"), "expiry": snapshot.get("expiry"), "tte_days": snapshot.get("tte_days")},
        "meta": {}  # reserve for replay_id and other meta
    }

    allowed = True
    try:
        allowed = risk_check_callback(trade_payload)
    except Exception:
        traceback.print_exc()
        allowed = False

    if not allowed:
        LOG.info("Candidate rejected by risk check (idx=%s sig=%s)", chosen_idx, signatures[chosen_idx] if chosen_idx is not None else "N/A")
        return {"status": "rejected_by_risk", "candidate": chosen_candidate}

    # append to replay buffer (reward None until closed)
    replay_id = None
    try:
        # store state and action; features as dict for future training
        replay_id = rb.append(snapshot, chosen_candidate, chosen_features, reward=None, meta={"source": "llm_bandit"})
    except Exception:
        traceback.print_exc()
        replay_id = None

    # inject replay_id into payload metadata using common key names
    try:
        if "meta" not in trade_payload or trade_payload["meta"] is None:
            trade_payload["meta"] = {}
        # Insert canonical key and some variants for broad compatibility
        if replay_id is not None:
            trade_payload["meta"]["replay_id"] = int(replay_id)
            trade_payload["meta"]["ml_replay_id"] = int(replay_id)
            trade_payload["meta"]["replayId"] = int(replay_id)
        else:
            # still include a placeholder so downstream knows no id present
            trade_payload["meta"]["replay_id"] = None
    except Exception:
        traceback.print_exc()

    # execute
    try:
        exec_res = execute_trade_callback(trade_payload)
        # ensure exec_res is a dict
        if exec_res is None:
            exec_res = {}
        if not isinstance(exec_res, dict):
            # try to coerce
            try:
                exec_res = dict(exec_res)
            except Exception:
                exec_res = {"result": exec_res}
    except Exception:
        traceback.print_exc()
        exec_res = {"placed": False, "error": "execute_callback_exception"}

    # guarantee replay_id present in execute_result for downstream recording
    try:
        if replay_id is not None:
            exec_res.setdefault("meta", {})
            # update meta in exec_res as well
            exec_res["meta"]["replay_id"] = int(replay_id)
            exec_res["meta"]["ml_replay_id"] = int(replay_id)
            exec_res["meta"]["replayId"] = int(replay_id)
        else:
            exec_res.setdefault("meta", {})
            exec_res["meta"].setdefault("replay_id", None)
    except Exception:
        traceback.print_exc()

    return {
        "status": "placed" if exec_res.get("placed", False) else "executed_callback_returned",
        "candidate": chosen_candidate,
        "features": chosen_features,
        "execute_result": exec_res,
        "replay_id": replay_id
    }


# If run as script, show brief usage example
if __name__ == "__main__":
    print("ml_pipeline: module providing run_selection_once(snapshot, ollama_client, execute_trade_callback, risk_check_callback)")
