#!/usr/bin/env python3
# Defensive ML scoring loader for SignalGenerator
# Place as src/scripts/deploy_scoring_hook.py (replace existing)

from __future__ import annotations
import os
import logging
import joblib
import pickle
import time
from typing import Any, Dict, Optional

LOG = logging.getLogger("deploy_scoring_hook")
LOG.setLevel(logging.INFO)

# Optional heavy deps imported lazily
_pd = None
_sklearn = None

def _lazy_import_pandas():
    global _pd
    if _pd is None:
        try:
            import pandas as pd  # type: ignore
            _pd = pd
        except Exception:
            _pd = None
    return _pd

def _ensure_sklearn_imported():
    global _sklearn
    if _sklearn is None:
        try:
            import sklearn  # noqa: F401
            _sklearn = True
        except Exception:
            _sklearn = False
    return _sklearn

class MLScoringEngine:
    """
    Robust loader & scorer for the saved ML bundle.

    Usage:
        engine = MLScoringEngine("models/llm_trades/model.pkl")
        result = engine.score(candidate_features_dict)
        # result = {"ok": True, "p_win": 0.72, "expected_pnl": 12.5, "score": 0.845}
    """

    def __init__(self, model_path: str, timeout_s: float = 3.0):
        self.model_path = model_path
        self.bundle = None
        self.feature_list = []
        self._disabled = False
        self._loaded_at = None

        if not model_path or not os.path.exists(model_path):
            LOG.info("MLScoringEngine: no model file at %s (scoring disabled)", model_path)
            self._disabled = True
            return

        # Make sure sklearn is importable before unpickling (helps pickle namespace)
        _ensure_sklearn_imported()

        start = time.time()
        try:
            # Prefer joblib.load for sklearn objects
            try:
                self.bundle = joblib.load(model_path)
                LOG.info("MLScoringEngine: loaded bundle via joblib from %s", model_path)
            except Exception as e_job:
                LOG.warning("MLScoringEngine: joblib.load() failed (%s). Trying pickle.load()", e_job)
                # pickle fallback
                try:
                    with open(model_path, "rb") as f:
                        self.bundle = pickle.load(f)
                    LOG.info("MLScoringEngine: loaded bundle via pickle from %s", model_path)
                except Exception as e_pick:
                    LOG.exception("MLScoringEngine: pickle.load() also failed: %s", e_pick)
                    self.bundle = None

            if not isinstance(self.bundle, dict):
                # Old-style bundles may be just classifier/regressor directly - normalize
                # Expect either dict with keys 'classifier' and 'regressor' and 'features'
                LOG.info("MLScoringEngine: loaded object type=%s", type(self.bundle))
                if hasattr(self.bundle, "get") and callable(getattr(self.bundle, "get")):
                    # fine, keep
                    pass
                else:
                    # attempt to wrap into dict assuming it's a classifier (best-effort)
                    try:
                        self.bundle = {"classifier": self.bundle, "regressor": None, "features": []}
                        LOG.info("MLScoringEngine: normalized legacy bundle into dict")
                    except Exception:
                        LOG.warning("MLScoringEngine: couldn't normalize loaded model; disabling scoring")
                        self._disabled = True
                        return

            # capture feature list if present
            fl = self.bundle.get("features") if isinstance(self.bundle, dict) else None
            if fl and isinstance(fl, (list, tuple)):
                self.feature_list = [str(x) for x in fl]
            else:
                self.feature_list = []

            self._loaded_at = time.time()
            LOG.info("MLScoringEngine: model loaded (features=%s)", self.feature_list or "<none>")
        except Exception:
            LOG.exception("MLScoringEngine: failed to initialize scoring engine")
            self._disabled = True
        finally:
            took = time.time() - start
            if took > timeout_s:
                LOG.warning("MLScoringEngine: loading took %.2fs (timeout_s=%s)", took, timeout_s)

    @property
    def enabled(self) -> bool:
        return (not self._disabled) and (self.bundle is not None)

    def _build_X(self, candidate_features: Dict[str, Any]):
        """
        Build a DataFrame or 2D array for prediction.
        If the saved model expects feature names, return a pandas DataFrame with those columns.
        Otherwise return a 2D list/array.
        """
        pd = _lazy_import_pandas()
        if self.feature_list and pd is not None:
            # Use pandas so sklearn recognizes feature names (avoid the warning)
            row = {}
            for f in self.feature_list:
                v = candidate_features.get(f, 0.0)
                try:
                    row[f] = float(v)
                except Exception:
                    row[f] = 0.0
            try:
                df = pd.DataFrame([row], columns=self.feature_list)
                return df
            except Exception:
                # fallback to simple list-of-lists
                try:
                    return [[float(candidate_features.get(f, 0.0) or 0.0) for f in self.feature_list]]
                except Exception:
                    return [[0.0 for _ in (self.feature_list or [0])]]
        else:
            # feature list not available; convert whatever numeric values we can
            # Use stable deterministic ordering of keys to ensure reproduciblity
            keys = sorted(candidate_features.keys())
            try:
                arr = [[float(candidate_features.get(k, 0.0) or 0.0) for k in keys]]
                return arr
            except Exception:
                return [[0.0]]

    def score(self, candidate_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a dictionary:
          {"ok": True/False, "p_win": float|None, "expected_pnl": float|None, "score": float|None}
        Always defensive: never raises; logs and returns ok=False on failure.
        """
        if not self.enabled:
            return {"ok": False, "reason": "disabled"}

        clf = self.bundle.get("classifier")
        reg = self.bundle.get("regressor")
        features = self.feature_list or []

        try:
            X = self._build_X(candidate_features)

            p_win = None
            expected_pnl = None

            # Classifier: prefer predict_proba, fallback to predict
            if clf is not None:
                try:
                    # sklearn classifiers accept DataFrame or array; this avoids the feature-name warning
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(X)
                        # class order unknown; try to pick prob for class '1' if present
                        try:
                            # if classes_ present, find index of label 1
                            classes = getattr(clf, "classes_", None)
                            if classes is not None:
                                idx = 1 if 1 in classes else (0 if len(classes) == 1 else 1)
                                p_win = float(probs[0][idx])
                            else:
                                p_win = float(probs[0][-1])
                        except Exception:
                            p_win = float(probs[0][-1])
                    else:
                        p = clf.predict(X)[0]
                        p_win = float(p)
                except Exception:
                    LOG.exception("MLScoringEngine: classifier predict failed")
                    p_win = None

            # Regressor: predict expected pnl
            if reg is not None:
                try:
                    pred = reg.predict(X)[0]
                    expected_pnl = float(pred)
                except Exception:
                    LOG.exception("MLScoringEngine: regressor predict failed")
                    expected_pnl = None

            # Combined score simple formula (same as auto-trainer): p_win + 0.01*expected_pnl
            score = None
            try:
                p_val = p_win if p_win is not None else 0.0
                ep_val = expected_pnl if expected_pnl is not None else 0.0
                score = float(p_val) + 0.01 * float(ep_val)
            except Exception:
                score = None

            return {"ok": True, "p_win": p_win, "expected_pnl": expected_pnl, "score": score}
        except Exception:
            LOG.exception("MLScoringEngine: scoring failed")
            return {"ok": False, "reason": "scoring_failed"}

# If executed stand-alone, allow a simple smoke-test
if __name__ == "__main__":
    import argparse, json, sys
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/llm_trades/model.pkl")
    p.add_argument("--example", default=None, help="json string or file path with candidate features")
    args = p.parse_args()

    engine = MLScoringEngine(args.model)
    print("enabled:", engine.enabled)
    if args.example:
        try:
            if args.example.strip().startswith("{"):
                ex = json.loads(args.example)
            else:
                with open(args.example, "r") as f:
                    ex = json.load(f)
            print("score:", engine.score(ex))
        except Exception as e:
            print("failed to parse example:", e, file=sys.stderr)
            sys.exit(2)
    else:
        print("no example provided")
