#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Self-improving ML model trainer for IC strategy.

Loads:
    - models/llm_trades/features.parquet  OR
    - models/llm_trades/features.csv

Trains:
    - Classifier: p(win)
    - Regressor: expected pnl

Outputs:
    models/llm_trades/model.pkl
"""

import os
import json
import joblib
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("train_model")

FEATURE_FILE_PARQUET = "models/llm_trades/features.parquet"
FEATURE_FILE_CSV = "models/llm_trades/features.csv"
MODEL_OUT = "models/llm_trades/model.pkl"


# -------------------------
# 1) LOAD FEATURES SAFELY
# -------------------------
def load_features():
    if os.path.exists(FEATURE_FILE_PARQUET):
        LOG.info("Loading features from Parquet")
        return pd.read_parquet(FEATURE_FILE_PARQUET)

    if os.path.exists(FEATURE_FILE_CSV):
        LOG.info("Loading features from CSV fallback")
        return pd.read_csv(FEATURE_FILE_CSV)

    raise FileNotFoundError("No feature file found (parquet or csv). Run build_daily_features first.")


# ---------------------------------------------------
# 2) BUILD ML MATRIX — AUTO-HANDLES MISSING FEATURES
# ---------------------------------------------------
def build_matrix(df: pd.DataFrame):
    # REQUIRED LABELS
    if "label_win" not in df.columns:
        raise KeyError("label_win missing from dataset")

    y_clf = df["label_win"].fillna(0).astype(int)

    # regression target (realized_pnl)
    y_reg = df["realized_pnl"].fillna(0.0).astype(float)

    # FEATURES WE WANT
    # (Only include features that actually exist in your parquet/csv)
    candidate_feature_cols = [
        "entry_credit",
        "spot",
        "short_put",
        "long_put",
        "short_call",
        "long_call",
        "delta_sp",
        "delta_sc",
        "avg_delta_abs",
        "iv_skew",
        "tte_days",
        "size_aggressiveness",
    ]

    # Filter only existing columns
    feature_cols = [c for c in candidate_feature_cols if c in df.columns]

    # Add missing ones as zeros (auto-expand)
    for c in candidate_feature_cols:
        if c not in df.columns:
            LOG.warning(f"Feature '{c}' missing — adding default zeros.")
            df[c] = 0.0
            feature_cols.append(c)

    # Create matrix
    X = df[feature_cols].fillna(0.0).astype(float)

    LOG.info("Using %d features: %s", len(feature_cols), feature_cols)
    LOG.info("Training samples: %d", len(X))

    return X, y_clf, y_reg, feature_cols


# ---------------------------------------------
# 3) TRAIN MODELS
# ---------------------------------------------
def train_models():
    LOG.info("Loading dataset...")
    df = load_features()

    X, y_clf, y_reg, feature_cols = build_matrix(df)

    # CLASSIFIER
    LOG.info("Training classifier...")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=3,
        random_state=42,
    )
    clf.fit(X, y_clf)

    # REGRESSOR
    LOG.info("Training regressor...")
    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )
    reg.fit(X, y_reg)

    # SAVE MODEL BUNDLE
    bundle = {
        "classifier": clf,
        "regressor": reg,
        "features": feature_cols,
    }

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(bundle, MODEL_OUT)

    LOG.info("Saved ML model: %s", MODEL_OUT)

    # Print summary
    LOG.info("--- Classifier importance ---")
    for name, imp in sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1]):
        LOG.info(f"{name}: {imp:.4f}")

    LOG.info("--- Regressor importance ---")
    for name, imp in sorted(zip(feature_cols, reg.feature_importances_), key=lambda x: -x[1]):
        LOG.info(f"{name}: {imp:.4f}")

    LOG.info("Training complete.")


if __name__ == "__main__":
    train_models()
