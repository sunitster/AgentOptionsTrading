# src/learning/train_challenger.py

import pandas as pd
import lightgbm as lgb
import os
import datetime as dt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


def train_lightgbm_challenger():

    print("Loading daily_features.parquet ...")

    df = pd.read_parquet("learning_data/daily_features.parquet")

    # Remove rows without labels
    df = df.dropna(subset=["next_day_pnl"])

    # Features
    X = df.drop(columns=["date", "next_day_pnl"])
    y = df["next_day_pnl"]

    print("Training LightGBM challenger model...")

    tscv = TimeSeriesSplit(n_splits=5)

    preds = []
    trues = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8
        )
        model.fit(X_train, y_train)

        p = model.predict(X_test)
        preds.extend(p)
        trues.extend(y_test)

    mse = mean_squared_error(trues, preds)
    print("Cross-validated MSE:", mse)

    # Fit final model
    final_model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8
    )
    final_model.fit(X, y)

    # Save
    os.makedirs("models", exist_ok=True)
    fname = f"models/lgb_challenger_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    final_model.booster_.save_model(fname)

    print("Saved LightGBM challenger:", fname)

    return fname
