# src/learning/train_from_trades.py

"""
Build supervised dataset from records.jsonl and train a baseline selector model.

Input:
    models/llm_trades/records.jsonl

Output:
    models/selector_baseline.pkl
    models/selector_dataset.parquet
"""

import os
import json
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


RECORDS_FILE = os.path.join("models", "llm_trades", "records.jsonl")
OUTPUT_DATASET = os.path.join("models", "selector_dataset.parquet")
OUTPUT_MODEL = os.path.join("models", "selector_baseline.pkl")


def load_records():
    """Load the JSONL file into list of dicts."""
    if not os.path.exists(RECORDS_FILE):
        raise FileNotFoundError(f"records.jsonl not found at {RECORDS_FILE}")

    rows = []
    with open(RECORDS_FILE, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def pair_entries_exits(records):
    """
    Pair entry and exit events using replay_id if available.
    Returns list of dicts with combined entry+exit info.
    """
    entries = {}
    exits = []
    paired = []

    # First collect by replay_id or fallback timestamp
    for r in records:
        if "candidate" in r:  # entry
            rid = r.get("replay_id")
            if rid is None:
                # timestamp fallback as key
                rid = f"ts_{r['time']}"
            entries[rid] = r

        elif "exit_pnl" in r:  # exit
            exits.append(r)

    # Match exits to entries
    for ex in exits:
        rid = ex.get("replay_id")
        if rid is not None and rid in entries:
            paired.append({"entry": entries[rid], "exit": ex})
        else:
            # fallback: naive – match closest prior entry in time
            ex_time = datetime.fromisoformat(ex["time"])
            best = None
            best_dt = None
            for k, e in entries.items():
                try:
                    et = datetime.fromisoformat(e["time"])
                    if et <= ex_time:
                        dt = (ex_time - et).total_seconds()
                        if (best_dt is None) or (dt < best_dt):
                            best = e
                            best_dt = dt
                except Exception:
                    continue
            if best is not None:
                paired.append({"entry": best, "exit": ex})

    return paired


def extract_features(entry_dict):
    """Convert entry record into flat numeric features for ML."""
    feats = {}

    # Engine config
    cfg = entry_dict.get("engine_config", {})
    for k in ["width", "lot_size", "size_aggressiveness"]:
        feats[k] = cfg.get(k)

    # Entry time-of-day
    try:
        t = datetime.fromisoformat(entry_dict["time"])
        feats["entry_hour"] = t.hour
        feats["entry_minute"] = t.minute
    except Exception:
        feats["entry_hour"] = None
        feats["entry_minute"] = None

    # Legs: IV/delta aggregates
    legs = entry_dict.get("legs_greeks") or []
    ivs = []
    deltas = []
    mprices = []
    for lg in legs:
        ivs.append(lg.get("iv"))
        deltas.append(lg.get("delta"))
        mprices.append(lg.get("market_price"))

    # Basic aggregates
    def safe_mean(v):
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None

    def safe_std(v):
        v = [x for x in v if x is not None]
        if len(v) < 2:
            return None
        m = safe_mean(v)
        return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5

    feats["iv_mean"] = safe_mean(ivs)
    feats["iv_std"] = safe_std(ivs)
    feats["delta_mean"] = safe_mean(deltas)
    feats["delta_std"] = safe_std(deltas)
    feats["legs_count"] = len(legs)

    # Chain snapshot: spot
    chain = entry_dict.get("chain_snapshot") or []
    if chain and isinstance(chain, list) and (len(chain) > 0):
        row0 = chain[0]
        feats["spot"] = (
            row0.get("spot")
            or row0.get("underlying_price")
            or row0.get("underlying")
        )
    else:
        feats["spot"] = None

    # Could add more features: IV skew, term structure, distance from ATM
    return feats


def build_dataset(paired):
    """
    Convert paired entry/exit into DataFrame with features + label.
    Label = 1 if exit_pnl > 0 else 0.
    """
    rows = []

    for item in paired:
        entry = item["entry"]
        exit = item["exit"]

        feats = extract_features(entry)
        pnl = exit.get("exit_pnl")

        label = 1 if (pnl is not None and pnl > 0) else 0

        feats["target"] = label
        feats["pnl"] = pnl
        feats["replay_id"] = exit.get("replay_id") or entry.get("replay_id")
        feats["entry_time"] = entry.get("time")
        feats["exit_time"] = exit.get("time")

        rows.append(feats)

    df = pd.DataFrame(rows)
    return df


def train_baseline_model(df):
    """
    Train logistic regression with scaling.
    Saves selector_baseline.pkl to models/.
    """
    # Numeric feature columns
    feature_cols = [
        "width", "lot_size", "size_aggressiveness",
        "entry_hour", "entry_minute",
        "iv_mean", "iv_std",
        "delta_mean", "delta_std",
        "legs_count", "spot"
    ]

    X = df[feature_cols]
    y = df["target"]

    # Drop rows where all features are NaN
    X = X.fillna(0)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Baseline selector model accuracy: {acc:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, OUTPUT_MODEL)

    print(f"Saved baseline selector model → {OUTPUT_MODEL}")
    return model


def main():
    print("Loading records…")
    records = load_records()

    print("Pairing entries and exits…")
    paired = pair_entries_exits(records)
    print(f"Paired trades: {len(paired)}")

    print("Building dataset…")
    df = build_dataset(paired)
    print(f"Features dataset size: {df.shape}")

    df.to_parquet(OUTPUT_DATASET, index=False)
    print(f"Saved dataset → {OUTPUT_DATASET}")

    print("Training baseline logistic model…")
    train_baseline_model(df)

    print("Done.")


if __name__ == "__main__":
    main()
