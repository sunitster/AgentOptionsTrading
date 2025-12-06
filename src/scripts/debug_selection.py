#!/usr/bin/env python3
"""
Pure LLM debug-selection tool.
NEVER imports KiteAPI or calls live/paper APIs.
This guarantees no hanging, even on weekends or offline.

Usage:
    python -m src.scripts.debug_selection --symbol NIFTY --n 5

It loads a snapshot from:
  models/llm_trades/latest_snapshot.json
  or stdin
and runs LLM + scoring + weights.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

try:
    from tabulate import tabulate
except:
    tabulate = None

# Import ONLY ml_pipeline (safe)
from src.ml_pipeline import (
    OllamaProposer,
    FeatureEngineer,
    ScorerEnsemble,
    ThompsonSelector,
    _signature_from_candidate,
)

# --------------------------------------------------------------------------
# SNAPSHOT LOADING (NO KiteAPI)
# --------------------------------------------------------------------------

def load_snapshot(symbol: str):
    """
    Load snapshot from file or stdin.
    Returns dict or None.
    """
    candidates = [
        Path("models/llm_trades/latest_snapshot.json"),
        Path("models/llm_trades/snapshot.json"),
        Path("debug_snapshot.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                if text.startswith("{"):
                    data = json.loads(text)
                    print(f"Loaded snapshot from {p}")
                    return data
                else:
                    # maybe NDJSON, load last line
                    lines = [l.strip() for l in text.splitlines() if l.strip()]
                    data = json.loads(lines[-1])
                    print(f"Loaded NDJSON snapshot from {p}")
                    return data
            except Exception:
                pass

    # fallback: stdin
    if not sys.stdin.isatty():
        try:
            text = sys.stdin.read()
            if text.strip():
                data = json.loads(text)
                print("Loaded snapshot from stdin.")
                return data
        except Exception:
            pass

    print("ERROR: No snapshot found. Put a snapshot at models/llm_trades/latest_snapshot.json")
    return None


# --------------------------------------------------------------------------
# MODEL WEIGHTS
# --------------------------------------------------------------------------

def load_weights(path="models/llm_trades/model_weights.json"):
    if not Path(path).exists():
        print("No weights file found.")
        return {}
    try:
        with open(path, "r") as f:
            w = json.load(f)
        print(f"Loaded {len(w)} weights.")
        return {str(k): float(v) for k, v in w.items()}
    except:
        print("Failed to load weights.")
        return {}


# --------------------------------------------------------------------------
# SCORING
# --------------------------------------------------------------------------

def score_candidates(cands, snapshot, weights):
    fe = FeatureEngineer()
    ensemble = ScorerEnsemble(n_models=5)

    Xs = []
    sigs = []
    for c in cands:
        try:
            vec, names = fe.candidate_to_vector(c, snapshot)
        except:
            import numpy as np
            vec = np.zeros(12)
            names = []
        Xs.append(vec)
        sigs.append(_signature_from_candidate(c))

    import numpy as np
    X = np.vstack(Xs)

    # predict mean/std
    try:
        mean, std = ensemble.predict_with_uncertainty(X)
    except Exception:
        mean = np.zeros(X.shape[0])
        std = np.ones(X.shape[0]) * 1e-6

    # apply weights
    rows = []
    for idx, c in enumerate(cands):
        sig = sigs[idx]
        w = float(weights.get(sig, weights.get(str(sig), 0.0)))
        m = float(mean[idx])
        weighted = m * (w if w > 0 else 0.01)
        rows.append({
            "idx": idx,
            "signature": sig,
            "mean": m,
            "std": float(std[idx]),
            "weight": w,
            "weighted_mean": weighted,
            "candidate": c,
        })

    rows_sorted = sorted(rows, key=lambda r: (r["weighted_mean"], r["mean"]), reverse=True)
    return rows_sorted


# --------------------------------------------------------------------------
# PRINTING
# --------------------------------------------------------------------------

def print_table(rows, limit=10):
    headers = ["rank", "idx", "signature", "mean", "std", "weight", "weighted"]
    table = []
    for i, r in enumerate(rows[:limit], start=1):
        table.append([
            i, r["idx"], r["signature"],
            f"{r['mean']:.4f}",
            f"{r['std']:.4f}",
            f"{r['weight']:.4f}",
            f"{r['weighted_mean']:.6f}",
        ])
    if tabulate:
        print(tabulate(table, headers=headers, tablefmt="github"))
    else:
        print(headers)
        for row in table:
            print(row)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--show-json", action="store_true")
    args = ap.parse_args()

    snapshot = load_snapshot(args.symbol)
    if not snapshot:
        sys.exit(1)

    # proposer (Ollama or fallback)
    proposer = OllamaProposer()

    print(f"Requesting {args.n} candidates...")
    try:
        candidates = proposer.propose(snapshot, n_candidates=args.n)
    except Exception as e:
        print("Proposer error:", e)
        sys.exit(1)

    if not candidates:
        print("Proposer returned 0 candidates.")
        sys.exit(1)

    weights = load_weights()
    rows = score_candidates(candidates, snapshot, weights)

    print_table(rows)

    if args.show-json and rows:
        print("\nTop candidate JSON:\n")
        print(json.dumps(rows[0]["candidate"], indent=2))


if __name__ == "__main__":
    main()
