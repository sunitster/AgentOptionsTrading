# src/learning/self_learning_loop.py

import os
import json
import datetime as dt
import numpy as np

CHAMPION_PATH = "models/regime_model_champion.json"


# -----------------------------
#  Load / Save Champion
# -----------------------------

def load_champion():
    if not os.path.exists(CHAMPION_PATH):
        print("No champion found.")
        return None

    with open(CHAMPION_PATH, "r") as f:
        return json.load(f)


def save_champion(model_dict: dict):
    os.makedirs("models", exist_ok=True)

    with open(CHAMPION_PATH, "w") as f:
        json.dump(model_dict, f, indent=2)

    print("Champion updated:", CHAMPION_PATH)


# -----------------------------
#  Evaluation Helper
# -----------------------------

def compute_metrics(daily_pnl):
    if len(daily_pnl) == 0:
        return {"sharpe": 0, "max_dd": 0, "total": 0}

    pnl = np.array(daily_pnl)
    total = float(pnl.sum())

    sharpe = float((pnl.mean() / (pnl.std() + 1e-9)) * np.sqrt(252))

    curve = pnl.cumsum()
    peak = np.maximum.accumulate(curve)
    dd = peak - curve
    max_dd = float(dd.max())

    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total": total
    }


def evaluate_challenger(backtest_result: dict):
    """
    backtest_result = {
        "daily_pnl": [...],
        "dates": [...],
        "summary": {...}
    }
    """
    daily = backtest_result.get("daily_pnl", [])
    return compute_metrics(daily)


# -----------------------------
#  Promotion Logic
# -----------------------------

def promote_if_better(champion_metrics, challenger_metrics):
    """
    Promotion Rule:
        - Challenger Sharpe must be higher
        - Challenger MaxDD must be <= Champion MaxDD
    """

    if champion_metrics is None:
        print("No champion yet → auto-promoting challenger.")
        return True

    if challenger_metrics["sharpe"] > champion_metrics["sharpe"]:
        if challenger_metrics["max_dd"] <= champion_metrics["max_dd"]:
            return True

    return False


# -----------------------------
#  Weekly Self-Learning Loop
# -----------------------------

def run_weekly_cycle(backtest_fn,
                     llm_candidates_fn=None,
                     start=None,
                     end=None):
    """
    backtest_fn(regime_model, start, end) → backtest_result

    llm_candidates_fn() → [candidate1, candidate2, ...]
    """

    print("\n==== Phase 5: Weekly Self-Learning Cycle ====\n")

    # Load champion (if exists)
    champion = load_champion()

    if champion:
        champ_res = backtest_fn(champion, start=start, end=end)
        champ_metrics = evaluate_challenger(champ_res)
        print("Champion metrics:", champ_metrics)
    else:
        champ_metrics = None

    # Ask LLM for challengers
    if llm_candidates_fn is None:
        print("No LLM challenger generator provided.")
        return

    print("\nGenerating LLM challengers...")
    challengers = llm_candidates_fn()

    if not challengers:
        print("No challengers produced.")
        return

    best_challenger = None
    best_metrics = None

    print("\nEvaluating challengers...")

    for cand in challengers:
        print("Testing challenger:", cand.get("version"))

        res = backtest_fn(cand, start=start, end=end)
        m = evaluate_challenger(res)

        print("Challenger metrics:", m)

        if best_metrics is None or m["sharpe"] > best_metrics["sharpe"]:
            best_challenger = cand
            best_metrics = m

    print("\nBest challenger:", best_challenger.get("version"))
    print("Best challenger metrics:", best_metrics)

    # Promotion logic
    should_promote = promote_if_better(champ_metrics, best_metrics)

    if should_promote:
        save_champion(best_challenger)
        print("\n🎉 Challenger promoted to NEW CHAMPION!")
    else:
        print("\n❌ Challenger NOT promoted.")

    print("\n==== Weekly Cycle Complete ====\n")
