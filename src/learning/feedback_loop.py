# src/learning/feedback_loop.py
"""
Daily feedback loop for ML-driven selection.

What it does (safe, offline):
1. Reads models/llm_trades/ml_replay.db -> table replays
2. Aggregates results grouped by a candidate signature (derived from ic_json)
   - metrics: count, mean_pnl, median_pnl, std_pnl
3. Loads existing model weights from models/llm_trades/model_weights.json (if exists)
4. Updates weights using a simple bandit-style rule:
     new_w = (1 - decay) * old_w + lr * clip(mean_pnl / scale)
   where scale normalizes currency into reasonable range (e.g., 1000)
   and we add small epsilon smoothing for exploration
5. Normalize weights to sum to 1 and store to model_weights.json
6. Write a timestamped report models/llm_trades/feedback_report-<date>.json

This is intentionally simple: it gives you a stable baseline to iterate on.
"""

from __future__ import annotations
import os
import json
import sqlite3
from typing import Dict, Any, Tuple
from collections import defaultdict
from datetime import datetime, date

MONITOR_PATH = os.path.join("models", "llm_trades")
REPLAY_DB_PATH = os.path.join(MONITOR_PATH, "ml_replay.db")
WEIGHTS_PATH = os.path.join(MONITOR_PATH, "model_weights.json")


def _ensure_replay_db_exists() -> bool:
    return os.path.exists(REPLAY_DB_PATH)


def _load_replays() -> list[Dict[str, Any]]:
    """Load all replays from DB. Returns list of dicts."""
    out = []
    if not _ensure_replay_db_exists():
        return out
    conn = sqlite3.connect(REPLAY_DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, ts, ic_json, entry_snapshot_json, exit_snapshot_json, realized_pnl, exit_reason, metadata_json FROM replays")
        rows = cur.fetchall()
        for r in rows:
            try:
                rid, ts, ic_json, entry_json, exit_json, pnl, exit_reason, meta_json = r
                ic = json.loads(ic_json) if ic_json else ic_json
                meta = json.loads(meta_json) if meta_json else {}
                out.append({
                    "id": rid,
                    "ts": ts,
                    "ic": ic,
                    "entry": json.loads(entry_json) if entry_json else {},
                    "exit": json.loads(exit_json) if exit_json else {},
                    "pnl": pnl,
                    "exit_reason": exit_reason,
                    "meta": meta,
                })
            except Exception:
                continue
    finally:
        conn.close()
    return out


def _signature_from_ic(ic_obj: Any) -> str:
    """
    Reduce an IC dict/object to a stable "candidate signature" string for aggregation.
    Strategy: use underlying symbol + expiry + strikes (short_put, long_put, short_call, long_call)
    """
    try:
        if isinstance(ic_obj, dict):
            fields = (
                ic_obj.get("symbol"),
                ic_obj.get("expiry"),
                int(float(ic_obj.get("short_put", 0))),
                int(float(ic_obj.get("long_put", 0))),
                int(float(ic_obj.get("short_call", 0))),
                int(float(ic_obj.get("long_call", 0))),
            )
            return "_".join([str(x) for x in fields])
        # fallback: string repr
        return str(ic_obj)
    except Exception:
        return str(ic_obj)


def aggregate_replays(replays: list[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate replays by signature, returning a dict:
      signature -> {count, mean_pnl, median_pnl, std_pnl, exit_reasons: {...}}
    """
    import math
    stats = {}
    groups = defaultdict(list)
    reasons = defaultdict(lambda: defaultdict(int))
    for r in replays:
        sig = _signature_from_ic(r.get("ic"))
        pnl = r.get("pnl") if r.get("pnl") is not None else 0.0
        groups[sig].append(float(pnl))
        er = r.get("exit_reason") or (r.get("meta", {}).get("exit_reason") if isinstance(r.get("meta"), dict) else None)
        if er:
            reasons[sig][er] += 1

    for sig, vals in groups.items():
        n = len(vals)
        mean = sum(vals) / n if n else 0.0
        # median
        svals = sorted(vals)
        med = svals[n//2] if n % 2 == 1 else (svals[n//2 - 1] + svals[n//2]) / 2.0 if n else 0.0
        # std
        var = sum((x - mean) ** 2 for x in vals) / n if n else 0.0
        std = var ** 0.5
        stats[sig] = {
            "count": n,
            "mean_pnl": mean,
            "median_pnl": med,
            "std_pnl": std,
            "exit_reasons": dict(reasons[sig]),
        }
    return stats


def _load_weights() -> Dict[str, float]:
    if not os.path.exists(WEIGHTS_PATH):
        return {}
    try:
        with open(WEIGHTS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_weights(weights: Dict[str, float]) -> None:
    try:
        with open(WEIGHTS_PATH, "w") as f:
            json.dump(weights, f, indent=2, sort_keys=True)
    except Exception:
        pass


def bandit_update(old_weights: Dict[str, float], stats: Dict[str, Dict[str, Any]], *, lr: float = 0.2, decay: float = 0.05, scale: float = 1000.0, epsilon: float = 0.05) -> Dict[str, float]:
    """
    A simple bandit-style update:

    new_score_raw = (1 - decay) * old_w + lr * clipped_reward
    where clipped_reward = tanh(mean_pnl/scale)  (keeps reward bounded)
    then normalize weights to sum=1 and add minimal epsilon mass for unseen candidates.

    Parameters:
      old_weights: existing weights mapping signature -> float
      stats: aggregated stats by signature
      lr: learning rate (how strongly to incorporate latest mean)
      decay: slows old weights down
      scale: normalizes currency -> 1.0 range
      epsilon: minimal exploration mass per candidate
    """
    import math
    new_raw = {}
    # initialize unseen candidates in old_weights with small mass
    all_sigs = set(old_weights.keys()) | set(stats.keys())
    for s in all_sigs:
        old = float(old_weights.get(s, 0.0))
        mean_pnl = float(stats.get(s, {}).get("mean_pnl", 0.0))
        # bounded reward using tanh
        reward = math.tanh(mean_pnl / float(scale))
        raw = (1.0 - decay) * old + lr * reward
        # keep non-negative
        new_raw[s] = max(raw, -1.0)
    # shift to positive domain
    min_raw = min(new_raw.values()) if new_raw else 0.0
    if min_raw < 0:
        for k in new_raw:
            new_raw[k] = new_raw[k] - min_raw + 1e-6

    # add epsilon exploration mass
    for k in new_raw:
        new_raw[k] = new_raw[k] + epsilon

    # normalize
    total = sum(new_raw.values()) or 1.0
    normalized = {k: float(v) / total for k, v in new_raw.items()}
    return normalized


def run_feedback_once(save_report: bool = True) -> Dict[str, Any]:
    replays = _load_replays()
    stats = aggregate_replays(replays)
    old_w = _load_weights()
    new_w = bandit_update(old_w, stats, lr=0.25, decay=0.02, scale=1500.0, epsilon=0.01)
    _save_weights(new_w)
    report = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "n_replays": len(replays),
        "n_signatures": len(stats),
        "weights_len": len(new_w),
        "top_stats": sorted(
            [
                (sig, stats[sig]["count"], stats[sig]["mean_pnl"], stats[sig]["exit_reasons"])
                for sig in stats
            ],
            key=lambda x: (x[2], x[1]),
            reverse=True,
        )[:20],
    }
    if save_report:
        fname = os.path.join(MONITOR_PATH, f"feedback_report-{date.today().isoformat()}.json")
        try:
            with open(fname, "w") as f:
                json.dump(report, f, indent=2, default=str)
        except Exception:
            pass
    return {"report": report, "stats": stats, "weights": new_w}


if __name__ == "__main__":
    out = run_feedback_once(save_report=True)
    print("Feedback run complete. Summary:")
    print(json.dumps(out["report"], indent=2))
