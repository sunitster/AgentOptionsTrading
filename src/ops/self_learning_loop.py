#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
self_learning_loop.py

Auto-scheduler: daily build -> train -> evaluate -> deploy model.

Usage:
    python -m src.ops.self_learning_loop --once
    python -m src.ops.self_learning_loop --daemon --time 02:00

Features:
 - safe locking to prevent overlapping runs
 - uses Python imports of your existing modules if available
 - fallback to subprocess calls if import fails
 - versioned model promotion (models/llm_trades/model-YYYYmmdd-HHMMSS.pkl)
 - JSON reports saved under models/llm_trades/auto_train_reports/
"""

from __future__ import annotations
import argparse
import importlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

# ML libs
try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, mean_absolute_error
except Exception:
    joblib = None
    np = None
    pd = None
    accuracy_score = None
    mean_absolute_error = None

LOG = logging.getLogger("self_learning_loop")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Paths
BASE_MONITOR = os.path.join("models", "llm_trades")
MODEL_MAIN = os.path.join(BASE_MONITOR, "model.pkl")
MODEL_DIR = BASE_MONITOR
FEATURE_PARQUET = os.path.join(BASE_MONITOR, "features.parquet")
FEATURE_CSV = os.path.join(BASE_MONITOR, "features.csv")
REPORT_DIR = os.path.join(BASE_MONITOR, "auto_train_reports")
LOCKFILE = os.path.join(BASE_MONITOR, "auto_train.lock")

# Ensure dirs
os.makedirs(BASE_MONITOR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Time utilities
def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# ----- locking -----
def acquire_lock(timeout_s: int = 0) -> bool:
    """
    Create a simple lock file to prevent concurrent runs.
    If timeout_s > 0, waits until lock removed or timeout.
    """
    start = time.time()
    while True:
        try:
            fd = os.open(LOCKFILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps({"pid": os.getpid(), "ts": now_iso()}))
            LOG.info("Acquired lock %s", LOCKFILE)
            return True
        except FileExistsError:
            # check timeout
            if timeout_s and (time.time() - start) >= timeout_s:
                LOG.error("Timeout while waiting for lock")
                return False
            LOG.info("Lockfile exists; waiting...")
            time.sleep(1)
        except Exception:
            LOG.exception("acquire_lock unexpected error")
            return False

def release_lock():
    try:
        if os.path.exists(LOCKFILE):
            os.remove(LOCKFILE)
            LOG.info("Released lock %s", LOCKFILE)
    except Exception:
        LOG.exception("Failed to release lock")

# ----- helpers to call existing modules -----
def run_build_features_via_import() -> bool:
    """
    Try to import src.apps.build_daily_features.build_features_and_labels and run it.
    Returns True on success, False otherwise.
    """
    try:
        mod = importlib.import_module("src.apps.build_daily_features")
        # function name in our provided script: build_features_and_labels
        if hasattr(mod, "build_features_and_labels"):
            LOG.info("Calling build_features_and_labels() via import")
            mod.build_features_and_labels(
                os.path.join(BASE_MONITOR, "records.jsonl"),
                os.path.join(BASE_MONITOR, "trades.db"),
                FEATURE_PARQUET,
            )
            return True
        # else call main() if exists
        if hasattr(mod, "main"):
            LOG.info("Calling build_daily_features.main() via import")
            mod.main()
            return True
    except Exception:
        LOG.exception("Import-based build_daily_features failed")
    return False

def run_build_features_via_subprocess() -> bool:
    """
    Fallback: run as subprocess - python -m src.apps.build_daily_features
    """
    try:
        LOG.info("Fallback: running build_daily_features via subprocess")
        cmd = [sys.executable, "-m", "src.apps.build_daily_features"]
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        LOG.info("Subprocess stdout: %s", proc.stdout.strip().splitlines()[-5:] if proc.stdout else "")
        return True
    except subprocess.CalledProcessError as e:
        LOG.error("Subprocess failed: %s\n%s", e, e.stderr)
    except Exception:
        LOG.exception("Failed to run build_daily_features subprocess")
    return False

def run_train_model_via_import() -> bool:
    """
    Try to import src.scripts.train_model.train_models (or train_models) and run it.
    """
    try:
        mod = importlib.import_module("src.scripts.train_model")
        # function names we expect: train_models or train_models()
        if hasattr(mod, "train_models"):
            LOG.info("Calling train_models() via import")
            mod.train_models()
            return True
        # if the module defines top-level train_models function under another name (older versions), try train_models
        if hasattr(mod, "train_model") and callable(mod.train_model):
            LOG.info("Calling train_model() via import")
            mod.train_model()
            return True
        # finally try running main() if present
        if hasattr(mod, "__main__") or hasattr(mod, "main"):
            LOG.info("Calling train_model.main() via import")
            try:
                if hasattr(mod, "main"):
                    mod.main()
                    return True
            except Exception:
                LOG.exception("train_model.main() failed via import")
    except Exception:
        LOG.exception("Import-based train_model failed")
    return False

def run_train_model_via_subprocess() -> bool:
    try:
        LOG.info("Fallback: running train_model via subprocess")
        cmd = [sys.executable, "src/scripts/train_model.py"]
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        LOG.info("train_model stdout: %s", proc.stdout.strip().splitlines()[-10:] if proc.stdout else "")
        return True
    except subprocess.CalledProcessError as e:
        LOG.error("train_model subprocess failed: %s\n%s", e, e.stderr)
    except Exception:
        LOG.exception("Failed to run train_model subprocess")
    return False

# ----- evaluation & deployment -----
def safe_load_features() -> Optional["pd.DataFrame"]:
    """
    Load features from parquet or csv. Returns pandas DataFrame or None.
    """
    try:
        if os.path.exists(FEATURE_PARQUET):
            return pd.read_parquet(FEATURE_PARQUET)
        if os.path.exists(FEATURE_CSV):
            return pd.read_csv(FEATURE_CSV)
    except Exception:
        LOG.exception("Failed to load features")
    return None

def evaluate_model(model_path: str, df: "pd.DataFrame") -> Dict[str, Any]:
    """
    Evaluate a model (joblib bundle with 'classifier','regressor','features') on given df.
    Returns metrics dict {clf_acc, reg_mae, combined_score, nsamples}
    """
    try:
        bundle = joblib.load(model_path)
        clf = bundle.get("classifier")
        reg = bundle.get("regressor")
        feature_cols = list(bundle.get("features", []))
    except Exception:
        LOG.exception("Failed to load model bundle: %s", model_path)
        return {"ok": False}

    # ensure df contains features
    if df is None or df.empty:
        LOG.warning("No features available for evaluation")
        return {"ok": False}

    # create X using available feature_cols; fill missing with zeros
    Xdf = df.copy()
    for c in feature_cols:
        if c not in Xdf.columns:
            Xdf[c] = 0.0
    try:
        X = Xdf[feature_cols].fillna(0.0).astype(float)
    except Exception:
        # fallback: try .infer_objects
        try:
            X = Xdf[feature_cols].fillna(0.0).infer_objects().astype(float)
        except Exception:
            LOG.exception("Failed to construct X for evaluation")
            return {"ok": False}

    # labels
    if "label_win" in Xdf.columns and accuracy_score is not None:
        y_true = Xdf["label_win"].fillna(0).astype(int)
        try:
            preds = clf.predict(X)
            acc = float(accuracy_score(y_true, preds))
        except Exception:
            LOG.exception("Classifier prediction failed")
            acc = float("nan")
    else:
        acc = float("nan")

    if "realized_pnl" in Xdf.columns and mean_absolute_error is not None:
        y_reg_true = Xdf["realized_pnl"].fillna(0.0).astype(float)
        try:
            preds_reg = reg.predict(X)
            mae = float(mean_absolute_error(y_reg_true, preds_reg))
        except Exception:
            LOG.exception("Regressor prediction failed")
            mae = float("nan")
    else:
        mae = float("nan")

    # combine into a single score — higher is better
    # If acc available, use acc - (mae_scaled)
    score = None
    try:
        acc_val = 0.0 if (acc is None or (isinstance(acc, float) and np.isnan(acc))) else float(acc)
        mae_val = 0.0 if (mae is None or (isinstance(mae, float) and np.isnan(mae))) else float(mae)
        # scale mae by typical magnitude: if spot present use spot avg, else 50
        scale = 50.0
        if "spot" in Xdf.columns:
            try:
                scale = max(1.0, float(Xdf["spot"].abs().mean()))
            except Exception:
                scale = 50.0
        score = acc_val - (mae_val / max(1.0, scale))
    except Exception:
        score = None

    metrics = {
        "ok": True,
        "clf_acc": None if np is None else (None if (acc is None or (isinstance(acc, float) and np.isnan(acc))) else float(acc)),
        "reg_mae": None if np is None else (None if (mae is None or (isinstance(mae, float) and np.isnan(mae))) else float(mae)),
        "combined_score": score,
        "nsamples": len(Xdf),
    }
    return metrics

def promote_model(new_model_path: str, report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Version and promote new_model_path to MODEL_MAIN with backup. Returns promotion metadata.
    """
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    dst_version = os.path.join(MODEL_DIR, f"model-{ts}.pkl")
    try:
        shutil.copy2(new_model_path, dst_version)
        # copy to main
        shutil.copy2(new_model_path, MODEL_MAIN)
        LOG.info("Promoted model to %s and %s", dst_version, MODEL_MAIN)
        return {"promoted": True, "version_path": dst_version, "deployed_path": MODEL_MAIN, "ts": ts}
    except Exception:
        LOG.exception("Failed to promote model")
        return {"promoted": False}

# ----- single run orchestration -----
def run_once(dry_run: bool = False) -> Dict[str, Any]:
    """
    Run the full pipeline once:
     - build features
     - train model
     - evaluate new model vs current deployed
     - promote if better
    Returns a run report dict.
    """
    run_report: Dict[str, Any] = {
        "started_at": now_iso(),
        "build_features": {"ok": False},
        "train_model": {"ok": False},
        "evaluate_new": None,
        "evaluate_current": None,
        "promote": None,
        "dry_run": bool(dry_run),
    }

    # Acquire lock
    if not acquire_lock(timeout_s=10):
        run_report["error"] = "could_not_acquire_lock"
        return run_report

    try:
        # 1) Build features
        ok = run_build_features_via_import()
        if not ok:
            ok = run_build_features_via_subprocess()
        run_report["build_features"]["ok"] = bool(ok)
        if not ok:
            run_report["build_features"]["error"] = "build_failed"
            LOG.error("Build features failed - aborting run")
            return run_report

        # 2) Train model
        ok = run_train_model_via_import()
        if not ok:
            ok = run_train_model_via_subprocess()
        run_report["train_model"]["ok"] = bool(ok)
        if not ok:
            run_report["train_model"]["error"] = "train_failed"
            LOG.error("Training failed - aborting run")
            return run_report

        # After training, new model should be at MODEL_MAIN or created by train script.
        # Some train scripts write to models/llm_trades/model.pkl directly. We'll locate the newest model file.
        candidate_model = MODEL_MAIN if os.path.exists(MODEL_MAIN) else None
        # fallback: find newest model-*.pkl
        if candidate_model is None:
            files = sorted([os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")], key=os.path.getmtime, reverse=True)
            candidate_model = files[0] if files else None

        if candidate_model is None:
            run_report["train_model"]["error"] = "no_model_file_found"
            LOG.error("No model file found after training")
            return run_report

        run_report["train_model"]["model_path"] = os.path.abspath(candidate_model)

        # 3) Evaluate new model on current features
        df = safe_load_features()
        metrics_new = evaluate_model(candidate_model, df) if df is not None else {"ok": False}
        run_report["evaluate_new"] = metrics_new

        # 4) evaluate current deployed (before promotion) if exists
        current_metrics = None
        current_model_path = None
        if os.path.exists(MODEL_MAIN) and os.path.abspath(candidate_model) != os.path.abspath(MODEL_MAIN):
            current_model_path = MODEL_MAIN
        else:
            # try to pick previous version (model-*.pkl) excluding candidate
            all_models = sorted([os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")], key=os.path.getmtime, reverse=True)
            # skip candidate_model (first) and pick next
            prev = None
            for p in all_models:
                try:
                    if os.path.abspath(p) != os.path.abspath(candidate_model):
                        prev = p
                        break
                except Exception:
                    continue
            current_model_path = prev

        if current_model_path:
            metrics_current = evaluate_model(current_model_path, df) if df is not None else {"ok": False}
            run_report["evaluate_current"] = {"model_path": os.path.abspath(current_model_path), "metrics": metrics_current}
        else:
            run_report["evaluate_current"] = None

        # 5) Decide promotion
        promote = False
        reason = None
        if dry_run:
            reason = "dry_run"
        else:
            # if current missing -> promote
            if run_report["evaluate_current"] is None:
                promote = True
                reason = "no_existing_model"
            else:
                try:
                    new_score = metrics_new.get("combined_score")
                    current_score = run_report["evaluate_current"].get("metrics", {}).get("combined_score")
                    LOG.info("Combined scores: new=%s current=%s", str(new_score), str(current_score))
                    # If both None -> promote by default
                    if new_score is None and current_score is None:
                        promote = True
                        reason = "no_scores_both"
                    elif new_score is not None and current_score is None:
                        promote = True
                        reason = "new_has_score"
                    elif new_score is None and current_score is not None:
                        promote = False
                        reason = "new_no_score"
                    else:
                        # promote if new_score >= current_score (allow equal to avoid churn)
                        if float(new_score) >= float(current_score):
                            promote = True
                            reason = "new_better_or_equal"
                        else:
                            promote = False
                            reason = "new_worse"
                except Exception:
                    LOG.exception("Failed to compare model metrics - not promoting")
                    promote = False
                    reason = "compare_failed"

        run_report["promote_decision"] = {"promote": promote, "reason": reason}

        if promote and not dry_run:
            prom = promote_model(candidate_model, run_report)
            run_report["promote"] = prom
        else:
            run_report["promote"] = {"promoted": False, "reason": reason}

        run_report["finished_at"] = now_iso()
        return run_report

    except Exception:
        LOG.exception("run_once failed unexpectedly")
        return {"ok": False, "error": "unexpected_exception"}
    finally:
        release_lock()

# ----- daemon loop -----
_stop_requested = False

def _signal_handler(signum, frame):
    global _stop_requested
    LOG.info("Signal %s received, stopping daemon loop", signum)
    _stop_requested = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def run_daemon(sched_time: str = "02:00", interval_days: int = 1, dry_run: bool = False):
    """
    Run the loop continuously. sched_time in HH:MM 24-hour (UTC).
    interval_days: how many days between runs.
    """
    try:
        hh, mm = [int(x) for x in sched_time.split(":")]
    except Exception:
        hh, mm = 2, 0
    LOG.info("Starting daemon. Schedule time (UTC): %02d:%02d every %d day(s).", hh, mm, interval_days)
    while not _stop_requested:
        now = datetime.utcnow()
        target = datetime.combine(now.date(), datetime.min.time()) + timedelta(hours=hh, minutes=mm)
        if target <= now:
            # already passed today -> schedule for next interval
            target = target + timedelta(days=interval_days)
        wait_seconds = (target - now).total_seconds()
        LOG.info("Next run scheduled at %s (UTC) — sleeping %.0f seconds", target.isoformat(), wait_seconds)
        slept = 0.0
        while slept < wait_seconds and not _stop_requested:
            time.sleep(min(60, wait_seconds - slept))
            slept = (datetime.utcnow() - now).total_seconds()
        if _stop_requested:
            break
        LOG.info("Daemon: starting scheduled run at %s", now_iso())
        report = run_once(dry_run=dry_run)
        # save report
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        report_path = os.path.join(REPORT_DIR, f"report-{ts}.json")
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            LOG.info("Saved run report to %s", report_path)
        except Exception:
            LOG.exception("Failed to write run report")
    LOG.info("Daemon loop ending")

# ----- CLI -----
def main(argv=None):
    parser = argparse.ArgumentParser(description="Self learning loop (build->train->eval->deploy)")
    parser.add_argument("--once", dest="once", action="store_true", help="Run once and exit")
    parser.add_argument("--daemon", dest="daemon", action="store_true", help="Run as daemon daily")
    parser.add_argument("--time", dest="time", type=str, default="02:00", help="Daily schedule time (UTC) in HH:MM")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Do not promote model (dry run)")
    parser.add_argument("--report", dest="report", type=str, default=None, help="Write run report to this path (optional)")
    args = parser.parse_args(argv)

    if args.once:
        LOG.info("Running single pipeline run")
        if not acquire_lock(timeout_s=5):
            LOG.error("Could not acquire lock for single run")
            return 2
        try:
            report = run_once(dry_run=args.dry_run)
            if args.report:
                try:
                    with open(args.report, "w") as f:
                        json.dump(report, f, indent=2)
                    LOG.info("Wrote report to %s", args.report)
                except Exception:
                    LOG.exception("Failed to write report to %s", args.report)
            else:
                # print short summary
                LOG.info("Run summary: %s", json.dumps(report, indent=2))
        finally:
            release_lock()
        return 0

    if args.daemon:
        try:
            run_daemon(sched_time=args.time, dry_run=args.dry_run)
        except KeyboardInterrupt:
            LOG.info("Stopped by user")
        return 0

    # default: run once
    LOG.info("Defaulting to single run")
    return main(["--once"])

if __name__ == "__main__":
    sys.exit(main())
