# src/apps/kite_historical_train.py
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, List

import pandas as pd

# --- Local imports (all already exist in your repo) ---
from src import full_chain_rebuilder
from src.scripts import unify_options_data
from src.learning.backfill_labels import backfill_labels
from src.learning.run_full_historical_training import main as run_full_training


LOG = logging.getLogger("kite_historical_train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[1]         # project root = src/..
DATA_DIR = ROOT / "data"
NIFTY_DIR = DATA_DIR / "nifty"
OPTIONS_ALL_DIR = DATA_DIR / "options_all"

LEARNING_DIR = ROOT / "learning_data"
FEATURES_ENRICHED_PATH = LEARNING_DIR / "daily_features_enriched.parquet"
LABELS_PATH = LEARNING_DIR / "daily_labels.parquet"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _load_nifty_close(date_ts: pd.Timestamp) -> Optional[float]:
    """
    Load NIFTY close for a given date from data/nifty/YYYY-MM-DD.parquet.
    Same idea as build_daily_features.py.
    """
    fname = f"{date_ts.date().isoformat()}.parquet"
    p = NIFTY_DIR / fname
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "close" not in df.columns:
        return None
    return float(df["close"].iloc[0])


def build_daily_features_enriched(
    options_dir: Path = OPTIONS_ALL_DIR,
    out_path: Path = FEATURES_ENRICHED_PATH,
) -> pd.DataFrame:
    """
    Build a daily features parquet from unified option data (data/options_all).

    For each date we compute:
        - nifty_close
        - iv_atm   (IV of strike closest to NIFTY close, if IV exists)
        - total_oi
        - n_options

    Output: learning_data/daily_features_enriched.parquet
    """
    if not options_dir.exists():
        raise FileNotFoundError(f"Options dir not found: {options_dir}")

    rows: List[dict] = []

    files = sorted(options_dir.glob("*.parquet"))
    LOG.info("Building daily_features_enriched from %d option files", len(files))

    for fp in files:
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            LOG.warning("Failed to read %s: %r", fp, e)
            continue

        if "date" not in df.columns:
            LOG.warning("File %s missing 'date' column, skipping", fp)
            continue

        # Assume single date per file
        date_val = pd.to_datetime(df["date"].iloc[0])
        nifty_close = _load_nifty_close(date_val)
        if nifty_close is None:
            LOG.info("No NIFTY close for %s (file %s) -> skipping", date_val.date(), fp.name)
            continue

        df = df.copy()
        # Ensure numeric types
        df["strike"] = pd.to_numeric(df.get("strike"), errors="coerce")
        df["iv"] = pd.to_numeric(df.get("iv"), errors="coerce")
        df["oi"] = pd.to_numeric(df.get("oi"), errors="coerce").fillna(0)

        df = df.dropna(subset=["strike"])
        if df.empty:
            continue

        # ATM IV: option (call/put) whose strike closest to NIFTY close
        iv_atm = None
        if "iv" in df.columns:
            try:
                atm_idx = (df["strike"] - nifty_close).abs().idxmin()
                iv_val = df.loc[atm_idx, "iv"]
                if pd.notna(iv_val):
                    iv_atm = float(iv_val)
            except Exception:
                iv_atm = None

        total_oi = float(df["oi"].sum())
        n_options = int(len(df))

        rows.append(
            {
                "date": date_val.normalize().strftime("%Y-%m-%d"),
                "nifty_close": float(nifty_close),
                "iv_atm": iv_atm,
                "total_oi": total_oi,
                "n_options": n_options,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    LOG.info("Wrote %d rows -> %s", len(out_df), out_path)
    return out_df


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def run_pipeline(
    with_kite_backfill: bool = False,
    unify_data: bool = True,
    rebuild_features: bool = True,
    rebuild_labels: bool = True,
    train_model: bool = True,
):
    """
    High-level pipeline:

    1) (optional) Backfill option chains via KITE + NSE fallback
    2) Unify data -> data/options_all
    3) Build daily_features_enriched
    4) Backfill labels (next_day_pnl)
    5) Run full historical training
    """
    LOG.info("=== KITE Historical Training Pipeline START ===")

    # 1) Optional: historical backfill via KITE + fallback (writes data/options_chain_kite)
    if with_kite_backfill:
        LOG.info("Step 1: Running full_chain_rebuilder.run_rebuilder()")
        full_chain_rebuilder.run_rebuilder()

    # 2) Unify option data from chain + snapshots -> data/options_all
    if unify_data:
        LOG.info("Step 2: Unifying options data into data/options_all/")
        unify_options_data.main()

    # 3) Build enriched daily features
    if rebuild_features:
        LOG.info("Step 3: Building learning_data/daily_features_enriched.parquet")
        build_daily_features_enriched()

    # 4) Build labels using backfill_labels, pointed at enriched features
    if rebuild_labels:
        LOG.info("Step 4: Backfilling labels (next_day_pnl) from enriched features")
        backfill_labels(
            features_path=str(FEATURES_ENRICHED_PATH),
            output_path=str(LABELS_PATH),
        )

    # 5) Run full historical training (LightGBM + LLM challengers)
    if train_model:
        LOG.info("Step 5: Running run_full_historical_training.main()")
        run_full_training()

    LOG.info("=== KITE Historical Training Pipeline COMPLETE ===")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Offline KITE historical training pipeline.")
    p.add_argument(
        "--with-kite-backfill",
        action="store_true",
        help="Also call full_chain_rebuilder.run_rebuilder() to fetch option chains via Kite+NSE.",
    )
    p.add_argument(
        "--no-unify",
        action="store_true",
        help="Skip unify_options_data.main() (assumes data/options_all already up-to-date).",
    )
    p.add_argument(
        "--no-features",
        action="store_true",
        help="Skip rebuilding daily_features_enriched.parquet.",
    )
    p.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip rebuilding daily_labels.parquet.",
    )
    p.add_argument(
        "--no-train",
        action="store_true",
        help="Skip model training; only prepare data.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    run_pipeline(
        with_kite_backfill=args.with_kite_backfill,
        unify_data=not args.no_unify,
        rebuild_features=not args.no_features,
        rebuild_labels=not args.no_labels,
        train_model=not args.no_train,
    )
