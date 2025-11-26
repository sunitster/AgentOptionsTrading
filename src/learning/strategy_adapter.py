"""
Strategy Adapter
----------------

Transforms LLM strategy JSON parameters into:
- filtered train/val subsets
- engineered feature subsets
- metadata summary

Supports:
- direction filtering (bull/bear/neutral)
- moneyness range
- IV min/max
- DTE range
- IV rank threshold
- allowed_regimes (NEW)
- regime generation (NEW)
"""

import pandas as pd
import numpy as np


def apply_strategy_adapter(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    strategy_json: dict,
    base_features: list,
    debug: bool = False,
):

    # Copy to avoid modifying originals
    train = train_df.copy()
    val = val_df.copy()

    # --- unify adapter state ---
    params = strategy_json.copy()
    meta = {"applied_filters": {}, "feature_deltas": {}}

    # ----------------------------------------
    # 1) Ensure regime column exists
    # ----------------------------------------
    def _ensure_regime(df):
        if "regime" in df.columns and df["regime"].notna().any():
            return df

        # try ATR-based regime
        if "atr_by_price" in df.columns:
            try:
                df["regime"] = pd.qcut(
                    df["atr_by_price"].rank(method="first"),
                    3,
                    labels=["low_vol", "mid_vol", "high_vol"],
                )
                return df
            except Exception:
                pass

        # try IV-based regime
        if "iv_bs" in df.columns:
            try:
                df["regime"] = pd.qcut(
                    df["iv_bs"].rank(method="first"),
                    3,
                    labels=["low_iv", "mid_iv", "high_iv"],
                )
                return df
            except Exception:
                pass

        # fallback
        df["regime"] = "all"
        return df

    train = _ensure_regime(train)
    val = _ensure_regime(val)

    # ----------------------------------------
    # 2) Build a filter mask
    # ----------------------------------------
    def _filter(df, params):

        mask = pd.Series(True, index=df.index)

        # ---- IV filters ----
        if "min_iv" in params and "max_iv" in params:
            if "iv_bs" in df.columns:
                mask &= df["iv_bs"].between(params["min_iv"], params["max_iv"], inclusive="both")
                meta["applied_filters"]["iv_range"] = (params["min_iv"], params["max_iv"])

        # ---- DTE filters ----
        if "dte_min" in params and "dte_max" in params:
            if "dte" in df.columns:
                mask &= df["dte"].between(params["dte_min"], params["dte_max"], inclusive="both")
                meta["applied_filters"]["dte_range"] = (params["dte_min"], params["dte_max"])

        # ---- Moneyness filters ----
        if "moneyness_min" in params and "moneyness_max" in params:
            if "moneyness" in df.columns:
                mask &= df["moneyness"].between(
                    params["moneyness_min"],
                    params["moneyness_max"],
                    inclusive="both",
                )
                meta["applied_filters"]["moneyness_range"] = (
                    params["moneyness_min"],
                    params["moneyness_max"],
                )

        # ---- IV Rank ----
        if params.get("iv_rank_threshold") is not None and "iv_rank" in df.columns:
            thr = float(params["iv_rank_threshold"])
            mask &= df["iv_rank"] >= thr
            meta["applied_filters"]["iv_rank_threshold"] = thr

        # ---- Direction ----
        direction = params.get("direction", "neutral")
        if direction == "bull":
            # bullish → keep CE legs or positive deltas
            if "delta" in df.columns:
                mask &= df["delta"] > 0
            meta["applied_filters"]["direction"] = "bull"
        elif direction == "bear":
            if "delta" in df.columns:
                mask &= df["delta"] < 0
            meta["applied_filters"]["direction"] = "bear"
        else:
            meta["applied_filters"]["direction"] = "neutral"

        # ---- Allowed Regimes (NEW) ----
        allowed = params.get("allowed_regimes")
        if allowed:
            allowed_set = set([str(x) for x in allowed])
            mask &= df["regime"].astype(str).isin(allowed_set)
            meta["applied_filters"]["allowed_regimes"] = list(allowed_set)

        return df[mask].reset_index(drop=True)

    # ----------------------------------------
    # 3) Apply filters
    # ----------------------------------------
    train_f = _filter(train, params)
    val_f = _filter(val, params)

    # ----------------------------------------
    # 4) Feature engineering deltas
    # (Keep original functionality — add more if needed)
    # ----------------------------------------
    final_features = list(base_features)

    # size aggressiveness → scale PnL prediction inputs
    if params.get("size_aggressiveness") is not None:
        multiplier = float(params["size_aggressiveness"])
        meta["feature_deltas"]["size_aggressiveness"] = multiplier

        # Example engineered feature:
        # scaled_pred = base model could learn from this if we pass it in
        if "iv_bs" in final_features and "iv_bs_scaled" not in final_features:
            train_f["iv_bs_scaled"] = train_f["iv_bs"] * multiplier
            val_f["iv_bs_scaled"] = val_f["iv_bs"] * multiplier
            final_features.append("iv_bs_scaled")

    # ----------------------------------------
    # 5) Return transformed datasets + final features
    # ----------------------------------------
    return train_f, val_f, final_features, meta



# =====================================================================
# StrategyAdapter class (thin wrapper)
# Required for Phase-7 integration tests
# =====================================================================

class StrategyAdapter:
    """
    Lightweight wrapper around apply_strategy_adapter() to maintain
    backward compatibility with earlier phases and the integration tests.

    This class does NOT change any logic — it only forwards calls.
    """

    def __init__(self, base_features=None, debug=False):
        self.base_features = base_features or []
        self.debug = debug

    def apply(self, train_df, val_df, strategy_json):
        """
        Phase-7 tests expect:
        adapter = StrategyAdapter(...)
        train_out, val_out, feats, meta = adapter.apply(train, val, strategy)
        """
        return apply_strategy_adapter(
            train_df=train_df,
            val_df=val_df,
            strategy_json=strategy_json,
            base_features=self.base_features,
            debug=self.debug,
        )
