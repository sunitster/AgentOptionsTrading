# engine/plan_builder.py

"""
plan_builder.py
----------------
Responsible for building deterministic trading plans from rule-based logic.

The output plan must be a DICT that the engine expects:
{
    "plan_id": str,
    "entry_price": float,
    "width": int,
    "underlying": float,
    "underlying_price": float,
    "iv_percentile": float,
    "days_to_expiry": int,
    "recent_pnl": float,
    ...
}

All non-AI logic should be handled here:
- strike selection (future: delta rules)
- width determination
- IVP-based regime classification
- expiry selection
- plan_id generation
"""

import time
import datetime
import uuid
from typing import Dict, Any


class PlanBuilder:
    """
    Builds deterministic plans for the trading engine.
    
    Responsibilities:
        - Generate plan IDs
        - Compute rule-based widths
        - Select underlying & price
        - Compute metadata
        - Produce clean plan dicts
    """

    def __init__(self):
        pass

    # ----------------------------------------------------
    # 1. Plan ID Generator
    # ----------------------------------------------------
    def _generate_plan_id(self) -> str:
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        uid = uuid.uuid4().hex[:12]
        return f"WPLAN_{ts}_{uid}"

    # ----------------------------------------------------
    # 2. Compute width (rule-based)
    # ----------------------------------------------------
    def _compute_width(self, iv_percentile: float) -> int:
        """
        Rule-based width logic.
        Replace with your real entry logic.

        Simple placeholder rules:
            - High IV → wider condor
            - Medium → standard width
            - Low IV → narrower condor
        """
        if iv_percentile >= 70:
            return 150
        elif iv_percentile <= 20:
            return 80
        else:
            return 100

    # ----------------------------------------------------
    # 3. Compute entry price (placeholder)
    # ----------------------------------------------------
    def _compute_entry_price(self, width: int, iv_percentile: float) -> float:
        """
        Placeholder rule:
            - wider wings & high IV → higher premium
        """
        base = 50
        vol_factor = iv_percentile * 0.3
        width_factor = width * 0.05
        return round(base + vol_factor + width_factor, 2)

    # ----------------------------------------------------
    # 4. Build Plan (Main Method)
    # ----------------------------------------------------
    def build_plan(
        self,
        *,
        underlying: float,
        underlying_price: float,
        iv_percentile: float,
        days_to_expiry: int,
        recent_pnl: float = 0.0,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Constructs the deterministic trading plan.

        Returns plan dict ready for engine.evaluate_plan().
        """

        width = self._compute_width(iv_percentile)
        entry_price = self._compute_entry_price(width, iv_percentile)
        plan_id = self._generate_plan_id()

        plan = {
            "plan_id": plan_id,
            "timestamp": int(time.time()),
            "entry_price": entry_price,
            "width": width,
            "underlying": underlying,
            "underlying_price": underlying_price,
            "iv_percentile": iv_percentile,
            "days_to_expiry": days_to_expiry,
            "recent_pnl": recent_pnl,
            "notes": notes or "deterministic-entry",
        }

        # ---- DEBUG PRINT REAL PLAN (only once) ----
        if not hasattr(self, "_printed_plan_example"):
            import json
            print("\n==== SAMPLE PLAN (from PlanBuilder) ====")
            print(json.dumps(plan, indent=2, default=str))
            print("========================================\n")
            self._printed_plan_example = True
        # -------------------------------------------

        return plan

    # ----------------------------------------------------
    # 5. Batch Builder (optional)
    # ----------------------------------------------------
    def build_plans_batch(
        self,
        market_data_list: list,
        days_to_expiry: int = 14
    ) -> list:
        """
        Build multiple plans from a list of simple market_data rows.

        market_data example:
            {
                "underlying": 18200,
                "price": 18210,
                "ivp": 55
            }
        """

        plans = []
        for md in market_data_list:
            plan = self.build_plan(
                underlying=md["underlying"],
                underlying_price=md["price"],
                iv_percentile=md["ivp"],
                days_to_expiry=days_to_expiry,
                recent_pnl=md.get("recent_pnl", 0),
                notes=md.get("notes", "")
            )
            plans.append(plan)
        return plans
