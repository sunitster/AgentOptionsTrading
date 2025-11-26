# ai/ai_prompt.py
# Simple prompt builder for AI client

from __future__ import annotations
from typing import Dict, Any, Tuple
import json

class AIPromptBuilder:
    """
    Builds a system/user prompt for the LLM from the engine payload.

    The build() method returns a tuple (prompt, sanitized_prompt).
    - prompt: human-readable prompt (for debugging / logging)
    - sanitized_prompt: machine-friendly JSON string to send to the model

    This simple builder formats the plan and key signals into a compact JSON string.
    """

    def __init__(self):
        pass

    def build(self, payload: Dict[str, Any]) -> Tuple[str, str]:
        plan = payload.get("plan", {})
        risk_signals = payload.get("risk_signals", {})
        risk_score = payload.get("risk_score", 0)
        last_adj = payload.get("last_adjustment", 0)

        # Human-readable prompt for logs
        human = (
            f"Plan {plan.get('plan_id')} | underlying={plan.get('underlying')} "
            f"price={plan.get('underlying_price')} ivp={plan.get('iv_percentile')} "
            f"width={plan.get('width')} days_to_expiry={plan.get('days_to_expiry')} "
            f"risk_score={risk_score} risk_signals={risk_signals} last_adj={last_adj}"
        )

        # Sanitized JSON prompt the model can easily parse
        safe = {
            "plan_id": plan.get("plan_id"),
            "underlying": plan.get("underlying"),
            "underlying_price": plan.get("underlying_price"),
            "iv_percentile": plan.get("iv_percentile"),
            "width": plan.get("width"),
            "days_to_expiry": plan.get("days_to_expiry"),
            "recent_pnl": plan.get("recent_pnl"),
            "risk_signals": risk_signals,
            "risk_score": risk_score,
            "last_adjustment": last_adj,
            # instructions: request a JSON object with specific fields
            "instructions": (
                "Return a JSON object with keys: suggestion (keep/reduce_width/increase_width), "
                "confidence (0..1), explanation (short), adjustments:{width_delta:int}, "
                "before_width:int, after_width:int"
            )
        }

        sanitized_prompt = json.dumps(safe, ensure_ascii=False)
        return human, sanitized_prompt
