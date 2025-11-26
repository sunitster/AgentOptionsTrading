# engine/engine.py
# Recreated with logging integration (Step 1)

from __future__ import annotations
from typing import Dict, Any
from uuid import uuid4

from engine.logging_routes import (
    log_decision,
    log_ai,
    log_risk,
    log_system,
)

from models.decision_types import FinalDecision, AIDecision
import engine.risk as risk_module


class EngineConfig:
    def __init__(self,
                 max_width: int = 300,
                 max_width_delta: int = 50):
        self.max_width = max_width
        self.max_width_delta = max_width_delta


class TradingEngine:
    def __init__(self, ai_client=None, config: EngineConfig = None, memory=None):
        self.ai_client = ai_client
        self.config = config or EngineConfig()
        self.memory = memory

    # ---------------------- RULE EVALUATION ----------------------
    def _evaluate_rules(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        width = plan["width"]
        ivp = plan["iv_percentile"]
        proximity = plan.get("proximity", 0)

        rule_result = {
            "iv_regime": "high" if ivp >= 70 else "low" if ivp <= 20 else "normal",
            "proximity_flag": proximity > 100,
            "width_ok": width <= self.config.max_width,
        }

        # Log rule-only stage
        log_decision(
            decision_id=f"rules-{plan['plan_id']}",
            rule_outputs=rule_result,
            ai_suggestion=None,
            final_decision={"stage": "rule_check"},
            context={"plan_id": plan["plan_id"]}
        )

        return rule_result

    # ---------------------- AI OVERLAY COMBINATION ----------------------
    def _combine_decisions(self,
                           plan: Dict[str, Any],
                           rule_result: Dict[str, Any],
                           ai_decision: AIDecision) -> FinalDecision:

        width = plan["width"]
        wd = ai_decision.adjustments.get("width_delta", 0)

        # Enforce engine-wide delta limit
        if abs(wd) > self.config.max_width_delta:
            wd = max(-self.config.max_width_delta,
                     min(self.config.max_width_delta, wd))

        before = width
        after = width + wd

        return FinalDecision(
            plan_id=plan["plan_id"],
            suggestion=ai_decision.suggestion,
            adjustments={"width_delta": wd},
            before_width=before,
            after_width=after,
        )

    # ---------------------- MAIN EVALUATE FUNCTION ----------------------
    def evaluate_plan(self, plan: Dict[str, Any], risk_thresholds: Dict[str, Any]) -> FinalDecision:
        decision_uuid = f"dec-{plan['plan_id']}-{uuid4()}"
        plan_id = plan["plan_id"]

        # 1) RULES
        rule_result = self._evaluate_rules(plan)

        # 2) RISK
        risk_signals = risk_module.compute_risk_signals(plan, risk_thresholds)
        risk_score = risk_module.compute_risk_score(risk_signals)

        log_risk(
            check_id=f"risk-{plan_id}",
            risk_state={"risk_signals": risk_signals, "risk_score": risk_score},
            triggers={k: v for k, v in risk_signals.items() if v}
        )

        # 3) AI OVERLAY
        if self.ai_client:
            ai_payload = {
                "plan": plan,
                "risk_signals": risk_signals,
                "risk_score": risk_score,
                "last_adjustment": self.memory.get_last_width_delta() if self.memory else 0,
            }

            ai_result: AIDecision = self.ai_client.get_decision(ai_payload)

            # Log AI event
            log_ai(
                request_id=f"ai-{plan_id}",
                prompt=getattr(self.ai_client, "last_prompt", "n/a"),
                sanitized_prompt=getattr(self.ai_client, "last_sanitized_prompt", "n/a"),
                raw_response=getattr(self.ai_client, "last_raw_response", {}),
                parsed_response=ai_result.to_dict(),
                context={"plan_id": plan_id, "risk_score": risk_score}
            )
        else:
            ai_result = AIDecision.from_dict({
                "suggestion": "keep",
                "confidence": 0.0,
                "explanation": "No AI client available",
                "adjustments": {"width_delta": 0},
                "before_width": plan["width"],
                "after_width": plan["width"],
            })

        # 4) COMBINE
        final = self._combine_decisions(plan, rule_result, ai_result)

        # 5) MEMORY UPDATE
        if self.memory:
            self.memory.record_adjustment(plan_id, final.adjustments.get("width_delta", 0))

        # FINAL DECISION LOG
        log_decision(
            decision_id=decision_uuid,
            rule_outputs=rule_result,
            ai_suggestion=ai_result.to_dict(),
            final_decision=final.to_dict(),
            context={
                "plan_id": plan_id,
                "risk_score": risk_score,
                "risk_signals": risk_signals
            }
        )

        return final
