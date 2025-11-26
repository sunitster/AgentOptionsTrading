# ai/overlay.py

import json
from typing import Dict, Any
from ai.client import LocalAIClient
from ai.prompts import AI_PROMPT_TMPL
from engine.logging_routes import log_ai


class AIOverlay:
    def __init__(self, model="llama3.2"):
        self.client = LocalAIClient(model=model)

    def run(self, plan: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
        context = {
            "plan": plan,
            "risk": risk
        }

        prompt = AI_PROMPT_TMPL.format(context=json.dumps(context, indent=2))
        raw_text = self.client.ask(prompt)

        # Log raw AI output
        log_ai(
            request_id=f"ai-{plan.get('plan_id')}",
            prompt=prompt,
            sanitized_prompt=prompt,
            raw_response={"text": raw_text}
        )

        # Safe parse
        parsed = self._safe_json(raw_text)

        # Final log
        log_ai(
            request_id=f"ai-{plan.get('plan_id')}-parsed",
            prompt="",
            sanitized_prompt="",
            raw_response={},
            parsed_response=parsed
        )

        return parsed

    @staticmethod
    def _safe_json(raw: str) -> Dict[str, Any]:
        """Try multiple parsing strategies."""
        try:
            return json.loads(raw)
        except:
            pass

        # Try extracting JSON substring
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except:
            return {
                "action": "keep",
                "width_adjustment": 0.0,
                "wing_offset": 0.0,
                "risk_score": 0.5,
                "reason": "fallback parser"
            }
