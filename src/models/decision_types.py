# models/decision_types.py

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AIDecision:
    suggestion: str
    confidence: float
    explanation: str
    adjustments: Dict[str, int]
    before_width: int
    after_width: int

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AIDecision":
        return AIDecision(
            suggestion=d.get("suggestion", "keep"),
            confidence=float(d.get("confidence", 0.0)),
            explanation=d.get("explanation", ""),
            adjustments=d.get("adjustments", {"width_delta": 0}),
            before_width=int(d.get("before_width", 0)),
            after_width=int(d.get("after_width", 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestion": self.suggestion,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "adjustments": self.adjustments,
            "before_width": self.before_width,
            "after_width": self.after_width,
        }


@dataclass
class FinalDecision:
    plan_id: str
    suggestion: str
    adjustments: Dict[str, int]
    before_width: int
    after_width: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "suggestion": self.suggestion,
            "adjustments": self.adjustments,
            "before_width": self.before_width,
            "after_width": self.after_width,
        }
