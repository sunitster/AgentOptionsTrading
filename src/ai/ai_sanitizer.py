# ai/ai_sanitizer.py
# Robust sanitizer / parser for LLM raw outputs

from __future__ import annotations
from typing import Dict, Any
import json
import re

class AISanitizer:
    """
    Tries to parse raw LLM output into a clean JSON dict matching AIDecision fields.

    parse(raw_text) -> Dict[str, Any]
    - If raw_text is valid JSON, use it.
    - Otherwise, attempt to extract the first {...} JSON object found by regex.
    - If parsing fails, return a safe default (keep, confidence 0.0)
    """

    JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

    def __init__(self):
        pass

    def parse(self, raw_text: str) -> Dict[str, Any]:
        if raw_text is None:
            return self._default()

        # Try direct JSON parse
        try:
            parsed = json.loads(raw_text)
            return self._normalize(parsed)
        except Exception:
            pass

        # Try to extract JSON object substring
        m = self.JSON_OBJ_RE.search(raw_text)
        if m:
            candidate = m.group(0)
            try:
                parsed = json.loads(candidate)
                return self._normalize(parsed)
            except Exception:
                pass

        # Try a permissive key-value extraction (very last resort)
        # Look for lines like suggestion: keep
        kv = {}
        for line in raw_text.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                kv[k.strip().strip('"')]=v.strip().strip('",')

        if kv:
            # build best-effort dict
            suggestion = kv.get("suggestion") or kv.get("action") or "keep"
            try:
                confidence = float(kv.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            try:
                wd = int(float(kv.get("width_delta", kv.get("delta", 0))))
            except Exception:
                wd = 0
            before = int(kv.get("before_width", 0) or 0)
            after = before + wd
            return {
                "suggestion": suggestion,
                "confidence": confidence,
                "explanation": kv.get("explanation", "parsed_kv"),
                "adjustments": {"width_delta": wd},
                "before_width": before,
                "after_width": after,
            }

        # fallback
        return self._default()

    def _normalize(self, parsed: Any) -> Dict[str, Any]:
        # If parsed is dict and contains expected keys, normalize types
        if not isinstance(parsed, dict):
            return self._default()

        suggestion = parsed.get("suggestion", parsed.get("action", "keep"))
        confidence = parsed.get("confidence", 0.0)
        explanation = parsed.get("explanation", parsed.get("reason", ""))
        adjustments = parsed.get("adjustments", parsed.get("delta", {}))
        if isinstance(adjustments, (int, float)):
            adjustments = {"width_delta": int(adjustments)}
        before = parsed.get("before_width", parsed.get("width", 0))
        after = parsed.get("after_width", before + adjustments.get("width_delta", 0))

        return {
            "suggestion": suggestion,
            "confidence": float(confidence or 0.0),
            "explanation": str(explanation or ""),
            "adjustments": {"width_delta": int(adjustments.get("width_delta", 0))},
            "before_width": int(before or 0),
            "after_width": int(after or 0),
        }

    def _default(self) -> Dict[str, Any]:
        return {
            "suggestion": "keep",
            "confidence": 0.0,
            "explanation": "default_parse_fallback",
            "adjustments": {"width_delta": 0},
            "before_width": 0,
            "after_width": 0,
        }
