# ai/ai_client.py
# Recreated with full logging instrumentation for real AI client

from __future__ import annotations
import json
from typing import Dict, Any
from uuid import uuid4

from engine.logging_routes import log_ai, log_system
from models.decision_types import AIDecision
from ai.ai_prompt import AIPromptBuilder
from ai.ai_sanitizer import AISanitizer


class AIClient:
    """
    Real AI client that communicates with Ollama (or any LLM backend).

    Responsibilities:
      - Build prompt using AIPromptBuilder
      - Send sanitized prompt to the model
      - Receive raw model output
      - Sanitize + parse JSON using AISanitizer
      - Convert to AIDecision
      - Emit full JSON logs for prompt, sanitized prompt, raw output, parsed JSON
    """

    def __init__(self, model_name: str = "llama3:8b"):
        self.model_name = model_name
        self.prompt_builder = AIPromptBuilder()
        self.sanitizer = AISanitizer()
        self.last_prompt = None
        self.last_sanitized_prompt = None
        self.last_raw_response = None

        # Lazy import so MockAI can run without this
        try:
            import ollama
            self.client = ollama
        except Exception:
            self.client = None
            log_system("ollama_import_failed", {"model": model_name})

    # ------------------------------------------------------------------
    def _call_model(self, prompt: str) -> str:
        """
        Makes the real API call. If Ollama is not available, returns an error string.
        """
        if self.client is None:
            return '{"error": "ollama_not_available"}'

        try:
            result = self.client.generate(model=self.model_name, prompt=prompt)
            # Ollama typically returns dict with "response"
            if isinstance(result, dict) and "response" in result:
                return result["response"]
            return json.dumps(result)
        except Exception as exc:
            log_system("ai_call_exception", {"error": str(exc)})
            return '{"error": "exception"}'

    # ------------------------------------------------------------------
    def get_decision(self, payload: Dict[str, Any]) -> AIDecision:
        req_id = f"ai-{uuid4()}"

        # 1) BUILD PROMPT
        prompt, sanitized = self.prompt_builder.build(payload)
        self.last_prompt = prompt
        self.last_sanitized_prompt = sanitized

        # 2) CALL MODEL
        raw_text = self._call_model(sanitized)
        self.last_raw_response = {"raw": raw_text}

        # 3) PARSE / SANITIZE
        parsed_json = self.sanitizer.parse(raw_text)
        decision = AIDecision.from_dict(parsed_json)

        # 4) LOG AI EVENT
        log_ai(
            request_id=req_id,
            prompt=prompt,
            sanitized_prompt=sanitized,
            raw_response={"raw": raw_text},
            parsed_response=parsed_json,
            context={"model": self.model_name}
        )

        return decision
