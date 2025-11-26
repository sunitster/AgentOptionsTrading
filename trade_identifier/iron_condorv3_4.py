#!/usr/bin/env python3
"""
iron_condorv3_4.py - Hybrid architecture (rules entry + AI overlay)
Balanced v4.2 — improvements:
  - A) KEEP-first logic (prefer keep unless strong risk triggers)
  - B) No repeated adjustments (engine local memory prevents repeated width deltas)
  - C) Risk triggers (IVP, proximity, recent losses) for adjustment recommendations
  - D1) Local Engine Memory (lightweight, deterministic)
  - F) AI returns before_width & after_width for traceability
  - Safe, fast Ollama client with retry, short timeout, cache_prompt support and deterministic parsing
Usage:
    python trade_identifier/iron_condorv3_4.py --backtest --weekly --ai --start 2025-01-01 --end 2025-06-30 --verbose
Notes:
 - Replace placeholder logic (IV percentile, underlying price, PnL simulation) with your real data sources.
 - This file focuses on robust prompt engineering, JSON strictness, and engine-side memory.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, Optional
import datetime as dt

# third-party
try:
    import yaml
except Exception:
    raise RuntimeError("Missing dependency 'pyyaml'. Install with: pip install pyyaml")

try:
    import requests
except Exception:
    raise RuntimeError("Missing dependency 'requests'. Install with: pip install requests")

# --- Paths / constants ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "config.yaml")
SYSTEM_PROMPT_PATH = os.path.join(ROOT, "config", "balanced_v4_2_system_prompt.txt")
LOG_DEFAULT = os.path.join(ROOT, "logs", "iron_condor_ai.log")

# --- Logging ---
def setup_logging(cfg: Dict[str, Any], verbose: bool = False):
    level_name = cfg.get("logging", {}).get("level", "INFO")
    file_path = cfg.get("logging", {}).get("file", LOG_DEFAULT)
    level = getattr(logging, level_name.upper(), logging.INFO)
    if verbose:
        level = logging.DEBUG
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
                        filename=file_path)
    # console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    logging.getLogger().addHandler(console)

logger = logging.getLogger("iron_condor_v3_4")

# --- Utilities ---
def now_utc_str():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def make_run_id(prefix: str = "WPLAN") -> str:
    return f"{prefix}_{now_utc_str()}_{uuid.uuid4().hex[:12]}"

# --- Config & prompts ---
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    p = path or CONFIG_PATH
    if not os.path.exists(p):
        logger.warning("Config file not found at %s — using empty config", p)
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_system_prompt(path: Optional[str] = None) -> str:
    p = path or SYSTEM_PROMPT_PATH
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    # fallback minimal system prompt for v4.2
    return (
        "You are Balanced v4.2 — a trading overlay model.\n\n"
        "STRICT MODE (IMPORTANT):\n"
        "- You MUST respond with ONLY valid JSON (one JSON object only).\n"
        "- No markdown, no commentary, no prefix/suffix.\n"
        "- Use double quotes for keys and values.\n"
        "- Fields required (exact): suggestion, confidence, explanation, adjustments, before_width, after_width.\n\n"
        "RESPONSE FORMAT (MANDATORY):\n"
        "{\n"
        '  "suggestion": "keep" | "skip" | "reduce_width" | "increase_width",\n'
        '  "confidence": float (0 to 1),\n'
        '  "explanation": "short reason",\n'
        '  "adjustments": {"width_delta": integer},\n'
        '  "before_width": integer,\n'
        '  "after_width": integer\n'
        "}\n\n"
        "RULES (engine-enforced):\n"
        "- Default suggestion is KEEP with confidence 0.50 unless a risk trigger is met.\n"
        "- Do NOT repeat the same width_delta that was applied in the previous decision (engine will pass last_adjustment).\n"
        "- Only suggest reduce_width when risk triggers exist (IVP>70, underlying proximity, prior loss, etc.).\n"
        "- Only suggest increase_width when clear low-vol conditions exist (IVP<20 etc.).\n"
        "- Confidence must be numeric. Width_delta must be integer.\n"
        "Your ONLY output must be exactly one JSON object following the schema above."
    )

# --- JSON extraction helpers (robust) ---
def _extract_first_json_object(s: str) -> Optional[str]:
    """Find a substring that looks like {...} or [...] and return it (first match)."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    # Find balanced braces by scanning — more robust than rfind() for nested or trailing text.
    stack = []
    start_idx = None
    for i, ch in enumerate(s):
        if ch == "{":
            if start_idx is None:
                start_idx = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    return s[start_idx:i+1]
    # fallback: array
    stack = []
    start_idx = None
    for i, ch in enumerate(s):
        if ch == "[":
            if start_idx is None:
                start_idx = i
            stack.append("[")
        elif ch == "]":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    return s[start_idx:i+1]
    return None

def _try_parse_json_in_string(s: str) -> Optional[Dict[str, Any]]:
    """Try to extract JSON contained inside a free-text assistant reply."""
    if not isinstance(s, str):
        return None
    stripped = s.strip()
    # If content wrapped in triple backticks, extract inside lines
    if stripped.startswith("```") and stripped.endswith("```"):
        try:
            inner = "\n".join(stripped.splitlines()[1:-1])
            return json.loads(inner)
        except Exception:
            pass
    # find first {...}
    candidate = _extract_first_json_object(stripped)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            # cautious replacement of single quotes with double quotes (best-effort)
            try:
                fixed = candidate.replace("'", '"')
                return json.loads(fixed)
            except Exception:
                return None
    # final attempt: parse the whole string
    try:
        return json.loads(stripped)
    except Exception:
        return None

# --- Ollama Chat Client (uses /api/chat) with safe + fast behavior ---
class OllamaChatClient:
    def __init__(self, base_url: str, model: str, timeout: int = 5, max_retries: int = 1, cache_prompt: bool = True):
        self.base_url = base_url.rstrip("/")
        if model.endswith(":latest"):
            # user requested not to use :latest — strip and log
            model = model.split(":", 1)[0]
            logger.warning("Configured model ends with ':latest' — stripped to avoid download issues.")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_prompt = cache_prompt
        logger.info("OllamaChatClient initialized (model=%s timeout=%ss retries=%s)", self.model, self.timeout, self.max_retries)

    def request(self, system_prompt: str, user_prompt: str,
                max_tokens: int = 150, temperature: float = 0.0) -> Dict[str, Any]:
        """
        SAFE + FAST + JSON-STRICT Ollama request
        - timeout default: short (5s)
        - retry up to max_retries on connection errors
        - optionally enable 'cache_prompt' in options for Ollama
        - on persistent failure returns a safe fallback dict (suggestion: skip)
        """
        url = f"{self.base_url}/api/chat?stream=false"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens)
            }
        }
        if self.cache_prompt:
            payload["options"]["cache_prompt"] = True

        def try_once(attempt: int):
            try:
                logger.debug("OLLAMA_REQUEST_ATTEMPT=%d payload=%s", attempt, json.dumps(payload)[:2000])
                r = requests.post(url, json=payload, timeout=self.timeout)
                return r
            except Exception as e:
                logger.warning("Ollama connection error (attempt %d): %s", attempt, e)
                return None

        r = try_once(1)
        if r is None:
            # small backoff then retry once
            time.sleep(0.25)
            for attempt in range(2, self.max_retries + 2):
                r = try_once(attempt)
                if r is not None:
                    break
                time.sleep(0.25)

        if r is None:
            logger.error("Ollama failed after %d attempts — returning fallback skip", self.max_retries + 1)
            return {
                "suggestion": "skip",
                "confidence": 1.0,
                "explanation": "ollama-connection-failure",
                "adjustments": {},
                "before_width": None,
                "after_width": None
            }

        raw = (r.text or "").strip()
        logger.debug("RAW_OLLAMA_RESPONSE = %s", raw[:2000])

        # If server returns lines prefixed with "data:" (stream style) — normalize
        if raw.startswith("data:"):
            try:
                parts = []
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        parts.append(line[len("data:"):].strip())
                    else:
                        parts.append(line)
                raw = "\n".join(parts)
            except Exception:
                logger.debug("Failed to normalize 'data:' streaming payload — continuing with raw")

        # Top-level parse attempt — Ollama returns JSON in many shapes
        data = None
        try:
            data = json.loads(raw)
        except Exception:
            logger.debug("Top-level JSON parse failed, attempting to extract JSON substring")
            extracted = _extract_first_json_object(raw)
            if extracted:
                try:
                    data = json.loads(extracted)
                except Exception:
                    logger.debug("Extracted substring still not JSON")
                    data = None

        assistant_text = None
        if isinstance(data, dict):
            # common shape: { "message": {"role":"assistant","content": "..."}, ... }
            msg = data.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                assistant_text = msg.get("content")
            # older/other shape: choices
            elif "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                first = data["choices"][0]
                if isinstance(first, dict) and "message" in first and isinstance(first["message"], dict):
                    assistant_text = first["message"].get("content")
                elif isinstance(first, dict) and "text" in first:
                    assistant_text = first.get("text")
            elif "text" in data and isinstance(data.get("text"), str):
                assistant_text = data.get("text")
        # If we couldn't parse top-level JSON to get assistant content, try raw string
        if assistant_text is None and isinstance(raw, str):
            assistant_text = raw

        if not assistant_text:
            logger.error("No assistant content found in Ollama response — returning fallback skip")
            return {
                "suggestion": "skip",
                "confidence": 1.0,
                "explanation": "ollama-no-content",
                "adjustments": {},
                "before_width": None,
                "after_width": None
            }

        logger.debug("RAW_ASSISTANT_TEXT = %s", assistant_text[:2000])

        parsed = _try_parse_json_in_string(assistant_text)
        logger.debug("PARSED_AI_JSON = %s", parsed)

        if parsed is None:
            # assistant_text did not contain JSON — return fallback skip but include raw for debugging
            logger.error("Assistant returned no valid JSON in content — returning fallback skip")
            return {
                "suggestion": "skip",
                "confidence": 1.0,
                "explanation": "ollama-invalid-json",
                "raw": assistant_text,
                "adjustments": {},
                "before_width": None,
                "after_width": None
            }

        # Ensure the structure contains required fields and types (engine will enforce stricter checks)
        parsed.setdefault("suggestion", "keep")
        parsed.setdefault("confidence", 0.5)
        parsed.setdefault("explanation", "")
        parsed.setdefault("adjustments", {})
        parsed.setdefault("before_width", None)
        parsed.setdefault("after_width", None)

        return parsed

# --- Minimal OpenAI fallback client (optional) ---
class OpenAIClient:
    def __init__(self, cfg: Dict[str, Any]):
        try:
            import openai
        except Exception:
            raise RuntimeError("openai package not available")
        self.openai = openai
        self.model = cfg.get("ai", {}).get("model")
        logger.info("OpenAI client initialized (model=%s)", self.model)

    def request(self, system_prompt: str, user_prompt: str, max_tokens: int = 150, temperature: float = 0.0) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = self.openai.ChatCompletion.create(model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        content = None
        if isinstance(resp, dict):
            choices = resp.get("choices", [])
            if choices and isinstance(choices[0], dict):
                content = choices[0].get("message", {}).get("content")
        if content:
            parsed = _try_parse_json_in_string(content)
            if parsed is not None:
                return parsed
            return {"suggestion": "keep", "confidence": 0.5, "explanation": "openai-unparsed", "raw": content}
        return {"suggestion": "skip", "confidence": 1.0, "explanation": "openai-no-content"}

# --- AI requester (chooses Ollama chat or OpenAI) ---
@dataclasses.dataclass
class AIRequester:
    cfg: Dict[str, Any]
    system_prompt: str
    client: Optional[Any] = None

    def __post_init__(self):
        ai_cfg = self.cfg.get("ai", {})
        if not ai_cfg.get("enabled", False):
            logger.info("AI disabled in config")
            self.client = None
            return
        client_name = ai_cfg.get("client", "ollama")
        if client_name == "ollama":
            ollama_cfg = self.cfg.get("ollama", {})
            base_url = ollama_cfg.get("base_url", "http://localhost:11434")
            model = ai_cfg.get("model") or ollama_cfg.get("model") or "llama3.2"
            timeout = ai_cfg.get("timeout_secs", ollama_cfg.get("timeout", 5))
            retries = ai_cfg.get("retries", ollama_cfg.get("retries", 1))
            cache_prompt = ai_cfg.get("cache_prompt", True)
            try:
                self.client = OllamaChatClient(base_url=base_url, model=model, timeout=timeout, max_retries=retries, cache_prompt=cache_prompt)
            except Exception as e:
                logger.warning("Failed to create OllamaChatClient: %s", e)
                self.client = None
        elif client_name == "openai":
            try:
                self.client = OpenAIClient(self.cfg)
            except Exception as e:
                logger.warning("Failed to create OpenAIClient: %s", e)
                self.client = None
        else:
            logger.warning("Unknown AI client '%s' — AI disabled", client_name)
            self.client = None

    def ask(self, user_prompt: str) -> Dict[str, Any]:
        if not self.client:
            # fallback: neutral keep suggestion
            return {"suggestion": "keep", "confidence": 0.5, "explanation": "ai-disabled", "adjustments": {}, "before_width": None, "after_width": None}
        try:
            ai_cfg = self.cfg.get("ai", {})
            max_tokens = int(ai_cfg.get("max_tokens", 150))
            temperature = float(ai_cfg.get("temperature", 0.0))
            # call client (OllamaChatClient returns already-parsed JSON when possible)
            return self.client.request(self.system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:
            logger.warning("AI request failed: %s", exc)
            return {"suggestion": "skip", "confidence": 1.0, "explanation": f"ai-request-error: {exc}", "adjustments": {}, "before_width": None, "after_width": None}

# --- Trading engine (Hybrid) ---
class TradeEngine:
    def __init__(self, cfg: Dict[str, Any], ai_requester: AIRequester):
        self.cfg = cfg
        self.ai = ai_requester
        self.lot_size = cfg.get("strategy", {}).get("lot_size", 25)
        self.ai_threshold = float(cfg.get("strategy", {}).get("ai_confidence_threshold", 0.75))
        # local memory to avoid repeated adjustments across plans
        # structure: { "last_adjustment": {"width_delta": int, "plan_id": str, "applied_width": int, "timestamp": str}, ... }
        self.memory: Dict[str, Any] = {"last_adjustment": None}
        # risk trigger thresholds (default values; configurable via config)
        self.ivp_reduce_threshold = float(cfg.get("strategy", {}).get("ivp_reduce_threshold", 70.0))
        self.ivp_increase_threshold = float(cfg.get("strategy", {}).get("ivp_increase_threshold", 20.0))
        # proximity threshold (in points) to strikes to consider "too close"
        self.proximity_threshold = int(cfg.get("strategy", {}).get("proximity_threshold", 100))
        # max allowed single width change magnitude (to prevent huge swings)
        self.max_width_delta = int(cfg.get("strategy", {}).get("max_width_delta", 50))

    # Deterministic candidate plan generator (placeholder) - replace with your real logic
    def deterministic_entry_rules(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dict describing a candidate plan from rules.
        Example fields: 'entry_price', 'strikes', 'width', 'days_to_expiry', 'underlying'
        NOTE: Replace placeholder values with real market-derived values.
        """
        base_strike = 18200  # placeholder underlying level
        width = 100
        plan = {
            "plan_id": make_run_id(),
            "index": idx,
            "entry_price": 93.75,
            "strikes": [base_strike - width, base_strike + width],
            "width": width,
            "days_to_expiry": 14,
            "underlying": base_strike,
            "notes": "deterministic-entry (placeholder)",
            # placeholders for market metadata (should be replaced by real values)
            "iv_percentile": 50.0,         # IV percentile
            "underlying_price": base_strike,
            "recent_pnl": 0.0,            # recent performance metric for same structure
            "last_applied_width": None
        }
        return plan

    def _build_ai_prompt(self, plan: Dict[str, Any], previous_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a compact prompt describing the deterministic plan and the overlay request.
        The prompt explicitly includes:
          - plan fields
          - risk trigger thresholds
          - previous_context (engine memory) to prevent repeated adjustments
          - strict JSON output requirements (we set system prompt too)
        Ask AI to return JSON only with keys:
          - suggestion: "keep" / "skip" / "reduce_width" / "increase_width"
          - confidence: 0.0 - 1.0
          - explanation: short text
          - adjustments: { width_delta: int }
          - before_width: int
          - after_width: int
        """
        # Minimal user-level instruction (system prompt contains strict mode)
        user = {
            "instruction": "Return ONLY JSON. No text. Follow schema strictly.",
            "plan": {
                "plan_id": plan.get("plan_id"),
                "entry_price": plan.get("entry_price"),
                "width": plan.get("width"),
                "days_to_expiry": plan.get("days_to_expiry"),
                "underlying": plan.get("underlying"),
                "iv_percentile": plan.get("iv_percentile"),
                "underlying_price": plan.get("underlying_price"),
                "recent_pnl": plan.get("recent_pnl"),
                "notes": plan.get("notes")
            },
            "risk_thresholds": {
                "ivp_reduce_threshold": self.ivp_reduce_threshold,
                "ivp_increase_threshold": self.ivp_increase_threshold,
                "proximity_threshold": self.proximity_threshold,
                "max_width_delta": self.max_width_delta
            },
            "previous_context": previous_context or {},
            "required_output": {
                "suggestion": "keep | skip | reduce_width | increase_width",
                "confidence": "0.0 - 1.0",
                "explanation": "string",
                "adjustments": {"width_delta": "int"},
                "before_width": "int",
                "after_width": "int"
            }
        }
        return json.dumps(user)

    def _engine_validate_and_apply_ai(self, plan: Dict[str, Any], ai_overlay: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take AI overlay (possibly noisy) and:
         - enforce schema and types
         - apply engine-level rules (KEEP preference, no repeated adjustments)
         - compute applied width and enforce max delta limits
         - update local memory if an adjustment was applied
        Returns the normalized overlay dict with applied fields.
        """
        normalized = {}
        # defensive parsing
        suggestion = str(ai_overlay.get("suggestion", "keep")).lower() if ai_overlay.get("suggestion") is not None else "keep"
        try:
            confidence = float(ai_overlay.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        explanation = str(ai_overlay.get("explanation", "") or "")
        adjustments = ai_overlay.get("adjustments") or {}
        try:
            width_delta = int(adjustments.get("width_delta", 0))
        except Exception:
            # fallback: try widthDelta
            try:
                width_delta = int(adjustments.get("widthDelta", 0))
            except Exception:
                width_delta = 0

        before_width = plan.get("width", 100)
        # if AI provided before/after explicitly, prefer those if sensible
        ai_before = ai_overlay.get("before_width")
        ai_after = ai_overlay.get("after_width")
        if isinstance(ai_before, int):
            before_width = ai_before

        # safety: clamp width_delta by max_width_delta
        if width_delta > self.max_width_delta:
            logger.debug("Clamping width_delta %s -> %s (max allowed %s)", width_delta, self.max_width_delta, self.max_width_delta)
            width_delta = self.max_width_delta
        if width_delta < -self.max_width_delta:
            logger.debug("Clamping width_delta %s -> %s (negative max allowed %s)", width_delta, -self.max_width_delta, self.max_width_delta)
            width_delta = -self.max_width_delta

        # Determine tentative after_width from AI suggestion
        if suggestion == "reduce_width":
            tentative_after = max(10, before_width + width_delta)  # width_delta likely negative
        elif suggestion == "increase_width":
            tentative_after = max(10, before_width + width_delta)
        else:
            # keep or skip -> no change
            tentative_after = before_width

        # Engine-level KEEP-first logic:
        # If suggestion is not 'keep' but confidence is low (<0.6) and no risk triggers, override to keep.
        # We will compute risk triggers now.
        ivp = float(plan.get("iv_percentile", 50.0) or 50.0)
        underlying_price = float(plan.get("underlying_price", plan.get("underlying", 0)) or 0)
        strikes = plan.get("strikes", [])
        proximity = 999999
        if strikes and isinstance(strikes, (list, tuple)) and underlying_price:
            # compute distance to nearest strike
            try:
                proximity = min(abs(underlying_price - s) for s in strikes if isinstance(s, (int, float)))
            except Exception:
                proximity = 999999

        recent_pnl = float(plan.get("recent_pnl", 0.0) or 0.0)

        # Risk trigger booleans
        risk_ivp = ivp >= self.ivp_reduce_threshold
        low_ivp = ivp <= self.ivp_increase_threshold
        risk_proximity = proximity <= self.proximity_threshold
        risk_recent_loss = recent_pnl < 0.0

        # Prevent repeating the same width_delta twice in a row:
        last_adj = self.memory.get("last_adjustment")
        repeated_adjustment = False
        if last_adj and isinstance(last_adj, dict):
            try:
                if int(last_adj.get("width_delta", 0)) == int(width_delta):
                    repeated_adjustment = True
            except Exception:
                repeated_adjustment = False

        # Enforce KEEP unless risk triggers or AI confidence strong and not repeated
        apply_adjustment = False
        applied_suggestion = suggestion

        if suggestion in ("reduce_width", "increase_width"):
            # must have at least one risk indicator OR high AI confidence
            if (risk_ivp or risk_proximity or risk_recent_loss or low_ivp) and not repeated_adjustment:
                # if AI confidence sufficiently high or risk strong, allow
                if confidence >= 0.6 or risk_ivp or risk_proximity:
                    apply_adjustment = True
                else:
                    logger.debug("AI suggested %s but confidence %.2f and no strong risk -> override to keep", suggestion, confidence)
                    applied_suggestion = "keep"
            else:
                # override to keep to avoid unnecessary changes or repeats
                if repeated_adjustment:
                    logger.debug("Detected repeated adjustment %s — overriding to keep", width_delta)
                else:
                    logger.debug("No risk triggers for suggestion %s (ivp=%.1f prox=%s recent_pnl=%.2f) -> override to keep", suggestion, ivp, proximity, recent_pnl)
                applied_suggestion = "keep"
                apply_adjustment = False
        elif suggestion in ("keep", "skip"):
            # keep or skip -> obey, but skip we treat as keep for entry; skip means engine shouldn't execute
            applied_suggestion = suggestion
            apply_adjustment = False
        else:
            # unknown suggestion -> keep
            logger.debug("Unknown suggestion '%s' from AI; defaulting to keep", suggestion)
            applied_suggestion = "keep"
            apply_adjustment = False

        # If apply_adjustment true but repeated was true -> block it
        if apply_adjustment and repeated_adjustment:
            logger.debug("Blocking repeated adjustment despite apply_adjustment=True")
            apply_adjustment = False
            applied_suggestion = "keep"

        # compute final after_width
        if apply_adjustment:
            after_width = max(10, before_width + width_delta)
            # additional guard: if after_width equals last_applied_width -> block
            if last_adj and last_adj.get("applied_width") == after_width:
                logger.debug("After width equals last applied width (%s) -> blocking to avoid repetition", after_width)
                apply_adjustment = False
                applied_suggestion = "keep"
                after_width = before_width
        else:
            after_width = before_width

        # update memory if we applied an adjustment
        if apply_adjustment and applied_suggestion in ("reduce_width", "increase_width"):
            self.memory["last_adjustment"] = {
                "width_delta": int(width_delta),
                "plan_id": plan.get("plan_id"),
                "applied_width": int(after_width),
                "timestamp": now_utc_str()
            }
            logger.debug("Updated engine memory last_adjustment=%s", self.memory["last_adjustment"])

        # Compose normalized result
        normalized["suggestion"] = applied_suggestion
        normalized["confidence"] = confidence
        normalized["explanation"] = explanation
        normalized["adjustments"] = {"width_delta": int(width_delta) if apply_adjustment else 0}
        normalized["before_width"] = int(before_width)
        normalized["after_width"] = int(after_width)

        return normalized

    def run_backtest(self, start: str, end: str, weekly: bool = True, verbose: bool = False) -> Dict[str, Any]:
        executed = 0
        skipped = 0
        total_pl = 0.0
        details = []

        # Simulate N weekly candidate plans (replace loop with actual plan generation)
        n = 52 if weekly else 10
        for i in range(1, n + 1):
            plan = self.deterministic_entry_rules(i)
            plan_id = plan["plan_id"]
            # include last adjustment in previous_context for AI prompt
            previous_context = self.memory.get("last_adjustment") or {}
            prompt = self._build_ai_prompt(plan, previous_context=previous_context)
            ai_overlay_raw = {"suggestion": "keep", "confidence": 0.5, "explanation": "no-ai", "adjustments": {}, "before_width": plan.get("width"), "after_width": plan.get("width")}

            if self.ai and self.ai.client:
                ai_overlay_raw = self.ai.ask(prompt)
            else:
                logger.debug("AI disabled or not configured; using deterministic plan as-is")

            # Defensive: ensure any raw overlays include before/after width where possible
            try:
                if "before_width" not in ai_overlay_raw or ai_overlay_raw.get("before_width") is None:
                    ai_overlay_raw["before_width"] = plan.get("width")
                if "after_width" not in ai_overlay_raw or ai_overlay_raw.get("after_width") is None:
                    ai_overlay_raw["after_width"] = plan.get("width")
            except Exception:
                ai_overlay_raw.setdefault("before_width", plan.get("width"))
                ai_overlay_raw.setdefault("after_width", plan.get("width"))

            # Normalize & enforce engine-level safeguards
            ai_overlay = self._engine_validate_and_apply_ai(plan, ai_overlay_raw)

            # Decide whether to execute
            suggestion = str(ai_overlay.get("suggestion", "keep")).lower()
            confidence = float(ai_overlay.get("confidence", 0.0) or 0.0)

            if suggestion in ("skip", "sell", "exit") and confidence >= self.ai_threshold:
                logger.info("SKIP by AI %s -> suggestion=%s conf=%.2f reason=%s", plan_id, suggestion, confidence, ai_overlay.get("explanation", ""))
                skipped += 1
                details.append({"plan_id": plan_id, "action": "skipped", "ai": ai_overlay})
                continue

            # Apply AI adjustments to plan before simulating execution
            applied_width = ai_overlay.get("after_width", plan.get("width"))
            plan["applied_width"] = applied_width

            # Simulate execution & P&L (placeholder: use entry_price as pnl dummy)
            pnl = float(plan.get("entry_price", 0.0))
            total_pl += pnl
            executed += 1

            logger.info("EXEC %s pnl=%.2f conf=%.2f (%s) adjustments=%s before=%s after=%s",
                        plan_id, pnl, confidence, ai_overlay.get("explanation", "") or "no explanation",
                        ai_overlay.get("adjustments", {}), ai_overlay.get("before_width"), ai_overlay.get("after_width"))

            details.append({"plan_id": plan_id, "action": "executed", "ai": ai_overlay, "pnl": pnl})

            # small sleep to avoid too-fast local requests
            time.sleep(0.02)

        logger.warning("Run complete executed=%s skipped=%s total_pl=%.2f", executed, skipped, total_pl)
        return {"executed": executed, "skipped": skipped, "total_pl": total_pl, "details": details}

# --- CLI parsing ---
def parse_args():
    p = argparse.ArgumentParser(description="Iron Condor v3.4 Hybrid backtester with AI overlay (Balanced v4.2)")
    p.add_argument("--backtest", action="store_true", help="Run backtest")
    p.add_argument("--weekly", action="store_true", help="Use weekly plans")
    p.add_argument("--ai", action="store_true", help="Enable AI (overrides config.enabled)")
    p.add_argument("--start", type=str, default=None, help="Backtest start date")
    p.add_argument("--end", type=str, default=None, help="Backtest end date")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()

# --- main ---
def main():
    args = parse_args()
    cfg = load_config(CONFIG_PATH)
    setup_logging(cfg, verbose=args.verbose)
    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH)
    logger.debug("SYSTEM_PROMPT = %s", system_prompt)

    # Respect CLI --ai to temporarily enable AI if needed
    if args.ai:
        cfg.setdefault("ai", {})["enabled"] = True

    ai_requester = AIRequester(cfg=cfg, system_prompt=system_prompt)

    engine = TradeEngine(cfg=cfg, ai_requester=ai_requester)

    if args.backtest:
        start = args.start or cfg.get("data", {}).get("start") or "2025-01-01"
        end = args.end or cfg.get("data", {}).get("end") or "2025-06-30"
        res = engine.run_backtest(start=start, end=end, weekly=args.weekly, verbose=args.verbose)
        logger.info("Backtest result: %s", res)
    else:
        logger.info("No action. Try --backtest to run.")

if __name__ == "__main__":
    main()
