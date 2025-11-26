"""
Updated LLM challenger generator (template-based + robust parsing)

- Template-based prompts for weak local models (llama3.2)
- Robust markdown stripping and JSON auto-repair
- Converts small LLM parameter outputs into full challenger JSON
- Defensive validation + clamping
- Detailed logging for debugging

Drop this file into: src/learning/llm_candidate_generator.py
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


##############################################
# Utilities
##############################################

def _strip_markdown_fences(text: str) -> str:
    """
    Remove common markdown/code fences (```json or ```), and leading/trailing whitespace.
    Works even if multiple fences or fences with language specifiers are present.
    """
    if not isinstance(text, str):
        return ""

    # Remove all ``` and ```json (case-insensitive) occurrences
    cleaned = re.sub(r"```\s*json", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")

    # Remove surrounding single backticks as well (inline code)
    cleaned = cleaned.replace("`", "")

    return cleaned.strip()


def _attempt_json_repair(text: str) -> str:
    """
    Repair common LLM JSON formatting issues to increase chance of json.loads success.

    Fixes applied (non-exhaustive):
      - Ensures arrays are closed (adds trailing ] if missing)
      - Removes trailing commas before ] or }
      - Wraps single object into an array if needed
      - Leaves other text intact for manual inspection
    """
    s = text.strip()

    # Quick guard
    if not s:
        return s

    # Remove stray non-JSON prefix/suffix whitespace lines
    # (we keep internal text as-is to not accidentally destroy structure)
    # If the snippet looks like an object only, wrap into an array
    if s.startswith("{") and s.endswith("}"):
        return f"[{s}]"

    # If it starts with [ but missing trailing ], add it
    if s.startswith("[") and not s.endswith("]"):
        s = s + "]"

    # Remove trailing commas before ] or }
    s = re.sub(r",\s*([\]}])", r"\1", s)

    return s


def _safe_json_loads(text: str) -> Optional[Any]:
    """
    Try to parse JSON with an auto-repair attempt. Returns parsed object or None.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    t0 = _strip_markdown_fences(text)

    # direct try
    try:
        return json.loads(t0)
    except Exception:
        pass

    # attempt repair
    repaired = _attempt_json_repair(t0)
    try:
        return json.loads(repaired)
    except Exception:
        # final fallback: try to extract the first { ... } or [ ... ] block heuristically
        # find first balanced {} block
        s = t0
        first_obj = None
        try:
            start = s.index("{")
            # find a closing } by scanning and counting braces
            depth = 0
            for i, ch in enumerate(s[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        first_obj = s[start:i + 1]
                        break
        except ValueError:
            first_obj = None

        if first_obj:
            try:
                return json.loads(first_obj)
            except Exception:
                return None

    return None


##############################################
# Template prompt builder
##############################################

def _build_template_prompt(num_candidates: int) -> str:
    """
    Template prompt for weak LLMs: ask for a tiny JSON array of parameter objects.

    The LLM only provides simple numeric/string fields. Python will map these to the
    full `rules` + `metadata` structure.
    """
    example = {"direction": "bull", "min_iv": 10, "max_iv": 90}

    prompt = (
        "SYSTEM:\n"
        "You output ONLY JSON. No markdown, no code fences, no explanation text.\n"
        f"Produce EXACTLY {num_candidates} JSON objects inside a JSON array.\n"
        "Each object must contain EXACTLY these keys (and nothing else):\n"
        "  direction  -> one of: bull, bear, both\n"
        "  min_iv     -> number (0-100)\n"
        "  max_iv     -> number (0-100)\n\n"
        "VALID EXAMPLE OUTPUT:\n"
        f"[{json.dumps(example)}]\n\n"
        "NOW OUTPUT ONLY THE JSON ARRAY:\n"
    )

    return prompt


##############################################
# Ollama runner (stdin)
##############################################

def call_ollama(prompt: str, model: str = "llama3.2", timeout_seconds: int = 30) -> Optional[str]:
    """
    Run Ollama using stdin (modern syntax). Returns raw stdout on success or None.
    """
    try:
        cmd = ["ollama", "run", model]
        logger.debug("Calling ollama: %s", cmd)

        proc = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=timeout_seconds,
        )

        if proc.returncode != 0:
            logger.warning("Ollama returned non-zero exit %s: %s", proc.returncode, proc.stderr.decode("utf-8", errors="ignore"))
            return None

        return proc.stdout.decode("utf-8", errors="ignore")

    except FileNotFoundError:
        logger.warning("Ollama CLI not found on PATH; skipping Ollama call")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Ollama call timed out")
        return None
    except Exception as e:
        logger.exception("Unexpected error calling Ollama: %s", e)
        return None


##############################################
# Convert tiny params -> full challenger
##############################################

def _build_challenger_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert simple parameter dict from LLM into full challenger JSON.
    Applies validation, clamping, and fills defaults.
    """
    # direction
    direction = str(params.get("direction", "both")).lower()
    if direction not in ("bull", "bear", "both"):
        logger.debug("Invalid direction %s, defaulting to 'both'", direction)
        direction = "both"

    # numeric helpers
    def _to_float(x, default):
        try:
            return float(x)
        except Exception:
            return default

    min_iv = _to_float(params.get("min_iv", 0), 0)
    max_iv = _to_float(params.get("max_iv", 100), 100)

    # clamp
    min_iv = max(0.0, min(min_iv, 100.0))
    max_iv = max(0.0, min(max_iv, 100.0))

    # ensure min <= max
    if min_iv > max_iv:
        logger.debug("min_iv > max_iv (%s > %s) — swapping", min_iv, max_iv)
        min_iv, max_iv = max_iv, min_iv

    challenger = {
        "rules": {
            "direction_filter": direction,
            "min_iv_percentile": min_iv,
            "max_iv_percentile": max_iv,
            # defaults for fields we do not ask the LLM for
            "min_trend_strength": -1.0,
            "max_trend_strength": 1.0,
        },
        "metadata": {
            "generator": "llm",
            "confidence": 0.5,
        },
    }

    return challenger


##############################################
# Main generator
##############################################

def generate_llm_challengers(
    base_label_sample: Dict[str, Any],
    num_candidates: int = 3,
    model: str = "llama3.2",
) -> List[Dict[str, Any]]:
    """
    Template-based LLM challenger generator.

    Steps:
      - Build a small template prompt requesting a JSON array of simple param objects
      - Call Ollama
      - Parse/repair the JSON
      - Convert each param object into a full challenger JSON
    """
    logger.info("generate_llm_challengers (TEMPLATE): requesting %d using %s", num_candidates, model)

    prompt = _build_template_prompt(num_candidates)
    raw = call_ollama(prompt, model=model)

    if raw is None:
        logger.warning("LLM unavailable — returning no candidates")
        return []

    # Always log raw output at debug level for offline inspection
    logger.debug("RAW LLM OUTPUT:\n%s", raw)

    # Attempt to parse JSON from the raw output
    parsed = _safe_json_loads(raw)
    if parsed is None:
        logger.warning("Failed to parse returned JSON. Raw output (first 1000 chars):\n%s", raw[:1000])
        return []

    # Expect an array
    if isinstance(parsed, dict):
        # LLM returned a single object — wrap
        tiny_list = [parsed]
    elif isinstance(parsed, list):
        tiny_list = parsed
    else:
        logger.warning("Parsed JSON is not a list or object: %s", type(parsed))
        return []

    challengers: List[Dict[str, Any]] = []
    for i, p in enumerate(tiny_list):
        if not isinstance(p, dict):
            logger.warning("Skipping non-dict LLM item at index %d: %s", i, type(p))
            continue

        # build full challenger
        try:
            ch = _build_challenger_from_params(p)
            challengers.append(ch)
        except Exception as e:
            logger.exception("Failed building challenger from params (index %d): %s", i, e)

    logger.info("Generated %d template-based LLM challengers", len(challengers))
    return challengers[:num_candidates]


##############################################
# CLI for quick debugging
##############################################

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    sample = {"spot": 100.0, "iv": 15.0, "date": "2025-11-18"}
    out = generate_llm_challengers(sample, num_candidates=3)
    print(json.dumps(out, indent=2))
