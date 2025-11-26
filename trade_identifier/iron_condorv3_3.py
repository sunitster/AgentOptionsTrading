# trade_identifier/iron_condorv3_3.py
"""
Iron Condor v3.3
- Uses Ollama Chat API (/api/chat?stream=false)
- Robust parsing of Ollama responses (handles `data:` prefix, multiple JSON blobs, non-JSON content)
- Loads config/config.yaml and config/balanced_v3_3_system_prompt.txt if present
- Minimal OpenAI fallback client included
- Placeholder trading loop (replace with your real trading logic)
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


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "config.yaml")
SYSTEM_PROMPT_PATH = os.path.join(ROOT, "config", "balanced_v3_3_system_prompt.txt")

# Logging
def setup_logging(cfg: Dict[str, Any]):
    level_name = cfg.get("logging", {}).get("level", "INFO")
    file_path = cfg.get("logging", {}).get("file", "logs/iron_condor_ai.log")
    level = getattr(logging, level_name.upper(), logging.INFO)
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
                        filename=file_path)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    logging.getLogger().addHandler(console)

logger = logging.getLogger("iron_condor_v3_3")

def now_utc_str():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def make_run_id(prefix: str = "WPLAN") -> str:
    return f"{prefix}_{now_utc_str()}_{uuid.uuid4().hex[:12]}"

def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    p = path or CONFIG_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(f"Config file not found at {p!s}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg

def load_system_prompt(path: Optional[str] = None) -> str:
    p = path or SYSTEM_PROMPT_PATH
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return "You are Balanced v3.3. Provide concise trading recommendations and return JSON when requested."

# --- Ollama Chat client (robust) ---
class OllamaChatClient:
    """
    Uses Ollama /api/chat?stream=false
    Handles responses:
      - JSON body with 'message': {'content': ...}
      - 'data: {...}\n' prefixed stream lines
      - responses where assistant content itself is JSON (we try to parse)
      - if parse fails, return raw assistant text
    """
    def __init__(self, base_url: str, model: str, timeout: int = 20, max_retries: int = 1):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = int(timeout or 20)
        self.max_retries = int(max_retries or 1)
        logger.info("OllamaChatClient initialized (model=%s)", self.model)

    @staticmethod
    def _first_json_object(s: str) -> Optional[str]:
        """
        Try to extract the first JSON object from a string s.
        Returns JSON string or None.
        """
        if not isinstance(s, str):
            return None
        s = s.strip()
        # If starts with "data:" remove all leading "data:" tokens and try again
        if s.startswith("data:"):
            # remove any "data:" prefixes per-line and rejoin
            lines = [ln[len("data:"):].strip() if ln.startswith("data:") else ln for ln in s.splitlines()]
            s = "\n".join(lines).strip()

        # find first '{'
        start = s.find("{")
        if start == -1:
            return None
        # naive match: find last '}' after start
        end = s.rfind("}")
        if end == -1 or end <= start:
            return None
        candidate = s[start:end+1]
        return candidate

    @staticmethod
    def _safe_json_load(text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except Exception:
            return None

    def _send_request(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat?stream=false"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # Ollama uses different option names; use 'options' to set temperature/num_predict
            "stream": False,
            "options": {
                "temperature": float(temperature),
                # Ollama expects num_predict (tokens to produce); keep int
                "num_predict": int(max_tokens)
            }
        }

        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout)
                text = (r.text or "").strip()
                # if 200 but body empty, return minimal structure
                if r.status_code != 200:
                    logger.warning("ollama chat request status=%s text=%s", r.status_code, text[:400])
                    raise RuntimeError(f"ollama returned status {r.status_code}")

                # Clean "data:" streaming prefixes often seen
                # If the body contains multiple JSONs or extra text, try to extract first {}
                # First attempt: parse whole body
                parsed = self._safe_json_load(text)
                if parsed is not None:
                    return parsed

                # If not parseable, try to extract first JSON object
                candidate = self._first_json_object(text)
                if candidate:
                    parsed = self._safe_json_load(candidate)
                    if parsed is not None:
                        return parsed

                # If still not parseable, but body looks like {"model":...,"message":{...}} truncated issues,
                # attempt to parse line-by-line for first valid JSON
                for ln in text.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    if ln.startswith("data:"):
                        ln = ln[len("data:"):].strip()
                    parsed = self._safe_json_load(ln)
                    if parsed is not None:
                        return parsed

                # As last resort, compose a fallback dict with raw text in 'raw'
                return {"raw": text, "model_response_status": "unparsed", "http_status": r.status_code}
            except Exception as exc:
                last_exc = exc
                logger.warning("ollama chat failure (attempt %d): %s", attempt + 1, exc)
                # small backoff
                time.sleep(0.1 + attempt * 0.1)
                continue
        # if reached here, all attempts failed
        raise RuntimeError(f"ollama chat failed after retries: {last_exc!s}")

    def request(self, system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> Dict[str, Any]:
        """
        Returns a dict. Typical successful shape (from Ollama chat) is:
          {"model": "...", "created_at": "...", "message": {"role":"assistant","content":"..."} , ...}
        We will attempt to extract assistant content and parse it as JSON if possible.
        If assistant content is JSON, return the parsed JSON. Otherwise return {"raw": assistant_text}
        """
        try:
            data = self._send_request(system_prompt, user_prompt, max_tokens, temperature)
        except Exception as exc:
            logger.warning("Ollama chat request failed permanently: %s", exc)
            return {"error": str(exc)}

        # Try to obtain assistant text in several possible shapes
        assistant_text = None
        # primary new chat API shape: 'message' -> 'content'
        if isinstance(data, dict):
            msg = data.get("message") or {}
            if isinstance(msg, dict):
                assistant_text = msg.get("content") or msg.get("text") or None

            # older/alternate shape: 'choices' list with 'message'
            if assistant_text is None:
                choices = data.get("choices") or []
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        m = first.get("message") or {}
                        if isinstance(m, dict):
                            assistant_text = m.get("content") or m.get("text")
                        else:
                            assistant_text = first.get("text") or first.get("content")

            # alternate top-level keys
            if assistant_text is None and "content" in data:
                assistant_text = data.get("content")

        # If still None, maybe the entire response is raw text inside 'raw'
        if assistant_text is None:
            if isinstance(data, dict) and "raw" in data:
                assistant_text = data.get("raw")
            else:
                # convert to string
                try:
                    assistant_text = json.dumps(data)
                except Exception:
                    assistant_text = str(data)

        assistant_text = (assistant_text or "").strip()

        # Try to parse assistant_text as JSON (assistant might reply with a JSON object)
        parsed = None
        if assistant_text:
            # First, direct JSON load
            parsed = self._safe_json_load(assistant_text)
            if parsed is None:
                # Maybe the assistant returned JSON inside markdown or with extra text; extract first {...}
                candidate = self._first_json_object(assistant_text)
                if candidate:
                    parsed = self._safe_json_load(candidate)

        if parsed is not None:
            # good: assistant returned JSON — return it
            return parsed

        # Not JSON: return the raw string in a consistent wrapper
        return {"suggestion": "keep", "confidence": 0.0, "explanation": "unparsed-assistant-text", "raw": assistant_text}

# Minimal OpenAI fallback wrapper
class OpenAIClient:
    def __init__(self, cfg: Dict[str, Any]):
        try:
            import openai
        except Exception:
            raise RuntimeError("openai package not available")
        self.openai = openai
        self.model = cfg.get("ai", {}).get("model")
        logger.info("OpenAI client initialized (model=%s)", self.model)

    def request(self, system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = self.openai.ChatCompletion.create(model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        return resp

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
            # prefer ai.model; allow overriding to map balanced_v3.3 -> llama3.2 if user configured
            model = ai_cfg.get("model") or ollama_cfg.get("model")
            # if user configured balanced_v3.3 but wants to use a local llama3.2, they can put "llama3.2" in config
            timeout = ai_cfg.get("timeout_secs", ollama_cfg.get("timeout", 20))
            retries = ai_cfg.get("retries", 1)
            try:
                self.client = OllamaChatClient(base_url=base_url, model=model, timeout=timeout, max_retries=retries)
            except Exception as e:
                logger.warning("Failed to create OllamaChatClient: %s", e)
                self.client = None
        elif client_name == "openai":
            try:
                self.client = OpenAIClient(self.cfg)
            except Exception as e:
                logger.warning("Failed to create OpenAI client: %s", e)
                self.client = None
        else:
            logger.warning("Unknown AI client '%s' configured", client_name)
            self.client = None

    def ask(self, user_prompt: str) -> Dict[str, Any]:
        if not self.client:
            return {"suggestion": "keep", "confidence": 0.0, "explanation": "fallback-no-client", "docs_used": 0}
        try:
            ai_cfg = self.cfg.get("ai", {})
            max_tokens = ai_cfg.get("max_tokens", 512)
            temperature = ai_cfg.get("temperature", 0.0)
            # call client.request — both clients return dict-like
            data = self.client.request(self.system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature)
            # If client returned a dict already representing suggestion/confidence, return it
            if isinstance(data, dict):
                # If returned raw wrapper from OllamaChatClient containing 'raw' and 'unparsed', convert to fallback
                if data.get("suggestion") is not None:
                    return data
                # If data contains assistant fields, attempt to extract:
                # some responses may directly be the model's JSON (like {"suggestion":"close","confidence":0.7,...})
                # so return data as-is if it looks like that (heuristic: contains 'suggestion' or 'confidence')
                if "suggestion" in data or "confidence" in data:
                    return data
                # Otherwise if it contains 'message' or 'raw', handle earlier in client; just return wrapper if present
                if "raw" in data:
                    return {"suggestion": "keep", "confidence": 0.0, "explanation": "unparsed-response", "raw": data.get("raw")}
                # fallback: return the dict stringified
                return {"suggestion": "keep", "confidence": 0.0, "explanation": "unknown-response-shape", "raw": json.dumps(data)}
            else:
                # unexpected non-dict response
                return {"suggestion": "keep", "confidence": 0.0, "explanation": "non-dict-response", "raw": str(data)}
        except Exception as exc:
            logger.warning("AI request error: %s", exc)
            return {"suggestion": "keep", "confidence": 0.0, "explanation": f"fallback-error: {exc}", "docs_used": 0}

# --- Placeholder trading logic (keep/replace with your real logic) ---
class TradeEngine:
    def __init__(self, cfg: Dict[str, Any], ai_requester: AIRequester):
        self.cfg = cfg
        self.ai = ai_requester
        self.lot_size = cfg.get("strategy", {}).get("lot_size", 25)
        self.ai_threshold = cfg.get("strategy", {}).get("ai_confidence_threshold", 0.75)

    def run_backtest(self, start: str, end: str, weekly: bool = True, verbose: bool = False):
        executed = 0
        skipped = 0
        runs = 52
        for i in range(1, runs + 1):
            plan_id = make_run_id()
            user_prompt = self._build_prompt_for_plan(plan_id, start, end, i)
            decision = self.ai.ask(user_prompt)
            # decision may be parsed JSON or wrapper; normalize
            suggestion = decision.get("suggestion", "keep")
            try:
                confidence = float(decision.get("confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0
            explanation = decision.get("explanation", "")
            if suggestion == "keep" or confidence < self.ai_threshold:
                logger.info("EXEC %s pnl=93.75 conf=%.2f (%s)", plan_id, confidence, explanation)
                executed += 1
            else:
                logger.info("SKIP %s suggestion=%s conf=%.2f (%s)", plan_id, suggestion, confidence, explanation)
                skipped += 1
            # avoid hammering local server
            time.sleep(0.05)
        logger.warning("Run complete executed=%s skipped=%s", executed, skipped)
        return {"executed": executed, "skipped": skipped}

    def _build_prompt_for_plan(self, plan_id: str, start: str, end: str, idx: int) -> str:
        return f"Assess weekly plan {plan_id} (index {idx}) between {start} and {end}. Return JSON with keys 'suggestion','confidence','explanation' and optional 'docs_used'."

# --- CLI ---
def parse_args():
    p = argparse.ArgumentParser(description="Iron Condor v3.3 backtester with Ollama Chat")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--weekly", action="store_true")
    p.add_argument("--ai", action="store_true")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config(CONFIG_PATH)
    setup_logging(cfg)
    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH)
    # If CLI ai flag set, temporarily enable
    if args.ai:
        cfg.setdefault("ai", {})["enabled"] = True
    ai_requester = AIRequester(cfg=cfg, system_prompt=system_prompt)
    engine = TradeEngine(cfg=cfg, ai_requester=ai_requester)

    if args.backtest:
        start = args.start or cfg.get("data", {}).get("start") or "2025-01-01"
        end = args.end or cfg.get("data", {}).get("end") or "2025-06-30"
        result = engine.run_backtest(start=start, end=end, weekly=args.weekly, verbose=args.verbose)
        logger.info("Backtest result: %s", result)
    else:
        logger.info("No action selected. Use --backtest to run.")

if __name__ == "__main__":
    main()
