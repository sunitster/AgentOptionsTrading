# src/engine/macro_filter.py
"""
Hybrid Macro Filter (Full) - Option C
- Combines numeric signals (nasdaq, sgx, vix) + news headlines + calendar events
- Optional LLM classifier (ollama/OpenAI) to classify headlines SAFE/DANGER
- Safe defaults: if data missing, do NOT block unless explicit trigger
- Exposes MacroFilter.fetch_and_update(fetcher) to pull fresh data from a fetcher adapter
- Exposes MacroFilter.is_risky_day() -> (bool, reason)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, date
import logging
import math
import json
import re
import os

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


@dataclass
class MacroConfig:
    # numeric thresholds (percent)
    nasdaq_drop_pct: float = 1.0        # Nasdaq down X% pre-open -> block
    sgx_drop_pct: float = 0.7           # SGX Nifty drop -> block
    india_vix_threshold: float = 18.0   # India VIX > this -> block
    sgx_abs_move_pct: float = 1.0       # SGX absolute move block
    # LLM confidence threshold (0..1): if LLM says DANGER with confidence >= this -> block
    llm_confidence_threshold: float = 0.8
    # keywords for hard-block (case-insensitive)
    hard_block_keywords: List[str] = field(default_factory=lambda: [
        "crash", "bubble", "selloff", "meltdown", "panic", "halt", "circuit", "collapse", "bankrun"
    ])
    # event calendar: list of ISO dates to block (strings "YYYY-MM-DD") OR a function
    block_event_dates: List[str] = field(default_factory=list)
    # minimum number of headlines before LLM is considered reliable
    min_headlines_for_llm: int = 2


class LLMAdapter:
    """
    Minimal LLM wrapper. You can pass your existing Ollama/OpenAI client instance here.
    The wrapper must implement .classify_risk(headlines: str) -> dict with keys:
       { "label": "SAFE"|"DANGER", "score": 0.0..1.0 }
    If no LLM client is provided, the MacroFilter will fall back to keyword rules.
    """

    def __init__(self, client: Optional[Any] = None, model_name: Optional[str] = None):
        self.client = client
        self.model_name = model_name

    def classify_risk(self, text: str) -> Dict[str, Any]:
        """
        Returns dict: {"label": "SAFE"|"DANGER", "score": float}
        Implementations:
          - Ollama/OpenAI: call client.chat or client.completions and parse result
          - Default fallback: use simple heuristics
        """
        if self.client is None:
            # fallback heuristic: check for hard keywords & negative words density
            lower = text.lower()
            for kw in ("crash", "bubble", "selloff", "panic", "meltdown", "collapse"):
                if kw in lower:
                    return {"label": "DANGER", "score": 0.9, "reason": f"keyword:{kw}"}
            # neutral
            return {"label": "SAFE", "score": 0.5, "reason": "heuristic_fallback"}

        # Try to call the client - adapt this section to your client API (ollama/OpenAI)
        try:
            # Example for Ollama/OpenAI-like wrapper that returns text
            prompt = (
                "You are an assistant that classifies market risk for short premium options.\n"
                "Given the following headlines, reply with JSON: {\"label\": \"SAFE\"|\"DANGER\", \"score\": 0.0-1.0}\n\n"
                "Headlines:\n" + text + "\n\nAnswer:"
            )
            # Example: if client uses Chat API
            if hasattr(self.client, "chat"):
                resp = self.client.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
                out = getattr(resp, "text", None) or str(resp)
            elif hasattr(self.client, "completion"):
                resp = self.client.completion(prompt=prompt, model=self.model_name, max_tokens=60)
                out = resp.get("text", "")
            else:
                out = str(self.client(prompt))
            # attempt to extract JSON from out
            import json, re
            m = re.search(r"\\{.*\\}", out, re.S)
            if m:
                parsed = json.loads(m.group(0))
                label = parsed.get("label", "SAFE")
                score = float(parsed.get("score", 0.5))
                return {"label": label.upper(), "score": score, "raw": out}
            # fallback: simple detection
            lower = out.lower()
            if "danger" in lower or "unsafe" in lower:
                return {"label": "DANGER", "score": 0.8, "raw": out}
            return {"label": "SAFE", "score": 0.6, "raw": out}
        except Exception as e:
            LOG.exception("LLM classification failed: %s", e)
            return {"label": "SAFE", "score": 0.5, "reason": "llm_error"}


class NewsProvider:
    """
    Adapter interface for news/headlines source. Implement fetch_headlines() -> List[str]
    Some example implementations provided: FileNewsProvider, RSSNewsProvider (skeleton)
    """

    def fetch_headlines(self) -> List[str]:
        return []


class FileNewsProvider(NewsProvider):
    """Read headlines from a local JSON/lines file. Accepts:
       - JSON array of {"title": "..."} OR
       - plain text with one headline per line
    """
    def __init__(self, path: str):
        self.path = path

    def fetch_headlines(self) -> List[str]:
        if not os.path.exists(self.path):
            return []
        try:
            text = open(self.path, "r", encoding="utf-8").read().strip()
            # try json array of dicts
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    heads = []
                    for item in data:
                        if isinstance(item, dict) and "title" in item:
                            heads.append(item["title"])
                        elif isinstance(item, str):
                            heads.append(item)
                    return heads
            except Exception:
                # not json: treat as lines
                return [line.strip() for line in text.splitlines() if line.strip()]
        except Exception:
            LOG.exception("Failed to read news file %s", self.path)
        return []


class SimpleMarketFetcher:
    """
    Example numeric fetcher API expected by MacroFilter.fetch_and_update().
    Implementations must provide:
      - get_nasdaq_change_pct() -> float (percent, negative if down)
      - get_sgx_change_pct() -> float
      - get_india_vix() -> float
      - get_sgx_abs_move_pct() -> float
      - event_calendar_today() -> List[str]
    You can implement a concrete fetcher that uses:
      - your KiteAPI (for SGX/Nifty/India VIX if available)
      - public web APIs (Yahoo Finance / yfinance) — adapt ourselves later
      - local files written by upstream processes
    """
    def __init__(self, providers: Dict[str, Any] = None):
        self.providers = providers or {}

    def get_nasdaq_change_pct(self) -> Optional[float]:
        fn = self.providers.get("nasdaq")
        if fn and callable(fn):
            try:
                return float(fn())
            except Exception:
                return None
        return None

    def get_sgx_change_pct(self) -> Optional[float]:
        fn = self.providers.get("sgx")
        if fn and callable(fn):
            try:
                return float(fn())
            except Exception:
                return None
        return None

    def get_india_vix(self) -> Optional[float]:
        fn = self.providers.get("vix")
        if fn and callable(fn):
            try:
                return float(fn())
            except Exception:
                return None
        return None

    def get_sgx_abs_move_pct(self) -> Optional[float]:
        fn = self.providers.get("sgx_abs")
        if fn and callable(fn):
            try:
                return float(fn())
            except Exception:
                return None
        return None

    def event_calendar_today(self) -> List[str]:
        fn = self.providers.get("events")
        if fn and callable(fn):
            try:
                return list(fn())
            except Exception:
                return []
        return []


class MacroFilter:
    def __init__(self, cfg: MacroConfig = None, llm_adapter: Optional[LLMAdapter] = None):
        self.cfg = cfg or MacroConfig()
        self.llm = llm_adapter or LLMAdapter()
        self._blocked = False
        self._reason = ""
        self._last_update = None
        self._metadata: Dict[str, Any] = {}

    def reset(self):
        self._blocked = False
        self._reason = ""
        self._last_update = None
        self._metadata = {}

    def update_from_numeric(self, nasdaq_pct: Optional[float], sgx_pct: Optional[float], india_vix: Optional[float], sgx_abs_pct: Optional[float]):
        """
        Apply numeric thresholds. If a strong numeric trigger fires, we block immediately.
        """
        reason_parts = []
        if nasdaq_pct is not None:
            self._metadata["nasdaq_pct"] = nasdaq_pct
            if nasdaq_pct <= -abs(self.cfg.nasdaq_drop_pct):
                self._blocked = True
                reason_parts.append(f"nasdaq_drop={nasdaq_pct}% <= -{self.cfg.nasdaq_drop_pct}%")
        if sgx_pct is not None:
            self._metadata["sgx_pct"] = sgx_pct
            if sgx_pct <= -abs(self.cfg.sgx_drop_pct):
                self._blocked = True
                reason_parts.append(f"sgx_drop={sgx_pct}% <= -{self.cfg.sgx_drop_pct}%")
        if sgx_abs_pct is not None:
            self._metadata["sgx_abs_pct"] = sgx_abs_pct
            if abs(sgx_abs_pct) >= abs(self.cfg.sgx_abs_move_pct):
                self._blocked = True
                reason_parts.append(f"sgx_move={sgx_abs_pct}% >= {self.cfg.sgx_abs_move_pct}%")
        if india_vix is not None:
            self._metadata["india_vix"] = india_vix
            if india_vix >= self.cfg.india_vix_threshold:
                self._blocked = True
                reason_parts.append(f"vix={india_vix} >= {self.cfg.india_vix_threshold}")
        if reason_parts:
            self._reason = \" & \".join(reason_parts)
            LOG.info(\"MacroFilter numeric triggered: %s\", self._reason)
            return

    def update_from_headlines(self, headlines: List[str]):
        """
        Use LLM if available; else fallback to keyword scanning.
        - If LLM returns DANGER above confidence threshold, block.
        - If ANY hard keyword present, block.
        """
        if not headlines:
            return

        txt = \"\\n\".join(headlines[:50])  # limit length
        self._metadata.setdefault(\"headlines_count\", len(headlines))
        # Hard keyword scan
        lowtxt = txt.lower()
        for kw in self.cfg.hard_block_keywords:
            if kw in lowtxt:
                self._blocked = True
                self._reason = f\"headline_keyword:{kw}\"
                LOG.info(\"MacroFilter headline keyword blocked: %s\", kw)
                return

        # LLM classification
        if len(headlines) >= self.cfg.min_headlines_for_llm:
            try:
                res = self.llm.classify_risk(txt)
                lab = res.get(\"label\", \"SAFE\").upper()
                score = float(res.get(\"score\", 0.0))
                self._metadata.setdefault(\"llm\", res)
                if lab == \"DANGER\" and score >= self.cfg.llm_confidence_threshold:
                    self._blocked = True
                    self._reason = f\"llm:{score:.2f}\"
                    LOG.info(\"MacroFilter LLM blocked: %s (score=%.2f)\", lab, score)
                    return
            except Exception:
                LOG.exception(\"LLM classify failed, falling back to keywords\")

    def update_from_events(self, event_dates: List[str]):
        try:
            today = date.today().isoformat()
            for d in event_dates:
                if d == today:
                    self._blocked = True
                    self._reason = f\"event_calendar:{d}\"
                    LOG.info(\"MacroFilter blocked due to event on %s\", d)
                    return
        except Exception:
            LOG.exception(\"Event calendar check failed\")

    def fetch_and_update(self, fetcher: SimpleMarketFetcher, news_provider: Optional[NewsProvider] = None):
        """
        Fetch numeric market data and headlines via adapters and update internal state.
        """
        self.reset()
        self._last_update = datetime.utcnow().isoformat()

        # numeric
        nasdaq_pct = None
        sgx_pct = None
        india_vix = None
        sgx_abs_pct = None
        try:
            nasdaq_pct = fetcher.get_nasdaq_change_pct()
            sgx_pct = fetcher.get_sgx_change_pct()
            india_vix = fetcher.get_india_vix()
            sgx_abs_pct = fetcher.get_sgx_abs_move_pct()
        except Exception:
            LOG.exception(\"Failed to fetch numeric data from fetcher\")

        self.update_from_numeric(nasdaq_pct, sgx_pct, india_vix, sgx_abs_pct)
        if self._blocked:
            return

        # events
        try:
            events = fetcher.event_calendar_today()
            self.update_from_events(events or [])
            if self._blocked:
                return
        except Exception:
            LOG.exception(\"Event fetch failed\")

        # headlines
        headlines = []
        try:
            if news_provider:
                headlines = news_provider.fetch_headlines()
            self.update_from_headlines(headlines)
            if self._blocked:
                return
        except Exception:
            LOG.exception(\"News fetch/update failed\")

        # explicit config-level block dates
        self.update_from_events(self.cfg.block_event_dates)

    def is_risky_day(self) -> Tuple[bool, str, Dict[str, Any]]:
        return (self._blocked, self._reason or \"ok\", self._metadata or {})

