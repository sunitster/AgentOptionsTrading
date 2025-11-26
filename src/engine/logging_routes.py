from typing import Dict, Any, Optional
from .logger import JSONLLogger
import datetime

# --- Universal serializer for safe JSON logging ---
def _serialize(obj):
    """Recursively convert payload into JSON-serializable objects.
    - datetime/date → ISO 8601 string
    - dict → serialize all values
    - list/tuple → serialize all elements
    - everything else returned as-is
    """
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    else:
        return obj


# Initialize or get existing logger singleton
def init_logging(root: str = "logs", app_name: str = None, flush_interval: float = 1.0):
    JSONLLogger.configure(root=root, app_name=app_name, flush_interval=flush_interval)
    return JSONLLogger.get()

def _get() -> JSONLLogger:
    return JSONLLogger.get()


# --- PATCHED LOGGING FUNCTIONS (no `default=` passed) ---

def log_decision(decision_id: str,
                 rule_outputs: Dict[str, Any],
                 ai_suggestion: Optional[Dict[str, Any]],
                 final_decision: Dict[str, Any],
                 context: Optional[Dict[str, Any]] = None):
    payload = _serialize({
        "decision_id": decision_id,
        "rule_outputs": rule_outputs,
        "ai_suggestion": ai_suggestion,
        "final_decision": final_decision,
        "context": context or {}
    })
    _get().log_decision(payload)


def log_ai(request_id: str,
           prompt: str,
           sanitized_prompt: str,
           raw_response: Dict[str, Any],
           parsed_response: Optional[Dict[str, Any]] = None,
           context: Optional[Dict[str, Any]] = None):
    payload = _serialize({
        "request_id": request_id,
        "prompt": prompt,
        "sanitized_prompt": sanitized_prompt,
        "raw_response": raw_response,
        "parsed_response": parsed_response or {},
        "context": context or {}
    })
    _get().log_ai(payload)


def log_risk(check_id: str, risk_state: Dict[str, Any], triggers: Optional[Dict[str, Any]] = None):
    payload = _serialize({
        "check_id": check_id,
        "risk_state": risk_state,
        "triggers": triggers or {}
    })
    _get().log_risk(payload)


def log_order(order_id: str, order: Dict[str, Any], reason: Optional[str] = None):
    payload = _serialize({
        "order_id": order_id,
        "order": order,
        "reason": reason or ""
    })
    _get().log_order(payload)


def log_trade(trade_id: str, order_id: str, fill: Dict[str, Any], pnl_update: Optional[Dict[str, Any]] = None):
    payload = _serialize({
        "trade_id": trade_id,
        "order_id": order_id,
        "fill": fill,
        "pnl_update": pnl_update or {}
    })
    _get().log_trade(payload)


def log_pnl(date: Any, summary: Dict[str, Any]):
    payload = _serialize({
        "date": date,
        "summary": summary
    })
    _get().log_pnl(payload)


def log_system(message: str, metadata: Optional[Dict[str, Any]] = None):
    payload = _serialize({
        "message": message,
        "metadata": metadata or {}
    })
    _get().log_system(payload)
