# src/dashboard/server.py
"""
FastAPI server for Live Paper Dashboard — Option B FINAL v2

Fixes:
- sanitize() now converts NaN -> None (JSON-safe).
- heatmap() reads instrument_type defensively (str(...).upper()).
- get_state() uses sanitize before returning JSON so NaNs never leak.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import math

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# ----------------------------------------------------------------------
# File paths used by engine
# ----------------------------------------------------------------------
BASE_DIR = Path("models") / "llm_trades"
BASE_DIR.mkdir(parents=True, exist_ok=True)

BROKER_STATE_FILE = BASE_DIR / "paper_broker_state.json"
PNL_HISTORY_FILE = BASE_DIR / "pnl_history.json"
SNAPSHOT_FILE = BASE_DIR / "latest_snapshot.json"
RISK_STATE_FILE = BASE_DIR / "risk_state.json"
MANUAL_EXIT_FILE = BASE_DIR / "manual_exit.json"
CURRENT_POS_FILE = BASE_DIR / "current_position.json"

# static dir relative to this file
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="IC Live Paper Dashboard - Advanced")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ----------------------------------------------------------------------
# In-memory LTP history
# ----------------------------------------------------------------------
LTP_MAX_POINTS = 500
ltp_history: List[Dict[str, Any]] = []


def safe_load_json(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        LOG.warning("Failed to load %s: %s", path, e)
        return default


def _append_ltp_point(spot: Optional[float], ts: Optional[str] = None):
    if spot is None:
        return
    try:
        if ts is None:
            ts = datetime.now().isoformat(timespec="seconds")

        ltp_history.append({"t": ts, "p": float(spot)})

        if len(ltp_history) > LTP_MAX_POINTS:
            del ltp_history[0 : len(ltp_history) - LTP_MAX_POINTS]

    except Exception:
        LOG.exception("Failed to append ltp point")


def write_manual_exit_flag():
    payload = {
        "force_exit": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": "fastapi_dashboard",
    }
    with MANUAL_EXIT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def sanitize(obj):
    """
    Convert objects to JSON-safe representation:
    - datetime -> isoformat
    - NaN (float('nan')) -> None
    - recursively sanitize lists/dicts
    """
    # primitives
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    # containers
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = sanitize(v)
        return out
    if isinstance(obj, list):
        return [sanitize(x) for x in obj]
    # fallback: convert to string
    try:
        return str(obj)
    except Exception:
        return None


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="index.html not found")
    return FileResponse(str(index_path))


@app.get("/api/state")
async def get_state():
    broker_state = safe_load_json(BROKER_STATE_FILE, default={})
    pnl_history = safe_load_json(PNL_HISTORY_FILE, default=[])
    snapshot = safe_load_json(SNAPSHOT_FILE, default=[])
    risk_state = safe_load_json(RISK_STATE_FILE, default={})
    current_pos = safe_load_json(CURRENT_POS_FILE, default={})

    starting_balance = float(broker_state.get("starting_balance", 0.0))
    capital = float(broker_state.get("capital", starting_balance))
    net_pnl = float(broker_state.get("pnl", capital - starting_balance))
    trades_count = len(broker_state.get("trades", []))

    flat_trades = []
    for t in broker_state.get("trades", []):
        flat_trades.append({
            "time": t.get("time"),
            "mode": t.get("mode"),
            "pnl": t.get("pnl"),
            "balance": t.get("balance"),
        })

    payload = {
        "server_time": datetime.now().isoformat(timespec="seconds"),
        "metrics": {
            "starting_balance": starting_balance,
            "capital": capital,
            "net_pnl": net_pnl,
            "trades_count": trades_count,
        },
        "pnl_history": pnl_history,
        "trades": flat_trades,
        "snapshot": snapshot,
        "risk_state": risk_state,
        "current_position": current_pos,
    }

    # append LTP point from snapshot
    try:
        if isinstance(snapshot, list) and len(snapshot) > 0:
            first_row = snapshot[0]
            spot = first_row.get("spot") if isinstance(first_row, dict) else None
            ts = first_row.get("timestamp") if isinstance(first_row, dict) else None

            if spot is not None:
                _append_ltp_point(spot, ts)
    except Exception:
        LOG.exception("Failed to update ltp from snapshot")

    return JSONResponse(sanitize(payload))


@app.post("/api/force-exit")
async def force_exit():
    write_manual_exit_flag()
    return {"status": "ok", "message": "Manual exit signal written"}


@app.get("/api/atm")
async def get_atm():
    snapshot = safe_load_json(SNAPSHOT_FILE, default=[])

    if not snapshot:
        return {"atm": None, "count_strikes": 0}

    strikes = sorted({int(r.get("strike")) for r in snapshot if r.get("strike") is not None})

    if not strikes:
        return {"atm": None, "count_strikes": 0}

    underlying_price = None
    for r in snapshot:
        if isinstance(r, dict) and r.get("instrument_type") == "UNDERLYING":
            underlying_price = r.get("ltp")
            break

    if underlying_price:
        atm = min(strikes, key=lambda s: abs(s - underlying_price))
    else:
        atm = strikes[len(strikes) // 2]

    return {"atm": atm, "count_strikes": len(strikes), "underlying": underlying_price}


@app.get("/api/ic-position")
async def ic_position():
    pos = safe_load_json(CURRENT_POS_FILE, default=None)
    return {"position": sanitize(pos) if pos else None}


@app.get("/api/greeks")
async def greeks():
    pos = safe_load_json(CURRENT_POS_FILE, default={})
    if not pos:
        return {"greeks": {}, "greeks_timeseries": []}

    legs = pos.get("legs", [])
    agg = {"delta": 0.0, "theta": 0.0, "vega": 0.0, "gamma": 0.0}

    for leg in legs:
        g = leg.get("greeks", {})
        agg["delta"] += float(g.get("delta", 0.0))
        agg["theta"] += float(g.get("theta", 0.0))
        agg["vega"] += float(g.get("vega", 0.0))
        agg["gamma"] += float(g.get("gamma", 0.0))

    ts = pos.get("greeks_timeseries")
    if ts:
        return {"greeks": agg, "greeks_timeseries": ts}
    else:
        return {
            "greeks": agg,
            "greeks_timeseries": [{"time": datetime.now().isoformat(timespec="seconds"), **agg}],
        }


@app.get("/api/margin")
async def margin():
    broker_state = safe_load_json(BROKER_STATE_FILE, default={})
    margin_info = broker_state.get("margin")

    if margin_info:
        return {"margin": margin_info}

    pos = safe_load_json(CURRENT_POS_FILE, default={})
    if pos and pos.get("margin"):
        return {"margin": pos.get("margin")}

    capital = float(broker_state.get("capital", 0.0))
    starting = float(broker_state.get("starting_balance", capital))
    used = max(0.0, starting - capital)

    return {"margin": {"used_estimate": used, "available": capital}}


@app.get("/api/heatmap")
async def heatmap():
    snapshot = safe_load_json(SNAPSHOT_FILE, default=[])
    if not snapshot:
        return {"rows": []}

    strike_map = {}

    for r in snapshot:
        strike = r.get("strike")
        if strike is None:
            continue

        try:
            strike = int(strike)
        except Exception:
            continue

        # defensive type handling: convert to string then uppercase
        typ = str(r.get("instrument_type") or "").upper()

        if strike not in strike_map:
            strike_map[strike] = {"strike": strike, "CE": None, "PE": None}

        entry = {
            "tradingsymbol": r.get("tradingsymbol"),
            "ltp": float(r.get("ltp") or 0.0),
            "oi": r.get("open_interest"),
        }

        if typ in ("CE", "CALL"):
            strike_map[strike]["CE"] = entry
        elif typ in ("PE", "PUT"):
            strike_map[strike]["PE"] = entry

    rows = [strike_map[k] for k in sorted(strike_map.keys(), reverse=True)]
    return {"rows": rows, "count": len(rows), "ok": True}


@app.get("/api/ltp-chart")
async def ltp_chart():
    snapshot = safe_load_json(SNAPSHOT_FILE, default=[])

    try:
        if isinstance(snapshot, list) and len(snapshot) > 0:
            first_row = snapshot[0]
            spot = None
            ts = None

            if isinstance(first_row, dict):
                spot = first_row.get("spot") or first_row.get("ltp")
                ts = first_row.get("timestamp") or first_row.get("time")

            if spot is not None:
                _append_ltp_point(spot, ts)

    except Exception:
        LOG.exception("Failed to read spot for ltp-chart")

    times = [p["t"] for p in ltp_history]
    prices = [p["p"] for p in ltp_history]

    return JSONResponse({"times": times, "prices": prices, "count": len(times)})


@app.get("/api/health")
async def health():
    return {"ok": True, "time": datetime.now().isoformat(timespec="seconds")}


# ----------------------------------------------------------------------
# DASHBOARD STARTER — required by agent_trade.py
# ----------------------------------------------------------------------
def start_dashboard(host="127.0.0.1", port=8000):
    """
    Start the FastAPI dashboard in a background thread.
    Called automatically by agent_trade.py when --dashboard is used.
    """
    import threading
    import uvicorn

    def _run():
        uvicorn.run(
            "src.dashboard.server:app",
            host=host,
            port=port,
            reload=False,
            log_level="info",
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
