# src/dashboard/server.py
"""
FastAPI server for Live Paper Dashboard — Advanced + WebSocket broadcaster

This file is a patched version of your original server that:
- Keeps all existing HTTP endpoints (/api/*) and behavior.
- Adds a WebSocket endpoint at /ws which streams a single JSON payload
  (snapshot + broker state + pnl_history + risk_state + current_position)
  to all connected clients at a configurable cadence.
- Accepts simple JSON control messages from clients (e.g. {"cmd":"force-exit"})
  and maps them to the same manual_exit.json mechanism used by /api/force-exit.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import math
import asyncio

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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

app = FastAPI(title="IC Live Paper Dashboard - Advanced (WS)")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ----------------------------------------------------------------------
# In-memory LTP history
# ----------------------------------------------------------------------
LTP_MAX_POINTS = 500
ltp_history: List[Dict[str, Any]] = []

# ----------------------------------------------------------------------
# WebSocket clients set and config
# ----------------------------------------------------------------------
_ws_clients: "set[WebSocket]" = set()
# Broadcast cadence in seconds (tune as needed)
BROADCAST_POLL_INTERVAL = 1.0


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
        "source": "fastapi_dashboard_ws",
    }
    try:
        with MANUAL_EXIT_FILE.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        LOG.exception("Failed to write manual exit flag")


def sanitize(obj):
    """
    Convert objects to JSON-safe representation:
    - datetime -> isoformat
    - NaN (float('nan')) or inf -> None
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
# Routes (unchanged behavior)
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


# ------------------------------
# Robust /api/atm
# ------------------------------
@app.get("/api/atm")
async def get_atm():
    """
    Return ATM-related data. Defensive parsing: ignore NaN/inf strikes.
    """
    snapshot = safe_load_json(SNAPSHOT_FILE, default=[])

    if not snapshot:
        return JSONResponse(sanitize({"atm": None, "count_strikes": 0}))

    # Build a clean set of strikes:
    # - skip None
    # - skip floats that are NaN or inf
    # - attempt safe numeric conversion from strings/floats -> integer strikes
    valid_strikes = set()
    underlying_price = None

    for r in snapshot:
        # pick underlying price if present (case-insensitive)
        try:
            if isinstance(r, dict):
                typ = str(r.get("instrument_type") or "").upper()
                if typ == "UNDERLYING":
                    underlying_price = r.get("ltp") or r.get("spot") or underlying_price
        except Exception:
            # keep robust
            pass

        # fetch strike safely
        s = None
        try:
            s_raw = r.get("strike") if isinstance(r, dict) else None
            if s_raw is None:
                s = None
            else:
                # if it's already int, accept
                if isinstance(s_raw, int):
                    s = s_raw
                elif isinstance(s_raw, float):
                    if math.isnan(s_raw) or math.isinf(s_raw):
                        s = None
                    else:
                        # 19500.0 -> 19500
                        if s_raw.is_integer():
                            s = int(s_raw)
                        else:
                            s = int(round(s_raw))
                elif isinstance(s_raw, str):
                    # strip and try float -> int
                    try:
                        sf = float(s_raw.strip())
                        if math.isnan(sf) or math.isinf(sf):
                            s = None
                        else:
                            if sf.is_integer():
                                s = int(sf)
                            else:
                                s = int(round(sf))
                    except Exception:
                        s = None
                else:
                    # unknown type, try to coerce
                    try:
                        sf = float(s_raw)
                        if math.isnan(sf) or math.isinf(sf):
                            s = None
                        else:
                            if sf.is_integer():
                                s = int(sf)
                            else:
                                s = int(round(sf))
                    except Exception:
                        s = None
        except Exception:
            s = None

        if s is not None:
            valid_strikes.add(s)

    strikes = sorted(valid_strikes)

    if not strikes:
        LOG.warning("get_atm: no valid strikes found in snapshot")
        return JSONResponse(sanitize({"atm": None, "count_strikes": 0, "underlying": underlying_price}))

    # Determine ATM: if underlying price present choose closest; otherwise middle strike
    try:
        if underlying_price is not None:
            # underlying_price might be string/float, make safe float
            try:
                up = float(underlying_price)
            except Exception:
                up = None

            if up is not None:
                atm = min(strikes, key=lambda s: abs(s - up))
            else:
                atm = strikes[len(strikes) // 2]
        else:
            atm = strikes[len(strikes) // 2]
    except Exception:
        atm = strikes[len(strikes) // 2]

    payload = {"atm": atm, "count_strikes": len(strikes), "underlying": underlying_price}
    return JSONResponse(sanitize(payload))


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
            # robust conversion for strike
            if isinstance(strike, float):
                if math.isnan(strike) or math.isinf(strike):
                    continue
                if strike.is_integer():
                    strike = int(strike)
                else:
                    strike = int(round(strike))
            elif isinstance(strike, str):
                try:
                    sf = float(strike.strip())
                    if math.isnan(sf) or math.isinf(sf):
                        continue
                    strike = int(round(sf))
                except Exception:
                    continue
            else:
                strike = int(strike)
        except Exception:
            continue

        # defensive type handling: convert to string then uppercase
        typ = str(r.get("instrument_type") or "").upper()

        if strike not in strike_map:
            strike_map[strike] = {"strike": strike, "CE": None, "PE": None}

        # ensure ltp is safe float
        try:
            ltp_val = r.get("ltp")
            if ltp_val is None:
                ltp = 0.0
            else:
                ltp = float(ltp_val)
                if math.isnan(ltp) or math.isinf(ltp):
                    ltp = 0.0
        except Exception:
            ltp = 0.0

        entry = {
            "tradingsymbol": r.get("tradingsymbol"),
            "ltp": ltp,
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
# WebSocket broadcaster and control
# ----------------------------------------------------------------------
async def _snapshot_broadcaster(poll_interval: float = BROADCAST_POLL_INTERVAL):
    """
    Background task that reads snapshot + states and broadcasts to connected websockets.
    """
    LOG.info("WebSocket broadcaster started (interval=%.3fs)", poll_interval)
    while True:
        try:
            snapshot = safe_load_json(SNAPSHOT_FILE, default=[])
            broker_state = safe_load_json(BROKER_STATE_FILE, default={})
            pnl_history = safe_load_json(PNL_HISTORY_FILE, default=[])
            risk_state = safe_load_json(RISK_STATE_FILE, default={})
            current_pos = safe_load_json(CURRENT_POS_FILE, default={})

            # attempt to append LTP point too
            try:
                if isinstance(snapshot, list) and len(snapshot) > 0:
                    first_row = snapshot[0]
                    spot = first_row.get("spot") if isinstance(first_row, dict) else None
                    ts = first_row.get("timestamp") if isinstance(first_row, dict) else None
                    if spot is not None:
                        _append_ltp_point(spot, ts)
            except Exception:
                LOG.exception("broadcaster failed to append ltp")

            payload = {
                "server_time": datetime.now().isoformat(timespec="seconds"),
                "snapshot": snapshot,
                "broker_state": broker_state,
                "pnl_history": pnl_history,
                "risk_state": risk_state,
                "current_position": current_pos,
                # include ltp timeseries for charts
                "ltp_times": [p["t"] for p in ltp_history],
                "ltp_prices": [p["p"] for p in ltp_history],
            }

            data = sanitize(payload)

            # send to all clients
            dead = []
            for ws in list(_ws_clients):
                try:
                    await ws.send_json(data)
                except Exception as e:
                    LOG.debug("WS send failed: %s", e)
                    dead.append(ws)

            # cleanup dead clients
            for d in dead:
                try:
                    _ws_clients.discard(d)
                    await d.close()
                except Exception:
                    pass

        except Exception:
            LOG.exception("snapshot_broadcaster error")
        await asyncio.sleep(poll_interval)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint:
    - pushes periodic snapshots to client via broadcaster
    - listens for simple control JSON messages from client, e.g. {"cmd":"force-exit"}
    """
    await websocket.accept()
    _ws_clients.add(websocket)
    LOG.info("WebSocket client connected. total=%d", len(_ws_clients))
    try:
        while True:
            # We expect the client mostly to be passive; await a small message for keepalive/control
            try:
                msg = await websocket.receive_text()
            except WebSocketDisconnect:
                raise
            except Exception:
                # Occasionally client may not send anything — continue listening
                await asyncio.sleep(0.1)
                continue

            # try to parse JSON commands
            try:
                obj = json.loads(msg)
                if isinstance(obj, dict):
                    cmd = obj.get("cmd")
                    if cmd == "force-exit":
                        write_manual_exit_flag()
                        await websocket.send_json({"ok": True, "message": "force-exit written"})
                    else:
                        # unknown command -> echo
                        await websocket.send_json({"ok": True, "message": f"unknown cmd {cmd}"})
                else:
                    await websocket.send_json({"ok": False, "message": "expected json object"})
            except json.JSONDecodeError:
                # ignore non-json text; optionally echo back
                await websocket.send_json({"ok": False, "message": "invalid json"})
            except Exception:
                LOG.exception("Error processing ws message")
                try:
                    await websocket.send_json({"ok": False, "message": "internal error"})
                except Exception:
                    pass

    except WebSocketDisconnect:
        LOG.info("WebSocket client disconnected")
    except Exception:
        LOG.exception("WebSocket endpoint error")
    finally:
        if websocket in _ws_clients:
            _ws_clients.discard(websocket)


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
        # set uvicorn access logger level to WARNING to reduce console spam (optional)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
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


# Ensure broadcaster is started when app starts
@app.on_event("startup")
async def _on_startup():
    # spawn broadcaster background task
    try:
        asyncio.create_task(_snapshot_broadcaster(poll_interval=BROADCAST_POLL_INTERVAL))
    except Exception:
        LOG.exception("Failed to start snapshot broadcaster")
