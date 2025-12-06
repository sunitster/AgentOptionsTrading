from __future__ import annotations
import os
import json
import logging
import time as _time
import sqlite3
from datetime import datetime, time, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# IV/delta helpers (best-effort)
try:
    from src.iv_utils import implied_volatility, bs_price
except Exception:
    def implied_volatility(*args, **kwargs):
        return None
    def bs_price(*args, **kwargs):
        return None

try:
    from scipy.stats import norm
    from math import log, sqrt
except Exception:
    # degrade gracefully if scipy missing
    class _FallbackNorm:
        @staticmethod
        def cdf(x):
            import math
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    norm = _FallbackNorm()
    def log(x):
        import math
        return math.log(x)
    def sqrt(x):
        import math
        return math.sqrt(x)


from src.live.kite_api import KiteAPI
from src.live.paper_broker import PaperBroker
from src.engine.risk_engine import RiskEngine
from src.engine.exit_engine import ExitEngine, ExitEngineConfig
from src.trading.signal_generator import SignalGenerator
from src.live.kite_data import full_option_chain, available_strikes, closest_strike

# ML pipeline + Ollama
from src.ml_pipeline import run_selection_once
import ollama  # local client


# live trade recorder (optional)
try:
    from src.live.live_trade_recorder import record_trade
except Exception:
    def record_trade(ic, entry_snapshot, exit_snapshot, realized_pnl, exit_reason, metadata=None):
        return {"status": "recorder_missing"}


# --- Logger ---
log = logging.getLogger("LivePaperEngine")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
log.setLevel(logging.INFO)


# Monitor folder used by dashboard
MONITOR_PATH = os.path.join("models", "llm_trades")
MANUAL_EXIT_FILE = os.path.join(MONITOR_PATH, "manual_exit.json")
LATEST_SNAPSHOT_FILE = os.path.join(MONITOR_PATH, "latest_snapshot.json")
RISK_STATE_FILE = os.path.join(MONITOR_PATH, "risk_state.json")
PAPER_BROKER_STATE_FILE = os.path.join(MONITOR_PATH, "paper_broker_state.json")
PNL_HISTORY_FILE = os.path.join(MONITOR_PATH, "pnl_history.json")
CURRENT_POS_FILE = os.path.join(MONITOR_PATH, "current_position.json")
RECOVERED_POS_FILE = os.path.join(MONITOR_PATH, "recovered_position.json")
TRADES_DB_FILE = os.path.join(MONITOR_PATH, "trades.db")

os.makedirs(MONITOR_PATH, exist_ok=True)



class LivePaperEngine:
    def __init__(
        self,
        symbol: str = "NIFTY",
        lot_size: int = 25,
        width: int = 150,
        minute_interval: int = 1,
        size_aggressiveness: float = 1.0,
        starting_capital: float = 1_000_000.0,
        risk_daily_loss_limit: float = 25000.0,
        risk_max_trade_pct: float = 0.02,
        max_daily_trades: int = 3,
        dashboard: bool = True,
        no_new_entries_after: time = time(15, 15),
        stale_exit_cooldown_seconds: int = 120,
        allow_overlap_open_ic: bool = False,
        **kwargs: Any,
    ) -> None:

        self.symbol = symbol
        self.lot_size = int(lot_size)
        self.width = int(width)
        self.minute_interval = int(minute_interval)
        self.size_aggressiveness = float(size_aggressiveness)
        self.max_daily_trades = int(max_daily_trades)
        self.starting_capital = float(starting_capital)
        self.allow_overlap_open_ic = allow_overlap_open_ic

        # stale exit cooldown after forced exit
        self._stale_exit_cooldown_until: Optional[datetime] = None
        self.stale_exit_cooldown_seconds = int(stale_exit_cooldown_seconds)

        # entry cutoff
        self.no_new_entries_after = no_new_entries_after


        # ======================= APIs =============================
        try:
            self.live_api = KiteAPI(mode="live")
            log.info("LivePaperEngine: Initialized LIVE API")
        except Exception:
            log.exception("LivePaperEngine: LIVE API init failed — continuing in PAPER-only mode")
            self.live_api = None

        try:
            self.paper_api = KiteAPI(mode="paper")
            log.info("LivePaperEngine: Initialized PAPER API")
        except Exception:
            log.info("LivePaperEngine: PAPER API unavailable; using in-memory PaperBroker")
            self.paper_api = None

        self.broker = PaperBroker(capital=self.starting_capital)


        # ======================= Risk Engine =============================
        dd_raw = float(risk_daily_loss_limit)
        if dd_raw <= 1.0:
            max_dd_frac = dd_raw
        elif dd_raw <= 100.0:
            max_dd_frac = dd_raw / 100.0
        else:
            max_dd_frac = dd_raw / max(1.0, self.starting_capital)

        tr_raw = float(risk_max_trade_pct)
        if tr_raw <= 1.0:
            max_trade_frac = tr_raw
        else:
            max_trade_frac = tr_raw / 100.0

        self.risk = RiskEngine(max_trade_risk_pct=max_trade_frac, max_daily_drawdown_pct=max_dd_frac)
        self.risk.reset_day_if_needed(datetime.now().date(), self.broker.balance)


        # ======================= Signal Generator =============================
        self.signal_gen = SignalGenerator(
            symbol=self.symbol,
            lot_size=self.lot_size,
            width=self.width,
            size_aggressiveness=self.size_aggressiveness
        )


        # ======================= Exit Engine =============================
        cfg = ExitEngineConfig()
        cfg.hard_close_time = time(15, 20)
        cfg.disable_time_exit = True
        self.exit_engine = ExitEngine(cfg)


        # ======================= Runtime State =============================
        self.trades_today = 0
        self.daily_history: List[Dict[str, Any]] = []
        self.open_ic = None
        self.open_ic_entry_time: Optional[datetime] = None
        self.open_ic_replay_id: Optional[int] = None
        self.pnl_history: List[Dict[str, Any]] = []

        os.makedirs(MONITOR_PATH, exist_ok=True)


        # ======================= Trades DB =============================
        try:
            self._trades_db_path = TRADES_DB_FILE
            self._trades_db = sqlite3.connect(self._trades_db_path, check_same_thread=False)
            self._trades_db.row_factory = sqlite3.Row

            cur = self._trades_db.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    event TEXT,
                    symbol_summary TEXT,
                    ic_json TEXT,
                    pnl REAL,
                    duration_seconds REAL,
                    note TEXT
                );
                """
            )
            self._trades_db.commit()
            log.info("LivePaperEngine: trades DB ready at %s", self._trades_db_path)

        except Exception:
            log.exception("LivePaperEngine: failed to initialize trades DB")
            self._trades_db = None
            self._trades_db_path = None


        # ======================= LLM Selector =============================
        self.use_llm_selector = bool(kwargs.get("use_llm_selector", False))
        self._ollama_client = None
        try:
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            self._ollama_client = ollama.Client(host=ollama_url)
            log.info("LivePaperEngine: Ollama client initialized")
        except Exception:
            log.exception("LivePaperEngine: Failed to create Ollama client")
            self._ollama_client = None


        log.info(
            "LivePaperEngine initialized: symbol=%s lot=%s width=%s starting_capital=%.2f no_new_after=%s stale_cooldown_s=%s",
            self.symbol, self.lot_size, self.width, self.starting_capital,
            self.no_new_entries_after, self.stale_exit_cooldown_seconds
        )

        # ==============================================================
        # NOTE: old `_restore_open_position()` removed completely.
        # We now use only:  self._reconcile_state_on_startup()
        # ==============================================================
        try:
            self._auto_clean_monitor_files()
        except Exception:
            log.exception("Auto-clean on startup failed (non-fatal)")

        try:
            self._reconcile_state_on_startup()
        except Exception:
            log.exception("Reconciliation on startup failed (non-fatal)")            

    # ---------------- helpers -----------------
    def _sanitize(self, obj: Any) -> Any:
        import datetime as _dt
        if isinstance(obj, _dt.datetime):
            return obj.isoformat()
        if isinstance(obj, _dt.date):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        return obj

    def _read_manual_exit_flag(self) -> bool:
        try:
            if not os.path.exists(MANUAL_EXIT_FILE):
                return False
            with open(MANUAL_EXIT_FILE, "r") as f:
                data = json.load(f)
            return bool(data.get("force_exit", False))
        except Exception:
            return False

    def _clear_manual_exit_flag(self) -> None:
        try:
            if os.path.exists(MANUAL_EXIT_FILE):
                os.remove(MANUAL_EXIT_FILE)
        except Exception:
            pass

    # ------------------ safe JSON helpers ------------------
    def _safe_read_json(self, path: str) -> Optional[dict]:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            log.exception("Failed to read JSON: %s", path)
        return None

    def _safe_write_json(self, path: str, obj: dict) -> bool:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._sanitize(obj), f, indent=2, default=str)
            return True
        except Exception:
            log.exception("Failed to write JSON to %s", path)
            return False

    # ------------------ validation + auto-clean helpers ------------------
    def _is_valid_open_positions(self, op: Any) -> bool:
        """
        Heuristic to decide whether broker_state['open_positions'] is a real, useful list.
        Returns True only if it's a non-empty list of dict-like positions that contain a
        tradingsymbol or symbol key. Conservative: empty / malformed lists are False.
        """
        try:
            if not isinstance(op, list):
                return False
            if len(op) == 0:
                return False
            for p in op:
                if not isinstance(p, dict):
                    return False
                if not ("tradingsymbol" in p or "symbol" in p):
                    return False
            return True
        except Exception:
            return False

    def _auto_clean_monitor_files(self) -> None:
        """
        Remove or neutralize obviously-stale monitor files to avoid phantom open-IC in the UI.
        Conservative rules:
          - If current_position.json *looks* like an IC (contains IC-like keys), but
            paper_broker_state.json has no valid open_positions according to _is_valid_open_positions,
            then clear current_position.json and write a minimal paper_broker_state.json.
          - If paper_broker_state.json exists but its open_positions are malformed, normalize it to {"open_positions": []}.
        This runs on startup only and logs operations. It will not delete trades.db or records.jsonl.
        """
        try:
            cur = self._safe_read_json(CURRENT_POS_FILE) or {}
            broker = self._safe_read_json(PAPER_BROKER_STATE_FILE) or {}

            # Detect "IC-like" current_position
            ic_like_keys = {"short_put", "short_call", "short_put_sym", "short_call_sym", "legs", "open_ic_entry_time"}
            cur_is_ic_like = isinstance(cur, dict) and any(k in cur for k in ic_like_keys)

            broker_open = broker.get("open_positions")
            broker_valid = self._is_valid_open_positions(broker_open)

            # If current looks like IC but broker has no valid open positions -> clear cur + normalize broker
            if cur_is_ic_like and not broker_valid:
                log.warning("Auto-clean: detected IC-like current_position but invalid/missing broker open_positions. Purging stale files.")
                try:
                    # clear current_position
                    self._safe_write_json(CURRENT_POS_FILE, {})
                except Exception:
                    log.exception("Auto-clean failed to clear current_position.json")

                try:
                    # ensure broker state minimal and consistent
                    minimal = {
                        "open_positions": [],
                        "capital": float(getattr(self, "starting_capital", 0.0)),
                        "starting_balance": float(getattr(self, "starting_capital", 0.0)),
                        "pnl": 0.0,
                    }
                    self._safe_write_json(PAPER_BROKER_STATE_FILE, minimal)
                except Exception:
                    log.exception("Auto-clean failed to write minimal paper_broker_state.json")

                try:
                    # clear recovered position file as it's stale too
                    self._safe_write_json(RECOVERED_POS_FILE, {})
                except Exception:
                    pass

            # If broker open_positions exists but malformed -> normalize it (no purge of current_position here)
            if broker.get("open_positions") is not None and not self._is_valid_open_positions(broker.get("open_positions")):
                try:
                    log.info("Auto-clean: normalizing malformed paper_broker_state.open_positions -> []")
                    broker["open_positions"] = []
                    self._safe_write_json(PAPER_BROKER_STATE_FILE, broker)
                except Exception:
                    log.exception("Auto-clean failed to normalize paper_broker_state")
        except Exception:
            log.exception("Auto-clean monitor files failed")


    # ------------------ reconstruction helper ------------------
    def _reconstruct_open_positions_from_ic(self, ic: dict) -> dict:
        """
        Build a minimal paper_broker_state structure containing open_positions list
        from an ic dict. Each leg becomes an open_position entry.
        """
        try:
            lot = int(ic.get("lot_size", 25))
            size_agg = float(ic.get("size_aggressiveness", 1.0))
            qty_per_leg = int(round(lot * size_agg))
            legs = []

            def leg_block(sym_key, price_key, side_sell=True):
                sym = ic.get(sym_key)
                px = ic.get(price_key)
                if sym:
                    return {
                        "tradingsymbol": sym,
                        "qty": -qty_per_leg if side_sell else qty_per_leg,
                        "entry_price": float(px) if px is not None else None,
                    }
                return None

            for k, p, s in [
                ("short_put_sym", "short_put_price", True),
                ("long_put_sym", "long_put_price", False),
                ("short_call_sym", "short_call_price", True),
                ("long_call_sym", "long_call_price", False),
            ]:
                block = leg_block(k, p, s)
                if block:
                    legs.append(block)

            return {
                "open_positions": legs,
                "capital": float(ic.get("capital", self.starting_capital)),
                "starting_balance": float(ic.get("starting_balance", self.starting_capital)),
                "pnl": float(ic.get("entry_pnl", 0.0)),
                "recovered_from_ic": True,
                "recovered_at": datetime.utcnow().isoformat(),
                "source_ic": ic,
            }
        except Exception:
            log.exception("Failed to reconstruct broker state from IC")
            return {"open_positions": [], "recovered_from_ic": True}

    # ------------------ recover IC from trades.db ------------------
    def _query_latest_entered_ic_from_trades_db(self) -> Optional[dict]:
        try:
            if not self._trades_db_path or not os.path.exists(self._trades_db_path):
                return None

            conn = sqlite3.connect(self._trades_db_path)
            cur = conn.cursor()
            cur.execute("SELECT ic_json FROM trades WHERE ic_json IS NOT NULL ORDER BY ts DESC LIMIT 1")
            row = cur.fetchone()
            conn.close()

            if row and row[0]:
                try:
                    return json.loads(row[0])
                except Exception:
                    try:
                        return eval(row[0])
                    except Exception:
                        return None
        except Exception:
            log.exception("Failed to query trades.db for latest IC")
        return None

    # ------------------ startup reconciliation ------------------
    def _reconcile_state_on_startup(self) -> None:
        """
        Ensure CURRENT_POS and PAPER_BROKER_STATE files agree.
        If current_position shows an IC but broker_state has no open_positions:
        → reconstruct using trades.db or current_position.
        """
        try:
            cur = self._safe_read_json(CURRENT_POS_FILE) or {}
            broker = self._safe_read_json(PAPER_BROKER_STATE_FILE) or {}

            open_ic_present = False
            if isinstance(cur, dict) and (
                "short_put" in cur or "short_call" in cur or "short_put_sym" in cur or "short_call_sym" in cur
            ):
                open_ic_present = True

            broker_has_positions = bool(broker.get("open_positions"))

            if open_ic_present and not broker_has_positions:
                log.warning(
                    "State inconsistency: current_position has IC but broker_state missing open_positions. Recovering…"
                )

                ic = self._query_latest_entered_ic_from_trades_db()
                if not ic:
                    ic = cur if cur else None

                if ic:
                    broker_state = self._reconstruct_open_positions_from_ic(ic)
                    self._safe_write_json(PAPER_BROKER_STATE_FILE, broker_state)
                    self._safe_write_json(
                        RECOVERED_POS_FILE,
                        {
                            "recovered_at": datetime.utcnow().isoformat(),
                            "ic_source": "trades_db_or_current_pos",
                            "ic": ic,
                        },
                    )
                    log.info("Reconstructed broker_state and wrote recovered_position.json")
                else:
                    placeholder = {
                        "open_positions": [],
                        "recovered_from_ic": False,
                        "note": "ghost_position_no_ic",
                        "detected_at": datetime.utcnow().isoformat(),
                    }
                    self._safe_write_json(PAPER_BROKER_STATE_FILE, placeholder)
                    self._safe_write_json(
                        RECOVERED_POS_FILE,
                        {"recovered_at": datetime.utcnow().isoformat(), "ic_source": None},
                    )
                    log.warning("Could not find IC → wrote placeholder placeholder broker_state.")

        except Exception:
            log.exception("Error during state reconciliation on startup")

    # ---------------- SQLite insertion helper -----------------
    def _insert_trade_db(
        self,
        event: str,
        ic_obj=None,
        pnl: Optional[float] = None,
        duration_seconds: Optional[float] = None,
        note: Optional[str] = None,
    ) -> None:
        """Insert entry/exit record into DB + write labeled record to records.jsonl."""
        try:
            if self._trades_db is None:
                return

            payload = None
            summary = None

            if ic_obj is not None:
                # Serialize IC
                try:
                    payload = json.dumps(
                        ic_obj.as_dict() if hasattr(ic_obj, "as_dict") else ic_obj,
                        default=str,
                    )
                except Exception:
                    try:
                        payload = json.dumps(str(ic_obj), default=str)
                    except Exception:
                        payload = None

                # Build symbol summary
                try:
                    if hasattr(ic_obj, "legs"):
                        summary = ",".join(
                            [str(getattr(l, "symbol", l)) for l in getattr(ic_obj, "legs", [])][:4]
                        )
                    else:
                        summary = getattr(ic_obj, "symbol", None) or str(ic_obj)[:200]
                except Exception:
                    summary = str(ic_obj)[:200]

            ts = datetime.now().isoformat()
            cur = self._trades_db.cursor()
            cur.execute(
                """
                INSERT INTO trades (ts,event,symbol_summary,ic_json,pnl,duration_seconds,note)
                VALUES (?,?,?,?,?,?,?)
                """,
                (ts, event, summary, payload, pnl, duration_seconds, note),
            )
            self._trades_db.commit()

            # Write labeled entry/exit to dataset
            try:
                if isinstance(event, str) and event.lower().startswith("entry"):
                    try:
                        self._append_record_entry(ic_obj, replay_id=self.open_ic_replay_id)
                    except Exception:
                        log.exception("failed entry record write")

                elif isinstance(event, str) and event.lower().startswith("exit"):
                    try:
                        os.makedirs(MONITOR_PATH, exist_ok=True)
                        rec = {
                            "time": datetime.now().isoformat(),
                            "exit_pnl": float(pnl) if pnl is not None else None,
                            "exit_reason": note,
                            "duration_seconds": duration_seconds,
                            "replay_id": self.open_ic_replay_id,
                        }
                        try:
                            with open(LATEST_SNAPSHOT_FILE, "r") as f:
                                rec["exit_snapshot"] = json.load(f)
                        except Exception:
                            rec["exit_snapshot"] = None

                        with open(os.path.join(MONITOR_PATH, "records.jsonl"), "a") as f:
                            f.write(json.dumps(rec, default=str) + "\n")
                    except Exception:
                        log.exception("exit record write failed")

            except Exception:
                log.exception("records append guard failed")

        except Exception:
            log.exception("_insert_trade_db failed")


    # ------------------ Append entry record (IV/delta included) ------------------
    def _append_record_entry(self, ic_obj, replay_id=None):
        """
        Full entry labeling logic including best-effort IV/delta for each leg.
        Preserved from your original implementation.
        """
        try:
            os.makedirs(MONITOR_PATH, exist_ok=True)

            rec = {
                "time": datetime.now().isoformat(),
                "candidate": ic_obj.as_dict() if hasattr(ic_obj, "as_dict") else ic_obj,
                "replay_id": replay_id,
            }

            # attach snapshot
            try:
                with open(LATEST_SNAPSHOT_FILE, "r") as f:
                    rec["chain_snapshot"] = json.load(f)
            except Exception:
                rec["chain_snapshot"] = None

            # minimal engine config
            rec["engine_config"] = {
                "width": self.width,
                "lot_size": self.lot_size,
                "size_aggressiveness": self.size_aggressiveness,
            }

            # ===== Attempt IV/delta computation per leg =====
            legs = []
            cand = rec["candidate"]
            if isinstance(cand, dict) and cand.get("legs"):
                legs = cand["legs"]
            else:
                try:
                    eo = cand.get("entry_orders") if isinstance(cand, dict) else None
                    if eo is None and hasattr(ic_obj, "entry_orders"):
                        eo = ic_obj.entry_orders()
                    if eo:
                        legs = eo
                except Exception:
                    legs = []

            chain = rec.get("chain_snapshot") or []
            leg_greeks = []

            def find_market_price(symbol=None, strike=None, opt_type=None):
                try:
                    if not chain:
                        return None
                    for row in chain:
                        try:
                            r_sym = str(row.get("tradingsymbol") or row.get("symbol") or "")
                            if symbol and r_sym == str(symbol):
                                return row.get("ltp") or row.get("last_price")
                            rstrike = row.get("strike")
                            rtype = row.get("option_type")
                            if strike is not None and rstrike is not None:
                                if float(rstrike) == float(strike):
                                    if opt_type is None or (
                                        rtype is not None and str(rtype).upper().startswith(str(opt_type).upper())
                                    ):
                                        return row.get("ltp") or row.get("last_price")
                        except Exception:
                            pass
                    return None
                except Exception:
                    return None

            now = datetime.now()
            for lg in legs:
                try:
                    if isinstance(lg, dict):
                        sym = lg.get("symbol")
                        strike = lg.get("strike")
                        opt = lg.get("option_type")
                        price = lg.get("price") or lg.get("ltp")
                    else:
                        sym = getattr(lg, "symbol", None)
                        strike = getattr(lg, "strike", None)
                        opt = getattr(lg, "option_type", None)
                        price = getattr(lg, "price", None)

                    # fallback on symbol for strike detection
                    if strike is None and isinstance(sym, str):
                        parts = sym.split("_")
                        if parts[-1].replace(".", "", 1).isdigit():
                            strike = float(parts[-1])

                    # market price fallback
                    if price is None:
                        price = find_market_price(sym, strike, opt)

                    # find spot / expiry
                    expiry = None
                    spot = None
                    if chain:
                        root = chain[0]
                        spot = root.get("spot") or root.get("underlying_price")
                        expiry = lg.get("expiry") if isinstance(lg, dict) else getattr(lg, "expiry", None)
                    if spot is not None:
                        try:
                            spot = float(spot)
                        except Exception:
                            spot = None

                    # parse expiry to year fraction
                    T = None
                    if expiry:
                        try:
                            if isinstance(expiry, str):
                                try:
                                    e = datetime.fromisoformat(expiry)
                                except Exception:
                                    e = datetime.fromisoformat(expiry.split("T")[0])
                            elif isinstance(expiry, date):
                                e = datetime.combine(expiry, time())
                            else:
                                e = expiry
                            if e:
                                T = max(0.0, (e - now).total_seconds() / (365 * 24 * 3600.0))
                        except Exception:
                            T = None

                    # compute greeks
                    iv_val = None
                    delta_val = None

                    if price is None or spot is None or strike is None or T is None:
                        leg_greeks.append(
                            {
                                "symbol": sym,
                                "strike": strike,
                                "option_type": opt,
                                "iv": None,
                                "delta": None,
                                "market_price": price,
                            }
                        )
                        continue

                    # standardize option type
                    opt_type = None
                    if opt:
                        o = str(opt).upper()
                        if o.startswith("C"):
                            opt_type = "CE"
                        elif o.startswith("P"):
                            opt_type = "PE"
                    else:
                        if isinstance(sym, str) and "_CE_" in sym.upper():
                            opt_type = "CE"
                        elif isinstance(sym, str) and "_PE_" in sym.upper():
                            opt_type = "PE"

                    # IV attempt
                    r = 0.06
                    try:
                        iv_val = implied_volatility(spot, strike, T, r, opt_type, price)
                        if (
                            iv_val is None
                            or (isinstance(iv_val, float) and (iv_val != iv_val or iv_val <= 0 or iv_val > 5))
                        ):
                            iv_val = None
                    except Exception:
                        iv_val = None

                    # delta attempt
                    try:
                        if iv_val is not None:
                            try:
                                d1 = (log(spot / strike) + (r + 0.5 * iv_val**2) * T) / (iv_val * sqrt(T))
                                if opt_type == "CE":
                                    delta_val = float(norm.cdf(d1))
                                else:
                                    delta_val = float(norm.cdf(d1) - 1.0)
                            except Exception:
                                delta_val = None
                        else:
                            # finite differences fallback
                            eps = max(0.01, spot * 0.001)
                            p_up = bs_price(spot + eps, strike, T, r, iv_val or 0.2, opt_type)
                            p_dn = bs_price(spot - eps, strike, T, r, iv_val or 0.2, opt_type)
                            delta_val = (p_up - p_dn) / (2 * eps)
                    except Exception:
                        delta_val = None

                    leg_greeks.append(
                        {
                            "symbol": sym,
                            "strike": strike,
                            "option_type": opt_type,
                            "iv": iv_val,
                            "delta": delta_val,
                            "market_price": price,
                        }
                    )
                except Exception:
                    leg_greeks.append(
                        {"symbol": None, "strike": None, "option_type": None, "iv": None, "delta": None}
                    )

            rec["legs_greeks"] = leg_greeks

            with open(os.path.join(MONITOR_PATH, "records.jsonl"), "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

        except Exception:
            log.exception("Failed to append entry record")

    # ---------------- snapshot writing -----------------
    def _write_snapshot_files(self, chain_df: Optional[pd.DataFrame], mtm: float) -> None:
        try:
            # latest_snapshot.json
            if chain_df is not None and not chain_df.empty:
                snap = chain_df.to_dict(orient="records")
                with open(LATEST_SNAPSHOT_FILE, "w") as f:
                    json.dump(self._sanitize(snap), f, indent=2)

            # risk-state
            try:
                risk_state = self.risk.get_state() if hasattr(self.risk, "get_state") else {}
            except Exception:
                risk_state = {}

            with open(RISK_STATE_FILE, "w") as f:
                json.dump(self._sanitize(risk_state), f, indent=2)

            # broker_state
            broker_state = {
                "capital": float(self.broker.balance),
                "starting_balance": float(getattr(self.broker, "starting_balance", self.starting_capital)),
                "pnl": float(self.broker.balance - getattr(self.broker, "starting_balance", self.starting_capital)),
                "trades": self.daily_history,
                "open_ic_replay_id": int(self.open_ic_replay_id) if self.open_ic_replay_id else None,
                "open_ic_entry_time": self.open_ic_entry_time.isoformat() if self.open_ic_entry_time else None,
            }

            # include open positions
            try:
                if hasattr(self.broker, "open_positions"):
                    op = self.broker.open_positions
                    if isinstance(op, list):
                        broker_state["open_positions"] = op
                if "open_positions" not in broker_state:
                    if self.open_ic is not None:
                        try:
                            icd = self.open_ic.as_dict() if hasattr(self.open_ic, "as_dict") else self.open_ic
                            rec = self._reconstruct_open_positions_from_ic(icd)
                            broker_state["open_positions"] = rec.get("open_positions", [])
                        except Exception:
                            broker_state["open_positions"] = []
                    else:
                        cur = self._safe_read_json(CURRENT_POS_FILE) or {}
                        if cur:
                            rec = self._reconstruct_open_positions_from_ic(cur)
                            broker_state["open_positions"] = rec.get("open_positions", [])
                        else:
                            broker_state["open_positions"] = []
            except Exception:
                broker_state["open_positions"] = []

            with open(PAPER_BROKER_STATE_FILE, "w") as f:
                json.dump(self._sanitize(broker_state), f, indent=2)

            # pnl history append
            now_iso = datetime.now().isoformat()
            self.pnl_history.append({"time": now_iso, "pnl": broker_state["pnl"]})
            with open(PNL_HISTORY_FILE, "w") as f:
                json.dump(self._sanitize(self.pnl_history), f, indent=2)

            # current_position.json
            cur = {}
            if self.open_ic is not None:
                try:
                    cur = self.open_ic.as_dict() if hasattr(self.open_ic, "as_dict") else self.open_ic
                    if self.open_ic_entry_time:
                        cur["open_ic_entry_time"] = self.open_ic_entry_time.isoformat()
                except Exception:
                    cur = {"note": "open_ic present but as_dict failed"}
            else:
                try:
                    bs = self._safe_read_json(PAPER_BROKER_STATE_FILE) or {}
                    if bs.get("open_positions"):
                        cur = {"recovered_open_positions": bs["open_positions"]}
                except Exception:
                    cur = {}

            with open(CURRENT_POS_FILE, "w") as f:
                json.dump(self._sanitize(cur), f, indent=2)

        except Exception:
            log.exception("Failed to write snapshot files")

    # ---------------- trade risk helpers -----------------
    def _estimate_trade_risk(self, ic) -> float:
        try:
            if hasattr(ic, "max_loss"):
                ml = ic.max_loss() if callable(ic.max_loss) else ic.max_loss
                return float(ml)
        except Exception:
            pass
        try:
            return abs(float(getattr(ic, "entry_credit", 0.0))) * self.lot_size
        except Exception:
            return max(1.0, self.starting_capital * 0.001)

    def _pre_trade_risk_check(self, trade_risk: float) -> bool:
        self.risk.reset_day_if_needed(datetime.now().date(), self.broker.balance)

        ok, reason = self.risk.can_open_trade(self.broker.balance, trade_risk)
        if not ok:
            log.warning("RiskEngine blocked trade: %s", reason)
            return False

        if self.trades_today >= self.max_daily_trades:
            log.warning("Max daily trades reached")
            return False

        if self.open_ic is not None and not self.allow_overlap_open_ic:
            log.info("Already in an open IC, skipping new entry (allow_overlap_open_ic=False)")
            return False

        # stale exit cooldown
        if self._stale_exit_cooldown_until is not None:
            if datetime.now() < self._stale_exit_cooldown_until:
                log.info("Cooldown active until %s", self._stale_exit_cooldown_until)
                return False
            self._stale_exit_cooldown_until = None

        # time cutoff
        try:
            nowt = datetime.now().time()
            if self.no_new_entries_after and nowt >= self.no_new_entries_after:
                log.info("Entry cutoff reached (%s ≥ %s)", nowt, self.no_new_entries_after)
                return False
        except Exception:
            pass

        return True


    # ---------------- symbol validation -----------------
    def _validate_ic_orders(self, ic_obj, chain_df):
        """
        Check if the candidate's entry orders reference symbols present in chain_df.
        Returns (ok, bad_symbol).
        """
        try:
            orders = []
            try:
                orders = ic_obj.entry_orders() or []
            except Exception:
                if hasattr(ic_obj, "legs"):
                    for l in ic_obj.legs:
                        try:
                            sym = getattr(l, "symbol", None) or getattr(l, "tradingsymbol", None)
                            orders.append({"symbol": sym})
                        except Exception:
                            continue

            # build set of available symbols
            ts_set = set()
            if chain_df is not None and not chain_df.empty:
                if "tradingsymbol" in chain_df.columns:
                    ts_set = set(chain_df["tradingsymbol"].astype(str).tolist())
                elif "symbol" in chain_df.columns:
                    ts_set = set(chain_df["symbol"].astype(str).tolist())
                elif "strike" in chain_df.columns:
                    return True, None

            if not ts_set:
                return True, None

            for o in orders:
                sym = o.get("symbol") if isinstance(o, dict) else getattr(o, "symbol", None)
                if sym and str(sym) not in ts_set:
                    return False, str(sym)

            return True, None

        except Exception:
            log.exception("_validate_ic_orders error")
            return True, None

    # ---------------- rebuild candidate (nearest strikes) -----------------
    def _rebuild_candidate_with_nearest_strikes(self, candidate, chain_df: pd.DataFrame):
        """
        Attempt to rebuild a candidate using the nearest available strikes.
        """
        try:
            if candidate is None:
                return None
            if chain_df is None or chain_df.empty:
                return None

            strikes = available_strikes(chain_df)
            if not strikes:
                return None

            # extract legs
            legs = getattr(candidate, "legs", None)
            if not legs:
                try:
                    eo = candidate.entry_orders()
                    legs = []
                    for o in eo:
                        sym = o.get("symbol")
                        strike = None
                        if isinstance(sym, str) and sym.split("_")[-1].isdigit():
                            strike = float(sym.split("_")[-1])
                        legs.append({"symbol": sym, "strike": strike, **o})
                except Exception:
                    return None

            changed = False
            new_legs = []

            for leg in legs:
                if isinstance(leg, dict):
                    sym = leg.get("symbol")
                    strike_val = leg.get("strike")
                else:
                    sym = getattr(leg, "symbol", None)
                    strike_val = getattr(leg, "strike", None)

                if strike_val is None and isinstance(sym, str):
                    parts = sym.split("_")
                    if parts[-1].replace(".", "", 1).isdigit():
                        strike_val = float(parts[-1])

                if strike_val is None:
                    new_legs.append(leg)
                    continue

                best = closest_strike(float(strike_val), strikes)
                if best is None:
                    new_legs.append(leg)
                    continue

                # detect option type
                opt = None
                if isinstance(sym, str):
                    u = sym.upper()
                    if "_PE_" in u:
                        opt = "PE"
                    elif "_CE_" in u:
                        opt = "CE"

                root = self.symbol
                if isinstance(sym, str) and sym.startswith(root):
                    prefix = sym.split("_")[0]
                    root = prefix

                new_sym = f"{root}_{opt}_{int(best)}" if opt else f"{root}_{int(best)}"

                if isinstance(leg, dict):
                    new_leg = leg.copy()
                    new_leg["symbol"] = new_sym
                    new_leg["strike"] = best
                else:
                    new_leg = {"symbol": new_sym, "strike": best}

                new_legs.append(new_leg)

                if best != strike_val:
                    changed = True

            if not changed:
                return None

            # build new candidate object
            class SimpleCand:
                pass

            nc = SimpleCand()

            for att in dir(candidate):
                if att.startswith("_"):
                    continue
                if att in ("legs", "entry_orders"):
                    continue
                try:
                    val = getattr(candidate, att)
                    if callable(val):
                        continue
                    setattr(nc, att, val)
                except Exception:
                    continue

            nc.legs = new_legs

            def entry_orders_func():
                out = []
                for l in new_legs:
                    if isinstance(l, dict):
                        out.append(
                            {
                                "symbol": l.get("symbol"),
                                "qty": l.get("qty"),
                                "side": l.get("side"),
                            }
                        )
                    else:
                        out.append({"symbol": getattr(l, "symbol", None)})
                return out

            def as_dict_func():
                return {"legs": nc.legs, "summary": getattr(candidate, "summary", "rebuilt")}

            nc.entry_orders = entry_orders_func
            nc.as_dict = as_dict_func

            try:
                if hasattr(candidate, "mark_to_market"):
                    nc.mark_to_market = candidate.mark_to_market
                if hasattr(candidate, "exit_pnl"):
                    nc.exit_pnl = candidate.exit_pnl
            except Exception:
                pass

            return nc
        except Exception:
            log.exception("_rebuild_candidate_with_nearest_strikes failed")
            return None

    # ------------------- main run_once loop -------------------
    def run_once(self) -> None:
        chain_df = pd.DataFrame()

        # ===== Daily reset =====
        try:
            if hasattr(self.signal_gen, "reset_daily"):
                self.signal_gen.reset_daily()

            today = date.today()
            if getattr(self, "_engine_last_date", None) != today:
                log.info(
                    "LivePaperEngine: date rolled %s -> %s. resetting counters",
                    getattr(self, "_engine_last_date", None),
                    today,
                )
                self._engine_last_date = today
                self.trades_today = 0
                self.daily_history = []
        except Exception:
            log.exception("Daily reset failed")

        # ===== 1) Fetch LIVE chain =====
        if self.live_api is not None:
            try:
                chain_df = full_option_chain(self.live_api, self.symbol)
            except Exception:
                log.exception("Failed to fetch live chain")
                chain_df = pd.DataFrame()

        # ===== 2) Fallback to PAPER chain =====
        if (chain_df is None or chain_df.empty) and self.paper_api is not None:
            try:
                chain_df = full_option_chain(self.paper_api, self.symbol)
                log.warning("Using PAPER chain (live chain empty)")
            except Exception:
                chain_df = pd.DataFrame()

        # ===== Compute spot =====
        spot = 0.0
        try:
            if chain_df is not None and not chain_df.empty and "spot" in chain_df.columns:
                raw = chain_df["spot"].iloc[0]
                spot = float(raw) if raw is not None else 0.0
        except Exception:
            spot = 0.0

        # ===== Option C: stale restored IC auto-exit =====
        try:
            if self.open_ic is not None and self.open_ic_entry_time is not None:
                now_dt = datetime.now()
                age = now_dt - self.open_ic_entry_time

                if age.total_seconds() > 24 * 3600 or self.open_ic_entry_time.date() < now_dt.date():
                    log.info("Stale restored IC — auto-exiting... age=%s", age)

                    try:
                        realized = (
                            self.open_ic.exit_pnl(chain_df)
                            if hasattr(self.open_ic, "exit_pnl")
                            else 0.0
                        )
                    except Exception:
                        realized = 0.0
                        log.exception("exit_pnl failed for stale IC")

                    # broker apply
                    try:
                        self.broker.realize_pnl(realized)
                    except Exception:
                        log.exception("broker.realize_pnl failed")

                    # ML linking
                    try:
                        metadata = {"source": "live_engine", "exit_reason": "stale_restore_auto_exit"}
                        if self.open_ic_replay_id is not None:
                            metadata["replay_id"] = int(self.open_ic_replay_id)
                            metadata["ml_replay_id"] = int(self.open_ic_replay_id)

                        try:
                            icd = (
                                self.open_ic.as_dict()
                                if hasattr(self.open_ic, "as_dict")
                                else self.open_ic
                            )
                        except Exception:
                            icd = {"repr": str(self.open_ic)}

                        entry_snap = {}
                        if self.open_ic_entry_time:
                            entry_snap["entry_time"] = self.open_ic_entry_time.isoformat()

                        exit_snap = {}
                        try:
                            if chain_df is not None and not chain_df.empty:
                                exit_snap["spot"] = float(chain_df["spot"].iloc[0])
                        except Exception:
                            pass

                        record_trade(
                            ic=icd,
                            entry_snapshot=entry_snap,
                            exit_snapshot=exit_snap,
                            realized_pnl=float(realized),
                            exit_reason="stale_restore_auto_exit",
                            metadata=metadata,
                        )
                    except Exception:
                        log.exception("record_trade failed for stale exit")

                    # DB insert
                    try:
                        dur = (
                            (datetime.now() - self.open_ic_entry_time).total_seconds()
                            if self.open_ic_entry_time
                            else None
                        )
                        self._insert_trade_db(
                            "exit",
                            ic_obj=self.open_ic,
                            pnl=float(realized),
                            duration_seconds=dur,
                            note="stale_restore_auto_exit",
                        )
                    except Exception:
                        log.exception("DB write failed for stale exit")

                    # history + clear
                    try:
                        self.daily_history.append(
                            {
                                "time": datetime.now().isoformat(),
                                "ic": self.open_ic.as_dict()
                                if hasattr(self.open_ic, "as_dict")
                                else {},
                                "mode": "stale_restore_auto_exit",
                                "balance": self.broker.balance,
                            }
                        )
                    except Exception:
                        pass

                    self.open_ic = None
                    self.open_ic_replay_id = None
                    self.open_ic_entry_time = None

                    # clear current_position.json
                    try:
                        with open(CURRENT_POS_FILE, "w") as f:
                            json.dump({}, f, indent=2)
                    except Exception:
                        log.exception("Failed to clear CURRENT_POS after stale exit")

                    # cooldown
                    try:
                        self._stale_exit_cooldown_until = datetime.now() + timedelta(
                            seconds=self.stale_exit_cooldown_seconds
                        )
                        log.info("Cooldown set → %s", self._stale_exit_cooldown_until)
                    except Exception:
                        self._stale_exit_cooldown_until = None

                    # unfreeze SG
                    try:
                        if hasattr(self.signal_gen, "unfreeze"):
                            self.signal_gen.unfreeze()
                    except Exception:
                        log.exception("SG unfreeze failed after stale exit")
        except Exception:
            log.exception("Stale exit handling failed")


        # ===== Candidate generation (SG or LLM) =====
        candidate = None

        if self._stale_exit_cooldown_until is not None:
            if datetime.now() < self._stale_exit_cooldown_until:
                log.info("Cooldown active → skip candidate building")
                try:
                    if hasattr(self.signal_gen, "unfreeze"):
                        self.signal_gen.unfreeze()
                except Exception:
                    pass
                try:
                    mtm = 0.0
                    if self.open_ic is not None and hasattr(self.open_ic, "mark_to_market"):
                        mtm = self.open_ic.mark_to_market(chain_df)
                    self._write_snapshot_files(chain_df, mtm)
                except Exception:
                    log.exception("snapshot failed in cooldown skip")
                return

        # ===== Try LLM selector =====
        try:
            if self.use_llm_selector and self._ollama_client is not None and self.open_ic is None:
                try:
                    snapshot = {
                        "symbol": self.symbol,
                        "spot": spot,
                        "expiry": None,
                        "tte_days": None,
                        "chain": chain_df.to_dict(orient="records") if not chain_df.empty else [],
                    }

                    # --- LLM execution callback ---
                    def execute_trade_callback(payload: dict):
                        try:
                            cand = payload.get("candidate")
                            from src.trading.iron_condor_builder import IronCondor

                            ic_obj = None
                            try:
                                if hasattr(cand, "as_dict"):
                                    cdict = cand.as_dict()
                                elif isinstance(cand, dict):
                                    cdict = cand
                                else:
                                    cdict = None

                                if cdict:
                                    required = [
                                        "short_put",
                                        "long_put",
                                        "short_call",
                                        "long_call",
                                    ]
                                    if all(k in cdict for k in required):
                                        ic_obj = IronCondor(
                                            symbol=self.symbol,
                                            expiry=cdict.get("expiry"),
                                            short_put=float(cdict.get("short_put")),
                                            long_put=float(cdict.get("long_put")),
                                            short_call=float(cdict.get("short_call")),
                                            long_call=float(cdict.get("long_call")),
                                            short_put_price=float(
                                                cdict.get("short_put_price", 0.0) or 0.0
                                            ),
                                            long_put_price=float(
                                                cdict.get("long_put_price", 0.0) or 0.0
                                            ),
                                            short_call_price=float(
                                                cdict.get("short_call_price", 0.0) or 0.0
                                            ),
                                            long_call_price=float(
                                                cdict.get("long_call_price", 0.0) or 0.0
                                            ),
                                            lot_size=self.lot_size,
                                            size_aggressiveness=float(
                                                cdict.get("size_aggressiveness", self.size_aggressiveness)
                                            ),
                                        )
                            except Exception:
                                ic_obj = cand

                            if ic_obj is None:
                                log.warning("LLM callback: invalid IC")
                                return {"placed": False}

                            try:
                                tr = self._estimate_trade_risk(ic_obj)
                                if not self._pre_trade_risk_check(tr):
                                    log.warning("LLM callback: risk-blocked")
                                    return {"placed": False}
                            except Exception:
                                log.exception("risk est failed in LLM callback")

                            try:
                                entry_info = self.broker.open_ic(ic_obj)
                                return {"placed": True, "trade_ref": getattr(entry_info, "id", None)}
                            except Exception:
                                log.exception("broker.open_ic failed in LLM")
                                return {"placed": False}

                        except Exception:
                            log.exception("LLM execute callback failed")
                            return {"placed": False}

                    # --- LLM risk-check callback ---
                    def risk_check_callback(payload: dict) -> bool:
                        try:
                            cand = payload.get("candidate")
                            try:
                                if hasattr(cand, "max_loss"):
                                    ra = cand.max_loss()
                                else:
                                    ra = abs(float(getattr(cand, "entry_credit", 0.0))) * self.lot_size
                            except Exception:
                                ra = self.starting_capital * 0.001
                            return self._pre_trade_risk_check(ra)
                        except Exception:
                            log.exception("risk_check_callback error")
                            return False

                    # --- run LLM selector ---
                    res = run_selection_once(
                        snapshot=snapshot,
                        ollama_client=self._ollama_client,
                        execute_trade_callback=execute_trade_callback,
                        risk_check_callback=risk_check_callback,
                        replay_db_path=os.path.join(MONITOR_PATH, "ml_replay.db"),
                        n_candidates=5,
                    )

                    log.info("LLM selector result: %s", res)

                    # capture replay_id
                    try:
                        rid = (
                            res.get("replay_id")
                            or res.get("execute_result", {})
                            .get("meta", {})
                            .get("replay_id")
                        )
                        self.open_ic_replay_id = int(rid) if rid is not None else None
                    except Exception:
                        self.open_ic_replay_id = None

                    # if placed, update open_ic
                    if isinstance(res, dict) and res.get("status") == "placed":
                        try:
                            if hasattr(self.broker, "last_open_ic"):
                                self.open_ic = self.broker.last_open_ic
                        except Exception:
                            self.open_ic = None

                        self.open_ic_entry_time = datetime.now()
                        self.trades_today += 1

                        try:
                            if hasattr(self.signal_gen, "freeze"):
                                self.signal_gen.freeze()
                        except Exception:
                            pass

                        try:
                            self._insert_trade_db("entry", ic_obj=self.open_ic, note="llm_entry")
                        except Exception:
                            log.exception("DB insert failed for llm entry")

                        candidate = self.open_ic

                    else:
                        candidate = None

                except Exception:
                    log.exception("LLM selector path failed")
                    candidate = None
        except Exception:
            log.exception("LLM section unexpected failure")
            candidate = None

        # ===== If no LLM candidate → use SignalGenerator =====
        if candidate is None and self.open_ic is None:
            try:
                try:
                    cands = self.signal_gen.generate(chain_df=chain_df, spot=spot)
                except TypeError:
                    cands = self.signal_gen.generate(chain_df, spot)

                if cands:
                    candidate = cands[0]
                    log.info("SignalGenerator produced a candidate")
            except Exception:
                log.exception("SignalGenerator failed")
                candidate = None

        # ===== Validate candidate =====
        if candidate is not None and self.open_ic is None:
            try:
                ok, bad = self._validate_ic_orders(candidate, chain_df)
                if not ok:
                    log.warning("Candidate has invalid symbol %s — attempting rebuild...", bad)
                    rebuilt = self._rebuild_candidate_with_nearest_strikes(candidate, chain_df)
                    if rebuilt is None:
                        log.warning("Rebuild failed → unfreezing SG")
                        try:
                            if hasattr(self.signal_gen, "unfreeze"):
                                self.signal_gen.unfreeze()
                        except Exception:
                            pass
                        candidate = None
                    else:
                        ok2, bad2 = self._validate_ic_orders(rebuilt, chain_df)
                        if not ok2:
                            log.warning("Rebuilt candidate still invalid (%s)", bad2)
                            try:
                                if hasattr(self.signal_gen, "unfreeze"):
                                    self.signal_gen.unfreeze()
                            except Exception:
                                pass
                            candidate = None
                        else:
                            candidate = rebuilt
            except Exception:
                log.exception("candidate validation failed")
                try:
                    if hasattr(self.signal_gen, "unfreeze"):
                        self.signal_gen.unfreeze()
                except Exception:
                    pass
                candidate = None

        # ===== Entry processing =====
        if candidate is not None and self.open_ic is None:
            try:
                # risk check
                tr = self._estimate_trade_risk(candidate)
                if self._pre_trade_risk_check(tr):
                    log.info("Opening IC")

                    try:
                        rs = self.broker.open_ic(candidate)
                        self.open_ic = getattr(self.broker, "last_open_ic", candidate)
                    except Exception:
                        log.exception("broker.open_ic failed for SG candidate")
                        self.open_ic = candidate

                    self.open_ic_entry_time = datetime.now()
                    self.trades_today += 1

                    try:
                        if hasattr(self.signal_gen, "freeze"):
                            self.signal_gen.freeze()
                    except Exception:
                        pass

                    try:
                        self._insert_trade_db("entry", ic_obj=candidate, note="sg_entry")
                    except Exception:
                        log.exception("DB insert failed for SG entry")

            except Exception:
                log.exception("Entry handling failed")


        # ===== Exit conditions =====
        if self.open_ic is not None:
            try:
                if hasattr(self.open_ic, "exit_signal"):
                    if self.open_ic.exit_signal(chain_df):
                        log.info("Exit signal triggered")
                        realized = 0.0
                        try:
                            if hasattr(self.open_ic, "exit_pnl"):
                                realized = self.open_ic.exit_pnl(chain_df)
                        except Exception:
                            realized = 0.0

                        try:
                            self.broker.realize_pnl(realized)
                        except Exception:
                            log.exception("broker.realize_pnl failed")

                        try:
                            metadata = {"source": "live_engine", "exit_reason": "exit_signal"}
                            if self.open_ic_replay_id is not None:
                                metadata["replay_id"] = int(self.open_ic_replay_id)
                                metadata["ml_replay_id"] = int(self.open_ic_replay_id)

                            icd = (
                                self.open_ic.as_dict()
                                if hasattr(self.open_ic, "as_dict")
                                else self.open_ic
                            )

                            entry_snap = {"entry_time": self.open_ic_entry_time.isoformat()} if self.open_ic_entry_time else {}
                            exit_snap = {}
                            if chain_df is not None and not chain_df.empty:
                                try:
                                    exit_snap["spot"] = float(chain_df["spot"].iloc[0])
                                except Exception:
                                    pass

                            record_trade(
                                ic=icd,
                                entry_snapshot=entry_snap,
                                exit_snapshot=exit_snap,
                                realized_pnl=float(realized),
                                exit_reason="exit_signal",
                                metadata=metadata,
                            )
                        except Exception:
                            log.exception("record_trade failed on exit signal")

                        # db
                        try:
                            dur = (
                                (datetime.now() - self.open_ic_entry_time).total_seconds()
                                if self.open_ic_entry_time
                                else None
                            )
                            self._insert_trade_db(
                                "exit",
                                ic_obj=self.open_ic,
                                pnl=float(realized),
                                duration_seconds=dur,
                                note="signal_exit",
                            )
                        except Exception:
                            log.exception("DB insert failed for exit signal")

                        try:
                            self.daily_history.append(
                                {
                                    "time": datetime.now().isoformat(),
                                    "ic": self.open_ic.as_dict()
                                    if hasattr(self.open_ic, "as_dict")
                                    else {},
                                    "mode": "exit_signal",
                                    "balance": self.broker.balance,
                                }
                            )
                        except Exception:
                            pass

                        self.open_ic = None
                        self.open_ic_replay_id = None
                        self.open_ic_entry_time = None

                        try:
                            with open(CURRENT_POS_FILE, "w") as f:
                                json.dump({}, f, indent=2)
                        except Exception:
                            pass

                        try:
                            if hasattr(self.signal_gen, "unfreeze"):
                                self.signal_gen.unfreeze()
                        except Exception:
                            pass

            except Exception:
                log.exception("Exit signal block failed")

        # ===== Manual exit =====
        try:
            if self.open_ic is not None and self._read_manual_exit_flag():
                log.info("Manual exit triggered")

                realized = 0.0
                try:
                    if hasattr(self.open_ic, "exit_pnl"):
                        realized = self.open_ic.exit_pnl(chain_df)
                except Exception:
                    realized = 0.0

                try:
                    self.broker.realize_pnl(realized)
                except Exception:
                    pass

                try:
                    metadata = {"source": "manual", "exit_reason": "manual_exit"}
                    if self.open_ic_replay_id is not None:
                        metadata["replay_id"] = int(self.open_ic_replay_id)
                        metadata["ml_replay_id"] = int(self.open_ic_replay_id)

                    icd = (
                        self.open_ic.as_dict()
                        if hasattr(self.open_ic, "as_dict")
                        else self.open_ic
                    )

                    entry_snap = {"entry_time": self.open_ic_entry_time.isoformat()} if self.open_ic_entry_time else {}
                    exit_snap = {}
                    if chain_df is not None and not chain_df.empty:
                        try:
                            exit_snap["spot"] = float(chain_df["spot"].iloc[0])
                        except Exception:
                            pass

                    record_trade(
                        ic=icd,
                        entry_snapshot=entry_snap,
                        exit_snapshot=exit_snap,
                        realized_pnl=float(realized),
                        exit_reason="manual_exit",
                        metadata=metadata,
                    )
                except Exception:
                    log.exception("record_trade failed on manual exit")

                # db
                try:
                    dur = (
                        (datetime.now() - self.open_ic_entry_time).total_seconds()
                        if self.open_ic_entry_time
                        else None
                    )

                    self._insert_trade_db(
                        "exit",
                        ic_obj=self.open_ic,
                        pnl=float(realized),
                        duration_seconds=dur,
                        note="manual_exit",
                    )
                except Exception:
                    log.exception("DB insert failed for manual exit")

                try:
                    self.daily_history.append(
                        {
                            "time": datetime.now().isoformat(),
                            "ic": self.open_ic.as_dict()
                            if hasattr(self.open_ic, "as_dict")
                            else {},
                            "mode": "manual_exit",
                            "balance": self.broker.balance,
                        }
                    )
                except Exception:
                    pass

                self.open_ic = None
                self.open_ic_replay_id = None
                self.open_ic_entry_time = None

                try:
                    with open(CURRENT_POS_FILE, "w") as f:
                        json.dump({}, f, indent=2)
                except Exception:
                    pass

                self._clear_manual_exit_flag()

                try:
                    if hasattr(self.signal_gen, "unfreeze"):
                        self.signal_gen.unfreeze()
                except Exception:
                    pass

        except Exception:
            log.exception("Manual exit block failed")

        # ===== Update snapshots + monitoring =====
        try:
            mtm = 0.0
            if self.open_ic is not None and hasattr(self.open_ic, "mark_to_market"):
                mtm = self.open_ic.mark_to_market(chain_df)
            self._write_snapshot_files(chain_df, mtm)
        except Exception:
            log.exception("snapshot writing after exits failed")

    # ========================== END OF FILE ==========================

