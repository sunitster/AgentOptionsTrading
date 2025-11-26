# src/live/live_paper_engine.py
"""
Ultra-stable LivePaperEngine (Option B).

Key improvements:
- Uses full_option_chain() (which guarantees underlying row).
- Writes monitor files used by dashboard:
    latest_snapshot.json (full chain list),
    paper_broker_state.json,
    pnl_history.json,
    risk_state.json,
    current_position.json  <-- NEW: ensures dashboard shows IC details/greeks
- Robust entry/exit bookkeeping and explicit current_position writes on open/close.
"""

import os
import json
import logging
import time as _time
from datetime import datetime, time, date
from typing import Any, Dict, List, Optional

import pandas as pd

from src.live.kite_api import KiteAPI
from src.live.paper_broker import PaperBroker
from src.engine.risk_engine import RiskEngine
from src.engine.exit_engine import ExitEngine, ExitEngineConfig
from src.trading.signal_generator import SignalGenerator
from src.live.kite_data import full_option_chain  # uses patched kite_data above

log = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

# Monitor folder used by dashboard
MONITOR_PATH = os.path.join("models", "llm_trades")
MANUAL_EXIT_FILE = os.path.join(MONITOR_PATH, "manual_exit.json")
LATEST_SNAPSHOT_FILE = os.path.join(MONITOR_PATH, "latest_snapshot.json")
RISK_STATE_FILE = os.path.join(MONITOR_PATH, "risk_state.json")
PAPER_BROKER_STATE_FILE = os.path.join(MONITOR_PATH, "paper_broker_state.json")
PNL_HISTORY_FILE = os.path.join(MONITOR_PATH, "pnl_history.json")
CURRENT_POS_FILE = os.path.join(MONITOR_PATH, "current_position.json")

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
        risk_daily_loss_limit: float = 20000.0,
        risk_max_trade_pct: float = 0.02,
        max_daily_trades: int = 3,
        dashboard: bool = True,
        **kwargs: Any,
    ) -> None:
        self.symbol = symbol
        self.lot_size = int(lot_size)
        self.width = int(width)
        self.minute_interval = int(minute_interval)
        self.size_aggressiveness = float(size_aggressiveness)
        self.max_daily_trades = int(max_daily_trades)

        self.starting_capital = float(starting_capital)

        # --- APIs ---
        try:
            self.live_api = KiteAPI(mode="live")
            log.info("LivePaperEngine: Initialized LIVE API")
        except Exception:
            log.exception("LivePaperEngine: LIVE API init failed — continuing in PAPER-only snapshot mode")
            self.live_api = None

        try:
            self.paper_api = KiteAPI(mode="paper")
            log.info("LivePaperEngine: Initialized PAPER API")
        except Exception:
            log.info("LivePaperEngine: PAPER API unavailable; using in-memory PaperBroker")
            self.paper_api = None

        self.broker = PaperBroker(capital=self.starting_capital)

        # --- Risk engine ---
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

        # Signal generator
        self.signal_gen = SignalGenerator(
            symbol=self.symbol,
            lot_size=self.lot_size,
            width=self.width,
            size_aggressiveness=self.size_aggressiveness
        )


        cfg = ExitEngineConfig()
        cfg.hard_close_time = time(15, 20)
        cfg.disable_time_exit = True
        self.exit_engine = ExitEngine(cfg)

        # runtime state
        self.trades_today: int = 0
        self.daily_history: List[Dict[str, Any]] = []
        self.open_ic = None
        self.open_ic_entry_time = None
        self.pnl_history: List[Dict[str, Any]] = []

        os.makedirs(MONITOR_PATH, exist_ok=True)

        log.info(
            "LivePaperEngine initialized: symbol=%s lot=%s width=%s starting_capital=%.2f",
            self.symbol,
            self.lot_size,
            self.width,
            self.starting_capital,
        )

    # ---------------- helpers -----------------
    def _sanitize(self, obj: Any) -> Any:
        import datetime
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, datetime.date):
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

    # ---------------- monitor writes -----------------
    def _write_snapshot_files(self, chain_df: Optional[pd.DataFrame], mtm: float) -> None:
        try:
            # latest_snapshot.json from LIVE chain
            if chain_df is not None and not chain_df.empty:
                snapshot = chain_df.to_dict(orient="records")
                with open(LATEST_SNAPSHOT_FILE, "w") as f:
                    json.dump(self._sanitize(snapshot), f, indent=2)

            # risk state
            risk_state = {}
            try:
                if hasattr(self.risk, "get_state"):
                    risk_state = self.risk.get_state()
            except Exception:
                risk_state = {}
            with open(RISK_STATE_FILE, "w") as f:
                json.dump(self._sanitize(risk_state), f, indent=2)

            # paper broker state
            broker_state = {
                "capital": float(getattr(self.broker, "balance", 0.0)),
                "starting_balance": float(getattr(self.broker, "starting_balance", self.starting_capital)),
                "pnl": float(getattr(self.broker, "balance", 0.0) - getattr(self.broker, "starting_balance", self.starting_capital)),
                "trades": self.daily_history,
            }
            with open(PAPER_BROKER_STATE_FILE, "w") as f:
                json.dump(self._sanitize(broker_state), f, indent=2)

            # pnl history append
            now_iso = datetime.now().isoformat()
            self.pnl_history.append({"time": now_iso, "pnl": broker_state["pnl"]})
            with open(PNL_HISTORY_FILE, "w") as f:
                json.dump(self._sanitize(self.pnl_history), f, indent=2)

            # current_position.json (dashboard depends on this)
            cur = {}
            if self.open_ic is not None and hasattr(self.open_ic, "as_dict"):
                try:
                    cur = self.open_ic.as_dict()
                except Exception:
                    cur = {"note": "open_ic present but as_dict failed"}
            else:
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

    def _pre_trade_risk_check(self, trade_risk_amount: float) -> bool:
        self.risk.reset_day_if_needed(datetime.now().date(), getattr(self.broker, "balance", self.starting_capital))

        allowed, reason = self.risk.can_open_trade(getattr(self.broker, "balance", self.starting_capital), trade_risk_amount)
        if not allowed:
            log.warning("RiskEngine blocked trade: %s", reason)
            return False

        if self.trades_today >= self.max_daily_trades:
            log.warning("Max daily trades reached: %d", self.max_daily_trades)
            return False

        if self.open_ic is not None:
            log.info("Already in an open IC, skipping new entry")
            return False

        return True

    # ---------------- main loop -----------------
    def run_once(self) -> None:
        chain_df = pd.DataFrame()

        # 1) Prefer LIVE API for market data
        if self.live_api is not None:
            try:
                chain_df = full_option_chain(self.live_api, self.symbol)
            except Exception:
                log.exception("Failed to fetch live chain")
                chain_df = pd.DataFrame()

        # 2) fallback to paper_api
        if (chain_df is None or chain_df.empty) and self.paper_api is not None:
            try:
                chain_df = full_option_chain(self.paper_api, self.symbol)
                log.warning("Using PAPER chain for snapshot — live chain empty")
            except Exception:
                chain_df = pd.DataFrame()

        # compute spot (safe)
        spot = 0.0
        try:
            if chain_df is not None and not chain_df.empty and "spot" in chain_df.columns:
                # underlying is expected at row 0
                spot = float(chain_df["spot"].iloc[0]) if chain_df["spot"].iloc[0] is not None else 0.0
        except Exception:
            spot = 0.0

        # 3) Build candidate using signal generator
        candidate = None
        try:
            candidates = self.signal_gen.generate(chain_df=chain_df, spot=spot)
            if not candidates:
                log.debug("SignalGenerator: no candidates built this tick")
            else:
                candidate = candidates[0]
                log.info("SignalGenerator: candidate built")
        except Exception:
            log.exception("SignalGenerator failed")
            candidate = None

        # 4) Enter trade (paper) if candidate and checks pass
        if candidate is not None and self.open_ic is None:
            try:
                ic_obj = candidate
                trade_risk = self._estimate_trade_risk(ic_obj)
                if self._pre_trade_risk_check(trade_risk):
                    entry_info = self.broker.open_ic(ic_obj)
                    self.open_ic = ic_obj
                    self.open_ic_entry_time = datetime.now()
                    self.trades_today += 1
                    self.daily_history.append({"time": datetime.now().isoformat(), "ic": ic_obj.as_dict() if hasattr(ic_obj, "as_dict") else {}, "mode": "live-paper-entry", "balance": getattr(self.broker, "balance", 0.0)})
                    log.info("Entered paper IC: %s", getattr(ic_obj, "as_dict", lambda: str(ic_obj))())
                    # write current position immediately for dashboard
                    try:
                        with open(CURRENT_POS_FILE, "w") as f:
                            json.dump(self._sanitize(ic_obj.as_dict() if hasattr(ic_obj, "as_dict") else {}), f, indent=2)
                    except Exception:
                        log.exception("Failed to write current_position after entry")
                    # --- ensure signal generator is frozen after an entry (defensive) ---
                    try:
                        if hasattr(self.signal_gen, "freeze"):
                            self.signal_gen.freeze()
                    except Exception:
                        pass
            except Exception:
                log.exception("Failed to enter paper trade")

        # 5) Manage exits for open position
        mtm = 0.0
        if self.open_ic is not None:
            try:
                mtm = self.open_ic.mark_to_market(chain_df)

                if self._read_manual_exit_flag():
                    realized = self.open_ic.exit_pnl(chain_df)
                    self.broker.realize_pnl(realized)
                    self.daily_history.append({"time": datetime.now().isoformat(), "ic": self.open_ic.as_dict() if hasattr(self.open_ic, "as_dict") else {}, "mode": "live-paper-exit-manual", "balance": getattr(self.broker, "balance", 0.0)})
                    self.open_ic = None
                    self._clear_manual_exit_flag()
                    # clear current_position file
                    try:
                        with open(CURRENT_POS_FILE, "w") as f:
                            json.dump({}, f, indent=2)
                    except Exception:
                        log.exception("Failed to clear current_position after manual exit")
                    # Unfreeze signal generator now that position closed
                    try:
                        if hasattr(self.signal_gen, "unfreeze"):
                            self.signal_gen.unfreeze()
                    except Exception:
                        pass
                else:
                    expiry_date = None
                    try:
                        if hasattr(self.open_ic, "expiry"):
                            e = getattr(self.open_ic, "expiry")
                            if isinstance(e, str):
                                expiry_date = datetime.fromisoformat(e).date()
                            elif isinstance(e, date):
                                expiry_date = e
                    except Exception:
                        expiry_date = None

                    if expiry_date is None and chain_df is not None and not chain_df.empty and "expiry" in chain_df.columns:
                        try:
                            uniq = chain_df["expiry"].dropna().unique().tolist()
                            if len(uniq) >= 1:
                                expiry_date = datetime.fromisoformat(str(uniq[0])).date()
                        except Exception:
                            expiry_date = None

                    now_dt = datetime.now()
                    today_date = now_dt.date()

                    # expiry-day hard close
                    if expiry_date is not None and today_date == expiry_date and now_dt.time() >= self.exit_engine.config.hard_close_time:
                        realized = self.open_ic.exit_pnl(chain_df)
                        self.broker.realize_pnl(realized)
                        self.daily_history.append({"time": now_dt.isoformat(), "ic": self.open_ic.as_dict() if hasattr(self.open_ic, "as_dict") else {}, "mode": "live-paper-exit-time", "balance": getattr(self.broker, "balance", 0.0)})
                        self.open_ic = None
                        try:
                            with open(CURRENT_POS_FILE, "w") as f:
                                json.dump({}, f, indent=2)
                        except Exception:
                            log.exception("Failed to clear current_position after expiry exit")
                        # Unfreeze signal generator now that position closed
                        try:
                            if hasattr(self.signal_gen, "unfreeze"):
                                self.signal_gen.unfreeze()
                        except Exception:
                            pass
                    else:
                        # exit engine scan
                        try:
                            actions = self.exit_engine.scan_and_exit(self.open_ic, chain_df, now_dt, mtm)
                            if actions:
                                realized = self.open_ic.exit_pnl(chain_df)
                                self.broker.realize_pnl(realized)
                                self.daily_history.append({"time": now_dt.isoformat(), "ic": self.open_ic.as_dict() if hasattr(self.open_ic, "as_dict") else {}, "mode": "live-paper-exit", "balance": getattr(self.broker, "balance", 0.0)})
                                self.open_ic = None
                                try:
                                    with open(CURRENT_POS_FILE, "w") as f:
                                        json.dump({}, f, indent=2)
                                except Exception:
                                    log.exception("Failed to clear current_position after exit_engine exit")
                                # Unfreeze signal generator now that position closed
                                try:
                                    if hasattr(self.signal_gen, "unfreeze"):
                                        self.signal_gen.unfreeze()
                                except Exception:
                                    pass
                        except Exception:
                            log.exception("ExitEngine scan failed")

            except Exception:
                log.exception("Error while managing open position")

        # 6) Write snapshot & monitor files
        try:
            self._write_snapshot_files(chain_df, mtm)
        except Exception:
            log.exception("Failed to write monitor state")

    # convenience runner
    def run_forever(self, sleep_seconds: float = 1.0) -> None:
        try:
            while True:
                self.run_once()
                _time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            log.info("LivePaperEngine stopped by user")
        except Exception:
            log.exception("LivePaperEngine encountered an error")
