"""
LivePaperTradingEngine (patched)

- Adds time-of-day / weekday / IV gating in _generate_candidates_from_snapshot
- Adds limit-order fill simulation & partial fills for PAPER mode in _place_ic_orders

Save as: src/live/paper_trading_engine.py
"""
from __future__ import annotations
import logging
import time
import json
import os
import random
from datetime import datetime, time as dtime, timezone
from typing import Optional, Dict, Any, List

# Local imports (adjust paths if your modules are in different places)
try:
    from src.live.kite_api import KiteAPI
except Exception:
    from live.kite_api import KiteAPI  # type: ignore

try:
    from src.execution.execution_engine import ExecutionEngine
except Exception:
    from execution_engine import ExecutionEngine  # type: ignore

# Iron Condor builder and signal generator
try:
    from src.trading.iron_condor_builder import build_iron_condor, IronCondor  # type: ignore
except Exception:
    from trading.iron_condor_builder import build_iron_condor, IronCondor  # type: ignore

try:
    from src.live.signal_generator import SignalGenerator
except Exception:
    SignalGenerator = None  # optional

LOG = logging.getLogger("LivePaperEngine")
stream_h = logging.StreamHandler()
stream_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
if not LOG.handlers:
    LOG.addHandler(stream_h)
LOG.setLevel(logging.INFO)


def _iso_ts_now_utc():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(timespec="seconds")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class LivePaperTradingEngine:
    def __init__(
        self,
        symbol: str = "NIFTY",
        lot_size: int = 25,
        width: int = 150,
        config_path: str = "config/config.yaml",
        cadence_s: int = 15,
        market_open: dtime = dtime(hour=9, minute=15),
        market_close: dtime = dtime(hour=15, minute=30),
        verbose: bool = True,
        # NEW gating params:
        enter_within_minutes_after_open: Optional[int] = None,  # e.g., 30 -> only enter during first 30 minutes after open
        allowed_weekdays: Optional[List[int]] = None,  # 0=Monday .. 6=Sunday ; e.g., [0,1,2,3,4]
        min_iv: Optional[float] = None,  # require IV >= this to enter
        # PAPER fill simulation params:
        paper_fill_max_delay_s: int = 3,  # max simulated seconds delay to "fill"
        paper_partial_fill_enabled: bool = True,
    ):
        self.symbol = symbol
        self.lot_size = lot_size
        self.width = width
        self.cadence = max(1, int(cadence_s))
        self.config_path = config_path
        self.market_open = market_open
        self.market_close = market_close
        self.verbose = verbose

        # gating rules
        self.enter_within_minutes_after_open = enter_within_minutes_after_open
        self.allowed_weekdays = allowed_weekdays or [0, 1, 2, 3, 4]  # default: weekdays
        self.min_iv = min_iv

        # paper fill simulation parameters
        self.paper_fill_max_delay_s = paper_fill_max_delay_s
        self.paper_partial_fill_enabled = paper_partial_fill_enabled

        # components
        self.kite = KiteAPI(mode="auto")  # KiteAPI auto-reads config/config.yaml
        try:
            self.engine = ExecutionEngine(starting_capital=1_000_000.0, paper=True)
        except TypeError:
            try:
                self.engine = ExecutionEngine()  # type: ignore
            except Exception:
                LOG.warning("Could not instantiate ExecutionEngine by signature; using simple fallback paper engine.")
                self.engine = None  # type: ignore

        # trades staged this run
        self.trades: List[Dict[str, Any]] = []

        # output folder
        self.out_dir = os.path.join("models", "llm_trades")
        _ensure_dir(self.out_dir)

    def _now_local_time(self):
        return datetime.now().time()

    def _in_market_hours(self) -> bool:
        t = self._now_local_time()
        return (t >= self.market_open) and (t <= self.market_close)

    def _snapshot(self):
        """Get the live snapshot from KiteAPI (or PAPER fallback provided inside)."""
        try:
            snap = self.kite.get_live_snapshot(self.symbol)
            return snap
        except Exception as e:
            LOG.error("Snapshot retrieval failed: %s", e, exc_info=False)
            return self.kite._paper_snapshot(self.symbol)

    def _extract_iv_from_snapshot(self, snapshot) -> Optional[float]:
        """Try common columns for IV in the snapshot (works for many chain formats)."""
        try:
            import pandas as pd
            if isinstance(snapshot, pd.DataFrame):
                for col in ("iv", "imp_vol", "implied_vol", "impliedVolatility"):
                    if col in snapshot.columns:
                        vals = pd.to_numeric(snapshot[col], errors="coerce").dropna()
                        if not vals.empty:
                            return float(vals.mean())
            elif isinstance(snapshot, dict):
                for k in ("iv", "implied_vol", "impliedVolatility"):
                    if k in snapshot:
                        v = snapshot.get(k)
                        try:
                            return float(v)
                        except Exception:
                            pass
            elif isinstance(snapshot, list) and len(snapshot) > 0 and isinstance(snapshot[0], dict):
                for k in ("iv", "implied_vol", "impliedVolatility"):
                    if k in snapshot[0]:
                        try:
                            return float(snapshot[0][k])
                        except Exception:
                            pass
        except Exception:
            LOG.debug("IV extraction failed.", exc_info=True)
        return None

    def _minutes_after_open(self) -> Optional[int]:
        """If market_open is known, compute minutes elapsed since market open; else None."""
        try:
            now = datetime.now()
            open_dt = datetime.combine(now.date(), self.market_open)
            delta = now - open_dt
            return int(delta.total_seconds() // 60)
        except Exception:
            return None

    def _generate_candidates_from_snapshot(self, snapshot) -> List[Dict[str, Any]]:
        """
        Convert snapshot -> candidate iron-condor plans with gating:
         - only during first X minutes after market open (if configured)
         - only on allowed weekdays (if configured)
         - only if IV >= min_iv (if configured)
        """
        candidates: List[Dict[str, Any]] = []

        # weekday gating
        weekday = datetime.now().weekday()
        if self.allowed_weekdays is not None and weekday not in self.allowed_weekdays:
            LOG.info("Weekday gating: today=%s not in allowed_weekdays=%s -> skipping.", weekday, self.allowed_weekdays)
            return []

        # minutes-after-open gating
        if self.enter_within_minutes_after_open is not None:
            mins = self._minutes_after_open()
            if mins is None:
                LOG.debug("Could not compute minutes after open; skipping minutes gating.")
            else:
                if mins < 0 or mins > int(self.enter_within_minutes_after_open):
                    LOG.info(
                        "Time gating: minutes_after_open=%s not within allowed(%s) -> skipping",
                        mins,
                        self.enter_within_minutes_after_open,
                    )
                    return []

        # IV gating
        if self.min_iv is not None:
            iv = self._extract_iv_from_snapshot(snapshot)
            if iv is None:
                LOG.debug("IV not found in snapshot; cannot apply min_iv gating.")
            else:
                if iv < float(self.min_iv):
                    LOG.info("IV gating: iv=%s < min_iv=%s -> skipping", iv, self.min_iv)
                    return []

        # Primary generator (user-provided SignalGenerator)
        try:
            if SignalGenerator:
                sg = SignalGenerator(symbol=self.symbol, lot_size=self.lot_size, width=self.width)
                cand = sg.generate(snapshot)
                if cand:
                    candidates.extend(cand if isinstance(cand, list) else [cand])
                    return candidates
        except Exception:
            LOG.debug("SignalGenerator failed; falling back to builder.", exc_info=True)

        # Fallback builder (needs spot or chain)
        try:
            import pandas as pd
            if isinstance(snapshot, pd.DataFrame):
                if "spot" in snapshot.columns:
                    spot = float(snapshot["spot"].iloc[0])
                elif "ltp" in snapshot.columns and "strike" in snapshot.columns:
                    # estimate spot as the ltp of ATM-like strike
                    spot = float(snapshot["spot"].iloc[0]) if "spot" in snapshot.columns else float(snapshot["ltp"].median())
                elif "strike" in snapshot.columns:
                    spot = float(snapshot["strike"].median())
                else:
                    spot = 0.0
                ic = build_iron_condor(snapshot, spot=spot, width=self.width, lot_size=self.lot_size, lots_per_leg=1)
                if ic:
                    candidates.append({"ic": ic, "spot": spot})
            else:
                # snapshot dict/list handling
                spot = None
                if isinstance(snapshot, dict):
                    spot = snapshot.get("spot")
                elif isinstance(snapshot, list) and len(snapshot) > 0 and isinstance(snapshot[0], dict):
                    spot = snapshot[0].get("spot")
                if spot is not None:
                    ic = build_iron_condor(snapshot, spot=spot, width=self.width, lot_size=self.lot_size, lots_per_leg=1)
                    if ic:
                        candidates.append({"ic": ic, "spot": spot})
        except Exception:
            LOG.exception("Failed building fallback IC from snapshot.")

        return candidates

    def _simulate_limit_fill(self, leg: Dict[str, Any], order_qty: int, limit_price: float) -> Dict[str, Any]:
        """
        Simple PAPER-mode limit order fill simulator.
        Inputs:
            leg: dict with at least 'ltp' (last traded price) or we fallback to 'best_bid'/'best_ask'
            order_qty: quantity requested
            limit_price: requested limit price (for BUY: max price, for SELL: min price)
        Returns:
            { 'filled_qty': int, 'avg_fill_price': float, 'fills': [ {qty, price, time} ] }
        Behavior:
            - If BUY and limit >= ltp -> high fill probability (near-full)
            - If BUY and limit < ltp -> lower probability, depending on gap
            - If SELL symmetrical.
            - Allows partial fill if enabled.
        """
        ltp = None
        # Leg may be object or dict
        if hasattr(leg, "ltp"):
            try:
                ltp = float(getattr(leg, "ltp"))
            except Exception:
                ltp = None
        elif isinstance(leg, dict):
            for k in ("ltp", "last_price", "mark"):
                if k in leg:
                    try:
                        ltp = float(leg[k])
                        break
                    except Exception:
                        pass
            if ltp is None:
                # try best_bid/best_ask midpoint
                if "best_bid" in leg and "best_ask" in leg:
                    try:
                        ltp = (float(leg["best_bid"]) + float(leg["best_ask"])) / 2.0
                    except Exception:
                        ltp = None

        if ltp is None:
            ltp = 0.0

        side = leg.side if hasattr(leg, "side") else leg.get("side", "BUY")
        side = side.upper()

        fills = []
        remaining = int(order_qty)
        # compute gap metric
        gap = 0.0
        if side == "BUY":
            gap = (limit_price - ltp) / (ltp + 1e-9)
        else:
            gap = (ltp - limit_price) / (ltp + 1e-9)

        # base fill probability and max partial fraction
        if gap >= 0:
            base_fill_prob = 0.95  # favorable limit
            max_fill_frac = 1.0
        else:
            # less favorable: reduce probability based on gap magnitude
            base_fill_prob = max(0.05, 0.6 + 0.4 * gap)  # gap negative -> reduce
            max_fill_frac = max(0.1, 0.8 + 0.2 * gap)

        # if limit is way away, further reduce
        if abs(gap) > 0.5:
            base_fill_prob *= 0.3
            max_fill_frac *= 0.5

        # simulate fill in potentially multiple chops
        # we'll produce between 1 and min(4, remaining) chops
        chops = min(4, max(1, remaining))
        for _ in range(chops):
            if remaining <= 0:
                break
            # decide whether this chop fills any
            if random.random() <= base_fill_prob:
                # fill fraction
                frac = 0.6 + 0.4 * random.random() if self.paper_partial_fill_enabled else 1.0
                frac = min(frac, max_fill_frac)
                qty = max(1, int(round(order_qty * frac)))
                qty = min(qty, remaining)
                # price: simulate as midway between limit and ltp with some noise
                noise = (random.random() - 0.5) * 0.02 * max(1.0, ltp)
                price = round((limit_price + ltp) / 2.0 + noise, 2)
                fills.append({"qty": qty, "price": price, "time": _iso_ts_now_utc()})
                remaining -= qty
            else:
                # no fill this chop (sleep small stochastic delay)
                pass

        filled_qty = order_qty - remaining
        if filled_qty == 0 and random.random() < 0.02:
            # tiny chance of a micro-fill (1 unit)
            filled_qty = 1
            fills.append({"qty": 1, "price": round(limit_price, 2), "time": _iso_ts_now_utc()})

        avg_price = None
        if fills:
            total_v = sum([f["qty"] * f["price"] for f in fills])
            total_q = sum([f["qty"] for f in fills])
            avg_price = round(total_v / total_q, 2) if total_q > 0 else None
        else:
            avg_price = None

        # simulate slight delay
        delay = random.random() * float(self.paper_fill_max_delay_s)
        time.sleep(min(0.1, delay))  # tiny sleep so run_once doesn't get stuck

        return {"filled_qty": filled_qty, "avg_fill_price": avg_price, "fills": fills}

    def _place_ic_orders(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place the 4 legs using kite or PAPER fallback. Supports limit-order simulation for paper mode.
        Candidate expected to have 'ic' key with IronCondor object (ic.legs iterable).
        """
        ic: IronCondor = candidate.get("ic")
        if ic is None:
            LOG.warning("No IC structure in candidate, skipping.")
            return {}

        ordered_ids = []
        fills_summary = []
        timestamp = datetime.utcnow().isoformat(timespec="seconds")

        for leg in ic.legs:
            # build order payload
            order_payload = {
                "exchange": "NFO",
                "tradingsymbol": getattr(leg, "tradingsymbol", leg.get("tradingsymbol") if isinstance(leg, dict) else None),
                "transaction_type": getattr(leg, "side", leg.get("side") if isinstance(leg, dict) else "BUY"),
                "quantity": int(getattr(leg, "lots", leg.get("lots", 1)) * self.lot_size),
                # allow order_type/limit to be available on leg.meta - default to MARKET
                "order_type": getattr(leg, "order_type", leg.get("order_type", "MARKET")),
                "limit_price": getattr(leg, "limit", leg.get("limit", None)),
                "product": "MIS",
            }

            if self.kite.mode == "live":
                # LIVE -> place real order; kite may still reject; handle exceptions
                try:
                    resp = self.kite.kite.place_order(
                        variety=self.kite.kite.VARIETY_REGULAR,
                        exchange=order_payload["exchange"],
                        tradingsymbol=order_payload["tradingsymbol"],
                        transaction_type=order_payload["transaction_type"],
                        quantity=order_payload["quantity"],
                        order_type=self.kite.kite.ORDER_TYPE_MARKET if order_payload["order_type"] == "MARKET" else self.kite.kite.ORDER_TYPE_LIMIT,
                        price=order_payload.get("limit_price", None),
                        product=self.kite.kite.PRODUCT_MIS,
                    )
                    ordered_ids.append(resp.get("order_id") if isinstance(resp, dict) else str(resp))
                    # NOTE: Kite returns order_id; you can query order status later
                    fills_summary.append({"leg": order_payload["tradingsymbol"], "status": "submitted_live", "kite_resp": resp})
                    LOG.info("Placed LIVE leg %s %s -> %s", order_payload["tradingsymbol"], order_payload["transaction_type"], ordered_ids[-1])
                except Exception as e:
                    LOG.error("Live order failed -> paper fallback: %s", e, exc_info=False)
                    # fallback to paper simulation below
                    sim = self._simulate_limit_fill(leg.__dict__ if hasattr(leg, "__dict__") else dict(leg), order_payload["quantity"], order_payload.get("limit_price") or 0.0)
                    paper_id = f"paper-{int(time.time()*1000)}"
                    ordered_ids.append(paper_id)
                    fills_summary.append({"leg": order_payload["tradingsymbol"], "status": "paper_simulated", "sim": sim})
            else:
                # PAPER mode -> simulate limit or market fills
                if order_payload["order_type"] == "MARKET" or not order_payload.get("limit_price"):
                    # market -> fill near LTP fully (simulate small slippage)
                    ltp = None
                    if hasattr(leg, "ltp"):
                        try:
                            ltp = float(getattr(leg, "ltp"))
                        except Exception:
                            ltp = None
                    elif isinstance(leg, dict):
                        ltp = float(leg.get("ltp") or leg.get("last_price") or 0.0)
                    if ltp is None:
                        ltp = 0.0
                    fill_price = round(ltp * (1.0 + (random.random() - 0.5) * 0.001), 2)
                    filled_qty = order_payload["quantity"]
                    sim = {"filled_qty": filled_qty, "avg_fill_price": fill_price, "fills": [{"qty": filled_qty, "price": fill_price, "time": _iso_ts_now_utc()}]}
                    paper_id = f"paper-{int(time.time()*1000)}"
                    ordered_ids.append(paper_id)
                    fills_summary.append({"leg": order_payload["tradingsymbol"], "status": "paper_market_fill", "sim": sim})
                else:
                    # LIMIT order simulate partial/limit behavior
                    limit_price = float(order_payload["limit_price"])
                    sim = self._simulate_limit_fill(leg.__dict__ if hasattr(leg, "__dict__") else dict(leg), order_payload["quantity"], limit_price)
                    paper_id = f"paper-{int(time.time()*1000)}"
                    ordered_ids.append(paper_id)
                    fills_summary.append({"leg": order_payload["tradingsymbol"], "status": "paper_limit_sim", "sim": sim})

        # compute net premium for filled legs (if any fills)
        net_pnl_entry = None
        try:
            total_side = 0.0
            total_notional = 0.0
            total_qty = 0
            for item in fills_summary:
                sim = item.get("sim")
                if not sim:
                    continue
                filled = sim.get("filled_qty", 0)
                avgp = sim.get("avg_fill_price")
                if avgp is None:
                    continue
                # get leg side to know sign
                # find leg object by tradingsymbol
                leg_obj = next((l for l in ic.legs if getattr(l, "tradingsymbol", getattr(l, "tradingsymbol", None)) == item["leg"]), None)
                sign = 1.0
                if leg_obj:
                    s = getattr(leg_obj, "side", None) or (leg_obj.get("side") if isinstance(leg_obj, dict) else None)
                    if s and s.upper().startswith("SELL"):
                        sign = 1.0
                    else:
                        sign = -1.0
                total_side += sign * filled * avgp
                total_notional += filled * avgp
                total_qty += filled
            if total_qty > 0:
                net_pnl_entry = round(total_side, 2)
        except Exception:
            net_pnl_entry = None

        record = {
            "time": _iso_ts_now_utc(),
            "symbol": self.symbol,
            "candidate": {
                "meta": ic.meta if hasattr(ic, "meta") else {},
                "legs": [getattr(l, "__dict__", dict(l)) for l in ic.legs],
            },
            "order_ids": ordered_ids,
            "fills": fills_summary,
            "entry_net_notional": net_pnl_entry,
            "status": "open",
        }
        self.trades.append(record)
        return record

    def _save_trades_snapshot(self):
        fname = os.path.join(self.out_dir, f"paper_trades_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
        try:
            with open(fname, "w") as f:
                json.dump({"runs": self.trades, "generated_at": _iso_ts_now_utc()}, f, indent=2, default=str)
            LOG.info("Paper trading day complete, summary saved: %s", fname)
        except Exception:
            LOG.exception("Failed to save trades to disk.")

    # ----------------------------
    # Public operations
    # ----------------------------
    def run_once(self) -> Dict[str, Any]:
        """Single snapshot evaluation and attempt to open ICs immediately."""
        LOG.info("LivePaperTradingEngine.run_once: symbol=%s lot=%d width=%d", self.symbol, self.lot_size, self.width)
        snapshot = self._snapshot()
        candidates = self._generate_candidates_from_snapshot(snapshot)
        LOG.info("Found %d candidate(s)", len(candidates))
        results = []
        for cand in candidates:
            try:
                rec = self._place_ic_orders(cand)
                results.append(rec)
            except Exception:
                LOG.exception("Failed to place IC orders for candidate.")
        self._save_trades_snapshot()
        return {"n_candidates": len(candidates), "records": results}

    def run_loop(self, run_until: Optional[datetime] = None, max_trades: Optional[int] = None):
        """
        Blocking loop. Samples snapshot every self.cadence seconds when within market hours.
        run_until: optional datetime to stop earlier.
        max_trades: optional cap on number of ICs opened this session.
        """
        LOG.info("Starting run_loop for %s — cadence=%ss", self.symbol, self.cadence)
        try:
            while True:
                if run_until and datetime.utcnow() >= run_until:
                    LOG.info("run_until reached, stopping loop.")
                    break
                if max_trades and len(self.trades) >= int(max_trades):
                    LOG.info("max_trades reached (%s), stopping loop.", max_trades)
                    break

                if not self._in_market_hours():
                    LOG.info("Outside market hours (%s-%s). Sleeping 60s.", self.market_open, self.market_close)
                    time.sleep(60)
                    continue

                snapshot = self._snapshot()
                candidates = self._generate_candidates_from_snapshot(snapshot)

                if candidates:
                    LOG.info("Candidate(s) discovered: %d", len(candidates))
                    for cand in candidates:
                        if max_trades and len(self.trades) >= int(max_trades):
                            LOG.info("Reached max_trades while placing orders.")
                            break
                        self._place_ic_orders(cand)

                time.sleep(self.cadence)
        except KeyboardInterrupt:
            LOG.info("Interrupted by user.")
        finally:
            self._save_trades_snapshot()
            LOG.info("Run loop finished. %d trades in this session.", len(self.trades))
        return {"n_trades": len(self.trades), "trades": self.trades}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(prog="live_paper_engine")
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--lot", type=int, default=25)
    p.add_argument("--width", type=int, default=150)
    p.add_argument("--once", action="store_true", help="Run a single evaluation and exit")
    p.add_argument("--cadence", type=int, default=15, help="Seconds between checks when in market hours")
    p.add_argument("--enter_mins", type=int, default=None, help="Only enter within X minutes after market open")
    p.add_argument("--min_iv", type=float, default=None, help="Minimum IV required to enter")
    args = p.parse_args()

    engine = LivePaperTradingEngine(
        symbol=args.symbol,
        lot_size=args.lot,
        width=args.width,
        cadence_s=args.cadence,
        enter_within_minutes_after_open=args.enter_mins,
        min_iv=args.min_iv,
    )
    if args.once:
        res = engine.run_once()
        print("run_once result:", res)
    else:
        engine.run_loop()
