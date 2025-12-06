# src/live/paper_broker.py
"""
PaperBroker (replaceable)

Compatibility:
- Keeps legacy APIs: place_order(), exit_order(), realize_pnl()
- Keeps multi-leg IC APIs: open_ic(), close_ic(ic_id, chain_df), close_ic_by_obj(...)
- Adds persistent state helpers: save_state_to_disk() and load_state_from_disk()
- Maintains self.open_positions (list of leg dicts) and self.last_open_ic (ICRecord)

Persisted state path:
  models/llm_trades/paper_broker_state.json
"""
from __future__ import annotations
import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List

MONITOR_DIR = os.path.join("models", "llm_trades")
PAPER_BROKER_STATE_FILE = os.path.join(MONITOR_DIR, "paper_broker_state.json")


class ICRecord:
    """
    Lightweight wrapper returned by open_ic.
    Provides attribute access (.id, .ic, .legs, .entry_time) and to_dict().
    """
    def __init__(self, ic_id: str, ic_obj: Any, legs: List[str], entry_time: datetime):
        self.id = ic_id
        self.ic = ic_obj
        self.legs = legs
        self.entry_time = entry_time

    def to_dict(self) -> Dict[str, Any]:
        try:
            icd = self.ic.as_dict() if hasattr(self.ic, "as_dict") else str(self.ic)
        except Exception:
            icd = str(self.ic)
        return {
            "ic_id": self.id,
            "ic": icd,
            "legs": list(self.legs),
            "entry_time": self.entry_time.isoformat() if isinstance(self.entry_time, datetime) else str(self.entry_time),
        }

    def __repr__(self):
        return f"<ICRecord id={self.id} legs={len(self.legs)} entry_time={self.entry_time}>"


class PaperBroker:
    """
    Simple in-memory simulated broker for live-paper execution with persistent state.
    """
    def __init__(self, capital: float = 100000.0):
        self.starting_balance = float(capital)
        self.balance = float(capital)
        self.capital = self.balance  # legacy alias

        # positions stores both single-leg orders and wrapper ICs
        # keyed by uuid: value is dict describing the position
        self.positions: Dict[str, Dict[str, Any]] = {}

        # history of closed positions
        self.history: List[Dict[str, Any]] = []

        # open_positions: list of dicts {tradingsymbol, qty, entry_price, entry_time}
        # kept for dashboard consumption and persisted to disk
        self.open_positions: List[Dict[str, Any]] = []

        # last opened ICRecord (object returned by open_ic) for quick engine pickup
        self.last_open_ic: Optional[ICRecord] = None

        # ensure monitor dir exists
        try:
            os.makedirs(MONITOR_DIR, exist_ok=True)
        except Exception:
            pass

        # try to load persisted broker state if available (non-fatal)
        try:
            self.load_state_from_disk()
        except Exception:
            # ignore errors; start fresh
            pass

    # ---------------- legacy single-order entry ----------------
    def place_order(self, order: dict) -> str:
        """
        Legacy single-leg order simulation.
        Returns generated order id.
        """
        oid = str(uuid.uuid4())
        symbol = order.get("symbol")
        qty = order.get("qty")
        side = order.get("side")
        price = order.get("price")

        self.positions[oid] = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "entry_price": price,
            "entry_time": datetime.now().isoformat(),
        }

        # update open_positions representation (for dashboard)
        try:
            entry = {
                "tradingsymbol": symbol,
                "qty": qty,
                "side": side,
                "entry_price": price,
                "entry_time": datetime.now().isoformat(),
                "position_id": oid,
            }
            self.open_positions.append(entry)
            self.save_state_to_disk()
        except Exception:
            pass

        return oid

    # ---------------- realize PnL ----------------
    def realize_pnl(self, pnl: float) -> float:
        """
        Apply realized PnL to broker balance.
        """
        self.balance += float(pnl)
        self.capital = self.balance
        # persist small ledger change
        try:
            self.save_state_to_disk()
        except Exception:
            pass
        return float(pnl)

    # ---------------- open multi-leg IC ----------------
    def open_ic(self, ic_obj: Any) -> ICRecord:
        """
        Opens all legs of an Iron Condor and returns an ICRecord.

        Expected ic_obj API:
            - ic_obj.entry_orders() -> list of dicts {symbol, qty, side, price}
            - ic_obj.as_dict() -> serializable dict for history
        """
        ic_id = str(uuid.uuid4())
        orders = []
        try:
            orders = ic_obj.entry_orders() or []
        except Exception:
            try:
                orders = getattr(ic_obj, "entry_orders", []) or []
            except Exception:
                orders = []

        leg_ids: List[str] = []
        created_legs: List[Dict[str, Any]] = []
        entry_time = datetime.now().isoformat()
        for od in orders:
            try:
                symbol = od.get("symbol") if isinstance(od, dict) else getattr(od, "symbol", None)
                qty = od.get("qty") if isinstance(od, dict) else getattr(od, "qty", None)
                side = od.get("side") if isinstance(od, dict) else getattr(od, "side", None)
                price = od.get("price") if isinstance(od, dict) else getattr(od, "price", None)
            except Exception:
                symbol = None; qty = None; side = None; price = None

            leg_id = str(uuid.uuid4())
            self.positions[leg_id] = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "entry_price": price,
                "entry_time": entry_time,
            }
            created_legs.append({
                "tradingsymbol": symbol,
                "qty": qty,
                "side": side,
                "entry_price": price,
                "entry_time": entry_time,
                "position_id": leg_id,
            })
            leg_ids.append(leg_id)

        # Wrapper IC entry
        self.positions[ic_id] = {
            "type": "IRON_CONDOR",
            "ic": ic_obj,
            "legs": leg_ids,
            "entry_time": entry_time,
        }

        ic_record = ICRecord(ic_id, ic_obj, leg_ids, datetime.fromisoformat(entry_time))
        self.last_open_ic = ic_record

        # update open_positions list for dashboard consumption (persist)
        try:
            # remove any identical symbols if present (defensive)
            for leg in created_legs:
                self.open_positions.append(leg)
            self.save_state_to_disk()
        except Exception:
            pass

        return ic_record

    # ---------------- helper: get IC record ----------------
    def get_ic_record(self, ic_id: str) -> Optional[ICRecord]:
        pos = self.positions.get(ic_id)
        if pos is None:
            return None
        if pos.get("type") != "IRON_CONDOR":
            return None
        return ICRecord(ic_id, pos.get("ic"), pos.get("legs", []), datetime.fromisoformat(pos.get("entry_time")) if pos.get("entry_time") else datetime.now())

    # ---------------- close IC by id ----------------
    def close_ic(self, ic_id: str, chain_df: Any) -> float:
        """
        Close Iron Condor wrapper and realize PnL.
        Returns realized pnl (float).
        """
        pos = self.positions.pop(ic_id, None)

        if pos is None or pos.get("type") != "IRON_CONDOR":
            return 0.0

        ic_obj = pos["ic"]

        # Compute realized pnl via IC API if available
        try:
            realized = float(ic_obj.exit_pnl(chain_df))
        except Exception:
            try:
                realized = float(getattr(ic_obj, "approx_pnl", 0.0))
            except Exception:
                realized = 0.0

        # Apply to balance
        self.balance += realized
        self.capital = self.balance

        # Remove leg positions from positions store and open_positions list
        for leg_id in pos.get("legs", []):
            try:
                leg = self.positions.pop(leg_id, None)
                # also remove from open_positions persistent list by matching position_id or symbol
                if leg is not None:
                    # attempt to remove entry by position_id
                    try:
                        self.open_positions = [op for op in self.open_positions if op.get("position_id") != leg_id]
                    except Exception:
                        # fallback: remove by symbol+qty
                        try:
                            self.open_positions = [op for op in self.open_positions if not (op.get("tradingsymbol") == leg.get("symbol") and op.get("qty") == leg.get("qty"))]
                        except Exception:
                            pass
            except Exception:
                pass

        # store history
        try:
            ic_snapshot = ic_obj.as_dict() if hasattr(ic_obj, "as_dict") else str(ic_obj)
        except Exception:
            ic_snapshot = str(ic_obj)

        self.history.append({
            "ic": ic_snapshot,
            "pnl": realized,
            "exit_time": datetime.now().isoformat(),
            "legs": pos.get("legs", []),
        })

        # clear last_open_ic if it references this IC
        try:
            if self.last_open_ic and self.last_open_ic.id == ic_id:
                self.last_open_ic = None
        except Exception:
            pass

        # persist updated state
        try:
            self.save_state_to_disk()
        except Exception:
            pass

        return float(realized)

    # ---------------- close IC by object convenience ----------------
    def close_ic_by_obj(self, ic_obj_or_record: Any, chain_df: Any) -> float:
        """
        Close by ICRecord or wrapper-like object. If original ic_obj passed, find wrapper.
        """
        try:
            if isinstance(ic_obj_or_record, ICRecord):
                return self.close_ic(ic_obj_or_record.id, chain_df)
            if hasattr(ic_obj_or_record, "id"):
                return self.close_ic(getattr(ic_obj_or_record, "id"), chain_df)
            # else match by identity of ic object
            for k, v in list(self.positions.items()):
                if v.get("type") == "IRON_CONDOR" and v.get("ic") is ic_obj_or_record:
                    return self.close_ic(k, chain_df)
        except Exception:
            pass
        return 0.0

    # ---------------- legacy single-leg exit ----------------
    def exit_order(self, oid: str, price: float) -> float:
        pos = self.positions.pop(oid, None)
        if pos is None:
            return 0.0
        qty = pos.get("qty", 0)
        entry = pos.get("entry_price", 0.0)
        side = pos.get("side", None)

        try:
            pnl = (float(price) - float(entry)) * float(qty)
        except Exception:
            pnl = 0.0

        if side == "SELL":
            pnl = -pnl

        self.history.append({
            **pos,
            "exit_price": price,
            "pnl": pnl,
            "exit_time": datetime.now().isoformat(),
        })

        self.balance += pnl
        self.capital = self.balance

        # remove from open_positions if present
        try:
            self.open_positions = [op for op in self.open_positions if op.get("position_id") != oid]
        except Exception:
            pass

        # persist
        try:
            self.save_state_to_disk()
        except Exception:
            pass

        return float(pnl)

    # ---------------- persistent state helpers ----------------
    def save_state_to_disk(self) -> bool:
        """
        Write a minimal broker state JSON used by the dashboard/engine:
        {
          "open_positions": [...],
          "capital": float,
          "starting_balance": float,
          "pnl": float,
          "history_len": int,
          "last_open_ic_id": str or None,
          "last_saved_at": iso
        }
        """
        try:
            os.makedirs(MONITOR_DIR, exist_ok=True)
            state = {
                "open_positions": self.open_positions,
                "capital": float(self.capital),
                "starting_balance": float(self.starting_balance),
                "pnl": float(self.balance - self.starting_balance),
                "history_len": len(self.history),
                "last_open_ic_id": self.last_open_ic.id if self.last_open_ic is not None else None,
                "last_saved_at": datetime.now().isoformat(),
            }
            with open(PAPER_BROKER_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            return True
        except Exception:
            try:
                # best-effort fallback to simple write
                with open(PAPER_BROKER_STATE_FILE, "w", encoding="utf-8") as f:
                    f.write("{}")
            except Exception:
                pass
            return False

    def load_state_from_disk(self) -> dict:
        """
        Load persisted broker state if exists and apply minimal restoration.
        Returns the loaded state dict or {}.
        """
        try:
            if not os.path.exists(PAPER_BROKER_STATE_FILE):
                return {}
            with open(PAPER_BROKER_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            # apply open_positions if present
            try:
                ops = state.get("open_positions")
                if isinstance(ops, list):
                    self.open_positions = ops
            except Exception:
                pass
            # capital/balance update is optional (do not overwrite if zero)
            try:
                cap = state.get("capital")
                if cap is not None:
                    self.capital = float(cap)
                    self.balance = float(cap)
            except Exception:
                pass
            return state
        except Exception:
            return {}

    # ---------------- housekeeping ----------------
    def get_positions_summary(self) -> Dict[str, Any]:
        """
        Return a summary useful for dashboards: counts and open_positions.
        """
        try:
            return {
                "open_count": len(self.open_positions),
                "open_positions": list(self.open_positions),
                "balance": float(self.balance),
                "starting_balance": float(self.starting_balance),
                "history_len": len(self.history),
                "last_open_ic_id": self.last_open_ic.id if self.last_open_ic is not None else None,
            }
        except Exception:
            return {"open_count": 0, "open_positions": [], "balance": float(self.balance)}

