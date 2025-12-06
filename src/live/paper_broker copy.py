# src/live/paper_broker.py
"""
PaperBroker (patched)

Enhancements:
- Sets `self.last_open_ic` when open_ic() is called.
- open_ic() returns a small wrapper object with `.id` and `.ic` attributes and dict-like access.
- Added get_ic_record(ic_id) helper.
- Added close_ic_by_obj(ic_obj, chain_df) to close using an object returned by open_ic().
- Backwards-compatible: legacy methods unchanged.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

class ICRecord:
    """
    Lightweight wrapper returned by open_ic.
    Provides attribute access (.id, .ic, .legs, .entry_time) and dict-like .to_dict().
    """
    def __init__(self, ic_id: str, ic_obj: Any, legs: list, entry_time: datetime):
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
    Simple in-memory simulated broker for live-paper execution.
    Supports:
        - place_order()   : legacy single-leg orders
        - open_ic()       : multi-leg iron condor open
        - close_ic()      : close IC at current MTM
        - realize_pnl()   : add MTM or realized PnL to balance

    Enhancements:
    - last_open_ic is set to the ICRecord for immediate engine pickup.
    - open_ic returns an ICRecord object with .id attribute and .to_dict().
    """
    def __init__(self, capital=100000):
        self.starting_balance = float(capital)
        self.balance = float(capital)
        self.capital = self.balance      # legacy name used by old code

        # All positions: both legs and wrapper ICs keyed by id
        # For wrapper ICs, value contains { "type":"IRON_CONDOR", "ic": ic_obj, "legs": [leg_ids], "entry_time": datetime }
        self.positions: Dict[str, Dict[str, Any]] = {}

        # List of closed positions (history)
        self.history: list = []

        # Last opened IC (ICRecord) for quick engine access; None when no recent open.
        self.last_open_ic: Optional[ICRecord] = None

    # ---------------------------------------------------------
    # Legacy single-order entry   (kept for compatibility)
    # ---------------------------------------------------------
    def place_order(self, order: dict) -> str:
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
            "entry_time": datetime.now(),
        }

        return oid

    # ---------------------------------------------------------
    # Realize PnL into capital
    # ---------------------------------------------------------
    def realize_pnl(self, pnl: float) -> float:
        self.balance += float(pnl)
        self.capital = self.balance
        return float(pnl)

    # ---------------------------------------------------------
    # NEW — Multi-leg Iron Condor entry
    # ---------------------------------------------------------
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
            # best-effort: attempt to use attribute or fallback
            try:
                orders = getattr(ic_obj, "entry_orders", []) or []
            except Exception:
                orders = []

        leg_ids = []
        for od in orders:
            # Normalize od shape (allow objects or dicts)
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
                "entry_time": datetime.now(),
            }
            leg_ids.append(leg_id)

        # Wrapper IC object
        self.positions[ic_id] = {
            "type": "IRON_CONDOR",
            "ic": ic_obj,
            "legs": leg_ids,
            "entry_time": datetime.now(),
        }

        # Record wrapper for engine
        ic_record = ICRecord(ic_id, ic_obj, leg_ids, self.positions[ic_id]["entry_time"])
        self.last_open_ic = ic_record

        # Return wrapper (object with .id attribute)
        return ic_record

    # ---------------------------------------------------------
    # Helper: get IC record by id
    # ---------------------------------------------------------
    def get_ic_record(self, ic_id: str) -> Optional[ICRecord]:
        pos = self.positions.get(ic_id)
        if pos is None:
            return None
        if pos.get("type") != "IRON_CONDOR":
            return None
        return ICRecord(ic_id, pos.get("ic"), pos.get("legs", []), pos.get("entry_time"))

    # ---------------------------------------------------------
    # Close an IC by id and realize PnL (existing behaviour)
    # ---------------------------------------------------------
    def close_ic(self, ic_id: str, chain_df: Any) -> float:
        """
        Close the Iron Condor and realize PnL.
        Returns realized pnl (float).
        """
        pos = self.positions.pop(ic_id, None)

        if pos is None or pos.get("type") != "IRON_CONDOR":
            return 0.0

        ic_obj = pos["ic"]

        # True MTM PnL (expects ic_obj.exit_pnl(chain_df))
        try:
            realized = float(ic_obj.exit_pnl(chain_df))
        except Exception:
            # Fallback: no exit_pnl; try to compute or return 0
            try:
                realized = float(getattr(ic_obj, "approx_pnl", 0.0))
            except Exception:
                realized = 0.0

        # Apply to broker balance
        self.balance += realized
        self.capital = self.balance

        # Remove leg positions as well
        for leg_id in pos.get("legs", []):
            try:
                self.positions.pop(leg_id, None)
            except Exception:
                pass

        # Store history
        try:
            ic_snapshot = ic_obj.as_dict() if hasattr(ic_obj, "as_dict") else str(ic_obj)
        except Exception:
            ic_snapshot = str(ic_obj)

        self.history.append({
            "ic": ic_snapshot,
            "pnl": realized,
            "exit_time": datetime.now(),
            "legs": pos.get("legs", []),
        })

        # Clear last_open_ic if it refers to this IC
        try:
            if self.last_open_ic and self.last_open_ic.id == ic_id:
                self.last_open_ic = None
        except Exception:
            pass

        return realized

    # ---------------------------------------------------------
    # Close an IC by passing the ICRecord / object (convenience)
    # ---------------------------------------------------------
    def close_ic_by_obj(self, ic_obj_or_record: Any, chain_df: Any) -> float:
        """
        If passed an ICRecord, use its id; if passed a wrapper object (with .id), handle that too.
        """
        try:
            # If ICRecord provided
            if isinstance(ic_obj_or_record, ICRecord):
                return self.close_ic(ic_obj_or_record.id, chain_df)
            # If it's an object with 'id' attribute (like returned by our open_ic)
            if hasattr(ic_obj_or_record, "id"):
                return self.close_ic(getattr(ic_obj_or_record, "id"), chain_df)
            # Otherwise, maybe they passed the original ic_obj: find matching wrapper
            for k, v in list(self.positions.items()):
                if v.get("type") == "IRON_CONDOR" and v.get("ic") is ic_obj_or_record:
                    return self.close_ic(k, chain_df)
        except Exception:
            pass
        return 0.0

    # ---------------------------------------------------------
    # Legacy exit of a single-leg order (kept for compatibility)
    # ---------------------------------------------------------
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
            "exit_time": datetime.now(),
        })

        self.balance += pnl
        self.capital = self.balance
        return float(pnl)
