# src/live/paper_broker.py

import uuid
from datetime import datetime


class PaperBroker:
    """
    Simple in-memory simulated broker for live-paper execution.
    Supports:
        - place_order()   : legacy single-leg orders
        - open_ic()       : multi-leg iron condor open
        - close_ic()      : close IC at current MTM
        - realize_pnl()   : add MTM or realized PnL to balance
    """

    def __init__(self, capital=100000):
        self.starting_balance = capital
        self.balance = capital
        self.capital = self.balance      # legacy name used by old code

        # All open positions keyed by order_id
        self.positions = {}

        # List of closed positions (history)
        self.history = []

    # ---------------------------------------------------------
    # Legacy single-order entry   (kept for compatibility)
    # ---------------------------------------------------------
    def place_order(self, order):
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
    def realize_pnl(self, pnl):
        self.balance += pnl
        self.capital = self.balance
        return pnl

    # ---------------------------------------------------------
    # NEW — Multi-leg Iron Condor entry
    # ---------------------------------------------------------
    def open_ic(self, ic_obj):
        """
        Opens all 4 legs of an Iron Condor.
        Creates:
            - 4 separate leg positions
            - 1 wrapper IC position referencing the 4 legs
        """
        ic_id = str(uuid.uuid4())
        orders = ic_obj.entry_orders()

        leg_ids = []
        for od in orders:
            leg_id = str(uuid.uuid4())

            self.positions[leg_id] = {
                "symbol": od["symbol"],
                "qty": od["qty"],
                "side": od["side"],
                "entry_price": od["price"],
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

        return {
            "ic_id": ic_id,
            "legs": leg_ids,
            "entry_time": self.positions[ic_id]["entry_time"],
        }

    # ---------------------------------------------------------
    # NEW — Close an IC at current MTM prices
    # ---------------------------------------------------------
    def close_ic(self, ic_id, chain_df):
        """
        Closes the Iron Condor and realizes PnL using IC exit_pnl().

        This matches the logic LivePaperEngine uses:
            realized = ic_obj.exit_pnl(chain_df)
        """
        pos = self.positions.pop(ic_id, None)

        if pos is None or pos.get("type") != "IRON_CONDOR":
            return 0.0

        ic_obj = pos["ic"]

        # True MTM PnL
        realized = ic_obj.exit_pnl(chain_df)

        # Apply to broker balance
        self.balance += realized
        self.capital = self.balance

        # Remove leg positions as well
        for leg_id in pos["legs"]:
            self.positions.pop(leg_id, None)

        # Store history
        self.history.append({
            "ic": ic_obj.as_dict(),
            "pnl": realized,
            "exit_time": datetime.now(),
            "legs": pos["legs"],
        })

        return realized

    # ---------------------------------------------------------
    # Legacy exit of a single-leg order (kept for compatibility)
    # ---------------------------------------------------------
    def exit_order(self, oid, price):
        pos = self.positions.pop(oid)
        qty = pos["qty"]
        entry = pos["entry_price"]
        side = pos["side"]

        pnl = (price - entry) * qty
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
        return pnl
