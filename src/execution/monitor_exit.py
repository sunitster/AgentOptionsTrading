# -------------------------
# file: monitor_exit.py
# -------------------------
"""
Monitor open trades and exit based on predefined rules:
- target_profit_pct = 0.5 (50% of max credit)
- max_loss_multiplier = 1.2 (exit if MTM loss > 1.2 × credit)
- Auto-stop when no open trades remain
- Spinner animation while monitoring
"""

import time
import threading
from itertools import cycle

from src.execution.exec_adapter import (
    get_open_trades,
    place_order,
    mark_closed,
    poll_and_update_mtm
)

TARGET_PROFIT_PCT = 0.5
MAX_LOSS_MULT = 1.2
POLL_SEC = 5.0


def _should_exit(row):
    """Determine whether a trade should be exited."""
    credit = row.credit
    unreal = row.unreal or 0.0

    # Target profit: e.g. 50% of entry credit
    target_unreal = credit * TARGET_PROFIT_PCT

    # Profit hit
    if unreal >= target_unreal:
        return dict(exit=True, reason="target_profit")

    # Max loss hit
    if unreal <= -credit * MAX_LOSS_MULT:
        return dict(exit=True, reason="max_loss")

    return dict(exit=False)


def monitor_loop(stop_event):
    """Main monitoring loop with spinner + auto-stop."""
    spinner = cycle(['|', '/', '-', '\\'])
    print("Starting monitor loop...")

    while not stop_event.is_set():
        # spinner animation
        print(f"\rMonitoring... {next(spinner)}", end="", flush=True)

        poll_and_update_mtm()
        rows = get_open_trades()

        # Auto-stop condition
        if not rows:
            print("\nNo open trades left. Exiting.")
            break

        for r in rows:
            decision = _should_exit(r)
            if decision["exit"]:
                plan = dict(
                    plan_id=r.plan_id,
                    short_strike=r.short_strike,
                    long_strike=r.long_strike,
                    underlying_price=r.meta.get("underlying_price", 100.0) if r.meta else 100.0,
                    entry_iv=r.meta.get("entry_iv", 30.0) if r.meta else 30.0,
                    credit=r.credit,
                )

                fill = place_order(plan, qty=r.qty, side="buy")
                close_price = fill["fill_price"]
                realised = (r.credit - close_price) * r.qty
                mark_closed(r.plan_id, realised)

                print(f"\nClosed {r.plan_id} | reason={decision['reason']} | realised={realised:.6f}")

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    stop = threading.Event()
    try:
        monitor_loop(stop)
    except KeyboardInterrupt:
        stop.set()
        print("\nStopped")
