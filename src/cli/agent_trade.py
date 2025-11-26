# Patched agent_trade.py
# Replace the entire file with this version.

import argparse
import logging
import time
import importlib
from datetime import datetime

from src.live.live_paper_engine import LivePaperEngine
from src.deploy.live_engine import LiveEngine
from src.live.kite_api import KiteAPI

log = logging.getLogger("agent_trade")


def cmd_live_paper(args):
    log.info("📘 LIVE PAPER TRADING STARTED...")

    # Start dashboard
    if args.dashboard:
        log.info("🚀 Starting FastAPI dashboard on http://127.0.0.1:8000 ...")
        try:
            from src.dashboard.server import start_dashboard
            start_dashboard()
            log.info("📊 FastAPI dashboard started successfully.")
        except Exception:
            log.exception("Dashboard failed to start")

    # Initialize hybrid engine
    engine = LivePaperEngine(
        symbol=args.symbol,
        lot_size=args.lot_size,
        width=args.width,
        minute_interval=args.interval,
        starting_capital=args.capital,
        size_aggressiveness=args.aggr,
        risk_daily_loss_limit=args.dd,
        risk_max_trade_pct=args.order_risk,
        max_daily_trades=args.max_trades,
        dashboard=args.dashboard,
    )

    log.info("📡 Entering live loop... Press CTRL+C to exit.")

    try:
        while True:
            engine.run_once()   # <--- FIXED: removed (api)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("⛔ Stopping live-paper trading...")
    except Exception:
        log.exception("❌ Critical error in live-paper")


def cmd_live(args):
    log.info("📘 LIVE TRADING STARTED...")

    api = KiteAPI(mode="live")
    engine = LiveEngine(api=api)

    try:
        while True:
            engine.run_once()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("⛔ Stopping live trading...")
    except Exception:
        log.exception("❌ Critical error in live trading")


def cmd_paper_live(args):
    log.info("📘 PAPER-LIVE MODE STARTED...")

    engine = LivePaperEngine(
        symbol=args.symbol,
        lot_size=args.lot_size,
        width=args.width,
        minute_interval=args.interval,
        starting_capital=args.capital,
        size_aggressiveness=args.aggr,
        risk_daily_loss_limit=args.dd,
        risk_max_trade_pct=args.order_risk,
        max_daily_trades=args.max_trades,
        dashboard=args.dashboard,
    )

    log.info("📡 Entering loop (paper-live)... Press CTRL+C to exit.")

    try:
        while True:
            engine.run_once()   # <--- FIXED
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("⛔ Stopping paper-live mode...")
    except Exception:
        log.exception("❌ Critical error in paper-live mode")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    # Shared args
    def add_shared(p):
        p.add_argument("--symbol", default="NIFTY")
        p.add_argument("--lot_size", type=int, default=25)
        p.add_argument("--width", type=int, default=150)
        p.add_argument("--interval", type=int, default=60)
        p.add_argument("--capital", type=float, default=1_000_000)
        p.add_argument("--aggr", type=float, default=1.0)
        p.add_argument("--dd", type=float, default=20000.0)
        p.add_argument("--order_risk", type=float, default=0.02)
        p.add_argument("--max_trades", type=int, default=3)
        p.add_argument("--dashboard", action="store_true")

    p_live = sub.add_parser("live")
    add_shared(p_live)
    p_live.set_defaults(func=cmd_live)

    p_live_paper = sub.add_parser("live-paper")
    add_shared(p_live_paper)
    p_live_paper.set_defaults(func=cmd_live_paper)

    p_paper_live = sub.add_parser("paper-live")
    add_shared(p_paper_live)
    p_paper_live.set_defaults(func=cmd_paper_live)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
