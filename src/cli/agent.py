#!/usr/bin/env python
import argparse
import logging
import yaml
from src.execution.execution_engine import ExecutionEngine

LOG = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_cfg(path="config/config.yaml"):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def cmd_paper_trade(args):
    cfg = load_cfg(args.config)
    kite_cfg = cfg.get("kite", {})
    symbol = args.symbol or kite_cfg.get("symbol", "NIFTY")
    lot_size = args.lot or kite_cfg.get("lot_size", 25)
    ic_width = args.width or kite_cfg.get("ic_width", 150)
    starting = args.starting or cfg.get("starting_capital", 1_000_000)

    LOG.info("📘 Running PAPER TRADING mode...")
    engine = ExecutionEngine(starting_capital=starting, lot_size=lot_size, ic_width=ic_width, cfg_path=args.config)
    summary = engine.run_paper_day(symbol=symbol, max_trades=args.max_trades)
    LOG.info("📘 Paper Trading Summary:\n%s", summary)
    print(summary)


def build_parser():
    p = argparse.ArgumentParser(prog="agent")
    sub = p.add_subparsers(dest="cmd")

    p_paper = sub.add_parser("paper-trade", help="Run a single day paper trading session using live option chain.")
    p_paper.add_argument("--config", default="config/config.yaml", help="path to config yaml")
    p_paper.add_argument("--symbol", default=None, help="Underlying symbol (NIFTY/BANKNIFTY)")
    p_paper.add_argument("--lot", type=int, default=None, help="lot size (overrides config)")
    p_paper.add_argument("--width", type=int, default=None, help="IC width (points)")
    p_paper.add_argument("--max-trades", type=int, default=2, help="Max ICs to open")
    p_paper.add_argument("--starting", type=float, default=None, help="Starting capital")
    p_paper.set_defaults(func=cmd_paper_trade)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
