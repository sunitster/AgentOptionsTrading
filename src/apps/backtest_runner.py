# apps/backtest_runner.py
# Recreated with full JSON logging integration (Step 1)

from __future__ import annotations
import datetime
from typing import Dict, Any, List
import logging

from engine.plan_builder import PlanBuilder
from engine.memory_store import MemoryStore
from engine.engine import TradingEngine, EngineConfig
import engine.risk as risk_module
from models.decision_types import FinalDecision
from ai.ai_client import AIClient
from apps.iron_condor_v4 import MockAIClient  # reuse mock

from engine.logging_routes import (
    init_logging,
    log_decision,
    log_ai,
    log_risk,
    log_pnl,
    log_system,
)


# ------------------------------------------------------------
def generate_weekly_market_data(start_date: datetime.date,
                                end_date: datetime.date) -> List[Dict[str, Any]]:
    """
    Minimal deterministic synthetic weekly data
    """
    import math

    rows = []
    underlying = 18200
    day = start_date
    i = 0

    while day <= end_date:
        total_weeks = max(1, ((end_date - start_date).days // 7) + 1)
        phase = (i / total_weeks) * 2 * math.pi
        ivp = 50 + 40 * math.sin(phase)
        price = underlying + 120 * math.sin(phase * 1.3)

        rows.append({
            "date": day.isoformat(),
            "underlying": underlying,
            "price": round(price, 2),
            "ivp": round(max(0, min(100, ivp)), 2),
            "recent_pnl": -1000 if (i % 7 == 3) else 100,
        })

        day += datetime.timedelta(days=7)
        i += 1

    return rows


# ------------------------------------------------------------
def run_backtest(start_date: datetime.date,
                 end_date: datetime.date,
                 weekly: bool,
                 use_ai: bool,
                 verbose: bool) -> Dict[str, Any]:

    logger = logging.getLogger("backtest")

    pb = PlanBuilder()
    memory = MemoryStore()
    config = EngineConfig()

    if use_ai:
        try:
            ai_client = AIClient()
            logger.info("Using real AIClient")
        except Exception:
            ai_client = MockAIClient()
            logger.warning("Real AI unavailable, falling back to MockAIClient")
    else:
        ai_client = MockAIClient()
        logger.info("Using MockAIClient (deterministic)")

    engine = TradingEngine(ai_client=ai_client, config=config, memory=memory)

    risk_thresholds = {
        "ivp_reduce_threshold": 70.0,
        "ivp_increase_threshold": 20.0,
        "proximity_threshold": 100.0,
        "max_width_delta": config.max_width_delta,
    }

    market_rows = generate_weekly_market_data(start_date, end_date)

    results = []
    total_pnl = 0.0
    memory.reset()

    for md in market_rows:
        # Build plan
        plan = pb.build_plan(
            underlying=md["underlying"],
            underlying_price=md["price"],
            iv_percentile=md["ivp"],
            days_to_expiry=14,
            recent_pnl=md.get("recent_pnl", 0.0),
            notes=f"backtest row {md['date']}"
        )

        # RISK
        risk_signals = risk_module.compute_risk_signals(plan, risk_thresholds)
        risk_score = risk_module.compute_risk_score(risk_signals)

        log_risk(
            check_id=f"risk-{plan['plan_id']}",
            risk_state={"risk_signals": risk_signals, "risk_score": risk_score},
            triggers={k: v for k, v in risk_signals.items() if v}
        )

        # DECISION (rules + AI)
        decision: FinalDecision = engine.evaluate_plan(plan, risk_thresholds)

        # AI LOG (mock)
        if isinstance(ai_client, MockAIClient):
            ai_payload = {
                "suggestion": decision.suggestion,
                "width_delta": decision.adjustments.get("width_delta", 0),
                "explanation": "Mock AI decision",
            }
            log_ai(
                request_id=f"mock-{plan['plan_id']}",
                prompt="mock",
                sanitized_prompt="mock",
                raw_response=ai_payload,
                parsed_response=ai_payload,
                context={"plan_id": plan["plan_id"]}
            )

        # DECISION LOG
        log_decision(
            decision_id=f"dec-{plan['plan_id']}-{md['date']}",
            rule_outputs={"risk_signals": risk_signals},
            ai_suggestion={"width_delta": decision.adjustments.get("width_delta", 0)},
            final_decision=decision.__dict__,
            context={"date": md["date"], "ivp": md["ivp"], "price": md["price"]}
        )

        # PnL (placeholder)
        pnl = float(plan["entry_price"]) - (decision.adjustments.get("width_delta", 0) * 0.1)
        total_pnl += pnl

        log_pnl(
            date=md["date"],
            summary={
                "plan_id": plan["plan_id"],
                "pnl": round(pnl, 2),
                "total_pnl": total_pnl,
                "risk_score": risk_score,
            }
        )

        results.append({
            "date": md["date"],
            "plan_id": plan["plan_id"],
            "ivp": md["ivp"],
            "price": md["price"],
            "risk_score": risk_score,
            "suggestion": decision.suggestion,
            "width_delta": decision.adjustments.get("width_delta", 0),
            "pnl": round(pnl, 2),
        })

        if verbose:
            logger.info(
                f"{md['date']} plan={plan['plan_id']} ivp={md['ivp']} "
                f"price={md['price']} risk={risk_score} -> {decision.suggestion} "
                f"wd={decision.adjustments['width_delta']} pnl={round(pnl,2)}"
            )

    return {
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "rows": len(results),
        "total_pnl": round(total_pnl, 2),
        "results": results,
    }


# ------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--weekly", action="store_true")
    p.add_argument("--ai", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    start = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(args.end, "%Y-%m-%d").date()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Initialize JSON logging
    init_logging(root="logs", app_name="backtest_runner", flush_interval=0.5)
    log_system("backtest_start", {"args": vars(args)})

    summary = run_backtest(start, end, args.weekly, args.ai, args.verbose)
    log_system("backtest_end", summary)

    print("SUMMARY:\n", summary)


if __name__ == "__main__":
    main()