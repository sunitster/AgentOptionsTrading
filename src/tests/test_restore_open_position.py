import os
import json
import shutil
from datetime import datetime, timedelta

import pytest

from src.live.live_paper_engine import LivePaperEngine
from src.trading.iron_condor_builder import IronCondor

MONITOR_PATH = "models/llm_trades"
CURRENT_POS_FILE = os.path.join(MONITOR_PATH, "current_position.json")
PAPER_BROKER_STATE_FILE = os.path.join(MONITOR_PATH, "paper_broker_state.json")


@pytest.fixture(autouse=True)
def clean_monitor_folder():
    """Ensures clean state before each test."""
    if os.path.exists(MONITOR_PATH):
        shutil.rmtree(MONITOR_PATH)
    os.makedirs(MONITOR_PATH, exist_ok=True)
    yield
    # leave folder for debugging if test fails


def test_restore_open_position_full_cycle():
    """
    Verifies the engine restores:
    - open_ic (from dict)
    - open_ic_replay_id
    - open_ic_entry_time
    """

    # --- 1) Prepare mock IronCondor dict for current_position.json
    ic_dict = {
        "symbol": "NIFTY",
        "expiry": "2025-12-26",
        "short_put": 24000,
        "long_put": 23800,
        "short_call": 24500,
        "long_call": 24700,
        "short_put_price": 120.0,
        "long_put_price": 40.0,
        "short_call_price": 100.0,
        "long_call_price": 30.0,
        "lot_size": 25,
        "size_aggressiveness": 1.0
    }
    with open(CURRENT_POS_FILE, "w") as f:
        json.dump(ic_dict, f, indent=2)

    # --- 2) Save replay + entry time
    replay_id = 777
    entry_time = (datetime.now() - timedelta(hours=1)).isoformat()

    with open(PAPER_BROKER_STATE_FILE, "w") as f:
        json.dump(
            {
                "capital": 1_000_000,
                "open_ic_replay_id": replay_id,
                "open_ic_entry_time": entry_time
            },
            f, indent=2
        )

    # --- 3) Construct engine (it should auto-restore state)
    engine = LivePaperEngine(
        symbol="NIFTY",
        use_llm_selector=False,    # ensure deterministic behaviour
        starting_capital=1_000_000
    )

    # --- 4) ASSERTIONS ---

    # open_ic must be restored as IronCondor object
    assert engine.open_ic is not None, "open_ic not restored"
    assert isinstance(engine.open_ic, IronCondor), "open_ic not restored as IronCondor"

    # verify legs and expiry
    restored = engine.open_ic.as_dict()
    assert restored["short_put"] == 24000
    assert restored["short_call"] == 24500
    assert restored["expiry"] == "2025-12-26"

    # replay id
    assert engine.open_ic_replay_id == replay_id, "replay_id not restored correctly"

    # entry time
    assert engine.open_ic_entry_time is not None, "entry time missing"
    # allow 2-second tolerance for parsing
    delta = abs((engine.open_ic_entry_time - datetime.fromisoformat(entry_time)).total_seconds())
    assert delta < 2, f"Entry time mismatch: expected {entry_time}, got {engine.open_ic_entry_time}"

    print("\nRestoration test passed successfully!\n")
