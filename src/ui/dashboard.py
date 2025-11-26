# src/dashboard.py
#
# Streamlit dashboard for LivePaperEngine (Python 3.14 compatible)
# ---------------------------------------------------------------
# Requires:
#   pip install streamlit pandas
#
# Run it with:
#   streamlit run src/dashboard.py
#
# Your live engine runs separately:
#   python -m src.cli.agent_trade live-paper

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

BASE_DIR = Path("models") / "llm_trades"
BROKER_STATE_FILE = BASE_DIR / "paper_broker_state.json"
PNL_HISTORY_FILE = BASE_DIR / "pnl_history.json"
SNAPSHOT_FILE = BASE_DIR / "latest_snapshot.json"
RISK_STATE_FILE = BASE_DIR / "risk_state.json"
MANUAL_EXIT_FILE = BASE_DIR / "manual_exit.json"


# -------------------------- helpers -------------------------- #

def safe_load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_manual_exit_flag() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "force_exit": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": "streamlit_dashboard",
    }
    with MANUAL_EXIT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# -------------------------- UI -------------------------- #

def main():
    st.set_page_config(
        page_title="IC Live Paper Dashboard",
        layout="wide",
    )

    st.title("📈 Iron Condor Live Paper Dashboard")

    st.caption(
        "Backed by Kite LIVE data (when market open). "
        "Engine: `python -m src.cli.agent_trade live-paper`"
    )

    # ---- Sidebar controls ----
    st.sidebar.header("Controls")

    refresh_hint = st.sidebar.radio(
        "Refresh mode",
        ["Manual (click button)", "Streamlit: use 'Rerun' / 'Always rerun'"],
        index=0,
    )

    if st.sidebar.button("🔁 Refresh now"):
        st.experimental_rerun()

    st.sidebar.markdown("---")

    if st.sidebar.button("🚨 Force Exit Current IC"):
        write_manual_exit_flag()
        st.sidebar.success("Exit signal sent to engine (manual_exit.json).")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Tip: In the Streamlit menu (top right), you can enable "
        "**'Always rerun'** to simulate auto-refresh."
    )

    # ---- Load data ----
    broker_state = safe_load_json(
        BROKER_STATE_FILE,
        default={"capital": 0.0, "starting_balance": 0.0, "pnl": 0.0, "trades": []},
    )
    pnl_history = safe_load_json(PNL_HISTORY_FILE, default=[])
    snapshot_data = safe_load_json(SNAPSHOT_FILE, default=[])
    risk_state = safe_load_json(RISK_STATE_FILE, default={})

    # ---- Top metrics ----
    col1, col2, col3, col4 = st.columns(4)

    starting_balance = float(broker_state.get("starting_balance", 0.0))
    capital = float(broker_state.get("capital", 0.0))
    net_pnl = float(broker_state.get("pnl", capital - starting_balance))
    trades_count = len(broker_state.get("trades", []))

    with col1:
        st.metric("Starting Balance", f"₹{starting_balance:,.2f}")
    with col2:
        st.metric("Current Capital", f"₹{capital:,.2f}")
    with col3:
        st.metric("Net PnL", f"₹{net_pnl:,.2f}")
    with col4:
        st.metric("Trades Today", trades_count)

    st.markdown("---")

    # ---- PnL History Chart + Risk State ----
    left, right = st.columns([2, 1])

    with left:
        st.subheader("📊 PnL Over Time")
        if pnl_history:
            df_pnl = pd.DataFrame(pnl_history)
            # Expecting keys: time, pnl
            if "time" in df_pnl.columns:
                df_pnl["time"] = pd.to_datetime(df_pnl["time"], errors="coerce")
                df_pnl = df_pnl.dropna(subset=["time"])
                df_pnl = df_pnl.sort_values("time")
                df_pnl.set_index("time", inplace=True)
            st.line_chart(df_pnl["pnl"])
        else:
            st.info("No PnL history yet. Start the engine to generate trades.")

    with right:
        st.subheader("🛡 Risk State")
        if risk_state:
            # Show as a table of key-value pairs
            kv = pd.DataFrame(
                [{"key": k, "value": str(v)} for k, v in risk_state.items()]
            )
            st.table(kv)
        else:
            st.info("No risk state found yet.")

    st.markdown("---")

    # ---- Trades Table ----
    st.subheader("📜 Trade Log")
    trades = broker_state.get("trades", [])
    if trades:
        # Flatten minimal view
        flat_rows: List[Dict[str, Any]] = []
        for t in trades:
            flat_rows.append(
                {
                    "time": t.get("time"),
                    "mode": t.get("mode"),
                    "pnl": t.get("pnl"),
                    "balance": t.get("balance"),
                }
            )
        df_trades = pd.DataFrame(flat_rows)
        df_trades = df_trades.sort_values("time", ascending=False)
        st.dataframe(df_trades, use_container_width=True, height=300)
    else:
        st.info("No trades recorded yet.")

    st.markdown("---")

    # ---- Option Snapshot ----
    st.subheader("📦 Latest Option Snapshot (top rows)")

    if snapshot_data:
        df_snap = pd.DataFrame(snapshot_data)
        # Show limited columns if many exist
        preferred_cols = [
            "tradingsymbol",
            "name",
            "expiry",
            "strike",
            "instrument_type",
            "ltp",
        ]
        cols = [c for c in preferred_cols if c in df_snap.columns]
        if cols:
            df_view = df_snap[cols]
        else:
            df_view = df_snap
        st.dataframe(df_view.head(40), use_container_width=True, height=400)
    else:
        st.info(
            "No snapshot found yet. Once the engine starts pulling chains, "
            "this table will populate."
        )


if __name__ == "__main__":
    main()
