


# -------------------------
# file: README.md
# -------------------------
"""
Phase 4


How to use
1) Put this document's files into your project (exec_adapter.py, monitor_exit.py, db_models.py, daily_report.py).
2) Ensure your local Postgres DBs are running and the connection strings in the top of exec_adapter.py and daily_report.py match your env.
3) Install dependencies: pip install sqlalchemy psycopg2-binary
4) Run quick test: python exec_adapter.py
5) Start monitor: python monitor_exit.py
6) Schedule daily_report.py via cron or Windows Task Scheduler to create daily PnL files.

Notes & Acceptance checks implemented
- Paper fills use midpoint ± slippage; latency simulated and kept under MAX_FILL_LATENCY_S (default 1.5s)
- monitor_exit closes on 50% profit or 1.2x credit loss
- flat_all_by schedules a background job to close positions at 15:15 IST
- All fills recorded to trade_logs DB; market features stored separately if needed


# -------------------------
# file: README.md
# -------------------------
"""
Phase 5 — Self-Learning Loop scaffold


Drop these files under src/learning/ and create the folders:
- src/learning/
- src/learning/prompts/
- learning_data/ (auto-created)
- models/ (auto-created)


How it works:
1. daily_report.py should append a row to learning_data/daily_features.parquet with features + pnl
2. run `python -m src.learning.self_learning_loop` weekly (or schedule it)
3. The loop calls local Ollama (llama3.2) to propose challengers, runs backtests, and promotes conservatively


Notes:
- You must implement/backfill a `backtest_fn(regime_model)` that runs your backtester with the supplied regime model and returns at least `{'daily_pnl': [...]}`
- Adapt the Ollama client call in self_learning_loop._call_ollama to your local Ollama client wrapper.
"""