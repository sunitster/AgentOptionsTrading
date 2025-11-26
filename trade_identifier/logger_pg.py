from sqlalchemy import create_engine, text
engine = create_engine("postgresql+psycopg2://trader:trader123@localhost:5432/trading")

def log_backtest(row: dict):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO logs_backtest (ts, payload)
                VALUES (NOW(), :p)
            """), {"p": json.dumps(row)}
        )

def log_tradeplan(plan):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO logs_tradeplan (ts, plan_id, payload)
                VALUES (NOW(), :pid, :p)
            """), {"pid": plan["plan_id"], "p": json.dumps(plan)}
        )
