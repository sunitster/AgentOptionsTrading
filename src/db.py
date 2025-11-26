# src/db.py
from sqlalchemy import create_engine, text
import yaml, os

def get_db_engine():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    db_url = cfg["database"]["url"]
    engine = create_engine(db_url)
    return engine

def init_tables(engine):
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS ohlc_data (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume DOUBLE PRECISION
        );
        """))
