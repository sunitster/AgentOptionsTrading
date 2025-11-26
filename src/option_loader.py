# src/option_loader.py
from kiteconnect import KiteConnect
import pandas as pd
from sqlalchemy import text
from datetime import datetime
from db import get_db_engine, init_tables
import yaml, os

def get_kite_client():
    """Initialize Kite Connect client from config."""
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    kite_cfg = cfg["kite"]
    kite = KiteConnect(api_key=kite_cfg["api_key"])
    kite.set_access_token(kite_cfg["access_token"])
    return kite, kite_cfg

def get_instrument_df(kite):
    """Fetch full instrument dump once (cache it locally)."""
    ins_file = os.path.join(os.path.dirname(__file__), "..", "data", "instruments.csv")
    if os.path.exists(ins_file) and (datetime.now().timestamp() - os.path.getmtime(ins_file)) < 86400:
        return pd.read_csv(ins_file)
    print("📥 Downloading instruments from Kite...")
    instruments = kite.instruments()
    df = pd.DataFrame(instruments)
    df.to_csv(ins_file, index=False)
    print(f"✅ Saved instruments to {ins_file}")
    return df

def get_latest_expiry(df, symbol):
    subset = df[(df.tradingsymbol.str.startswith(symbol)) & (df.instrument_type.isin(["CE","PE"]))]
    expiries = pd.to_datetime(subset.expiry.unique())
    latest = expiries.sort_values().iloc[0]
    return latest.date()

def fetch_option_chain(kite, symbol="NIFTY", expiry=None, strikes_around=10):
    """Fetch CE+PE around ATM for given symbol."""
    df = get_instrument_df(kite)
    spot = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["last_price"]
    print(f"📈 {symbol} spot = {spot}")

    if not expiry:
        expiry = get_latest_expiry(df, symbol)
    print(f"📅 Using expiry: {expiry}")

    df_chain = df[
        (df.name == symbol) &
        (df.expiry == str(expiry)) &
        (df.instrument_type.isin(["CE", "PE"]))
    ].copy()

    df_chain["distance"] = abs(df_chain["strike"] - spot)
    atm_strike = df_chain.loc[df_chain["distance"].idxmin(), "strike"]
    selected = df_chain[df_chain["strike"].between(atm_strike - strikes_around*100, atm_strike + strikes_around*100)]

    symbols = [f"{row.exchange}:{row.tradingsymbol}" for _, row in selected.iterrows()]
    quotes = kite.ltp(symbols)

    rows = []
    for sym, data in quotes.items():
        row = selected[selected.tradingsymbol == sym.split(":")[1]].iloc[0].to_dict()
        row.update({
            "last_price": data["last_price"],
            "timestamp": datetime.now(),
        })
        rows.append(row)
    return pd.DataFrame(rows)

def init_option_table(engine):
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS option_chain (
            id SERIAL PRIMARY KEY,
            symbol TEXT,
            tradingsymbol TEXT,
            expiry DATE,
            strike DOUBLE PRECISION,
            option_type TEXT,
            last_price DOUBLE PRECISION,
            open_interest DOUBLE PRECISION,
            timestamp TIMESTAMPTZ
        );
        """))

def save_option_chain(df, engine):
    df_to_save = df[["name", "tradingsymbol", "expiry", "strike", "instrument_type", "last_price", "timestamp"]].copy()
    df_to_save.columns = ["symbol", "tradingsymbol", "expiry", "strike", "option_type", "last_price", "timestamp"]
    df_to_save.to_sql("option_chain", engine, if_exists="append", index=False)
    print(f"✅ Saved {len(df_to_save)} option rows to PostgreSQL")

if __name__ == "__main__":
    kite, kite_cfg = get_kite_client()
    engine = get_db_engine()
    init_option_table(engine)

    df = fetch_option_chain(kite, symbol=kite_cfg["symbol"], expiry=kite_cfg.get("expiry") or None)
    save_option_chain(df, engine)
    print(df[["tradingsymbol", "strike", "option_type", "last_price"]].head(10))
