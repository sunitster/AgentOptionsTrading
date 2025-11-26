# diagnose_chain.py
import json
import pandas as pd
from src.trading.signal_generator import SignalGenerator
from src.live.kite_data import KiteData  # adjust import path if different

def dump_df_info(df, name="chain_df", n=10):
    print(f"==== {name} info ====")
    if df is None:
        print("None")
        return
    print("shape:", df.shape)
    print("columns:", df.columns.tolist())
    print("non-null counts:")
    print(df.notna().sum().to_dict())
    for col in ["strike", "option_type", "expiry", "ltp", "best_bid", "best_ask", "delta", "iv", "spot"]:
        print(f"  {col} present? {col in df.columns}")
    # show first rows
    print("\nfirst rows:\n", df.head(n).to_dict(orient="records"))
    # show unique expiries (up to 10)
    if "expiry" in df.columns:
        try:
            exps = df["expiry"].dropna().unique()[:10]
            print("unique expiry samples:", list(map(str, exps)))
        except Exception as e:
            print("could not list expiry unique:", e)
    # show available strikes (sorted sample)
    if "strike" in df.columns:
        try:
            strikes = sorted(set([float(x) for x in df["strike"].dropna().unique()]))
            print("sample strikes (first 20):", strikes[:20])
        except Exception as e:
            print("could not list strikes:", e)

def main():
    # create kite data helper and get live snapshot for your symbol
    kd = KiteData()
    # if your runner uses specific symbol, set it; default in engine is "NIFTY"
    symbol = "NIFTY"
    print("Fetching option chain snapshot for", symbol)
    try:
        # there are multiple helpers; try common ones
        df = kd.get_option_chain_snapshot(symbol)  # typical
    except Exception as e:
        print("get_option_chain_snapshot failed:", e)
        try:
            df = kd.full_option_chain(symbol)
        except Exception as e2:
            print("full_option_chain failed:", e2)
            df = None

    dump_df_info(df, "live_chain_df")

    # show what SignalGenerator would pick (without placing orders)
    sg = SignalGenerator()
    spot = None
    if df is not None:
        if "spot" in df.columns:
            try:
                spot = float(df["spot"].dropna().iloc[0])
            except Exception:
                spot = None
    print("Inferred spot:", spot)
    cands = sg.generate(df, spot=spot)
    print("generate() returned", len(cands), "candidates")
    if cands:
        for c in cands:
            try:
                print("candidate as_dict:", c.as_dict())
            except Exception:
                print("candidate object:", c)

if __name__ == "__main__":
    main()
