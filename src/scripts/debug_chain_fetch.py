import json, os
from src.live.kite_api import KiteAPI
from src.live.kite_data import full_option_chain

def main():
    print("Initializing KiteAPI...")
    api = KiteAPI(mode="live")
    print("KiteAPI init OK")

    try:
        print("Fetching chain for NIFTY...")
        df = full_option_chain(api, "NIFTY")
        if df is None:
            print("full_option_chain returned None")
            return

        print("Rows:", len(df))
        print(df.head())

        out = os.path.join("models","llm_trades","debug_snapshot.json")
        df.to_json(out, orient="records")
        print("Wrote debug snapshot:", out)

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
