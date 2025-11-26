import pandas as pd
import os
import glob

OPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "options")

bad_files = []
good_files = []
empty_files = []

for f in sorted(glob.glob(os.path.join(OPT_DIR, "*.parquet"))):
    try:
        df = pd.read_parquet(f)
        if df.empty:
            empty_files.append(f)
            continue

        required_cols = {"SYMBOL", "OPTION_TYP", "STRIKE_PR", "EXPIRY_DT", "CLOSE"}
        if not required_cols.issubset(set(df.columns)):
            bad_files.append((f, "missing columns"))
            continue

        if not (df["SYMBOL"] == "NIFTY").all():
            bad_files.append((f, "wrong symbol"))
            continue

        good_files.append(f)

    except Exception as e:
        bad_files.append((f, str(e)))

print("======== Verification Report ========")
print("Total files:", len(good_files) + len(bad_files) + len(empty_files))
print("Valid files:", len(good_files))
print("Empty files:", len(empty_files))
print("Corrupt/Invalid files:", len(bad_files))

if bad_files or empty_files:
    print("\nList of bad files:")
    for f in bad_files[:20]:
        print("❌", f)
    print("\nList of empty files:")
    for f in empty_files[:20]:
        print("⚠", f)
