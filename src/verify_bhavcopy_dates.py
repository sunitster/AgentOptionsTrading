import pandas as pd
import os
import glob
from datetime import datetime, timedelta


OPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "options")

mismatched_files = []
matched_files = []

for f in sorted(glob.glob(os.path.join(OPT_DIR, "*.parquet"))):
    file_date_str = os.path.basename(f).replace(".parquet", "")
    file_date = datetime.strptime(file_date_str, "%Y-%m-%d").date()

    try:
        df = pd.read_parquet(f)

        if df.empty:
            mismatched_files.append((f, "empty"))
            continue

        # Bhavcopy DO NOT have exact timestamp, so we check:
        # - EXPIRY_DT must be >= file_date
        # - EXPIRY_DT must be <= file_date + 120 days
        exp_dates = pd.to_datetime(df["EXPIRY_DT"]).dt.date

        if (exp_dates < file_date).any() or (exp_dates > file_date + timedelta(days=120)).any():
            mismatched_files.append((f, "expiry out of range"))
            continue

        matched_files.append(f)

    except Exception as e:
        mismatched_files.append((f, str(e)))

print("======== Deep Bhavcopy Verification ========")
print("Total files:", len(matched_files) + len(mismatched_files))
print("Correct files:", len(matched_files))
print("Incorrect files:", len(mismatched_files))

if mismatched_files:
    print("\nIncorrect Samples:")
    for f, reason in mismatched_files[:20]:
        print("❌", f, "→", reason)
