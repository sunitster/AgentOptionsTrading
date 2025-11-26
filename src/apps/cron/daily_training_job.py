import subprocess
from datetime import datetime
import pandas as pd

class DailyTrainingJob:
    def __init__(self):
        pass

    def run(self):
        print("📥 Collecting today's paper trades...")
        # Save broker history to disk

        print("📘 Rebuilding features from today's data...")

        print("🧠 Retraining model...")
        subprocess.run(["python", "-m", "src.learning.run_full_historical_training"])

        print("🏆 Promoting if improved...")
        # call scoring + promote logic

        print("✨ Done EOD pipeline")
