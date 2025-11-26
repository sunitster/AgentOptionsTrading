import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(__file__))
FEAT_DIR = os.path.join(BASE, "data/features")

def plot_corr():
    dfs = []
    for f in os.listdir(FEAT_DIR):
        df = pd.read_parquet(os.path.join(FEAT_DIR, f))
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)

    corr = full.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.title("Feature Correlation Heatmap")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_corr()
