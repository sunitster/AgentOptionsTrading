import json
import pickle
import os

def load_champion_model(model_dir="models"):
    """
    Loads champion model + feature list.
    Used by SignalGenerator in live/paper trading.
    """
    champion_path = os.path.join(model_dir, "lightgbm_champion.pkl")
    features_path = os.path.join(model_dir, "feature_list.json")

    if not os.path.exists(champion_path):
        raise FileNotFoundError(f"Champion model not found: {champion_path}")

    with open(champion_path, "rb") as f:
        model = pickle.load(f)

    # Feature list
    if os.path.exists(features_path):
        with open(features_path, "r") as f:
            features = json.load(f)
    else:
        features = []

    return model, features
