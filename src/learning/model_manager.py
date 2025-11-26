# -------------------------
# file: model_manager.py
# -------------------------
"""
Model manager: load/save/promote/rollback JSON rule-models and metadata.
Champion stored at models/champion.json
Challengers saved as models/challenger_<ts>.json
"""
import os
import json
from datetime import datetime

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
CHAMPION_PATH = os.path.join(MODELS_DIR, 'regime_model_champion.json')


def save_challenger(candidate: dict) -> str:
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    path = os.path.join(MODELS_DIR, f'regime_model_challenger_{ts}.json')
    with open(path, 'w') as f:
        json.dump(candidate, f, indent=2)
    return path


def promote_challenger(challenger_path: str):
    # backup current champion
    if os.path.exists(CHAMPION_PATH):
        bak = CHAMPION_PATH + '.' + datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        os.replace(CHAMPION_PATH, bak)
    os.replace(challenger_path, CHAMPION_PATH)
    return CHAMPION_PATH


def load_champion() -> dict:
    if not os.path.exists(CHAMPION_PATH):
        return {}
    with open(CHAMPION_PATH, 'r') as f:
        return json.load(f)