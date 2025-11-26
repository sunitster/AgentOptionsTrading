from sentence_transformers import SentenceTransformer
import numpy as np
from sqlalchemy import text
from logger_pg import engine

model = SentenceTransformer("all-MiniLM-L6-v2")

def fetch_similar(ivp, atr, trend):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT payload FROM logs_tradeplan
            ORDER BY id DESC LIMIT 5000
        """)).fetchall()

    items = [r[0] for r in rows]
    if not items:
        return []

    keys = []
    for item in items:
        keys.append(f"ivp:{item['ivp']},atr:{item['atr']},trend:{item['trend']},win:{item['pnl']>0}")

    q = f"ivp:{ivp} atr:{atr} trend:{trend}"
    qvec = model.encode([q])
    ivec = model.encode(keys)

    scores = np.dot(ivec, qvec.T).flatten()
    idx = scores.argsort()[::-1][:10]

    return [items[i] for i in idx]
