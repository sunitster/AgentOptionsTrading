import pandas as pd
from src.learning.strategy_adapter import apply_strategy_adapter


def test_strategy_adapter_basic():
    df_train = pd.DataFrame({
        "iv": [0.2, 0.3, 0.5],
        "dte": [10, 20, 30],
        "moneyness": [0.9, 1.0, 1.1],
        "target": [1, 2, 3]
    })
    df_val = df_train.copy()

    bounds = {
        "iv_min": 0.1,
        "iv_max": 1.0,
        "dte_min": 0,
        "dte_max": 200,
        "moneyness_min": 0.5,
        "moneyness_max": 2.0
    }

    train_adj, val_adj, features, meta = apply_strategy_adapter(
        df_train=df_train,
        df_val=df_val,
        bounds=bounds
    )

    assert len(train_adj) == 3
    assert "iv" in features
    assert meta["survived_fraction"] == 1.0
