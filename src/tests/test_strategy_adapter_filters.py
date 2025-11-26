import pandas as pd
from src.learning.strategy_adapter import apply_strategy_adapter


def test_strategy_adapter_filters_bounds():
    df_train = pd.DataFrame({
        "iv": [0.05, 0.2, 0.9],
        "dte": [1, 50, 400],
        "moneyness": [0.4, 1.0, 3.0],
        "target": [1, 2, 3]
    })
    df_val = df_train.copy()

    bounds = {
        "iv_min": 0.1,
        "iv_max": 1.0,
        "dte_min": 0,
        "dte_max": 300,
        "moneyness_min": 0.5,
        "moneyness_max": 2.0
    }

    train_adj, _, _, meta = apply_strategy_adapter(
        df_train=df_train,
        df_val=df_val,
        bounds=bounds
    )

    # Only row 1 survives
    assert len(train_adj) == 1
    assert meta["survived_fraction"] == 1 / 3
