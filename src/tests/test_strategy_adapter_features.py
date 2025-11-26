import pandas as pd
from src.learning.strategy_adapter import apply_strategy_adapter


def test_strategy_adapter_feature_selection():
    df_train = pd.DataFrame({
        "iv": [0.2, 0.2, 0.2],
        "dte": [10, 10, 10],
        "moneyness": [1.0, 1.0, 1.0],
        "symbol": ["A", "B", "C"],  # non-numeric
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

    _, _, features, _ = apply_strategy_adapter(
        df_train=df_train,
        df_val=df_val,
        bounds=bounds
    )

    assert "symbol" not in features
    assert "iv" in features
    assert "dte" in features
    assert "moneyness" in features
