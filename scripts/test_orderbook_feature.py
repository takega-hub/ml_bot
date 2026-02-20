"""
Проверка фичи Order Book Imbalance: расчёт из ответа API и наличие колонок в create_technical_indicators.
Запуск: python scripts/test_orderbook_feature.py
"""
import sys
from pathlib import Path

# Корень проекта
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.ml.feature_engineering import compute_orderbook_imbalance, FeatureEngineer
import pandas as pd
import numpy as np


def test_compute_orderbook_imbalance():
    """Тест расчёта imbalance из сырого ответа Bybit."""
    # Пример ответа Bybit /v5/market/orderbook
    ob_response = {
        "retCode": 0,
        "result": {
            "s": "BTCUSDT",
            "b": [["65000", "1.5"], ["64900", "2.0"], ["64800", "1.0"]],  # bid_vol = 4.5
            "a": [["65100", "1.0"], ["65200", "1.0"], ["65300", "0.5"]],  # ask_vol = 2.5
        },
    }
    imb = compute_orderbook_imbalance(ob_response, depth=10)
    expected = (4.5 - 2.5) / (4.5 + 2.5)  # 2/7 ≈ 0.2857
    assert abs(imb - expected) < 1e-6, f"Expected {expected}, got {imb}"
    print(f"  compute_orderbook_imbalance: OK (imbalance={imb:.4f})")

    # Пустой ответ
    assert compute_orderbook_imbalance({}, depth=10) == 0.0
    assert compute_orderbook_imbalance({"retCode": 1}, depth=10) == 0.0
    print("  Empty/invalid response: OK (returns 0.0)")


def test_create_technical_indicators_has_ob_columns():
    """Проверка, что create_technical_indicators добавляет колонки ob_imbalance."""
    fe = FeatureEngineer()
    # Минимальный OHLCV датафрейм
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    df = pd.DataFrame(
        {
            "open": np.random.rand(n) * 100 + 100,
            "high": np.random.rand(n) * 100 + 101,
            "low": np.random.rand(n) * 100 + 99,
            "close": np.random.rand(n) * 100 + 100,
            "volume": np.random.rand(n) * 1e6,
        },
        index=idx,
    )
    df["timestamp"] = df.index
    out = fe.create_technical_indicators(df)
    assert "ob_imbalance" in out.columns, "ob_imbalance column missing"
    assert "ob_imbalance_5" in out.columns, "ob_imbalance_5 column missing"
    assert "ob_imbalance_20" in out.columns, "ob_imbalance_20 column missing"
    assert (out["ob_imbalance"] == 0.0).all(), "ob_imbalance should be 0 when no snapshot"
    print("  create_technical_indicators: ob_imbalance, ob_imbalance_5, ob_imbalance_20 present and zero: OK")


def main():
    print("Testing Order Book Imbalance feature...")
    test_compute_orderbook_imbalance()
    test_create_technical_indicators_has_ob_columns()
    print("All checks passed.")


if __name__ == "__main__":
    main()
