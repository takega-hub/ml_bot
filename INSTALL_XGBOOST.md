# Установка XGBoost

## Проблема

При обучении моделей возникает ошибка:
```
ModuleNotFoundError: No module named 'xgboost'
```

## Решение

Установите XGBoost с помощью pip:

```bash
pip install xgboost
```

Или если используете conda:

```bash
conda install -c conda-forge xgboost
```

## Проверка установки

После установки проверьте:

```python
python -c "import xgboost; print(xgboost.__version__)"
```

Должна вывестись версия XGBoost (например, `2.0.0`).

## Альтернатива

Если по какой-то причине не можете установить XGBoost, вы можете обучать только Random Forest модели:

```bash
# Обучение только RF моделей (без XGBoost)
python retrain_ml_optimized.py --interval 60m --no-mtf --symbol BTCUSDT
```

XGBoost модели будут пропущены, но RF модели будут обучены.

## Примечание

XGBoost является обязательной зависимостью для:
- XGBoost моделей (xgb_*.pkl)
- Ensemble моделей (ensemble_*.pkl)
- TripleEnsemble моделей (triple_ensemble_*.pkl)
- QuadEnsemble моделей (quad_ensemble_*.pkl)

Random Forest модели (rf_*.pkl) не требуют XGBoost.
