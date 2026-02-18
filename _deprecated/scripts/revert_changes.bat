@echo off
echo Откат изменений в файлах...
git checkout HEAD -- backtest_ml_strategy.py
git checkout HEAD -- bot/ml/strategy_ml.py
echo Готово! Изменения откачены.
