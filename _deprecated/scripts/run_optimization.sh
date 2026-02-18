#!/bin/bash
# Скрипт для запуска оптимизации лучших моделей

echo "=========================================="
echo "🚀 ОПТИМИЗАЦИЯ ЛУЧШИХ МОДЕЛЕЙ"
echo "=========================================="
echo ""

# Оптимизация гиперпараметров для RF моделей
echo "📊 Этап 1: Оптимизация гиперпараметров RF"
echo "----------------------------------------"
python optimize_hyperparameters.py --model rf --symbols SOLUSDT,ADAUSDT --interval 15

# Оптимизация гиперпараметров для XGBoost моделей
echo ""
echo "📊 Этап 2: Оптимизация гиперпараметров XGBoost"
echo "----------------------------------------"
python optimize_hyperparameters.py --model xgb --symbols SOLUSDT --interval 15

# Оптимизация весов ансамблей
echo ""
echo "⚖️  Этап 3: Оптимизация весов ансамблей"
echo "----------------------------------------"

# SOLUSDT ensemble
echo "   - SOLUSDT ensemble..."
python optimize_ensemble_weights.py \
    --symbol SOLUSDT \
    --days 30 \
    --models "ensemble_SOLUSDT_15_15m.pkl,triple_ensemble_SOLUSDT_15_15m.pkl"

# ADAUSDT ensemble
echo "   - ADAUSDT ensemble..."
python optimize_ensemble_weights.py \
    --symbol ADAUSDT \
    --days 30 \
    --models "ensemble_ADAUSDT_15_15m.pkl"

# ETHUSDT quad_ensemble
echo "   - ETHUSDT quad_ensemble..."
python optimize_ensemble_weights.py \
    --symbol ETHUSDT \
    --days 30 \
    --models "quad_ensemble_ETHUSDT_15_15m.pkl"

# BTCUSDT ensemble
echo "   - BTCUSDT ensemble..."
python optimize_ensemble_weights.py \
    --symbol BTCUSDT \
    --days 30 \
    --models "ensemble_BTCUSDT_15_15m.pkl"

# BNBUSDT ensemble
echo "   - BNBUSDT ensemble..."
python optimize_ensemble_weights.py \
    --symbol BNBUSDT \
    --days 30 \
    --models "ensemble_BNBUSDT_15_15m.pkl"

echo ""
echo "✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА"
echo "=========================================="
echo ""
echo "📋 Следующие шаги:"
echo "1. Проверьте результаты в JSON файлах"
echo "2. Переобучите модели с оптимизированными параметрами"
echo "3. Протестируйте оптимизированные модели"
