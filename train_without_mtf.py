#!/usr/bin/env python3
"""
Скрипт для обучения моделей БЕЗ MTF фичей и проверки прибыльности.
"""
import os
import subprocess
import sys
from pathlib import Path

# Устанавливаем переменную окружения для отключения MTF
os.environ['ML_MTF_ENABLED'] = '0'

print("=" * 80)
print("🚀 ОБУЧЕНИЕ МОДЕЛЕЙ БЕЗ MTF ФИЧЕЙ")
print("=" * 80)
print(f"ML_MTF_ENABLED = {os.environ.get('ML_MTF_ENABLED', 'не установлено')}")
print()

# Запускаем обучение
print("📚 Запуск обучения моделей...")
print("=" * 80)

try:
    result = subprocess.run(
        [sys.executable, 'retrain_ml_optimized.py'],
        env=os.environ.copy(),
        encoding='utf-8',
        errors='replace'
    )
    
    if result.returncode == 0:
        print("\n✅ Обучение завершено успешно!")
        print("\n" + "=" * 80)
        print("📊 СЛЕДУЮЩИЙ ШАГ: Проверка прибыльности")
        print("=" * 80)
        print("\nЗапустите бэктесты для проверки прибыльности:")
        print("\nДля одного символа:")
        print("  python backtest_ml_strategy.py --model ml_models/xgb_SOLUSDT_15_15m.pkl --symbol SOLUSDT --days 14")
        print("\nДля всех моделей одного символа:")
        print("  python run_all_backtests.py --symbol SOLUSDT --days 14")
        print("\nДля сравнения всех моделей:")
        print("  python compare_ml_models.py --days 14")
    else:
        print(f"\n❌ Ошибка при обучении (код: {result.returncode})")
        sys.exit(1)
        
except KeyboardInterrupt:
    print("\n⚠️ Обучение прервано пользователем")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
