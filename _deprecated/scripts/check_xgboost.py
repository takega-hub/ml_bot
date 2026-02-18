"""
Скрипт для проверки доступности XGBoost в текущем окружении.
"""
import sys

print("=" * 80)
print("ПРОВЕРКА XGBOOST")
print("=" * 80)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print()

# Проверка 1: Прямой импорт
print("1. Прямой импорт xgboost:")
try:
    import xgboost as xgb
    print(f"   ✅ XGBoost импортирован успешно")
    print(f"   Версия: {xgb.__version__}")
except ImportError as e:
    print(f"   ❌ Ошибка импорта: {e}")
    print(f"   XGBoost не установлен в текущем окружении")
except Exception as e:
    print(f"   ❌ Неожиданная ошибка: {e}")

print()

# Проверка 2: Через model_trainer
print("2. Импорт через bot.ml.model_trainer:")
try:
    from bot.ml.model_trainer import XGBOOST_AVAILABLE, xgb
    print(f"   XGBOOST_AVAILABLE: {XGBOOST_AVAILABLE}")
    if XGBOOST_AVAILABLE:
        print(f"   ✅ XGBoost доступен через model_trainer")
        if xgb is not None:
            print(f"   Версия: {xgb.__version__}")
    else:
        print(f"   ❌ XGBoost недоступен через model_trainer")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("РЕКОМЕНДАЦИИ:")
print("=" * 80)

# Проверяем, установлен ли xgboost
try:
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pip", "show", "xgboost"], 
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("✅ XGBoost найден через pip show")
        # Извлекаем Location
        for line in result.stdout.split('\n'):
            if line.startswith('Location:'):
                print(f"   {line}")
    else:
        print("❌ XGBoost не найден через pip show")
        print("   Установите: pip install xgboost")
except Exception as e:
    print(f"⚠️  Не удалось проверить через pip: {e}")

print()
print("Если XGBoost не установлен, выполните:")
print("  pip install xgboost")
print("=" * 80)
