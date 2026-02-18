#!/usr/bin/env python3
"""
Скрипт для упрощения фильтров стратегии для увеличения использования сигналов.

Изменения:
1. Снизить базовый confidence_threshold с 0.45 до 0.40
2. Увеличить max_signals_per_day с 10 до 20
3. Ослабить stability_filter с 1.1x до 1.05x
4. Ослабить RSI фильтры с >90/<10 до >95/<5
5. Ослабить volume фильтры
"""

import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

def simplify_filters():
    """Упрощает фильтры в strategy_ml.py и config.py"""
    
    # 1. Изменения в bot/ml/strategy_ml.py
    strategy_file = Path("bot/ml/strategy_ml.py")
    if not strategy_file.exists():
        print(f"[ERROR] Файл не найден: {strategy_file}")
        return False
    
    content = strategy_file.read_text(encoding='utf-8')
    original_content = content
    
    # Изменение 1: Снизить базовый confidence_threshold
    content = content.replace(
        'def __init__(self, model_path: str, confidence_threshold: float = 0.45,',
        'def __init__(self, model_path: str, confidence_threshold: float = 0.40,'
    )
    
    # Изменение 2: Ослабить stability_filter
    content = content.replace(
        'stability_threshold = max(confidence * 1.1, min_strength * 1.1)  # УМЕНЬШЕНО с 1.3/1.5 до 1.1',
        'stability_threshold = max(confidence * 1.05, min_strength * 1.05)  # УМЕНЬШЕНО с 1.1 до 1.05'
    )
    
    # Изменение 3: Ослабить RSI фильтры
    content = content.replace(
        'if (prediction == 1 and rsi > 90) or (prediction == -1 and rsi < 10):',
        'if (prediction == 1 and rsi > 95) or (prediction == -1 and rsi < 5):'
    )
    content = content.replace(
        'extreme_threshold = confidence * 1.1  # УМЕНЬШЕНО с 1.2 до 1.1',
        'extreme_threshold = confidence * 1.05  # УМЕНЬШЕНО с 1.1 до 1.05'
    )
    
    # Изменение 4: Ослабить volume фильтры
    content = content.replace(
        'if confidence > 0.8:  # УВЕЛИЧЕНО с 0.7 до 0.8 (применяется реже)',
        'if confidence > 0.85:  # УВЕЛИЧЕНО с 0.8 до 0.85 (применяется еще реже)'
    )
    content = content.replace(
        'if volume_ratio < 0.3:  # УМЕНЬШЕНО с 0.5 до 0.3 (более мягкий порог)',
        'if volume_ratio < 0.2:  # УМЕНЬШЕНО с 0.3 до 0.2 (еще более мягкий порог)'
    )
    
    # Изменение 5: Обновить default в build_ml_signals
    content = content.replace(
        'confidence_threshold: float = 0.45,  # Снижено с 0.5 до 0.45 для увеличения использования сигналов',
        'confidence_threshold: float = 0.40,  # Снижено с 0.45 до 0.40 для увеличения использования сигналов'
    )
    
    if content != original_content:
        strategy_file.write_text(content, encoding='utf-8')
        print(f"[OK] Обновлен файл: {strategy_file}")
    else:
        print(f"[INFO] Файл {strategy_file} не требует изменений")
    
    # 2. Изменения в bot/config.py
    config_file = Path("bot/config.py")
    if not config_file.exists():
        print(f"[ERROR] Файл не найден: {config_file}")
        return False
    
    content = config_file.read_text(encoding='utf-8')
    original_content = content
    
    # Изменение 1: Снизить confidence_threshold
    content = content.replace(
        'confidence_threshold: float = 0.75  # Минимальная уверенность модели для открытия позиции (75% по умолчанию)',
        'confidence_threshold: float = 0.40  # Минимальная уверенность модели для открытия позиции (40% по умолчанию)'
    )
    
    # Изменение 2: Увеличить max_signals_per_day
    content = content.replace(
        'max_signals_per_day: int = 10  # Максимум сигналов в день',
        'max_signals_per_day: int = 20  # Максимум сигналов в день'
    )
    
    if content != original_content:
        config_file.write_text(content, encoding='utf-8')
        print(f"[OK] Обновлен файл: {config_file}")
    else:
        print(f"[INFO] Файл {config_file} не требует изменений")
    
    # 3. Изменения в retrain_ml_optimized.py
    retrain_file = Path("retrain_ml_optimized.py")
    if not retrain_file.exists():
        print(f"[WARNING] Файл не найден: {retrain_file}")
        return True
    
    content = retrain_file.read_text(encoding='utf-8')
    original_content = content
    
    # Изменение 1: Уменьшить forward_periods
    content = content.replace(
        'forward_periods=5,  # 5 * 15m = 75 минут',
        'forward_periods=4,  # 4 * 15m = 60 минут (уменьшено для большего охвата)'
    )
    
    # Изменение 2: Увеличить max_hold_periods
    content = content.replace(
        'max_hold_periods=96,  # УВЕЛИЧЕНО с 48 до 96 (24 часа)',
        'max_hold_periods=120,  # УВЕЛИЧЕНО с 96 до 120 (30 часов)'
    )
    
    if content != original_content:
        retrain_file.write_text(content, encoding='utf-8')
        print(f"[OK] Обновлен файл: {retrain_file}")
    else:
        print(f"[INFO] Файл {retrain_file} не требует изменений")
    
    print("\n" + "=" * 80)
    print("[OK] ВСЕ ИЗМЕНЕНИЯ ПРИМЕНЕНЫ")
    print("=" * 80)
    print("\nСледующие шаги:")
    print("1. Переобучить модели: python retrain_ml_optimized.py --no-mtf")
    print("2. Протестировать: python compare_ml_models.py --detailed-analysis")
    print("3. Проверить улучшение использования сигналов (должно быть 10-15% вместо 2.5%)")
    
    return True

if __name__ == "__main__":
    print("=" * 80)
    print("УПРОЩЕНИЕ ФИЛЬТРОВ СТРАТЕГИИ")
    print("=" * 80)
    print("\nЭтот скрипт упростит фильтры для увеличения использования сигналов.")
    print("Изменения:")
    print("  - confidence_threshold: 0.45 → 0.40")
    print("  - max_signals_per_day: 10 → 20")
    print("  - stability_filter: 1.1x → 1.05x")
    print("  - RSI фильтры: >90/<10 → >95/<5")
    print("  - volume фильтры: ослаблены")
    print("  - forward_periods: 5 → 4")
    print("  - max_hold_periods: 96 → 120")
    print("\n" + "=" * 80)
    
    response = input("\nПродолжить? (y/n): ")
    if response.lower() != 'y':
        print("Отменено.")
        sys.exit(0)
    
    success = simplify_filters()
    if success:
        print("\n[OK] Готово!")
    else:
        print("\n[ERROR] Произошли ошибки при применении изменений.")
        sys.exit(1)
