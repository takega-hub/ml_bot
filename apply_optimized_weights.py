#!/usr/bin/env python3
"""
Скрипт для применения оптимизированных весов к ансамблям при переобучении.

Использование:
    python apply_optimized_weights.py --weights-file ensemble_weights_all_20260209_022646.json
    python retrain_ml_optimized.py --use-optimized-weights
"""

import argparse
import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

# Устанавливаем UTF-8 кодировку для Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))


def find_latest_weights_file(pattern: str = "ensemble_weights_all_*.json") -> Optional[Path]:
    """Находит последний файл с оптимизированными весами."""
    weights_files = sorted(
        Path(".").glob(pattern),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True
    )
    return weights_files[0] if weights_files else None


def parse_weights_from_output(output_text: str) -> Optional[Dict[str, float]]:
    """
    Парсит веса из текстового вывода optimize_ensemble_weights.py.
    
    Ищет строки вида:
        rf_SOLUSDT_15_15m.pkl: 1.000 (100.0%)
        xgb_SOLUSDT_15_15m.pkl: 0.000 (0.0%)
    """
    weights = {}
    # Ищем паттерн: "имя_модели.pkl: вес (процент%)"
    pattern = r'([a-zA-Z_]+_[A-Z]+USDT_\d+[_\w]*\.pkl):\s+([\d.]+)\s+\([\d.]+%\)'
    matches = re.findall(pattern, output_text)
    
    for model_name, weight_str in matches:
        try:
            weight = float(weight_str)
            weights[model_name] = weight
        except ValueError:
            continue
    
    return weights if weights else None


def load_optimized_weights(weights_file: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """
    Загружает оптимизированные веса из JSON файла.
    
    Returns:
        Словарь вида: {symbol: {model_name: weight}}
    """
    if weights_file is None:
        weights_file = find_latest_weights_file()
    
    if weights_file is None or not weights_file.exists():
        print(f"[WARNING] Файл с весами не найден: {weights_file}")
        return {}
    
    print(f"[OK] Загрузка весов из: {weights_file}")
    
    with open(weights_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    weights_dict = {}
    
    # Обрабатываем разные форматы JSON
    if isinstance(data, list):
        # Формат: [{"symbol": "SOLUSDT", "weights": {...}}, ...]
        for item in data:
            symbol = item.get("symbol", "").upper()
            weights = item.get("weights", {})
            if symbol and weights:
                weights_dict[symbol] = weights
    elif isinstance(data, dict):
        # Формат: {"SOLUSDT": {"status": "completed", "output": "..."}, ...}
        for symbol, item in data.items():
            symbol = symbol.upper()
            if isinstance(item, dict):
                # Пробуем извлечь веса из поля "weights"
                if "weights" in item:
                    weights_dict[symbol] = item["weights"]
                # Или парсим из поля "output"
                elif "output" in item:
                    weights = parse_weights_from_output(item["output"])
                    if weights:
                        weights_dict[symbol] = weights
    
    print(f"[OK] Загружены веса для {len(weights_dict)} символов: {', '.join(weights_dict.keys())}")
    return weights_dict


def get_weights_for_symbol(
    symbol: str,
    weights_dict: Dict[str, Dict[str, float]],
    mode_suffix: str = "15m"
) -> Optional[Tuple[float, float]]:
    """
    Извлекает веса RF и XGB для указанного символа.
    
    Args:
        symbol: Торговая пара (например, "SOLUSDT")
        weights_dict: Словарь с весами для всех символов
        mode_suffix: Суффикс модели ("15m" или "mtf")
    
    Returns:
        Кортеж (rf_weight, xgb_weight) или None если не найдено
    """
    symbol = symbol.upper()
    if symbol not in weights_dict:
        return None
    
    symbol_weights = weights_dict[symbol]
    
    # Ищем веса для RF и XGB моделей
    rf_weight = None
    xgb_weight = None
    
    for model_name, weight in symbol_weights.items():
        # Проверяем суффикс модели
        if mode_suffix not in model_name:
            continue
        
        if model_name.startswith("rf_") and symbol in model_name:
            rf_weight = weight
        elif model_name.startswith("xgb_") and symbol in model_name:
            xgb_weight = weight
    
    if rf_weight is not None and xgb_weight is not None:
        # Нормализуем веса (на случай если они не в сумме 1.0)
        total = rf_weight + xgb_weight
        if total > 0:
            rf_weight = rf_weight / total
            xgb_weight = xgb_weight / total
            return (rf_weight, xgb_weight)
    
    return None


def save_weights_config(weights_dict: Dict[str, Dict[str, float]], output_file: str = "optimized_weights_config.json"):
    """Сохраняет веса в удобном формате для использования в retrain_ml_optimized.py."""
    config = {
        "weights": weights_dict,
        "timestamp": datetime.now().isoformat(),
        "description": "Оптимизированные веса ансамблей на основе Sharpe ratio из бэктеста"
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Конфигурация весов сохранена в: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Загружает и сохраняет оптимизированные веса ансамблей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Использовать последний файл с весами
  python apply_optimized_weights.py

  # Указать конкретный файл
  python apply_optimized_weights.py --weights-file ensemble_weights_all_20260209_022646.json

  # Сохранить в другой файл
  python apply_optimized_weights.py --output optimized_weights_config.json
        """
    )
    
    parser.add_argument(
        "--weights-file",
        type=str,
        help="Путь к JSON файлу с оптимизированными весами (по умолчанию ищет последний)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="optimized_weights_config.json",
        help="Имя выходного файла конфигурации (по умолчанию: optimized_weights_config.json)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Показать список доступных файлов с весами"
    )
    
    args = parser.parse_args()
    
    if args.list:
        weights_files = sorted(
            Path(".").glob("ensemble_weights_all_*.json"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True
        )
        print(f"\nДоступные файлы с весами ({len(weights_files)} файлов):")
        for i, f in enumerate(weights_files[:10], 1):  # Показываем последние 10
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {i}. {f.name} ({mtime})")
        if len(weights_files) > 10:
            print(f"  ... и еще {len(weights_files) - 10} файлов")
        return
    
    # Загружаем веса
    weights_file = Path(args.weights_file) if args.weights_file else None
    weights_dict = load_optimized_weights(weights_file)
    
    if not weights_dict:
        print("[ERROR] Не удалось загрузить веса. Используйте --list для просмотра доступных файлов.")
        return
    
    # Показываем загруженные веса
    print("\n" + "=" * 80)
    print("ЗАГРУЖЕННЫЕ ВЕСА:")
    print("=" * 80)
    for symbol, weights in weights_dict.items():
        print(f"\n{symbol}:")
        for model_name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model_name}: {weight:.6f} ({weight*100:.2f}%)")
    
    # Сохраняем конфигурацию
    config_file = save_weights_config(weights_dict, args.output)
    
    print("\n" + "=" * 80)
    print("[OK] ГОТОВО")
    print("=" * 80)
    print(f"\nКонфигурация сохранена в: {config_file}")
    print("\nДля применения весов при переобучении:")
    print("  1. Используйте флаг --use-optimized-weights в retrain_ml_optimized.py")
    print("  2. Или модифицируйте retrain_ml_optimized.py для автоматической загрузки весов")


if __name__ == "__main__":
    main()
