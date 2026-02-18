#!/usr/bin/env python3
"""
Автоматизированный скрипт для оптимизации лучших моделей.

Оптимизирует гиперпараметры для RF и XGBoost моделей,
и веса для ансамблей на основе результатов бэктеста.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import subprocess

# Устанавливаем UTF-8 кодировку для Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import load_settings


# Лучшие модели из последнего бэктеста (2026-02-08)
BEST_MODELS = {
    "SOLUSDT": {
        "rf": "rf_SOLUSDT_15_15m.pkl",
        "xgb": "xgb_SOLUSDT_15_15m.pkl",
        "ensemble": "ensemble_SOLUSDT_15_15m.pkl",
        "triple_ensemble": "triple_ensemble_SOLUSDT_15_15m.pkl",
    },
    "ADAUSDT": {
        "rf": "rf_ADAUSDT_15_15m.pkl",
        "xgb": "xgb_ADAUSDT_15_15m.pkl",
        "ensemble": "ensemble_ADAUSDT_15_15m.pkl",
    },
    "ETHUSDT": {
        "quad_ensemble": "quad_ensemble_ETHUSDT_15_15m.pkl",
    },
    "BTCUSDT": {
        "ensemble": "ensemble_BTCUSDT_15_15m.pkl",
    },
    "BNBUSDT": {
        "ensemble": "ensemble_BNBUSDT_15_15m.pkl",
    },
}


def find_model_path(model_name: str, models_dir: Path = Path("ml_models")) -> str:
    """Находит путь к модели по имени."""
    model_path = models_dir / model_name
    if model_path.exists():
        return str(model_path)
    
    # Пробуем найти по паттерну
    pattern = model_name.replace(".pkl", "*.pkl")
    matches = list(models_dir.glob(pattern))
    if matches:
        return str(matches[0])
    
    raise FileNotFoundError(f"Модель {model_name} не найдена в {models_dir}")


def optimize_hyperparameters(
    model_type: str,
    symbol: str,
    models_dir: Path = Path("ml_models")
) -> Dict:
    """
    Оптимизирует гиперпараметры для модели.
    
    Args:
        model_type: Тип модели (rf, xgb, lgb)
        symbol: Торговая пара (SOLUSDT, ADAUSDT и т.д.)
        models_dir: Директория с моделями
    
    Returns:
        Словарь с результатами оптимизации
    """
    print(f"\n{'='*80}")
    print(f"ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ: {model_type.upper()} для {symbol}")
    print(f"{'='*80}")
    
    # Запускаем скрипт оптимизации
    cmd = [
        sys.executable,
        "optimize_hyperparameters.py",
        "--model", model_type,
        "--symbol", symbol,
        "--interval", "15",
    ]
    
    print(f"Команда: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    
    if result.returncode != 0:
        print(f"[ERROR] Ошибка при оптимизации {model_type} для {symbol}:")
        print(result.stderr)
        return None
    
    print(result.stdout)
    
    # Парсим результаты (если они сохранены в файл)
    results_file = Path(f"hyperparameters_optimization_{datetime.now().strftime('%Y%m%d')}.json")
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results
    
    return {"status": "completed", "output": result.stdout}


def optimize_ensemble_weights(
    symbol: str,
    ensemble_models: List[str],
    models_dir: Path = Path("ml_models"),
    days: int = 30
) -> Dict:
    """
    Оптимизирует веса ансамбля.
    
    Args:
        symbol: Торговая пара
        ensemble_models: Список имен моделей для ансамбля (только имена файлов)
        models_dir: Директория с моделями
        days: Количество дней для бэктеста
    
    Returns:
        Словарь с оптимизированными весами
    """
    print(f"\n{'='*80}")
    print(f"ОПТИМИЗАЦИЯ ВЕСОВ АНСАМБЛЯ для {symbol}")
    print(f"{'='*80}")
    
    # Находим пути к моделям
    model_paths = []
    for model_name in ensemble_models:
        try:
            path = find_model_path(model_name, models_dir)
            model_paths.append(path)
            print(f"   [OK] Найдена модель: {model_name}")
        except FileNotFoundError as e:
            print(f"   [WARNING] {e}")
            continue
    
    if not model_paths:
        print(f"[ERROR] Не найдено моделей для ансамбля {symbol}")
        return None
    
    if len(model_paths) < 2:
        print(f"[WARNING] Недостаточно моделей для ансамбля (нужно минимум 2, найдено: {len(model_paths)})")
        return None
    
    # Запускаем скрипт оптимизации весов
    # Объединяем пути в одну строку через запятую
    models_str = ",".join(model_paths)
    cmd = [
        sys.executable,
        "optimize_ensemble_weights.py",
        "--symbol", symbol,
        "--days", str(days),
        "--models", models_str
    ]
    
    print(f"Команда: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    
    if result.returncode != 0:
        print(f"[ERROR] Ошибка при оптимизации весов для {symbol}:")
        print(result.stderr)
        return None
    
    print(result.stdout)
    
    # Парсим результаты (если они сохранены в файл)
    # Ищем последний созданный файл с весами
    results_files = sorted(Path(".").glob(f"ensemble_weights_{symbol}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if results_files:
        with open(results_files[0], 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    
    return {"status": "completed", "output": result.stdout}


def main():
    parser = argparse.ArgumentParser(
        description='Оптимизация лучших моделей: гиперпараметры и веса ансамблей',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Оптимизировать все лучшие модели
  python optimize_best_models.py --all
  
  # Оптимизировать только RF модели
  python optimize_best_models.py --model-type rf
  
  # Оптимизировать только для SOLUSDT
  python optimize_best_models.py --symbol SOLUSDT
  
  # Оптимизировать только веса ансамблей
  python optimize_best_models.py --ensemble-only
  
  # Оптимизировать только гиперпараметры
  python optimize_best_models.py --hyperparameters-only
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Оптимизировать все лучшие модели')
    parser.add_argument('--model-type', type=str, choices=['rf', 'xgb', 'lgb'],
                       help='Тип модели для оптимизации (rf, xgb, lgb)')
    parser.add_argument('--symbol', type=str,
                       help='Торговая пара для оптимизации (SOLUSDT, ADAUSDT и т.д.)')
    parser.add_argument('--ensemble-only', action='store_true',
                       help='Оптимизировать только веса ансамблей')
    parser.add_argument('--hyperparameters-only', action='store_true',
                       help='Оптимизировать только гиперпараметры')
    parser.add_argument('--days', type=int, default=30,
                       help='Количество дней для бэктеста при оптимизации весов (default: 30)')
    parser.add_argument('--models-dir', type=str, default='ml_models',
                       help='Директория с моделями (default: ml_models)')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"[ERROR] Директория {models_dir} не существует!")
        return
    
    print("=" * 80)
    print("ОПТИМИЗАЦИЯ ЛУЧШИХ МОДЕЛЕЙ")
    print("=" * 80)
    print(f"Директория моделей: {models_dir}")
    print(f"Дней для бэктеста: {args.days}")
    print("=" * 80)
    
    results = {
        "hyperparameters": {},
        "ensemble_weights": {},
        "timestamp": datetime.now().isoformat(),
    }
    
    # Определяем, что оптимизировать
    optimize_hyperparams = not args.ensemble_only
    optimize_ensembles = not args.hyperparameters_only
    
    # Оптимизация гиперпараметров
    if optimize_hyperparams:
        print("\n" + "=" * 80)
        print("ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ")
        print("=" * 80)
        
        for symbol, models in BEST_MODELS.items():
            if args.symbol and symbol != args.symbol.upper():
                continue
            
            # RF модели
            if 'rf' in models and (args.all or args.model_type == 'rf' or not args.model_type):
                try:
                    result = optimize_hyperparameters('rf', symbol, models_dir)
                    if result:
                        results["hyperparameters"][f"rf_{symbol}"] = result
                except Exception as e:
                    print(f"[ERROR] Ошибка при оптимизации RF для {symbol}: {e}")
            
            # XGBoost модели
            if 'xgb' in models and (args.all or args.model_type == 'xgb' or not args.model_type):
                try:
                    result = optimize_hyperparameters('xgb', symbol, models_dir)
                    if result:
                        results["hyperparameters"][f"xgb_{symbol}"] = result
                except Exception as e:
                    print(f"[ERROR] Ошибка при оптимизации XGBoost для {symbol}: {e}")
    
    # Оптимизация весов ансамблей
    if optimize_ensembles:
        print("\n" + "=" * 80)
        print("ОПТИМИЗАЦИЯ ВЕСОВ АНСАМБЛЕЙ")
        print("=" * 80)
        
        for symbol, models in BEST_MODELS.items():
            if args.symbol and symbol != args.symbol.upper():
                continue
            
            # Ensemble модели - находим компоненты (RF, XGBoost, LightGBM)
            # Для оптимизации весов нужны компоненты, а не сам ансамбль
            ensemble_components = []
            
            # Ищем компоненты ансамбля (rf_*, xgb_*, lgb_*)
            component_patterns = [
                f"rf_{symbol}_15_15m.pkl",
                f"xgb_{symbol}_15_15m.pkl",
                f"lgb_{symbol}_15_15m.pkl",
            ]
            
            for pattern in component_patterns:
                try:
                    path = find_model_path(pattern, models_dir)
                    ensemble_components.append(path)
                    print(f"   [OK] Найден компонент: {pattern}")
                except FileNotFoundError:
                    # Пробуем найти без суффикса 15m
                    pattern_alt = pattern.replace("_15_15m.pkl", "_15.pkl")
                    try:
                        path = find_model_path(pattern_alt, models_dir)
                        ensemble_components.append(path)
                        print(f"   [OK] Найден компонент: {pattern_alt}")
                    except FileNotFoundError:
                        print(f"   [WARNING] Компонент не найден: {pattern}")
            
            if len(ensemble_components) >= 2:  # Нужно минимум 2 компонента
                try:
                    result = optimize_ensemble_weights(
                        symbol,
                        [Path(p).name for p in ensemble_components],  # Передаем только имена
                        models_dir,
                        args.days
                    )
                    if result:
                        results["ensemble_weights"][symbol] = result
                except Exception as e:
                    print(f"[ERROR] Ошибка при оптимизации весов для {symbol}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"   [WARNING] Недостаточно компонентов для ансамбля {symbol} (найдено: {len(ensemble_components)})")
    
    # Сохраняем результаты
    results_file = Path(f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("[OK] ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
    print("=" * 80)
    print(f"Результаты сохранены в: {results_file}")
    print("\nСледующие шаги:")
    print("1. Проверьте результаты оптимизации в JSON файле")
    print("2. Переобучите модели с оптимизированными параметрами")
    print("3. Протестируйте оптимизированные модели")
    print("4. Сравните результаты с исходными моделями")


if __name__ == "__main__":
    main()
