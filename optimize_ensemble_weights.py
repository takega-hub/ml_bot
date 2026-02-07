"""
Скрипт для оптимизации весов ансамблей на основе Sharpe ratio.

Использование:
    python optimize_ensemble_weights.py --symbol BTCUSDT --days 30
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))

from backtest_ml_strategy import run_exact_backtest
from bot.config import load_settings


def calculate_sharpe_from_backtest(metrics) -> float:
    """Вычисляет Sharpe ratio из метрик бэктеста."""
    if metrics is None or metrics.total_trades == 0:
        return 0.0
    return metrics.sharpe_ratio if hasattr(metrics, 'sharpe_ratio') else 0.0


def optimize_ensemble_weights(
    model_paths: List[str],
    symbol: str,
    days: int = 30,
    interval: str = "15",
    initial_balance: float = 100.0,
    risk_per_trade: float = 0.02,
    leverage: int = 10,
) -> Dict[str, float]:
    """
    Оптимизирует веса ансамбля на основе Sharpe ratio.
    
    Использует оптимизацию портфеля (Markowitz-style) для максимизации Sharpe ratio.
    """
    print(f"\n🔍 Тестирование {len(model_paths)} моделей...")
    
    # Получаем метрики для каждой модели
    model_sharpes = {}
    for model_path in model_paths:
        try:
            metrics = run_exact_backtest(
                model_path=model_path,
                symbol=symbol,
                days_back=days,
                interval=interval,
                initial_balance=initial_balance,
                risk_per_trade=risk_per_trade,
                leverage=leverage,
            )
            sharpe = calculate_sharpe_from_backtest(metrics)
            model_sharpes[model_path] = sharpe
            print(f"   {Path(model_path).name}: Sharpe = {sharpe:.2f}")
        except Exception as e:
            print(f"   ⚠️  Ошибка для {Path(model_path).name}: {e}")
            model_sharpes[model_path] = 0.0
    
    # Функция для максимизации (отрицательный Sharpe, т.к. minimize)
    def objective(weights):
        weighted_sharpe = sum(w * model_sharpes[path] for w, path in zip(weights, model_paths))
        # Добавляем штраф за неравномерность весов (регуляризация)
        penalty = 0.1 * sum((w - 1/len(weights))**2 for w in weights)
        return -(weighted_sharpe - penalty)
    
    # Ограничения: веса должны быть >= 0 и сумма = 1
    constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in model_paths]
    
    # Начальные веса (равномерные)
    initial_weights = [1.0 / len(model_paths)] * len(model_paths)
    
    # Оптимизация
    print(f"\n⚙️  Оптимизация весов...")
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if not result.success:
        print(f"⚠️  Оптимизация не сошлась, используем равномерные веса")
        optimal_weights = initial_weights
    else:
        optimal_weights = result.x
    
    # Формируем результат
    weights_dict = {
        Path(path).name: float(w) for path, w in zip(model_paths, optimal_weights)
    }
    
    return weights_dict


def main():
    parser = argparse.ArgumentParser(description='Optimize ensemble weights based on Sharpe ratio')
    parser.add_argument('--symbol', type=str, default=None,
                       help='Symbol (e.g., BTCUSDT). If not specified, optimizes for all symbols')
    parser.add_argument('--symbols', type=str, default=None,
                       help='Comma-separated list of symbols (e.g., BTCUSDT,ETHUSDT,SOLUSDT)')
    parser.add_argument('--models', type=str, required=True,
                       help='Comma-separated list of model paths or pattern (e.g., "rf_*_15_mtf.pkl")')
    parser.add_argument('--days', type=int, default=30,
                       help='Days to backtest (default: 30)')
    parser.add_argument('--interval', type=str, default='15',
                       help='Interval (default: 15)')
    parser.add_argument('--balance', type=float, default=100.0,
                       help='Initial balance (default: 100.0)')
    parser.add_argument('--risk', type=float, default=0.02,
                       help='Risk per trade (default: 0.02)')
    parser.add_argument('--leverage', type=int, default=10,
                       help='Leverage (default: 10)')
    
    args = parser.parse_args()
    
    # Определяем список символов
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        # По умолчанию все 6 торговых пар
        symbols = ["SOLUSDT", "BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "BNBUSDT"]
    
    print("=" * 80)
    print("⚖️  ОПТИМИЗАЦИЯ ВЕСОВ АНСАМБЛЕЙ")
    print("=" * 80)
    print(f"Символы: {', '.join(symbols)} ({len(symbols)} символов)")
    print(f"Дни: {args.days}")
    print(f"Модели: {args.models}")
    print("=" * 80)
    
    # Парсим пути к моделям или ищем по паттерну
    if '*' in args.models or '?' in args.models:
        # Это паттерн, ищем модели
        from glob import glob
        models_dir = Path("ml_models")
        model_paths = list(models_dir.glob(args.models))
        if not model_paths:
            print(f"❌ Модели не найдены по паттерну: {args.models}")
            return
        model_paths = [str(p) for p in model_paths]
    else:
        # Это список путей
        model_paths = [p.strip() for p in args.models.split(',')]
    
    # Проверяем существование файлов
    valid_paths = []
    for path in model_paths:
        if Path(path).exists():
            valid_paths.append(path)
        else:
            print(f"⚠️  Модель не найдена: {path}")
    
    if not valid_paths:
        print("❌ Нет валидных моделей")
        return
    
    print(f"✅ Найдено {len(valid_paths)} моделей")
    
    all_results = []
    
    # Оптимизируем веса для каждого символа
    for symbol in symbols:
        print("\n" + "=" * 80)
        print(f"📊 ОПТИМИЗАЦИЯ ДЛЯ {symbol}")
        print("=" * 80)
        
        # Фильтруем модели для этого символа
        symbol_models = [p for p in valid_paths if symbol in Path(p).name]
        
        if not symbol_models:
            print(f"⚠️  Нет моделей для {symbol}, пропускаем")
            continue
        
        print(f"   Модели: {len(symbol_models)}")
        for m in symbol_models:
            print(f"      - {Path(m).name}")
        
        try:
            # Оптимизируем веса
            weights = optimize_ensemble_weights(
                model_paths=symbol_models,
                symbol=symbol,
                days=args.days,
                interval=args.interval,
                initial_balance=args.balance,
                risk_per_trade=args.risk,
                leverage=args.leverage,
            )
            
            result = {
                'symbol': symbol,
                'days': args.days,
                'weights': weights,
                'timestamp': datetime.now().isoformat(),
            }
            
            all_results.append(result)
            
            print(f"\n✅ {symbol} - Оптимальные веса:")
            for model, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                print(f"   {model}: {weight:.3f} ({weight*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ Ошибка при оптимизации {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Сохраняем все результаты
    if all_results:
        output_file = f"ensemble_weights_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print("✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА ДЛЯ ВСЕХ СИМВОЛОВ")
        print("=" * 80)
        print(f"Обработано символов: {len(all_results)}/{len(symbols)}")
        print(f"\n💾 Все результаты сохранены в: {output_file}")
    else:
        print("\n❌ Не удалось оптимизировать ни один символ")


if __name__ == "__main__":
    main()
