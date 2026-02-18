"""
Скрипт для автоматического запуска бэктестов всех моделей по символу.

Использование:
    python run_all_backtests.py --symbol SOLUSDT --days 14
    python run_all_backtests.py --symbol BTCUSDT --days 30 --output results.csv
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import json

def find_models_for_symbol(symbol: str, models_dir: str = "ml_models") -> List[str]:
    """
    Находит все модели для указанного символа.
    
    Args:
        symbol: Торговый символ (например, SOLUSDT)
        models_dir: Директория с моделями
        
    Returns:
        Список путей к моделям
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"❌ Директория {models_dir} не найдена")
        return []
    
    # Ищем все .pkl файлы, содержащие символ
    symbol_upper = symbol.upper()
    models = []
    
    for model_file in models_path.glob("*.pkl"):
        model_name = model_file.name
        # Проверяем, содержит ли имя модели символ
        if symbol_upper in model_name.upper():
            models.append(str(model_file))
    
    # Сортируем для консистентности
    models.sort()
    return models

def run_backtest(
    model_path: str,
    symbol: str,
    days: int = 14,
    interval: str = "15m",
    balance: float = 100.0,
    risk: float = 0.02,
    leverage: int = 10
) -> Dict:
    """
    Запускает бэктест для одной модели.
    
    Returns:
        Словарь с результатами или None при ошибке
    """
    cmd = [
        sys.executable,
        "backtest_ml_strategy.py",
        "--model", model_path,
        "--symbol", symbol,
        "--days", str(days),
        "--interval", interval,
        "--balance", str(balance),
        "--risk", str(risk),
        "--leverage", str(leverage),
    ]
    
    print(f"\n{'='*80}")
    print(f"🚀 Запуск бэктеста: {Path(model_path).name}")
    print(f"{'='*80}")
    
    try:
        # Устанавливаем UTF-8 кодировку для Windows
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Заменяем проблемные символы вместо ошибки
            timeout=3600,  # 1 час максимум на модель
            env=env
        )
        
        if result.returncode != 0:
            print(f"❌ Ошибка при запуске бэктеста:")
            print(result.stderr)
            return None
        
        # Парсим результаты из вывода
        output = result.stdout
        
        # Извлекаем ключевые метрики из вывода
        metrics = {
            "model": Path(model_path).name,
            "symbol": symbol,
            "status": "completed",
            "output": output
        }
        
        # Пытаемся извлечь метрики из вывода
        try:
            for line in output.split("\n"):
                # Win Rate
                if "Win Rate:" in line and "win_rate" not in metrics:
                    try:
                        parts = line.split("Win Rate:")[1].strip()
                        win_rate_str = parts.split("%")[0].strip()
                        metrics["win_rate"] = float(win_rate_str)
                    except:
                        pass
                
                # Profit Factor
                if "Profit Factor:" in line and "profit_factor" not in metrics:
                    try:
                        parts = line.split("Profit Factor:")[1].strip()
                        metrics["profit_factor"] = float(parts.split()[0].strip())
                    except:
                        pass
                
                # Total PnL
                if ("Общий PnL:" in line or "Total PnL:" in line) and "total_pnl" not in metrics:
                    try:
                        if "$" in line:
                            pnl_str = line.split("$")[1].split("(")[0].strip()
                            metrics["total_pnl"] = float(pnl_str)
                    except:
                        pass
                
                # Total trades
                if ("Всего сделок:" in line or "Total trades:" in line) and "total_trades" not in metrics:
                    try:
                        parts = line.split(":")[1].strip()
                        trades_str = parts.split()[0]
                        metrics["total_trades"] = int(trades_str)
                    except:
                        pass
                
                # Return %
                if ("Return:" in line or "Доходность:" in line) and "return_pct" not in metrics:
                    try:
                        if "%" in line:
                            return_str = line.split("%")[0].split()[-1].strip()
                            metrics["return_pct"] = float(return_str.replace("+", "").replace("(", "").replace(")", ""))
                    except:
                        pass
                
                # Max Drawdown
                if ("Max Drawdown:" in line or "Макс. просадка:" in line) and "max_drawdown_pct" not in metrics:
                    try:
                        if "%" in line:
                            dd_str = line.split("%")[0].split()[-1].strip()
                            metrics["max_drawdown_pct"] = float(dd_str.replace("(", "").replace(")", ""))
                    except:
                        pass
        except Exception as e:
            print(f"⚠️  Не удалось извлечь метрики: {e}")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"⏱️  Бэктест превысил лимит времени (1 час)")
        return {"model": Path(model_path).name, "status": "timeout"}
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return {"model": Path(model_path).name, "status": "error", "error": str(e)}

def main():
    parser = argparse.ArgumentParser(
        description="Запуск бэктестов для всех моделей по символу",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Запустить все модели для SOLUSDT на 14 дней
  python run_all_backtests.py --symbol SOLUSDT --days 14
  
  # Запустить с сохранением результатов в CSV
  python run_all_backtests.py --symbol BTCUSDT --days 30 --output results.csv
  
  # Запустить только MTF модели
  python run_all_backtests.py --symbol ETHUSDT --days 14 --filter mtf
        """
    )
    
    parser.add_argument('--symbol', type=str, required=True,
                       help='Торговый символ (например, SOLUSDT)')
    parser.add_argument('--days', type=int, default=14,
                       help='Количество дней для бэктеста (по умолчанию: 14)')
    parser.add_argument('--interval', type=str, default='15m',
                       help='Таймфрейм (по умолчанию: 15m)')
    parser.add_argument('--balance', type=float, default=100.0,
                       help='Начальный баланс (по умолчанию: 100.0)')
    parser.add_argument('--risk', type=float, default=0.02,
                       help='Риск на сделку (по умолчанию: 0.02 = 2%%)')
    parser.add_argument('--leverage', type=int, default=10,
                       help='Плечо (по умолчанию: 10)')
    parser.add_argument('--output', type=str, default=None,
                       help='Путь к CSV файлу для сохранения результатов')
    parser.add_argument('--models-dir', type=str, default='ml_models',
                       help='Директория с моделями (по умолчанию: ml_models)')
    parser.add_argument('--filter', type=str, default=None,
                       choices=['mtf', 'non-mtf'],
                       help='Фильтр моделей: mtf (только MTF), non-mtf (только не-MTF)')
    parser.add_argument('--skip-errors', action='store_true',
                       help='Продолжать при ошибках (по умолчанию: останавливаться)')
    
    args = parser.parse_args()
    
    # Находим все модели для символа
    print(f"🔍 Поиск моделей для {args.symbol}...")
    all_models = find_models_for_symbol(args.symbol, args.models_dir)
    
    if not all_models:
        print(f"❌ Не найдено моделей для {args.symbol} в {args.models_dir}")
        return
    
    # Применяем фильтр если указан
    if args.filter == 'mtf':
        all_models = [m for m in all_models if '_mtf' in m.lower()]
        print(f"   Фильтр: только MTF модели")
    elif args.filter == 'non-mtf':
        all_models = [m for m in all_models if '_mtf' not in m.lower()]
        print(f"   Фильтр: только не-MTF модели")
    
    print(f"✅ Найдено {len(all_models)} моделей:")
    for i, model in enumerate(all_models, 1):
        print(f"   {i}. {Path(model).name}")
    
    if not all_models:
        print(f"❌ Нет моделей для запуска")
        return
    
    # Запрашиваем подтверждение
    print(f"\n⚠️  Будет запущено {len(all_models)} бэктестов")
    print(f"   Символ: {args.symbol}")
    print(f"   Дней: {args.days}")
    print(f"   Баланс: ${args.balance}")
    print(f"   Риск: {args.risk*100}%")
    
    response = input("\nПродолжить? (y/n): ")
    if response.lower() not in ['y', 'yes', 'да', 'д']:
        print("Отменено")
        return
    
    # Запускаем бэктесты
    results = []
    start_time = datetime.now()
    
    for i, model_path in enumerate(all_models, 1):
        print(f"\n{'='*80}")
        print(f"📊 Модель {i}/{len(all_models)}: {Path(model_path).name}")
        print(f"{'='*80}")
        
        result = run_backtest(
            model_path=model_path,
            symbol=args.symbol,
            days=args.days,
            interval=args.interval,
            balance=args.balance,
            risk=args.risk,
            leverage=args.leverage
        )
        
        if result:
            results.append(result)
        elif not args.skip_errors:
            print(f"\n❌ Остановка из-за ошибки. Используйте --skip-errors для продолжения.")
            break
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Выводим сводку
    print(f"\n{'='*80}")
    print(f"📊 СВОДКА РЕЗУЛЬТАТОВ")
    print(f"{'='*80}")
    print(f"Всего моделей: {len(all_models)}")
    print(f"Успешно: {len(results)}")
    print(f"Время выполнения: {elapsed/60:.1f} минут")
    
    if results:
        print(f"\n📈 Результаты по моделям:")
        for result in results:
            model_name = result.get("model", "unknown")
            status = result.get("status", "unknown")
            win_rate = result.get("win_rate", "N/A")
            profit_factor = result.get("profit_factor", "N/A")
            total_pnl = result.get("total_pnl", "N/A")
            trades = result.get("total_trades", "N/A")
            
            print(f"\n  {model_name}:")
            print(f"    Статус: {status}")
            if win_rate != "N/A":
                print(f"    Win Rate: {win_rate:.2f}%")
            if profit_factor != "N/A":
                print(f"    Profit Factor: {profit_factor:.2f}")
            if total_pnl != "N/A":
                print(f"    Total PnL: ${total_pnl:.2f}")
            if trades != "N/A":
                print(f"    Сделок: {trades}")
    
    # Сохраняем результаты в CSV если указано
    if args.output and results:
        try:
            # Создаем DataFrame из результатов
            df_results = pd.DataFrame(results)
            
            # Сохраняем в CSV
            output_path = Path(args.output)
            df_results.to_csv(output_path, index=False)
            print(f"\n✅ Результаты сохранены в {output_path}")
            
            # Также сохраняем JSON с полными данными
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"✅ Полные результаты сохранены в {json_path}")
        except Exception as e:
            print(f"⚠️  Ошибка сохранения результатов: {e}")

if __name__ == "__main__":
    main()
