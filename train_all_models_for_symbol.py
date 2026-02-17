"""
Скрипт для обучения всех моделей по всем таймфреймам (15m и 1h) 
с MTF фичами и без MTF фичей для определенного символа.

Использование:
    python train_all_models_for_symbol.py --symbol BTCUSDT
"""
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

def safe_print(*args, **kwargs):
    """Безопасный print для Windows"""
    try:
        print(*args, **kwargs)
        sys.stdout.flush()
    except (UnicodeEncodeError, IOError):
        text = ' '.join(str(arg) for arg in args)
        text = ''.join(c for c in text if ord(c) < 128)
        print(text, **kwargs)

def train_models(symbol: str, interval: str, use_mtf: bool):
    """Обучает модели для указанного символа, таймфрейма и режима MTF"""
    python_exe = sys.executable
    cmd = [python_exe, "retrain_ml_optimized.py", "--symbol", symbol, "--interval", interval]
    
    if use_mtf:
        cmd.append("--mtf")
        mtf_status = "С MTF"
    else:
        cmd.append("--no-mtf")
        mtf_status = "БЕЗ MTF"
    
    interval_display = "1h" if interval == "60m" else "15m"
    
    safe_print(f"\n{'=' * 80}")
    safe_print(f"📊 Обучение моделей: {symbol} | {interval_display} | {mtf_status}")
    safe_print(f"{'=' * 80}")
    safe_print(f"Команда: {' '.join(cmd)}")
    safe_print(f"{'=' * 80}\n")
    
    try:
        env = os.environ.copy()
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).parent,
            env=env,
            encoding='utf-8',
            errors='replace'
        )
        safe_print(f"\n✅ Успешно: {symbol} | {interval_display} | {mtf_status}")
        return True
    except subprocess.CalledProcessError as e:
        safe_print(f"\n❌ Ошибка: {symbol} | {interval_display} | {mtf_status}")
        safe_print(f"   Код возврата: {e.returncode}")
        if hasattr(e, 'stdout') and e.stdout:
            safe_print(f"   Вывод: {e.stdout[-500:]}")
        if hasattr(e, 'stderr') and e.stderr:
            safe_print(f"   Ошибки: {e.stderr[-500:]}")
        return False
    except KeyboardInterrupt:
        safe_print(f"\n⚠️ Прервано пользователем")
        raise
    except Exception as e:
        safe_print(f"\n❌ Неожиданная ошибка: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Обучение всех моделей по всем таймфреймам с MTF и без MTF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Обучение всех моделей для BTCUSDT (15m и 1h)
  python train_all_models_for_symbol.py --symbol BTCUSDT
  
  # Обучение только 15m моделей
  python train_all_models_for_symbol.py --symbol BTCUSDT --timeframe 15m
  
  # Обучение только 1h моделей
  python train_all_models_for_symbol.py --symbol BTCUSDT --timeframe 1h
  
  # Обучение с подробным выводом
  python train_all_models_for_symbol.py --symbol ETHUSDT --verbose
        """
    )
    parser.add_argument("--symbol", type=str, required=True, 
                       help="Торговая пара (например, BTCUSDT)")
    parser.add_argument("--timeframe", type=str, choices=["15m", "1h", "all"],
                       default="all",
                       help="Таймфрейм для обучения: 15m, 1h или all (по умолчанию all)")
    parser.add_argument("--verbose", action="store_true",
                       help="Подробный вывод")
    
    args = parser.parse_args()
    symbol = args.symbol.upper()
    timeframe = args.timeframe.lower()
    
    safe_print("=" * 80)
    if timeframe == "all":
        safe_print("🚀 ОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ ПО ВСЕМ ТАЙМФРЕЙМАМ")
        safe_print(f"⏰ Таймфреймы: 15m, 1h")
    else:
        safe_print(f"🚀 ОБУЧЕНИЕ МОДЕЛЕЙ ДЛЯ ТАЙМФРЕЙМА {timeframe.upper()}")
        safe_print(f"⏰ Таймфрейм: {timeframe}")
    safe_print("=" * 80)
    safe_print(f"📊 Символ: {symbol}")
    safe_print(f"🔧 Режимы: БЕЗ MTF, С MTF")
    safe_print(f"📅 Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print("=" * 80)
    
    # Конфигурация обучения: (интервал, использовать_MTF)
    if timeframe == "all":
        training_configs = [
            ("15m", False),  # 15m без MTF
            ("15m", True),   # 15m с MTF
            ("60m", False),  # 1h без MTF
            ("60m", True),   # 1h с MTF
        ]
    elif timeframe == "15m":
        training_configs = [
            ("15m", False),  # 15m без MTF
            ("15m", True),   # 15m с MTF
        ]
    else:  # timeframe == "1h"
        training_configs = [
            ("60m", False),  # 1h без MTF
            ("60m", True),   # 1h с MTF
        ]
    
    results = {}
    start_time = datetime.now()
    
    for i, (interval, use_mtf) in enumerate(training_configs, 1):
        interval_display = "1h" if interval == "60m" else "15m"
        mtf_status = "С MTF" if use_mtf else "БЕЗ MTF"
        config_name = f"{interval_display} {mtf_status}"
        
        safe_print(f"\n[{i}/{len(training_configs)}] {config_name}")
        safe_print("-" * 80)
        
        success = train_models(symbol, interval, use_mtf)
        results[config_name] = success
        
        if not success:
            safe_print(f"\n⚠️ Предупреждение: Обучение {config_name} завершилось с ошибкой")
            try:
                response = input("Продолжить обучение остальных конфигураций? (y/n): ")
                if response.lower() != 'y':
                    safe_print("\n❌ Обучение прервано пользователем")
                    break
            except (EOFError, KeyboardInterrupt):
                safe_print("\n❌ Обучение прервано")
                break
    
    # Итоговая сводка
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60  # минуты
    
    safe_print("\n" + "=" * 80)
    safe_print("📊 ИТОГОВАЯ СВОДКА")
    safe_print("=" * 80)
    safe_print(f"Символ: {symbol}")
    safe_print(f"Время выполнения: {duration:.1f} минут")
    safe_print(f"\nРезультаты:")
    
    for config_name, success in results.items():
        status = "✅ Успешно" if success else "❌ Ошибка"
        safe_print(f"  {config_name:20s}: {status}")
    
    successful = sum(1 for s in results.values() if s)
    total = len(results)
    
    safe_print(f"\nУспешно: {successful}/{total}")
    
    if successful == total:
        safe_print("\n✅ ВСЕ МОДЕЛИ УСПЕШНО ОБУЧЕНЫ!")
        safe_print("\n💡 Следующие шаги:")
        safe_print("   1. Протестировать модели:")
        safe_print(f"      python compare_ml_models.py --symbols {symbol} --detailed-analysis")
        safe_print("   2. Сравнить результаты разных конфигураций")
        safe_print("   3. Выбрать лучшие модели для продакшена")
    else:
        safe_print(f"\n⚠️ Некоторые модели не были обучены ({total - successful} ошибок)")
        safe_print("   Проверьте логи выше для деталей")
    
    safe_print("=" * 80)

if __name__ == "__main__":
    main()
