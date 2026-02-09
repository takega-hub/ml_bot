"""
Скрипт для обучения ML моделей на 1-часовом таймфрейме.

Использование:
    # Обучение всех моделей для всех символов на 1h без MTF
    python train_1h_models.py --no-mtf
    
    # Обучение всех моделей для всех символов на 1h с MTF (4h, 1d)
    python train_1h_models.py --mtf
    
    # Обучение для конкретного символа
    python train_1h_models.py --symbol BTCUSDT --no-mtf
"""
import subprocess
import sys
import os
from pathlib import Path

# Символы для обучения
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "BNBUSDT"]

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Обучение ML моделей на 1-часовом таймфрейме",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Обучение всех моделей на 1h без MTF
  python train_1h_models.py --no-mtf
  
  # Обучение всех моделей на 1h с MTF
  python train_1h_models.py --mtf
  
  # Обучение для конкретного символа
  python train_1h_models.py --symbol BTCUSDT --no-mtf
        """
    )
    parser.add_argument("--symbol", type=str, help="Торговая пара (если не указано, обучаются все)")
    parser.add_argument("--mtf", action="store_true", help="Использовать MTF фичи (4h, 1d)")
    parser.add_argument("--no-mtf", action="store_true", help="НЕ использовать MTF фичи (только 1h)")
    parser.add_argument("--use-optimized-weights", action="store_true", 
                       help="Использовать оптимизированные веса ансамблей")
    
    args = parser.parse_args()
    
    # Определяем символы
    symbols = [args.symbol] if args.symbol else SYMBOLS
    
    # Формируем команду - используем sys.executable для использования того же Python
    python_exe = sys.executable
    cmd = [python_exe, "retrain_ml_optimized.py", "--interval", "60m"]
    
    if args.mtf:
        cmd.append("--mtf")
    elif args.no_mtf:
        cmd.append("--no-mtf")
    
    if args.use_optimized_weights:
        cmd.append("--use-optimized-weights")
    
    print("=" * 80)
    print("🚀 ОБУЧЕНИЕ МОДЕЛЕЙ НА 1-ЧАСОВОМ ТАЙМФРЕЙМЕ")
    print("=" * 80)
    print(f"📊 Символы: {', '.join(symbols)}")
    print(f"⏰ Таймфрейм: 1h")
    print(f"🔧 MTF: {'Включено (4h, 1d)' if args.mtf else 'Выключено' if args.no_mtf else 'По умолчанию'}")
    print("=" * 80)
    
    # Обучаем для каждого символа
    for symbol in symbols:
        print(f"\n📈 Обучение моделей для {symbol}...")
        symbol_cmd = cmd + ["--symbol", symbol]
        
        try:
            # Используем тот же Python и окружение
            env = os.environ.copy()
            result = subprocess.run(
                symbol_cmd, 
                check=True, 
                cwd=Path(__file__).parent,
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            print(f"✅ Модели для {symbol} успешно обучены")
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка при обучении моделей для {symbol}: {e}")
            if hasattr(e, 'stdout') and e.stdout:
                print(f"   Вывод: {e.stdout[-500:]}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"   Ошибки: {e.stderr[-500:]}")
            continue
        except KeyboardInterrupt:
            print(f"\n⚠️ Прервано пользователем")
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 80)
    print("\n💡 Следующие шаги:")
    print("   1. Протестировать модели:")
    print("      python compare_ml_models.py --detailed-analysis")
    print("   2. Сравнить результаты 15m и 1h моделей")
    print("   3. Выбрать лучшие модели для продакшена")

if __name__ == "__main__":
    main()
