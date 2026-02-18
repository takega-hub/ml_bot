"""
Удобная обертка для ручного запуска оптимизации стратегий.

Использование:
    python optimize_strategies.py --now  # Запустить немедленно
    python optimize_strategies.py --full  # Полный цикл (обучение + сравнение + MTF)
    python optimize_strategies.py --quick  # Только сравнение и MTF тестирование
    python optimize_strategies.py --symbols BTCUSDT,ETHUSDT  # Конкретные символы
"""
import argparse
import sys
from auto_strategy_optimizer import StrategyOptimizer
from bot.state import BotState


def main():
    parser = argparse.ArgumentParser(
        description="Ручной запуск оптимизации стратегий",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Быстрый запуск (без обучения)
  python optimize_strategies.py --quick
  
  # Полный цикл оптимизации
  python optimize_strategies.py --full
  
  # Для конкретных символов
  python optimize_strategies.py --now --symbols BTCUSDT,ETHUSDT
  
  # Только MTF тестирование (используя существующие модели)
  python optimize_strategies.py --skip-training --skip-comparison
        """
    )
    
    parser.add_argument("--now", action="store_true",
                       help="Запустить оптимизацию немедленно")
    parser.add_argument("--full", action="store_true",
                       help="Полный цикл: обучение + сравнение + MTF тестирование")
    parser.add_argument("--quick", action="store_true",
                       help="Быстрый режим: только сравнение существующих моделей и MTF тестирование")
    parser.add_argument("--symbols", type=str, default=None,
                       help="Список символов через запятую (по умолчанию из state.active_symbols)")
    parser.add_argument("--days", type=int, default=30,
                       help="Количество дней для бэктеста (по умолчанию 30)")
    parser.add_argument("--output-dir", type=str, default="optimization_results",
                       help="Директория для сохранения результатов")
    
    args = parser.parse_args()
    
    # Если не указан ни один режим, показываем помощь
    if not (args.now or args.full or args.quick):
        parser.print_help()
        print("\n⚠️  Укажите режим: --now, --full или --quick")
        sys.exit(1)
    
    # Определяем параметры в зависимости от режима
    if args.full:
        skip_training = False
        skip_comparison = False
        skip_mtf_testing = False
    elif args.quick:
        skip_training = True
        skip_comparison = False
        skip_mtf_testing = False
    else:  # --now
        # По умолчанию пропускаем обучение (быстрее)
        skip_training = True
        skip_comparison = False
        skip_mtf_testing = False
    
    # Определяем символы
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        # Загружаем из state
        state = BotState()
        symbols = state.active_symbols
        if not symbols:
            symbols = ["BTCUSDT"]  # Fallback
    
    print("=" * 80)
    print("🚀 РУЧНОЙ ЗАПУСК ОПТИМИЗАЦИИ СТРАТЕГИЙ")
    print("=" * 80)
    print(f"Символы: {', '.join(symbols)}")
    print(f"Дни бэктеста: {args.days}")
    print(f"Режим: {'Полный цикл' if args.full else 'Быстрый' if args.quick else 'Стандартный'}")
    print(f"Пропуск обучения: {skip_training}")
    print(f"Пропуск сравнения: {skip_comparison}")
    print(f"Пропуск MTF тестирования: {skip_mtf_testing}")
    print("=" * 80)
    print()
    
    # Создаем оптимизатор
    optimizer = StrategyOptimizer(
        symbols=symbols,
        days=args.days,
        output_dir=args.output_dir,
        skip_training=skip_training,
        skip_comparison=skip_comparison,
        skip_mtf_testing=skip_mtf_testing,
    )
    
    # Запускаем оптимизацию
    try:
        optimizer.run()
        print("\n✅ Оптимизация завершена успешно!")
    except KeyboardInterrupt:
        print("\n⚠️  Оптимизация прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
