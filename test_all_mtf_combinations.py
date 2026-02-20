"""
Скрипт для тестирования всех MTF комбинаций моделей для определенного символа.

Использование:
    python test_all_mtf_combinations.py --symbol BTCUSDT
"""
import argparse
import sys
from pathlib import Path
from backtest_mtf_strategy import run_mtf_backtest_all_combinations


def main():
    parser = argparse.ArgumentParser(
        description="Тестирование всех MTF комбинаций моделей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Тестирование всех комбинаций для BTCUSDT
  python test_all_mtf_combinations.py --symbol BTCUSDT

  # С кастомными параметрами
  python test_all_mtf_combinations.py --symbol ETHUSDT --days 60
        """,
    )
    parser.add_argument("--symbol", type=str, required=True, help="Торговая пара (например, BTCUSDT)")
    parser.add_argument("--days", type=int, default=30, help="Количество дней для бэктеста (по умолчанию 30)")
    parser.add_argument("--conf-1h", type=float, default=0.50, help="Порог уверенности для 1h модели (по умолчанию 0.50)")
    parser.add_argument("--conf-15m", type=float, default=0.35, help="Порог уверенности для 15m модели (по умолчанию 0.35)")

    args = parser.parse_args()
    symbol = args.symbol.upper()

    print("=" * 80)
    print("🧪 ТЕСТИРОВАНИЕ ВСЕХ MTF КОМБИНАЦИЙ")
    print("=" * 80)
    print(f"Символ: {symbol}")
    print(f"Период: {args.days} дней")
    print(f"Пороги уверенности: 1h={args.conf_1h}, 15m={args.conf_15m}")
    print("=" * 80)
    print()

    try:
        df_results = run_mtf_backtest_all_combinations(
            symbol=symbol,
            days_back=args.days,
            initial_balance=100.0,
            risk_per_trade=0.02,
            leverage=10,
            confidence_threshold_1h=args.conf_1h,
            confidence_threshold_15m=args.conf_15m,
            alignment_mode="strict",
            require_alignment=True,
        )

        if df_results is not None and not df_results.empty:
            print("\n" + "=" * 80)
            print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
            print("=" * 80)
            print(f"Протестировано комбинаций: {len(df_results)}")
            print("Лучшая комбинация:")
            best = df_results.iloc[0]
            print(f"  1h: {best['model_1h']}")
            print(f"  15m: {best['model_15m']}")
            print(f"  PnL: {best['total_pnl_pct']:.2f}%")
            print(f"  Win Rate: {best['win_rate']:.1f}%")
            print(f"  Profit Factor: {best['profit_factor']:.2f}")
            print(f"  Sharpe Ratio: {best['sharpe_ratio']:.2f}")
            print("=" * 80)
            return 0
        else:
            print("\n❌ Не удалось получить результаты тестирования")
            return 1

    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
