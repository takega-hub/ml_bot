"""
Сравнение двух ML моделей на одном бэктесте (одинаковые параметры и период).

Использование:
  python scripts/compare_two_models.py ml_models/rf_BTCUSDT_15_15m.pkl ml_models/rf_BTCUSDT_15_15m_ob.pkl --symbol BTCUSDT --days 30

  # С pullback (как в live)
  python scripts/compare_two_models.py ml_models/rf_BTCUSDT_15_15m.pkl ml_models/rf_BTCUSDT_15_15m_ob.pkl --symbol BTCUSDT --days 30 --pullback
"""

import argparse
import os
import sys
from pathlib import Path

# Корень проекта
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_ml_strategy import run_exact_backtest, BacktestMetrics


def load_settings_safe():
    try:
        from bot.config import load_settings
        s = load_settings()
        atr_filter = getattr(s.ml_strategy, "atr_filter_enabled", False)
        atr_min = getattr(s.ml_strategy, "atr_min_pct", 0.3)
        atr_max = getattr(s.ml_strategy, "atr_max_pct", 2.0)
        return atr_filter, atr_min, atr_max
    except Exception:
        return False, 0.3, 2.0


def main():
    parser = argparse.ArgumentParser(description="Сравнение двух ML моделей на одном бэктесте")
    parser.add_argument("model_baseline", type=str, help="Путь к базовой модели (обычная)")
    parser.add_argument("model_new", type=str, help="Путь к новой модели (например с ob)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--interval", type=str, default="15m")
    parser.add_argument("--balance", type=float, default=100.0)
    parser.add_argument("--risk", type=float, default=0.02)
    parser.add_argument("--pullback", action="store_true", help="Включить вход по откату")
    parser.add_argument("--pullback-ema-period", type=int, default=9)
    parser.add_argument("--pullback-pct", type=float, default=0.3)
    parser.add_argument("--pullback-max-bars", type=int, default=3)
    args = parser.parse_args()

    atr_filter, atr_min, atr_max = load_settings_safe()

    def run_one(path: str, label: str) -> BacktestMetrics:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / path
        if not p.exists():
            print(f"[ERROR] Файл не найден: {p}")
            sys.exit(1)
        print(f"\n--- Backtest: {label} ({p.name}) ---")
        return run_exact_backtest(
            model_path=str(p),
            symbol=args.symbol,
            days_back=args.days,
            interval=args.interval,
            initial_balance=args.balance,
            risk_per_trade=args.risk,
            leverage=10,
            atr_filter_enabled=atr_filter,
            atr_min_pct=atr_min,
            atr_max_pct=atr_max,
            partial_tp_enabled=False,
            partial_tp_pct=0.015,
            trailing_activation_pct=0.03,
            trailing_distance_pct=0.02,
            pullback_enabled=args.pullback,
            pullback_ema_period=args.pullback_ema_period,
            pullback_pct=args.pullback_pct / 100.0 if args.pullback_pct >= 1.0 else args.pullback_pct,
            pullback_max_bars=args.pullback_max_bars,
        )

    m1 = run_one(args.model_baseline, "Базовая (обычная)")
    m2 = run_one(args.model_new, "Новая (с новыми фичами)")

    # Сводная таблица
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 70)
    print(f"  {'Метрика':<28}  {'Базовая':>12}  {'Новая':>12}  {'Разница':>12}")
    print("  " + "-" * 66)

    def row(name: str, key: str, as_int: bool = False):
        v1 = getattr(m1, key, None)
        v2 = getattr(m2, key, None)
        if v1 is None or v2 is None:
            return
        if as_int:
            print(f"  {name:<28} {int(v1):>12d}  {int(v2):>12d}  {int(v2)-int(v1):>+12d}")
        else:
            print(f"  {name:<28} {v1:>12.2f}  {v2:>12.2f}  {v2-v1:>+12.2f}")

    row("Total PnL ($)", "total_pnl")
    row("Total PnL (%)", "total_pnl_pct")
    row("Win Rate (%)", "win_rate")
    row("Total Trades", "total_trades", as_int=True)
    row("Profit Factor", "profit_factor")
    row("Max Drawdown (%)", "max_drawdown_pct")
    row("Sharpe Ratio", "sharpe_ratio")
    row("Expectancy ($)", "expectancy_usd")
    print("=" * 70)

    if m2.total_pnl > m1.total_pnl and m2.win_rate >= m1.win_rate - 5:
        print("\n[OK] Новая модель дала лучший или сопоставимый результат по PnL и Win Rate.")
    elif m2.total_pnl > m1.total_pnl:
        print("\n[OK] Новая модель дала больший PnL; Win Rate можно проверить отдельно.")
    else:
        print("\n[INFO] Базовая модель в этом периоде показала лучший результат.")


if __name__ == "__main__":
    main()
