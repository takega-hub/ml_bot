"""
Тест гипотезы: адаптивный порог уверенности по ATR.

Формула из роадмэпа: при высокой волатильности — чуть снижать порог (больше сигналов),
при низкой — повышать (меньше шума).
effective_threshold = base_threshold * (1 + k * (atr_median - atr_current) / atr_median),
с ограничением множителя [adaptive_confidence_min, adaptive_confidence_max].

Сравнивает бэктест с фиксированным порогом (baseline) и с адаптацией по ATR.

Использование:
  python scripts/test_adaptive_confidence_atr.py --model ml_models/rf_BTCUSDT_15_15m.pkl --symbol BTCUSDT --days 30
  python scripts/test_adaptive_confidence_atr.py --model ml_models/triple_ensemble_BTCUSDT_15_15m.pkl --days 60 --k 0.4
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest_ml_strategy import run_exact_backtest, BacktestMetrics


def _metrics_row(m: BacktestMetrics, label: str) -> list:
    if m is None or m.total_trades == 0:
        return [label, "-", "-", "-", "-", "-", "-"]
    wr = m.win_rate if m.win_rate > 1 else m.win_rate * 100
    return [
        label,
        str(m.total_trades),
        f"{wr:.1f}%",
        f"{m.total_pnl:.2f}",
        f"{m.total_pnl_pct:.2f}%",
        f"{m.sharpe_ratio:.2f}" if getattr(m, "sharpe_ratio", None) is not None else "-",
        f"{m.max_drawdown_pct:.1f}%" if getattr(m, "max_drawdown_pct", None) is not None else "-",
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Тест гипотезы: адаптивный порог уверенности по ATR"
    )
    parser.add_argument("--model", type=str, required=True, help="Путь к модели (.pkl)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--interval", type=str, default="15")
    parser.add_argument("--balance", type=float, default=5000.0)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--k", type=float, default=0.3, help="Коэффициент k в формуле ATR")
    parser.add_argument("--min-mul", type=float, default=0.8, help="Мин. множитель порога")
    parser.add_argument("--max-mul", type=float, default=1.2, help="Макс. множитель порога")
    args = parser.parse_args()

    model_path = args.model
    if not Path(model_path).exists():
        model_path = str(Path("ml_models") / Path(args.model).name)
    if not Path(model_path).exists():
        print(f"Модель не найдена: {args.model}")
        sys.exit(1)

    print("=" * 70)
    print("ТЕСТ: Адаптивный порог уверенности по ATR")
    print("=" * 70)
    print(f"Модель: {model_path}")
    print(f"Символ: {args.symbol}, период: {args.days} дней")
    print(f"Формула: effective_threshold = base * (1 + k*(atr_median - atr_current)/atr_median)")
    print(f"k={args.k}, множитель в [{args.min_mul}, {args.max_mul}]")
    print()

    # 1) Baseline: фиксированный порог (без динамики и без ATR-формулы)
    print("Запуск бэктеста BASELINE (фиксированный порог)...")
    metrics_baseline = run_exact_backtest(
        model_path=model_path,
        symbol=args.symbol,
        days_back=args.days,
        interval=args.interval,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        leverage=args.leverage,
        use_dynamic_threshold=False,
        use_adaptive_confidence_by_atr=False,
    )
    print()

    # 2) ATR-адаптивный порог
    print("Запуск бэктеста с АДАПТИВНЫМ порогом по ATR...")
    metrics_atr = run_exact_backtest(
        model_path=model_path,
        symbol=args.symbol,
        days_back=args.days,
        interval=args.interval,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        leverage=args.leverage,
        use_dynamic_threshold=False,
        use_adaptive_confidence_by_atr=True,
        adaptive_confidence_k=args.k,
        adaptive_confidence_min=args.min_mul,
        adaptive_confidence_max=args.max_mul,
    )
    print()

    # 3) Сравнение
    print("=" * 70)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 70)
    headers = ["Вариант", "Сделок", "Win rate", "PnL $", "PnL %", "Sharpe", "Max DD %"]
    rows = [
        _metrics_row(metrics_baseline, "Baseline (фикс. порог)"),
        _metrics_row(metrics_atr, "ATR-адаптивный порог"),
    ]
    col_widths = [max(len(str(h)), 8) for h in headers]
    for r in rows:
        for j, c in enumerate(r):
            col_widths[j] = max(col_widths[j], len(str(c)))
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-" * (sum(col_widths) + 2 * (len(headers) - 1)))
    for r in rows:
        print(fmt.format(*r))

    if metrics_baseline and metrics_atr and metrics_baseline.total_trades > 0 and metrics_atr.total_trades > 0:
        better_pnl = "ATR-адаптивный" if metrics_atr.total_pnl > metrics_baseline.total_pnl else "Baseline"
        better_sharpe = "ATR-адаптивный" if (getattr(metrics_atr, "sharpe_ratio", 0) or 0) > (getattr(metrics_baseline, "sharpe_ratio", 0) or 0) else "Baseline"
        print()
        print(f"По PnL лучше: {better_pnl}")
        print(f"По Sharpe лучше: {better_sharpe}")
        if better_pnl == "ATR-адаптивный" or better_sharpe == "ATR-адаптивный":
            print("Вывод: гипотеза адаптивного порога по ATR дала улучшение на этом периоде.")
        else:
            print("Вывод: на этом периоде фиксированный порог показал себя не хуже; можно попробовать другие k/min/max.")
    print("=" * 70)


if __name__ == "__main__":
    main()
