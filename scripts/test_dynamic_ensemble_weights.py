"""
Тест гипотезы: динамические веса ансамбля (тренд vs флэт по ADX).

Сравнивает бэктест с фиксированными весами (baseline) и с переключением весов по режиму:
- ADX > 25 → тренд → trend_weights (больше вес XGB)
- ADX < 20 → флэт → flat_weights (больше вес RF)
- иначе → стандартные веса модели

Использование:
  python scripts/test_dynamic_ensemble_weights.py --model ml_models/triple_ensemble_BTCUSDT_15.pkl --symbol BTCUSDT --days 30
  python scripts/test_dynamic_ensemble_weights.py --model ml_models/quad_ensemble_BTCUSDT_15_mtf.pkl --days 60 --trend-xgb 0.5 --flat-rf 0.5
"""
import sys
import argparse
from pathlib import Path

# Корень проекта
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest_ml_strategy import run_exact_backtest, BacktestMetrics


# Веса по умолчанию для теста (тренд: больше XGB, флэт: больше RF)
DEFAULT_TREND_WEIGHTS_TRIPLE = {"rf_weight": 0.2, "xgb_weight": 0.5, "lgb_weight": 0.3}
DEFAULT_FLAT_WEIGHTS_TRIPLE = {"rf_weight": 0.5, "xgb_weight": 0.2, "lgb_weight": 0.3}
DEFAULT_TREND_WEIGHTS_QUAD = {"rf_weight": 0.2, "xgb_weight": 0.4, "lgb_weight": 0.2, "lstm_weight": 0.2}
DEFAULT_FLAT_WEIGHTS_QUAD = {"rf_weight": 0.4, "xgb_weight": 0.2, "lgb_weight": 0.2, "lstm_weight": 0.2}


def _is_quad_model(model_path: str) -> bool:
    name = Path(model_path).stem.lower()
    return "quad" in name


def _metrics_row(m: BacktestMetrics, label: str) -> list:
    if m is None or m.total_trades == 0:
        return [label, "-", "-", "-", "-", "-", "-"]
    # win_rate в BacktestMetrics уже в процентах (0–100)
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
        description="Тест гипотезы: динамические веса ансамбля по режиму тренд/флэт (ADX)"
    )
    parser.add_argument("--model", type=str, required=True, help="Путь к ансамблевой модели (triple или quad)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--interval", type=str, default="15")
    parser.add_argument("--balance", type=float, default=5000.0)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--leverage", type=int, default=10)
    # Границы ADX
    parser.add_argument("--adx-trend", type=float, default=25.0, help="ADX > этого = тренд")
    parser.add_argument("--adx-flat", type=float, default=20.0, help="ADX < этого = флэт")
    # Кастомные веса (опционально)
    parser.add_argument("--trend-rf", type=float, default=None)
    parser.add_argument("--trend-xgb", type=float, default=None)
    parser.add_argument("--trend-lgb", type=float, default=None)
    parser.add_argument("--flat-rf", type=float, default=None)
    parser.add_argument("--flat-xgb", type=float, default=None)
    parser.add_argument("--flat-lgb", type=float, default=None)
    args = parser.parse_args()

    model_path = args.model
    if not Path(model_path).exists():
        model_path = str(Path("ml_models") / Path(args.model).name)
    if not Path(model_path).exists():
        print(f"Модель не найдена: {args.model}")
        sys.exit(1)

    is_quad = _is_quad_model(model_path)
    if is_quad:
        trend_weights = DEFAULT_TREND_WEIGHTS_QUAD.copy()
        flat_weights = DEFAULT_FLAT_WEIGHTS_QUAD.copy()
    else:
        trend_weights = DEFAULT_TREND_WEIGHTS_TRIPLE.copy()
        flat_weights = DEFAULT_FLAT_WEIGHTS_TRIPLE.copy()

    if args.trend_rf is not None:
        trend_weights["rf_weight"] = args.trend_rf
    if args.trend_xgb is not None:
        trend_weights["xgb_weight"] = args.trend_xgb
    if args.trend_lgb is not None:
        trend_weights["lgb_weight"] = args.trend_lgb
    if args.flat_rf is not None:
        flat_weights["rf_weight"] = args.flat_rf
    if args.flat_xgb is not None:
        flat_weights["xgb_weight"] = args.flat_xgb
    if args.flat_lgb is not None:
        flat_weights["lgb_weight"] = args.flat_lgb

    print("=" * 70)
    print("ТЕСТ: Динамические веса ансамбля (тренд vs флэт по ADX)")
    print("=" * 70)
    print(f"Модель: {model_path}")
    print(f"Символ: {args.symbol}, период: {args.days} дней, интервал: {args.interval}")
    print(f"ADX: тренд > {args.adx_trend}, флэт < {args.adx_flat}")
    print(f"Веса тренд: {trend_weights}")
    print(f"Веса флэт:  {flat_weights}")
    print()

    # 1) Бэктест baseline (фиксированные веса модели)
    print("Запуск бэктеста BASELINE (фиксированные веса)...")
    metrics_baseline = run_exact_backtest(
        model_path=model_path,
        symbol=args.symbol,
        days_back=args.days,
        interval=args.interval,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        leverage=args.leverage,
        use_dynamic_ensemble_weights=False,
    )
    print()

    # 2) Бэктест с динамическими весами
    print("Запуск бэктеста с ДИНАМИЧЕСКИМИ весами (тренд/флэт)...")
    metrics_dynamic = run_exact_backtest(
        model_path=model_path,
        symbol=args.symbol,
        days_back=args.days,
        interval=args.interval,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        leverage=args.leverage,
        use_dynamic_ensemble_weights=True,
        adx_trend_threshold=args.adx_trend,
        adx_flat_threshold=args.adx_flat,
        trend_weights=trend_weights,
        flat_weights=flat_weights,
    )
    print()

    # 3) Сравнение
    print("=" * 70)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 70)
    headers = ["Вариант", "Сделок", "Win rate", "PnL $", "PnL %", "Sharpe", "Max DD %"]
    rows = [
        _metrics_row(metrics_baseline, "Baseline (фикс. веса)"),
        _metrics_row(metrics_dynamic, "Dynamic (тренд/флэт)"),
    ]
    col_widths = [max(len(str(h)), 8) for h in headers]
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            col_widths[j] = max(col_widths[j], len(str(c)))
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-" * (sum(col_widths) + 2 * (len(headers) - 1)))
    for r in rows:
        print(fmt.format(*r))

    # Итог
    if metrics_baseline and metrics_dynamic and metrics_baseline.total_trades > 0 and metrics_dynamic.total_trades > 0:
        better_pnl = "Dynamic" if metrics_dynamic.total_pnl > metrics_baseline.total_pnl else "Baseline"
        better_sharpe = "Dynamic" if (getattr(metrics_dynamic, "sharpe_ratio", 0) or 0) > (getattr(metrics_baseline, "sharpe_ratio", 0) or 0) else "Baseline"
        print()
        print(f"По PnL лучше: {better_pnl}")
        print(f"По Sharpe лучше: {better_sharpe}")
        if better_pnl == "Dynamic" or better_sharpe == "Dynamic":
            print("Вывод: гипотеза динамических весов дала улучшение на этом периоде.")
        else:
            print("Вывод: на этом периоде baseline показал себя не хуже; можно попробовать другие веса или период.")
    print("=" * 70)


if __name__ == "__main__":
    main()
