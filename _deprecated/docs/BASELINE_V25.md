# Baseline V2.5 - Первая прибыльная модель

## Дата: 2026-02-05

## Результаты OOS (30 дней)

| Метрика | Значение |
|---------|----------|
| **Net PnL** | **+$273.10 (+2.73%)** |
| Total Trades | 49 |
| Win Rate | 40.82% |
| Profit Factor | 1.68 |
| Avg Win | $106.51 |
| Avg Loss | $43.85 |
| Avg RR | 2.43 |
| Max Drawdown | 6.24% |
| Sharpe Ratio | 1.27 |
| Sortino Ratio | 0.86 |

## Распределение сделок

- SHORT: 47 (96%)
- LONG: 2 (4%)

**Примечание**: Тестовый период был медвежьим, поэтому преобладание SHORT ожидаемо.

## Параметры модели

```
min_bars_between_trades: 48 (12 часов)
min_adx: 15.0
min_atr_pct: 0.08
rr_default: 4.0
rr_min: 3.0
rr_max: 5.0
max_sl_pct: 0.01 (1%)
commission_rate: 0.0006 (0.06%)
slippage_bps: 2
entropy_coef: 0.1
use_trend_filter: False
use_balance_filter: False
iterations: 200
```

## Стресс-тесты

| Сценарий | PnL | Return |
|----------|-----|--------|
| Base | +$273.10 | +2.73% |
| Comm x2 | -$1,611.07 | -16.11% |
| Slip x2 | -$238.37 | -2.38% |
| Both x2 | -$807.58 | -8.08% |

## Файлы

- Модель: `ppo_models/ppo_v2_BTCUSDT_baseline_v25.pth`
- Отчёт: `backtest_reports/ppo_v2_backtest_BTCUSDT_20260205_120910.txt`
- Сделки: `backtest_reports/ppo_v2_trades_BTCUSDT_20260205_120910.csv`
