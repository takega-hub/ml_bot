# Эталонная модель для тестирования изменений

Все новые изменения (фильтр волатильности, тайм-фильтр, условный trailing и т.д.) рекомендуется тестировать на одной и той же модели для сопоставимости результатов.

## Выбранная модель

**Файл:** `ml_models/rf_BTCUSDT_15_15m.pkl`  
**Символ:** BTCUSDT  
**Таймфрейм:** 15m  
**Тип:** Random Forest (rf)

### Показатели из сравнения (ml_models_comparison_20260215_131500.csv)

| Метрика | Значение |
|---------|----------|
| Total PnL % | 68.06% |
| Win Rate | 57.95% |
| Sharpe Ratio | 6.51 |
| Profit Factor | 2.37 |
| Max Drawdown % | 4.68% |
| Total trades | 88 |
| Risk category | Low |

### Почему эта модель

- **Главная пара (BTCUSDT)** — базовая для бота, по ней удобно сравнивать до/после.
- **15m** — тот же таймфрейм, что используется для входа в live и в бэктестах.
- **Хороший баланс:** высокий Sharpe, приемлемая просадка, win rate > 55%, profit factor > 2.
- **Risk category Low** — устойчивая на истории.

## Как использовать

### Бэктест без фильтра волатильности
```bash
python backtest_ml_strategy.py --model ml_models/rf_BTCUSDT_15_15m.pkl --symbol BTCUSDT --days 30
```

### Бэктест с фильтром волатильности (ATR 1h)
```bash
python backtest_ml_strategy.py --model ml_models/rf_BTCUSDT_15_15m.pkl --symbol BTCUSDT --days 30 --atr-filter --atr-min 0.3 --atr-max 2.0
```

### Сравнение порогов ATR (сетка)
Запускать бэктест с разными `--atr-min` / `--atr-max` и сравнивать PnL / Sharpe в отчёте.

## Альтернативные модели (если нужен другой символ или тип)

- **Максимальный composite_score среди 15m:** `quad_ensemble_ADAUSDT_15_15m.pkl` (composite 26.89, PnL 123.5%, Sharpe 7.1).
- **Лучший PnL среди 15m:** `triple_ensemble_SOLUSDT_15_15m.pkl` (PnL 139.3%, Sharpe 7.51, Win 57.6%).

Для единообразия тестов новых фич рекомендуется держать эталоном **rf_BTCUSDT_15_15m.pkl**.

---

## MTF-комбинация для тестирования

Для тестов с **включённой MTF-стратегией** (1h фильтр + 15m вход) используется комбинация из [mtf_combinations_BTCUSDT_20260212_194950.csv](mtf_combinations_BTCUSDT_20260212_194950.csv).

### Выбранная комбинация

| Роль | Модель |
|------|--------|
| **1h (фильтр)** | `ml_models/quad_ensemble_BTCUSDT_60_1h.pkl` |
| **15m (вход)** | `ml_models/xgb_BTCUSDT_15_15m.pkl` |

**Символ:** BTCUSDT

### Показатели из MTF-сравнения

| Метрика | Значение |
|---------|----------|
| Total PnL % | 96.36% |
| Win Rate | 70.97% |
| Sharpe Ratio | **13.97** (максимум среди комбинаций) |
| Profit Factor | 5.80 |
| Max Drawdown % | 1.46% |
| Total trades | 62 |

### Почему эта комбинация

- **Максимальный Sharpe** среди всех MTF-комбинаций BTCUSDT — лучшая доходность с учётом риска.
- **Минимальная просадка** (1.46%) при высокой доходности.
- Win rate > 70%, profit factor > 5.

### Альтернатива (максимальный PnL)

**quad_ensemble_BTCUSDT_60_1h.pkl** + **ensemble_BTCUSDT_15_15m.pkl** — PnL 99.9%, Win 71.2%, Sharpe 13.86, DD 1.47%. Чуть выше прибыль, чуть ниже Sharpe.
