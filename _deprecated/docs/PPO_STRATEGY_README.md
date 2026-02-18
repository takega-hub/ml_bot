# PPO MTF Trading Strategy

Реализация PPO (Proximal Policy Optimization) торговой стратегии с мультитаймфреймовыми признаками (15m/1h/4h) и детерминированным риск-менеджментом.

## ⚡ Быстрый старт (Entry-Only V2 - РЕКОМЕНДУЕТСЯ)

**Entry-Only V2** - упрощённый подход, где агент ТОЛЬКО выбирает точки входа. Выходы полностью автоматические по TP/SL.

### 1. Обучение

```bash
python train_ppo_v2.py --csv "ml_data/BTCUSDT_15_*.csv" --symbol BTCUSDT --iterations 200
```

### 2. Бэктест

```bash
python backtest_ppo_v2.py --checkpoint ppo_models/ppo_v2_BTCUSDT_best.pth --csv "ml_data/BTCUSDT_15_*.csv" --symbol BTCUSDT
```

---

## Архитектура

### V2 (Entry-Only) - РЕКОМЕНДУЕТСЯ

- **TradingEnvV2**: Агент только входит (HOLD/LONG/SHORT), выходы только по TP/SL
- **Простая reward**: только PnL закрытых сделок
- **Меньше сделок**: cooldown 48 баров (12 часов) между сделками
- **Низкие комиссии**: 0.06% maker rate

### V1 (Legacy)

- **TradingEnv**: Агент может входить и закрывать позиции
- **Сложная reward**: множество бонусов и штрафов
- **Проблема**: преждевременное закрытие позиций съедает прибыль

## Структура файлов

```
bot/rl/
  ├── __init__.py
  ├── trading_env.py          # V1 торговое окружение
  ├── trading_env_v2.py       # V2 Entry-Only окружение
  ├── risk_manager.py         # Расчет TP/SL
  ├── ppo_agent.py           # Actor-Critic сеть
  ├── ppo_trainer.py         # PPO обучение
  ├── data_preparation.py    # Подготовка данных
  └── metrics.py             # Метрики производительности

train_ppo.py                 # V1 обучение
train_ppo_v2.py              # V2 Entry-Only обучение
backtest_ppo.py              # V1 бэктест
backtest_ppo_v2.py           # V2 Entry-Only бэктест
```

## Параметры V2

| Параметр | Значение | Описание |
|----------|----------|----------|
| `--iterations` | 200 | Итерации обучения |
| `--min-bars-between-trades` | 48 | Минимум 12 часов между сделками |
| `--min-adx` | 12.0 | Фильтр: минимум силы тренда |
| `--min-atr-pct` | 0.08 | Фильтр: минимум волатильности |
| `--rr-default` | 3.0 | Risk-Reward по умолчанию |
| `--rr-min` | 2.5 | Минимальный RR |
| `--rr-max` | 4.0 | Максимальный RR |
| `--commission-rate` | 0.0006 | 0.06% (Bybit maker) |
| `--slippage-bps` | 3.0 | 0.03% слиппедж |
| `--entropy-coef` | 0.05 | Коэффициент энтропии |

## Почему V2 лучше V1

### Проблема V1

В V1 агент мог закрывать позиции вручную (`manual_close`, `flip`). Это приводило к:
- **59% сделок закрывались преждевременно** (вместо TP/SL)
- **Только 7% сделок достигали TP**
- **Комиссии съедали прибыль** (283% от gross profit!)

### Решение V2

1. **Убрали возможность закрывать позиции** - только TP/SL
2. **Упростили reward** - только PnL сделки
3. **Увеличили cooldown** - меньше сделок = меньше комиссий
4. **Снизили комиссию** - 0.06% вместо 0.1%
5. **Увеличили RR** - 3.0 вместо 2.2

## Особенности реализации

### MTF Признаки

- **15m**: Базовые технические индикаторы (RSI, ADX, ATR, BB, MACD, etc.)
- **1h/4h**: Ключевые индикаторы с суффиксами `_60` и `_240`
- **Уровни S/R**: Расстояния до поддержки/сопротивления в долях ATR

### Risk Management

- **SL**: За ближайшим уровнем S/R + буфер (0.2 * ATR)
- **TP**: По RR 2.5-4.0 от SL
- **Размер позиции**: 1% риска на сделку

### Reward Function (V2)

```python
# Простая и понятная:
reward = (net_pnl / initial_capital) * reward_scale
# Где net_pnl = pnl - commission
```

### PPO Hyperparameters

- `γ=0.99` (discount factor)
- `λ=0.95` (GAE lambda)
- `clip_eps=0.2` (PPO clip epsilon)
- `entropy_coef=0.05` (entropy bonus)
- `learning_rate=1e-4`

## Метрики бэктеста

- **Trade Metrics**: Total trades, Win rate, Profit factor, Avg win/loss, Avg RR
- **PnL Metrics**: Gross profit/loss, Total commission, Net PnL, Return %
- **Equity Metrics**: Final equity, Max drawdown, Sharpe/Sortino ratio
- **Exit Reasons**: % Take Profit vs Stop Loss

## Стресс-тесты

Автоматически выполняются:
1. Комиссия x2
2. Слиппедж x2
3. Комиссия + Слиппедж x2

## Важные замечания

1. **Lookahead Prevention**: Все MTF фичи используют `ffill` из уже закрытых HTF свечей
2. **Уровни S/R**: Рассчитываются только из lookback окна ≤ текущего времени
3. **OOS Test**: Последние 30 дней - чистый out-of-sample тест

## Требования

- Python 3.8+
- PyTorch >= 2.0.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- pandas-ta >= 0.3.14b0

## Legacy V1

Если нужен старый подход с возможностью закрытия позиций:

```bash
# Обучение V1
python train_ppo.py --csv "ml_data/BTCUSDT_15_*.csv" --symbol BTCUSDT --preset moderate

# Бэктест V1
python backtest_ppo.py --checkpoint ppo_models/ppo_BTCUSDT_best.pth --csv "ml_data/BTCUSDT_15_*.csv" --preset moderate
```
