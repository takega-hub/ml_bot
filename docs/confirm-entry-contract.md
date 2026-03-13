# confirm_entry JSON-контракт

## Назначение
`confirm_entry` — функция AI Agent, которая получает сигнал ML и рыночный/ботовый контекст и возвращает решение: разрешить вход или запретить вход.

Контракт рассчитан на вызов непосредственно перед выставлением ордера (перед `execute_trade()`/в начале `execute_trade()`), чтобы решение можно было однозначно связать с фактическим результатом сделки через `decision_id`.

## Запрос (payload)
Объект JSON со следующими обязательными полями:

```json
{
  "request_id": "uuid-string",
  "timestamp_utc": "2026-03-13T12:34:56.789+00:00",
  "symbol": "BTCUSDT",
  "timeframe": "15m",
  "signal": {
    "action": "LONG",
    "reason": "string",
    "price": 65000.0,
    "stop_loss": 64000.0,
    "take_profit": 66500.0,
    "signal_timestamp": "2026-03-13 12:30:00",
    "confidence": 0.62,
    "strength": "слабое"
  },
  "bot_context": {
    "side": "Buy",
    "leverage": 10,
    "position_horizon": "short_term",
    "risk_settings": {
      "base_order_usd": 50.0,
      "margin_pct_balance": 0.2,
      "stop_loss_pct": 0.03,
      "take_profit_pct": 0.015,
      "max_position_usd": 200.0
    },
    "ai_fallback_policy": {
      "force_enabled": false,
      "spread_reduce_pct": 0.10,
      "spread_veto_pct": 0.25,
      "min_depth_usd_5": 0.0,
      "imbalance_abs_reduce": 0.60,
      "orderflow_ratio_low": 0.40,
      "orderflow_ratio_high": 2.50
    }
  },
  "market_context": {
    "ohlcv": [
      { "time": 1710000000000, "open": 0, "high": 0, "low": 0, "close": 0, "volume": 0 }
    ],
    "orderbook": {
      "best_bid": 0,
      "best_ask": 0,
      "spread": 0,
      "spread_pct": 0,
      "imbalance_5": 0,
      "imbalance_20": 0,
      "imbalance_50": 0,
      "bid_vol_5": 0,
      "ask_vol_5": 0,
      "bid_vol_20": 0,
      "ask_vol_20": 0
    },
    "recent_trades": {
      "trades_count": 0,
      "buy_volume": 0,
      "sell_volume": 0,
      "buy_sell_ratio": 0,
      "total_volume": 0,
      "last_price": 0,
      "last_time": 0
    }
  }
}
```

### Инварианты
- `request_id` должен быть уникален для каждого вызова (UUID).
- `symbol` в верхнем регистре и соответствует торговой паре.
- `signal.action` принимает только `LONG` или `SHORT`.
- `signal.price` — число.
- `timestamp_utc` — ISO-строка в UTC.
- `bot_context.ai_fallback_policy.force_enabled=true` принудительно включает rule-based fallback даже при доступном AI.

## Ответ (decision)
Строго JSON-объект:

```json
{
  "decision": "allow",
  "confidence": 0.0,
  "risk_score": 50,
  "size_multiplier": 1.0,
  "reason_codes": ["..."],
  "notes": "Коротко (до 30 слов)",
  "decision_id": "uuid-string",
  "timestamp_utc": "2026-03-13T12:34:57.000+00:00",
  "latency_ms": 123
}
```

### Поля ответа
- `decision`: `allow`, `reduce` или `veto`.
- `confidence`: 0..1 (уверенность именно в решении allow/veto).
- `risk_score`: 0..100 (100 = наиболее безопасно).
- `size_multiplier`: множитель размера позиции (применяется к марже/ноционалу на входе), по умолчанию 1.0. Для `reduce` должен быть одним из 0.1 / 0.25 / 0.5.
- `reason_codes`: массив коротких кодов (для статистики и фильтрации).
- `notes`: короткий текст для аудита.
- `decision_id`: должен совпадать с `request_id` (используется как ключ связи).
- `timestamp_utc`, `latency_ms`: метаданные вызова.

## Связка с результатом сделки (аудит)
Для аудита используется единый `decision_id`:
- Событие `confirm_entry` логирует запрос и ответ.
- Событие `trade_outcome` логирует результат сделки и содержит тот же `decision_id`.

Это позволяет постфактум агрегировать статистику (veto-rate, PnL по группам allow/veto, ошибки/latency) без необходимости парсинга строк `trades.log`.
