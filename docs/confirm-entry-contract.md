# Система принятия решения на вход (confirm_entry + Decision Engine)

## Назначение
Перед выставлением ордера бот выполняет **проверку входа**: оценивает качество сигнала, рыночный контекст и риск, и возвращает стандартизированное решение:
- `allow` — разрешить вход
- `reduce` — разрешить вход, но уменьшить размер позиции
- `veto` — запретить вход

Система состоит из двух методов, которые могут работать независимо или совместно:
1) **Decision Engine** — детерминированный движок агрегирования факторов (rule-based + статистика по истории).
2) **AI Agent confirm_entry** — AI-агент, который принимает решение по JSON-контракту (LLM/офлайн-fallback).

Обе системы рассчитаны на вызов непосредственно перед постановкой ордера (перед `execute_trade()`/в начале `execute_trade()`), чтобы решение можно было однозначно связать с фактическим результатом сделки.

## 1) Decision Engine (агрегатор факторов)
### Идея
Decision Engine не “думает” текстом — он **считает скоринг** на основании набора факторов и настраиваемых весов. Это делает поведение объяснимым, воспроизводимым и пригодным для бэктестинга.

### Факторы (текущая реализация)
- **ML/MTF уверенность**: базовая уверенность ML, либо усреднённая MTF (1h/15m/4h), если есть данные.
- **MTF alignment**: согласованность направлений 1h/15m/4h с учётом уверенности каждого ТФ.
- **ATR regime (волатильность)**: предпочтительный диапазон ATR% (слишком низкий = флэт, слишком высокий = паника) с мягким штрафом.
- **S/R proximity (уровни)**: приблизительные уровни поддержки/сопротивления на основе pivot high/low и расстояния до них, нормированного на ATR.
- **Trend slope**: простой нормированный наклон (по close за lookback), как лёгкая поправка.
- **History edge**: поправка по исторической статистике (насколько данный символ лучше/хуже базового win-rate).

### Режимы (как применяется)
Decision Engine имеет переключатели:
- `decision_engine_enabled: true/false`
- `decision_engine_mode: shadow|enforce`

Поведение:
- `shadow`: движок **считает и логирует** решение, но **не блокирует** вход и не меняет размер.
- `enforce`: движок влияет на исполнение:
  - `veto` блокирует вход до вызова AI
  - `reduce` применяет `size_multiplier` (уменьшение размера)

### Что логируется
В `logs/signals.log` пишется:
- `ENGINE EVAL: ... score=... decision=... mult=... codes=...`
- `ENGINE DETAILS: {...}` — компактный JSON с вкладом факторов, весами и порогами

## 2) AI Agent confirm_entry (JSON-контракт)
### Идея
AI Agent получает тот же сигнал и контекст, и возвращает стандартизированное решение. Он может работать:
- как LLM-решатель,
- либо как **офлайн-fallback** (правила по стакану/спреду/ликвидности), когда LLM отключён или принудительно запрещён.

### Режимы (как применяется)
AI подтверждение имеет переключатели:
- `ai_entry_confirmation_enabled: true/false`
- `ai_entry_confirmation_mode: shadow|enforce`
- `ai_fallback_force_enabled: true/false` (принудительный rule-based fallback вместо LLM)

Поведение:
- `shadow`: решение AI **логируется/аудитится**, но **не блокирует вход** и **не меняет размер**.
- `enforce`: решение AI влияет на исполнение:
  - `veto` блокирует вход
  - `reduce` уменьшает размер позиции (`size_multiplier`)

### Запрос (payload)
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
    "strength": "слабое",
    "1h_pred": 1,
    "1h_conf": 0.55,
    "15m_pred": 1,
    "15m_conf": 0.68,
    "4h_pred": 0,
    "4h_conf": 0.0
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
- `request_id` уникален для каждого вызова (UUID).
- `symbol` в верхнем регистре и соответствует торговой паре.
- `signal.action` принимает только `LONG` или `SHORT`.
- `timestamp_utc` — ISO-строка UTC.
- `decision_id` в ответе должен совпадать с `request_id` (связка аудита).

### Ответ (decision)
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
- `size_multiplier`: множитель размера позиции (по умолчанию 1.0). Для `reduce` должен быть разумным (0.1/0.25/0.5).
- `reason_codes`: массив коротких кодов (для статистики и фильтрации).
- `notes`: короткий текст для аудита.
- `decision_id`: должен совпадать с `request_id`.
- `timestamp_utc`, `latency_ms`: метаданные вызова.

## Совместная работа двух методов (правила приоритета)
Система поддерживает независимое включение:
- Decision Engine может работать один (enforce/shadow).
- AI confirm_entry может работать один (enforce/shadow).
- Оба могут быть включены одновременно.

Практический пайплайн (рекомендуемый):
1) Генерация сигнала ML/MTF.
2) Decision Engine вычисляет `ENGINE EVAL` и пишет подробности в `signals.log`.
3) Если Decision Engine `enforce` и `veto` — вход блокируется сразу.
4) Если AI подтверждение включено:
   - `shadow`: только аудит/лог, не влияет на вход.
   - `enforce`: AI может заблокировать вход или уменьшить размер.

Рекомендация по управлению риском:
- Для “интеллектуальной” системы лучше держать **Decision Engine в enforce**, а **AI в shadow** (как второй независимый взгляд и источник данных для анализа), затем постепенно переводить AI в enforce при достаточной статистике.

## Аудит и связка с результатом сделки
Для аудита используется единый идентификатор и JSONL-лог:
- `confirm_entry`: логирует запрос/ответ AI и дополнительно может сохранять `decision_engine` (расчёт движка).
- `entry_blocked`: логирует факт блокировки входа (указан `decision_source`: `engine` или `ai`).
- `trade_outcome`: логирует исход сделки и содержит тот же `decision_id`.

Это позволяет строить постфактум статистику (частота veto, PnL по группам allow/reduce/veto, качество фильтров, latency) без парсинга строк `trades.log`.

## Настройки (ml_settings.json и /api/ml)
Ключи управления:
- Decision Engine:
  - `decision_engine_enabled`, `decision_engine_mode`
  - `decision_engine_allow_score`, `decision_engine_reduce_score`
  - `decision_engine_w_*` (веса факторов)
  - `decision_engine_atr_prefer_min_pct`, `decision_engine_atr_prefer_max_pct`
- AI confirmation:
  - `ai_entry_confirmation_enabled`, `ai_entry_confirmation_mode`
  - `ai_fallback_force_enabled` + параметры `ai_fallback_*`

## Возможные дополнения и развитие (roadmap)
Чтобы прийти к “интеллектуальной” системе с глубоким пониманием рынка, развитие логично строить слоями:
1) **Режим рынка (regime detection)**: классификация (trend/mean-reversion/flat/panic) на основе ADX/volatility/structure и автоматическое переключение весов.
2) **Уровни S/R следующего поколения**:
   - multi-timeframe уровни (15m/1h/4h),
   - volume profile / VPVR,
   - orderbook walls / liquidity zones,
   - структурные уровни (HH/HL/LH/LL).
3) **Orderflow/ликвидность**:
   - дельта, агрессор, дисбаланс,
   - спред/глубина/скольжение,
   - фильтр “слишком тонкий рынок”.
4) **Оценка качества сделки до входа**:
   - прогнозируемый RR, вероятность достижения TP/SL,
   - risk-of-ruin / expected value,
   - адаптивный размер позиции (не только reduce, но и плавный sizing).
5) **Обучение на истории решений**:
   - supervised модель на признаках Decision Engine + исходах (trade_outcome),
   - калибровка вероятностей (Platt/Isotonic),
   - авто-подбор весов/порогов (grid/random/bayes opt) с регуляризацией.
6) **Интерпретируемость и аудит**:
   - сохранение вкладов факторов, фичей, версий моделей/порогов,
   - отчёты по “почему вход был запрещён”, Top reasons и деградации.
7) **Безопасные контуры (safety rails)**:
   - лимит на частоту сделок, лимит по дневному убытку,
   - защита от “догона” и переобучения,
   - мониторинг drift (изменение распределения рынка/признаков).

Цель: сделать Decision Engine базовым объяснимым слоем (контроль риска и структуры), а AI/ML — надстройкой, которая добавляет контекст и адаптивность, но не может “сломать” безопасность системы.
