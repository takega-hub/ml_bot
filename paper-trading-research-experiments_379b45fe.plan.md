---
name: paper-trading-research-experiments
overview: Добавим модуль PaperTrading для онлайн-тестирования экспериментальных моделей из AI исследований (experiments.json), используя тот же поток real-time свечей, что и основной `TradingLoop`, но без реальных ордеров. Управление и метрики будут доступны через FastAPI.
todos:
  - id: inspect-research-format
    content: "Использовать `experiments.json` формат из `run_research.py` как контракт: `results.models` + метрики."
    status: pending
  - id: paper-module
    content: Спроектировать и реализовать `bot/paper_trading.py` (manager/session/broker/metrics) с виртуальным исполнением на свечах.
    status: pending
  - id: trading-loop-hook
    content: Встроить вызов paper-менеджера в `bot/trading_loop.py` так, чтобы он получал те же данные `df/row/high/low/current_price` без повторных запросов.
    status: pending
  - id: api-endpoints
    content: Добавить эндпоинты управления и просмотра paper-тестов в `bot/api_server.py` и связать их с AI research (experiment_id).
    status: pending
  - id: smoke-test
    content: "Сделать минимальную проверку: запустить paper-сессию по существующему experiment_id и убедиться, что идут виртуальные сделки и метрики."
    status: pending
isProject: false
---

## Контекст и точки интеграции

- Эксперименты AI исследований создаются `run_research.py` и сохраняются в `experiments.json` через `update_experiment_status()` с ключами `id`, `symbol`, `type`, `status`, `results`, `results.models` (пути к `15m` и `1h`) и метриками бэктеста внутри `results`.
- Основной поток real-time данных/сигналов уже есть в `bot/trading_loop.py`: внутри `process_symbol()` формируются `df`, `row`, `current_price`, `high`, `low` и вызывается `strategy.generate_signal()` в `asyncio.to_thread(...)`.
- Формат сигнала унифицирован в `bot/strategy.py` (`Signal`, `Action`, поля `stop_loss`, `take_profit`, `trailing`, `indicators_info`). Это позволит paper-движку потреблять сигналы без изменения стратегий.

## Архитектура PaperTrading

- Добавить модуль `[bot/paper_trading.py](c:\Users\takeg\OneDrive\Документы\vibecodding\ml_bot\bot\paper_trading.py)`.
- Основные сущности:
  - `PaperTradingManager`: хранит активные paper-сессии (по `experiment_id`), даёт API для start/stop/status.
  - `PaperSession`: одна виртуальная “копия торговли” для конкретного эксперимента (symbol + стратегия/модели + виртуальный портфель).
  - `PaperBroker` (внутри сессии): виртуальная позиция, расчёт qty, комиссия/проскальзывание (параметризуемо), TP/SL/trailing, логика закрытия по `high/low` свечи.
  - `PaperMetrics`: накопление метрик (PnL$, PnL%, winrate, drawdown, avg duration, count signals/trades).
- Стратегии для paper-сессий:
  - `Single`: `MLStrategy(model_path=results.models["15m"])`
  - `MTF`: `MultiTimeframeMLStrategy(model_1h_path=results.models["1h"], model_15m_path=results.models["15m"])` (если оба пути есть)
  - Выбор режима делаем автоматически из наличия путей в `experiments.json`.

## Поток данных (без дубля получения свечей)

- В `bot/trading_loop.py` добавить “tap” в `process_symbol()` после подготовки `df/row/high/low/current_price`, чтобы **передать те же данные** в `PaperTradingManager.on_bar(...)`.
- `PaperTradingManager.on_bar(symbol, row, df, current_price, high, low, candle_timestamp)`:
  - Для каждой активной paper-сессии по этому `symbol`:
    - сначала проверить `broker.check_exit(...)` на текущей свече (TP/SL/trailing/time-stop)
    - затем, если нет открытой позиции, сгенерировать сигнал через `asyncio.to_thread(session.strategy.generate_signal, ...)` (точно так же, как в основном лупе)
    - при `LONG/SHORT` — открыть виртуальную позицию (без реального API)

## API управления (интеграция с AI Research разделом)

- В `[bot/api_server.py](c:\Users\takeg\OneDrive\Документы\vibecodding\ml_bot\bot\api_server.py)` добавить эндпоинты:
  - `POST /api/paper/start` с `experiment_id` (берём модели/символ из `experiments.json`, стартуем сессию)
  - `POST /api/paper/stop` с `experiment_id`
  - `GET /api/paper/status` (список активных сессий + метрики)
  - `GET /api/paper/trades?experiment_id=...&limit=...`
- В `GET /api/ai/research/status` (уже существует) расширить вывод: если по `experiment_id` запущен paper-тест, вернуть `paper_status`/`paper_metrics` рядом с `recommendation`.

## Хранение данных

- По умолчанию: хранить paper-трейды и метрики **в памяти процесса** (быстро, без риска загрязнить `runtime_state.json`).
- Опционально (в следующем шаге, если понадобится): лёгкая персистентность в `paper_state.json` или расширение `BotState` отдельными списками `paper_trades/paper_sessions`.

## Метрики и реализм исполнения

- Поддержать параметры:
  - комиссия (дефолт как в бэктесте `0.0006`)
  - простое проскальзывание (bps) на вход/выход
  - sizing: фиксированный `base_order_usd` или % от виртуального баланса (похоже на `MLBacktestSimulator`)
- Для TP/SL использовать цены из `Signal.stop_loss/take_profit` (или из `signal.indicators_info['stop_loss'/'take_profit']` как fallback).

## Проверка корректности

- Локальный smoke-test: запустить бота в режиме без реальных сделок (state.is_running=True, но отключить real execution настройкой/флагом если есть), включить paper на одном `experiment_id`, проверить что:
  - paper-сессия получает бары, открывает/закрывает виртуальные позиции
  - метрики обновляются
  - API отдаёт статус/трейды без блокировок event loop.

## Диаграмма

```mermaid
flowchart TD
  api[FastAPI_api_server] -->|startPaper(experiment_id)| paperMgr[PaperTradingManager]
  paperMgr -->|loadExperiment| expFile[experiments.json]
  tradingLoop[TradingLoop.process_symbol] -->|row_df_price_high_low| paperMgr
  paperMgr --> paperSession[PaperSession]
  paperSession --> strategy[MLStrategy_or_MTFStrategy]
  paperSession --> broker[PaperBroker]
  broker --> metrics[PaperMetrics]
  api -->|status_trades_metrics| paperMgr
```



