# AI Agent — Research (Исследования)

Документ описывает текущую реализацию вкладки **Research (Исследования)** в мобильном приложении и её взаимодействие с backend.

## Термины

- **Research‑эксперимент** — оффлайн-процесс: обучение + бэктест, который создаёт модели и пишет результат в `experiments.json`.
- **Virtual Testing (paper trading)** — онлайн‑тест выбранного эксперимента на живых барах. Работает в фоне на сервере и пишет состояние в `paper_trading_state.json`. UI только читает и визуализирует.
- **Working model / Real strategy** — текущая “рабочая” стратегия бота (реальные сделки), используется для сравнения на графике и в статах.

## Файлы и точки входа

### Mobile (Flutter)

- Экран вкладок AI Agent: [ai_agent_screen.dart](file:///c:/Users/takeg/OneDrive/%D0%94%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82%D1%8B/vibecodding/ml_bot/ml_bot_app/lib/screens/ai_agent_screen.dart)
  - Родитель: `AIAgentScreen` / `_AIAgentScreenState`
  - Вкладка исследований: `_ResearchTab` / `_ResearchTabState`
  - Карточка эксперимента: `_ResearchResultCard`
- API-клиент: `ml_bot_app/lib/api/api_service.dart`
- Виджет графика: `ml_bot_app/lib/widgets/virtual_test_chart.dart`

### Backend (FastAPI + trading loop)

- HTTP API: [bot/api_server.py](file:///c:/Users/takeg/OneDrive/%D0%94%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82%D1%8B/vibecodding/ml_bot/bot/api_server.py)
- Сервис Research: [bot/ai_agent_service.py](file:///c:/Users/takeg/OneDrive/%D0%94%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82%D1%8B/vibecodding/ml_bot/bot/ai_agent_service.py)
- Скрипт эксперимента: [run_research.py](file:///c:/Users/takeg/OneDrive/%D0%94%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82%D1%8B/vibecodding/ml_bot/run_research.py)
- Paper trading: [bot/paper_trading.py](file:///c:/Users/takeg/OneDrive/%D0%94%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82%D1%8B/vibecodding/ml_bot/bot/paper_trading.py)
- Основной цикл: [bot/trading_loop.py](file:///c:/Users/takeg/OneDrive/%D0%94%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82%D1%8B/vibecodding/ml_bot/bot/trading_loop.py)

### Persisted storage

- `experiments.json` — реестр research‑экспериментов (статус, прогресс, результаты, пути моделей).
- `paper_trading_state.json` — снимки paper trading сессий (статусы, equity_curve, метрики, трейды).

## API контракт (ключевые эндпоинты)

### Research (training/backtest)

- `POST /api/ai/research/start`
  - Запускает research‑эксперимент (отдельный процесс `run_research.py`).
  - Тело: `{ "symbol": "ETHUSDT", "type": "aggressive|conservative|balanced", "metadata": {...}, "allow_duplicate": false }`
  - Ответ: `{ ok, pid, experiment_id, symbol, type, started_at }`
  - Повторы:
    - если найден дубликат с “неэффективными” метриками, сервер вернёт ошибку `409` (механизм предотвращения повторов).

- `GET /api/ai/research/status`
  - Возвращает список экспериментов из `experiments.json`.
  - Дополнительно может добавлять:
    - `current_strategy` (текущая стратегия/модели/метрики по реальным сделкам),
    - `comparison` и `recommendation`,
    - `paper_status` / `paper_metrics` (если доступно).

- `POST /api/ai/research/apply`
  - Применяет модели эксперимента как текущую стратегию (HOT SWAP).
  - Тело: `{ "experiment_id": "exp_..." }`
  - Использует пути моделей из `experiment.results.models`.
  - Если `experiment.results.recommended_tactic` определён, применяется выбранная тактика:
    - `mtf` (1h+15m) или `single_1h`/`single_15m` (одна модель).

### Experiment Management (insights + reports)

- `GET /api/ai/experiments/insights?symbol=ETHUSDT&type=conservative`
  - Возвращает агрегированный анализ прошлых экспериментов и список гипотез.
- `GET /api/ai/experiments/report/{experiment_id}`
  - Возвращает markdown‑отчёт для выбранного эксперимента.

### Virtual Testing (paper trading)

- `POST /api/paper/start`
  - Запускает paper trading сессию по `experiment_id`.
  - Ответ содержит `symbol` для сессии.

- `POST /api/paper/stop`
  - Останавливает paper trading сессию по `experiment_id`.

- `GET /api/paper/status`
  - Возвращает список статусов сессий (объединение active + persisted).
  - Используется UI для восстановления состояния и для проверки “идёт ли тест”.

- `GET /api/paper/realtime_chart/{experiment_id}`
  - Возвращает данные графика и метрик paper trading сессии.
  - Важно: данные строятся сервером из `paper_trading_state.json` и/или active session.

### Данные рабочей стратегии (для сравнения)

- `GET /api/bot/realtime_data?symbol=ETHUSDT`
  - Возвращает метрики и equity_curve “реальной” (working) стратегии, чтобы сравнить с virtual.

- `POST /api/bot/update_settings`
  - Синхронизирует runtime-настройки бота/стратегии на сервере перед расчётами.

## Как работает UI вкладки Research

### 1) Загрузка списка символов для “New Experiment”

- При инициализации `_ResearchTabState` вызывается:
  - `_loadSymbols()` → `GET /api/pairs` → берётся `active_symbols` → наполняется dropdown “Target Symbol”.

### 2) Запуск research‑эксперимента

- Пользователь выбирает `Target Symbol`.
- Нажимает карточку “Aggressive Growth” или “Conservative Trend”.
- UI вызывает `POST /api/ai/research/start`.
- После успешного старта UI обновляет список экспериментов.

### 3) Polling статуса экспериментов

- Вкладка Research обновляет историю экспериментов раз в 5 секунд.
- UI отображает `status`, `progress`, `last_log`, а при `completed` — `results` и рекомендацию.

### 4) HOT SWAP (Apply Strategy)

- На карточке эксперимента UI вызывает `POST /api/ai/research/apply`.
- Сервер обновляет конфигурацию стратегии для символа, после чего бот начинает использовать выбранные модели.

## Как работает Virtual Testing (paper trading) внутри Research

### Ключевой принцип

Virtual Testing выполняется на сервере “в фоне”. UI не отвечает за исполнение стратегии и запись истории. UI только:

- стартует/останавливает сессию,
- периодически читает статусы и данные графика,
- показывает persisted данные при повторном заходе.

### 1) Старт/стоп

- Нажатие **Play** на карточке эксперимента вызывает `POST /api/paper/start`.
- Нажатие **Stop** — `POST /api/paper/stop`.

На мобильном стороне `AIAgentScreen` сохраняет `active_paper_experiment_id` в `SharedPreferences`, чтобы при следующем открытии экрана восстановить эксперимент.

### 2) Polling данных

Если сервер сообщает, что сессия active, UI запускает таймер 1 сек:

1) `POST /api/bot/update_settings`
2) `GET /api/paper/realtime_chart/{experiment_id}` → virtualChartData
3) `GET /api/bot/realtime_data?symbol=...` → workingModelData
4) `GET /api/paper/status` → чтобы обновить флаг активной сессии

### 3) Отображение графика

`_ResearchResultCard` рисует график из:

- `virtualChartData.equity_curve + virtualChartData.timestamps`
- `workingModelData.equity_curve`

и показывает статы:

- Virtual Balance / Virtual PnL / Virtual Trades / Virtual Open
- Real PnL / Real Win Rate / Real Trades

### 4) Восстановление после перезахода

При открытии экрана AI Agent UI:

1) читает `active_paper_experiment_id` из `SharedPreferences`,
2) делает `GET /api/paper/status`,
3) если находит `status=active` — включает polling,
4) если активной нет, но есть persisted сессия — один раз загружает график для отображения истории.

Таким образом “данные должны отображаться при новом заходе” выполняется за счёт persisted state на сервере.

## Как работает серверная часть Virtual Testing

- `PaperTradingManager` хранит:
  - `sessions` (активные),
  - `persisted_sessions` (сохранённые).
- При старте бота manager читает `paper_trading_state.json` и пытается восстановить все persisted с `status="active"` (если есть данные эксперимента).
- На каждом баре `TradingLoop` вызывает `paper_trading_manager.on_bar(...)`, и manager:
  - прокидывает бар в `PaperSession.process_bar(...)`,
  - сохраняет снапшот обратно в `paper_trading_state.json`.

## Сценарий end‑to‑end (шаги 1 → 5)

Ниже — эталонный поток “создал эксперимент → дождался completed → нажал Play Virtual Testing → проверил persisted JSON → повторно зашёл и увидел данные”.

### Шаг 1. Создаю эксперимент

**UI действие**

- Research → “New Experiment” → выбрать `Target Symbol` → нажать карточку типа эксперимента.

**HTTP**

- `POST /api/ai/research/start` с `{symbol, type}`.

**Сервер**

- Создаётся/обновляется запись в `experiments.json`.
- Запускается процесс `run_research.py` (обучение + бэктест), который периодически обновляет `experiments.json`.

**Что должно появиться в `experiments.json` сразу**

Минимальные поля, которые записываются на старте:

```json
{
  "exp_...": {
    "id": "exp_...",
    "created_at": "2026-03-10T20:18:00.459984",
    "updated_at": "2026-03-10T20:18:00.500000",
    "status": "starting",
    "symbol": "ETHUSDT",
    "type": "conservative",
    "params": {
      "interval": "1h",
      "no_mtf": true
    },
    "param_signature": "sha256...",
    "code_version": {
      "git_sha": "....",
      "git_dirty": false
    }
  }
}
```

Источник записи: [run_research.py](file:///c:/Users/takeg/OneDrive/%D0%94%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82%D1%8B/vibecodding/ml_bot/run_research.py).

### Шаг 2. Эксперимент становится completed

**UI**

- Вкладка Research раз в 5 секунд делает `GET /api/ai/research/status` и отображает прогресс.

**Сервер**

- В `experiments.json` постепенно обновляются:
  - `status` (training → backtesting → completed/failed),
  - `progress`,
  - `last_log`,
  - а на завершении добавляется `results` и `completed_at`.

**Что UI ожидает в completed эксперименте (минимум)**

- `id`, `symbol`, `type`, `status="completed"`, `progress=100`
- `results` (для summary) — UI использует `results.total_pnl_pct`, `results.win_rate`, `results.total_trades` и опционально `results.model_name`.
- `results.models` — для Apply (HOT SWAP) и для server-side создания paper trading стратегии.
- `results.recommended_tactic` и `results.tactics` (если эксперимент сравнивает single vs mtf):
  - `recommended_tactic`: `"mtf" | "single_1h" | "single_15m"`
  - `tactics`: словарь с результатами всех проверенных тактик

Пример из вашего текущего `experiments.json` (укорочено):

```json
{
  "id": "exp_9621987_84ce94f9",
  "status": "completed",
  "symbol": "ETHUSDT",
  "type": "conservative",
  "progress": 100,
  "results": {
    "model_name": "...",
    "total_trades": 155,
    "win_rate": 69.03,
    "total_pnl_pct": 121.00,
    "models": {
      "15m": null,
      "1h": "ml_models/quad_ensemble_ETHUSDT_60_1h_conservative_exp.pkl"
    }
  },
  "completed_at": "2026-03-10T21:24:36.389548"
}
```

### Шаг 3. Нажимаю Play Virtual Testing

**UI действие**

- На карточке completed‑эксперимента нажать кнопку Play (Virtual Testing).

**HTTP**

- `POST /api/paper/start` с `{experiment_id}`.

**Сервер**

- `PaperTradingManager.start_session(experiment_id)`:
  - читает эксперимент из `experiments.json`,
  - создаёт стратегию по `results.models` с учётом `results.recommended_tactic` (если тактика single — используется одна модель),
  - создаёт `PaperSession(symbol=..., strategy=...)`,
  - выставляет `status=active`,
  - сразу сохраняет снапшот в `paper_trading_state.json`.

**UI (после старта)**

- Сохраняет `active_paper_experiment_id` в `SharedPreferences`.
- Запускает polling (1 сек), который регулярно читает `GET /api/paper/realtime_chart/{experimentId}`.

### Шаг 4. Что должно быть в paper_trading_state.json и что ждёт UI

#### 4.1. Формат `paper_trading_state.json` (persisted)

Файл пишется сервером и выглядит так:

```json
{
  "updated_at": "2026-03-13T10:05:00.000000",
  "sessions": {
    "exp_9621987_84ce94f9": {
      "experiment_id": "exp_9621987_84ce94f9",
      "symbol": "ETHUSDT",
      "status": "active",
      "status_reason": "active",
      "start_time": "2026-03-13T10:00:00.000000",
      "end_time": null,
      "last_bar_time": "2026-03-13T10:00:00.000000",
      "last_snapshot_at": "2026-03-13T10:00:01.000000",
      "last_error": null,
      "broker": {
        "initial_balance": 10000.0,
        "balance": 10000.0,
        "commission": 0.0006,
        "slippage_bps": 0.0,
        "base_order_usd": 100.0,
        "position": null,
        "trades": []
      },
      "chart_data": {
        "experiment_id": "exp_9621987_84ce94f9",
        "symbol": "ETHUSDT",
        "equity_curve": [10000.0],
        "timestamps": ["2026-03-13T10:00:00.000000"],
        "current_balance": 10000.0,
        "initial_balance": 10000.0,
        "total_trades": 0,
        "open_trades": 0,
        "closed_trades": 0,
        "is_active": true,
        "status": "active",
        "last_bar_time": "2026-03-13T10:00:00.000000",
        "last_snapshot_at": "2026-03-13T10:00:01.000000",
        "last_error": null,
        "status_reason": "active"
      },
      "metrics": {
        "experiment_id": "exp_9621987_84ce94f9",
        "symbol": "ETHUSDT",
        "metrics": {},
        "balance": 10000.0,
        "initial_balance": 10000.0,
        "status": "active",
        "last_bar_time": "2026-03-13T10:00:00.000000",
        "last_snapshot_at": "2026-03-13T10:00:01.000000",
        "last_error": null,
        "status_reason": "active"
      }
    }
  }
}
```

Источник схемы: `PaperSession.to_persisted_dict()` и `PaperTradingManager._write_state()` в [paper_trading.py](file:///c:/Users/takeg/OneDrive/%D0%94%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82%D1%8B/vibecodding/ml_bot/bot/paper_trading.py).

#### 4.2. Что UI реально ожидает для графика (через API)

UI **не читает файл напрямую**. UI получает `chart_data` через:

- `GET /api/paper/realtime_chart/{experiment_id}`

и ожидает ключи:

- обязательные для графика:
  - `equity_curve: number[]`
  - `timestamps: string[]`
- обязательные для статов:
  - `current_balance: number`
  - `initial_balance: number`
  - `total_trades: number`
  - `open_trades: number`
- для статуса:
  - `status: "active"|"completed"|"interrupted"|"error"|"stopped"`
  - `symbol` (используется также для запроса realtime_data)

См. генерацию `chart_data`: `PaperSession.get_chart_data()` в [paper_trading.py](file:///c:/Users/takeg/OneDrive/%D0%94%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82%D1%8B/vibecodding/ml_bot/bot/paper_trading.py#L525-L553) и использование в UI: `_ResearchResultCard` в [ai_agent_screen.dart](file:///c:/Users/takeg/OneDrive/%D0%94%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82%D1%8B/vibecodding/ml_bot/ml_bot_app/lib/screens/ai_agent_screen.dart#L1594-L1692).

### Шаг 5. Как выглядит корректный experiments.json (ожидаемый минимум)

`experiments.json` — это map по `experiment_id`. Для корректной работы Research + Virtual Testing достаточно:

- `id` (ключ и значение должны совпадать),
- `symbol` (например, `ETHUSDT`),
- `type` (aggressive|conservative|balanced),
- `status` и `updated_at`,
- при `completed`:
  - `results.models` (пути моделей; допустимо только `1h` или только `15m`),
  - метрики для отображения summary (минимум `total_trades`, `win_rate`, `total_pnl_pct`).

Рекомендуемый минимальный completed‑объект:

```json
{
  "id": "exp_...",
  "created_at": "2026-03-10T20:18:00.459984",
  "updated_at": "2026-03-10T21:24:36.389801",
  "completed_at": "2026-03-10T21:24:36.389548",
  "status": "completed",
  "symbol": "ETHUSDT",
  "type": "conservative",
  "params": { "interval": "1h", "no_mtf": true },
  "progress": 100,
  "last_log": "…",
  "results": {
    "model_name": "…",
    "total_trades": 155,
    "win_rate": 69.03,
    "total_pnl_pct": 121.00,
    "models": {
      "15m": null,
      "1h": "ml_models/quad_ensemble_ETHUSDT_60_1h_conservative_exp.pkl"
    }
  }
}
```

## Частые вопросы/симптомы

### “Эксперимент идёт, но данных на графике нет”

Проверьте:

- что `POST /api/paper/start` вернул `{ ok: true }`,
- что `GET /api/paper/status` показывает `status=active` для experiment_id,
- что `GET /api/paper/realtime_chart/{id}` отдаёт `equity_curve` и `timestamps`,
- что серверный `TradingLoop` действительно получает бары по `symbol` сессии (иначе `on_bar` не обновит кривую).

### “Cached 15m data is outdated … updating…”

Это информирование о том, что кэш 15m свечей устарел и выполняется обновление. Обычно не является ошибкой.

### “Cannot open position: signal is too old …”

Это защита от слишком старых сигналов при открытии сделки. Может проявляться, если вход откладывается (например, pullback ждёт несколько свечей).
