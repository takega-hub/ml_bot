# AI Chat Agent — статус реализации и план внедрения

## 1) Что уже реализовано

### 1.1 Tool-calling core
- Введён whitelist-контракт инструментов: `docs/ai_chat_tools.json`.
- Реализован backend tool executor в чате:
  - валидация аргументов по `input_schema`,
  - уровни риска `low/medium/high/critical`,
  - подтверждение для high/critical.
- Добавлены команды управления ожиданием:
  - подтверждение через фразу `ПОДТВЕРЖДАЮ`,
  - отмена: `/cancel`, `cancel`, `отмена`, `отменить`, `отбой`,
  - TTL для pending-действий с авто-истечением.

### 1.2 API для управления и отладки
- `GET /api/ai/chat/tools` — список доступных инструментов, confirmation phrase, TTL и текущий pending.
- `GET /api/ai/chat/tool_execution_log` — журнал tool execution с фильтрами:
  - `limit`,
  - `event_type`,
  - `tool_name`,
  - `risk_tier`,
  - `ok`.
- В `POST /api/ai/chat` добавлены:
  - обработка `/tools`,
  - обработка `/pending`,
  - обработка подтверждения/отмены/истечения pending.

### 1.3 Audit trail
- JSONL журнал: `logs/tool_execution_log.jsonl`.
- Человекочитаемый журнал: `logs/tool_execution.log`.
- Логируются события:
  - `pending_created`,
  - `pending_cancelled`,
  - `pending_expired`,
  - `tool_executed`.

### 1.4 Mobile/UI слой
- `ApiService` расширен методами:
  - `getChatTools()`,
  - `getChatToolExecutionLog(...)`.
- В AI Chat UI добавлены:
  - быстрые кнопки `Tools`, `Pending`, `Cancel`, `Audit`,
  - окно отладки с доступными инструментами и последними событиями,
  - переключатель автопрокрутки `ON/OFF`.

## 2) Что ещё предстоит сделать

### 2.1 Надёжность и безопасность
- Добавить защиту от повторного выполнения критичных действий по `request_id`.
- Ввести rate limit на tool-вызовы из одного чата.
- Добавить role-based policy для инструментов (оператор/админ).
- Добавить deny-list аргументов для потенциально опасных полей.

### 2.2 Наблюдаемость
- Добавить endpoint статистики по tool usage:
  - количество вызовов по инструментам,
  - доля `ok/error`,
  - средняя задержка.
- Добавить UI-таблица аудита с фильтрами и экспортом.

### 2.3 Качество агентного поведения
- Добавить planner-процедуру для многошаговых задач (plan → execute → verify → summarize).
- Добавить self-check перед критическими действиями:
  - обязательная проверка состояния бота,
  - наличие свежих метрик по символу/эксперименту.
- Добавить стандартные “safe pipelines” для типовых операций.

## 3) План внедрения типовых запросов

### 3.1 Типовой запрос: анализ рынка по активным символам
**Цель:** по выбранному активному символу выдавать сжатый actionable-отчёт.

Статус: реализован в чате командой `/market` (или `/market SYMBOL`) с быстрым UI-кнопкой `Market`.

**Сценарий:**
1. Получить список активных символов.
2. Пользователь выбирает символ.
3. Выполнить market insight + статус + краткий срез по открытым позициям.
4. Сформировать ответ в формате:
   - режим рынка,
   - волатильность,
   - рекомендации (что делать/чего избегать),
   - уровень уверенности.

**Инструменты:**
- `get_bot_status`
- `get_ai_research_status` (для контекста экспериментов)
- `getAIMarketInsight`/`/api/ai/market_insight`

### 3.2 Типовой запрос: проверка последних 10 сделок и предложения по рискам
**Цель:** быстро диагностировать качество торговли и дать корректировки risk settings.

Статус: реализован в чате командой `/riskcheck` с быстрым UI-кнопкой `Risk Check`.

**Сценарий:**
1. Получить последние 10 закрытых сделок.
2. Посчитать win/loss, средний pnl, выбросы.
3. Запросить AI risk analysis.
4. Сформировать предложения по 2–4 параметрам риска.
5. При согласии пользователя — применить через `update_risk_settings` с подтверждением.

**Инструменты:**
- `get_trade_history` (`limit=10`)
- `get_bot_stats`
- `getAIRiskAnalysis`/`/api/ai/analyze_risks`
- `update_risk_settings` (high-risk с подтверждением)

### 3.3 Типовой запрос: статус и управление экспериментами
**Цель:** управлять исследовательским циклом без ручного переключения экранов.

**Сценарий:**
1. Получить текущие эксперименты.
2. Показать кандидатов в проблемном статусе.
3. Предложить действие: `pause/resume/stop/start`.
4. Выполнить действие и вернуть подтверждённый результат.

**Инструменты:**
- `get_ai_research_status`
- `get_experiment_health`
- `control_research_campaign`
- `start_research_experiment`

## 4) Что ещё добавить в типовые запросы инструментов

### 4.1 Рекомендуемые новые шаблоны
1. **“Ежедневный pre-trade чек”**
   - статус бота, позиции, риск-профиль, аномалии.
2. **“Paper validation before apply”**
   - старт paper, контроль метрик N минут, решение apply/skip.
3. **“Incident response”**
   - при резком drawdown: диагностика + рекомендованное действие.
4. **“Campaign watchdog”**
   - обнаружение stale/зависших экспериментов и рекомендации.
5. **“Post-mortem сделки”**
   - разбор последней убыточной сделки + точечные улучшения.

### 4.2 Рекомендуемые системные возможности
- Макросы команд: сохранённые многошаговые сценарии.
- Режим “только чтение” для безопасной аналитики.
- Планировщик фоновых отчётов по расписанию.
- Фича флаги для постепенного включения инструментов.

## 5) Порядок ближайшего внедрения

### Этап A (быстро)
1. Типовые запросы:
   - анализ рынка по выбранному активному символу,
   - проверка последних 10 сделок и риск-рекомендации.
2. UI-шаблоны кнопок для этих запросов в чате.

### Этап B (управляемость)
1. Полноценная страница аудита tool execution в UI.
2. Rate limit и request-id idempotency.

Статус:
- Выполнено: отдельный экран аудита в UI, backend rate limit для chat tool execution, idempotency cache для mutating инструментов, endpoint `GET /api/ai/chat/limits`.
- Выполнено: UI-предзаполнения request_id для high/critical tool-команд (`update_risk_settings`, `apply_research_experiment`, `stop_bot`, `emergency_stop_all`).

### Этап C (автономность)
1. Многошаговый planner с верификацией.
2. Полуавтоматические runbook-сценарии для research/paper/apply.

Статус:
- Выполнено (базово): runbook-команда `/runbook paper_validate_apply [experiment_id]` с safety-checks (completed, health, oos, drift, stress) и генерацией safe apply команды с request_id.
- Выполнено (базово): runbook-команды `/runbook incident_response [symbol]`, `/runbook campaign_watchdog`, `/runbook paper_auto_validation [experiment_id]`.
- Выполнено: paper auto-validation поддерживает окно наблюдения (`/runbook paper_auto_validation [experiment_id] [window_minutes]`) и выдаёт автоматические рекомендации continue/stop/start/apply с командными шаблонами.
- Выполнено: расширены метрики окна (rolling volatility, consecutive losses) и добавлена авто-эскалация incident_response при повторяющихся high/critical инцидентах за 24ч.
- Выполнено (базово): planner-режим `/runbook planner paper_auto_validation ...` с этапами plan → execute → verify → summarize и флагом auto_apply.
- Выполнено: planner расширен на `incident_response` и `campaign_watchdog`.
- Выполнено: добавлена политика `auto_apply_max_risk` для mixed-risk цепочек (блокировка auto-apply при превышении риск-порога).
- Выполнено: policy-профили auto-apply (`auto_profile=conservative|balanced|aggressive`) с маппингом в риск-порог.
- Выполнено: журнал причин блокировки/применения planner policy в audit trail (`planner_policy_blocked`, `planner_auto_apply_executed`, `planner_auto_apply_skipped`).
- Выполнено: endpoint `GET /api/ai/chat/planner_policy_stats` и виджет policy analytics в Audit (окно 6h/24h/72h/7d, totals, top profile/scenario).
- Выполнено: расширенная визуализация policy-трендов (time buckets) и сравнение профилей/сценариев по conversion (actionable и blocked→executed).
- Выполнено: экспорт policy analytics (`GET /api/ai/chat/planner_policy_export`) и UI-действие Export CSV.
- Выполнено: алерты деградации conversion по активному профилю и по падению последнего bucket.
- Выполнено: push/notification-канал policy alerting через Telegram + audit события `planner_policy_alert` с cooldown.
- Выполнено: конфиг порогов policy alerts из UI (profile, min actionable, min conversion, cooldown) с backend persistence.
- Выполнено: долгосрочные тренды policy analytics (daily/weekly агрегаты) и отображение в Audit UI.
- Выполнено: backend-рекомендация `suggested_profile` и UI-действие Apply Suggested Profile.
- В работе: авто-тюнинг порогов policy alerts на основе долгосрочной конверсии и волатильности.
