# Полный гайд по главному AI-агенту

## 1) Что это за агент и как он работает

Главный AI-агент — это диалоговый слой управления ботом.  
Он умеет:

- отвечать на вопросы по рынку, рискам, сделкам и экспериментам;
- вызывать инструменты (API) по вашей задаче;
- выполнять многошаговые сценарии (`tool_chain`);
- показывать статус и ожидать подтверждение для рискованных действий;
- формировать итоговый ответ простым человеческим языком.

Базовая модель работы:

1. Агент анализирует ваш запрос.
2. Определяет, нужно ли вызвать инструменты.
3. При необходимости вызывает один или несколько инструментов.
4. Объединяет результат и возвращает понятный итог.

---

## 2) Быстрый старт (что делать сразу)

1. Откройте вкладку **AI Chat**.
2. Для знакомства отправьте:
   - `/tools` — список доступных инструментов;
   - `/limits` — текущие лимиты;
   - `/pending` — есть ли ожидающее подтверждение.
3. Для базовой аналитики:
   - `/market BTCUSDT`
   - `/riskcheck`
4. Для runbook-помощников:
   - `/runbook help`

---

## 3) Форматы команд в чате

## 3.1 Служебные команды

- `/tools` — список инструментов (с risk-tier и назначением)
- `/limits` — rate limit + idempotency состояние
- `/pending` — текущее ожидающее действие (если есть)
- `/cancel` — отменить ожидающее действие
- `ПОДТВЕРЖДАЮ` — подтвердить high/critical действие

Отмена также срабатывает на: `cancel`, `отмена`, `отменить`, `отбой`.

## 3.2 Типовые команды аналитики

- `/market [SYMBOL]`  
  Пример: `/market BTCUSDT`

- `/riskcheck`  
  Анализ последних 10 сделок + предложения по риску.

## 3.3 Ручной вызов инструмента

Формат:

```text
/tool <tool_name> <json_arguments>
```

Примеры:

```text
/tool get_trade_history {"limit":10}
/tool update_risk_settings {"request_id":"req_1742123001","updates":{"base_order_usd":20}}
/tool apply_research_experiment {"request_id":"req_1742123002","experiment_id":"exp_123"}
```

Если JSON некорректный, команда не будет принята.

## 3.4 Runbook-команды

- `/runbook help`
- `/runbook paper_validate_apply [experiment_id]`
- `/runbook incident_response [symbol]`
- `/runbook campaign_watchdog`
- `/runbook paper_auto_validation [experiment_id] [window_minutes]`
- `/runbook planner paper_auto_validation [experiment_id] [window_minutes] [auto_apply=true|false] [auto_profile=conservative|balanced|aggressive]`

Planner также поддерживает сценарии:

- `incident_response`
- `campaign_watchdog`

---

## 4) Уровни риска (risk-tier) и подтверждения

Система делит инструменты на 4 уровня:

- **low** — чтение, без изменения состояния;
- **medium** — ограниченные изменения;
- **high** — существенные изменения (обычно нужен `request_id`);
- **critical** — самые опасные действия.

Для **high/critical**:

1. Агент создаёт pending-действие.
2. Показывает предпросмотр и TTL.
3. Вы подтверждаете фразой `ПОДТВЕРЖДАЮ` или отменяете `/cancel`.

Если TTL истёк — действие сбрасывается, нужно сформировать его заново.

---

## 5) Rate limit и idempotency (очень важно)

### Rate limit

Для чат-инструментов действует лимит вызовов в минуту.  
Проверка:

- `/limits`

Если лимит превышен, агент вернёт `retry_after_seconds`.

### Idempotency

Для изменяющих действий используйте `request_id`, чтобы:

- не применить одно и то же изменение дважды;
- безопасно повторить команду при сетевой ошибке.

Рекомендуемый формат:

```text
req_<timestamp_ms>_<short_tag>
```

Пример:

```text
req_1742123555123_risk
```

---

## 6) Основные инструменты (что для чего)

Ниже практическая группировка из whitelist-контракта.

## 6.1 Мониторинг и чтение

- `get_bot_status` — состояние бота
- `get_bot_stats` — агрегированные метрики
- `get_trade_history` — история сделок
- `get_ai_research_status` — список экспериментов
- `get_experiment_health` — health эксперимента
- `get_experiment_report` — markdown-отчёт по эксперименту
- `get_paper_metrics` — paper-метрики
- `get_paper_realtime_chart` — realtime equity paper

## 6.2 Управление исследованиями и paper

- `start_research_experiment`
- `control_research_campaign` (`pause/resume/stop`)
- `start_paper_trading`
- `stop_paper_trading`

## 6.3 Управление риском и боевые действия

- `update_risk_settings` (high)
- `apply_research_experiment` (critical)
- `stop_bot` (critical)
- `emergency_stop_all` (critical)

---

## 7) Сценарии использования (пошагово)

## Сценарий A: Быстрый рынок + решение

1. `/market BTCUSDT`
2. Смотрите trend/volatility/confidence/advice.
3. При необходимости запрашиваете уточнение обычным текстом:
   - `Какие риски для лонга сейчас и какой стоп логичен?`

Когда применять: быстрый вход в контекст перед решением.

## Сценарий B: Проверка качества торговли

1. `/riskcheck`
2. Смотрите win/loss, total pnl, ai risk score.
3. Если рекомендации адекватны — применяете через `/tool update_risk_settings ...` с `request_id`.
4. Подтверждаете `ПОДТВЕРЖДАЮ`.

Когда применять: серия убыточных/нестабильных сделок.

## Сценарий C: Безопасный apply эксперимента

1. `/runbook paper_validate_apply exp_...`
2. Агент проверяет гейты и формирует safe apply команду.
3. Выполняете предложенную команду с `request_id`.
4. Подтверждаете `ПОДТВЕРЖДАЮ`.

Когда применять: перевод кандидата в рабочую стратегию.

## Сценарий D: Автономный planner

1. `/runbook planner paper_auto_validation exp_... 30 auto_apply=true auto_profile=balanced`
2. Агент пройдёт этапы `plan → execute → verify → summarize`.
3. Если риск команды выше разрешённого профиля — auto-apply заблокируется политикой.

Когда применять: полуавтоматическая операционная рутина.

## Сценарий E: Инцидент и эскалация

1. `/runbook incident_response BTCUSDT`
2. Получаете оценку риска и рекомендованное действие.
3. При критике — переходите к `emergency_stop_all` с подтверждением.

Когда применять: резкий drawdown, каскад ошибок, abnormal рынок.

---

## 8) Planner policy, аналитика и экспорт

Для контроля planner-политик есть API:

- `GET /api/ai/chat/planner_policy_stats`
- `GET /api/ai/chat/planner_policy_config`
- `PUT /api/ai/chat/planner_policy_config`
- `GET /api/ai/chat/planner_policy_export`

Что вы получаете:

- blocked/executed/skipped статистику;
- conversion по profile/scenario;
- time buckets;
- long-term daily/weekly;
- suggested profile;
- alerts (включая planner_policy_alert).

---

## 9) Как писать запросы, чтобы агент отвечал лучше

Плохой запрос:

- `проверь`

Хороший запрос:

- `Проанализируй BTCUSDT на 1ч и 15м, дай сценарий лонга и шорта с риском не выше 1.5% на сделку.`

Ещё лучше:

- `Сначала оцени рынок по BTCUSDT, потом проверь последние 10 сделок и предложи, менять ли base_order_usd.`

Шаблон:

- **Цель:** что хотите получить;
- **Контекст:** символ, горизонт, режим;
- **Ограничения:** риск, просадка, плечо;
- **Формат ответа:** кратко/детально, с шагами.

---

## 10) Частые ошибки и как исправить

### Ошибка 1: “Ожидающее действие не выполняется”

Причина: не подтверждено или истёк TTL.  
Решение: `/pending` → `ПОДТВЕРЖДАЮ` или сформировать команду заново.

### Ошибка 2: “Превышен лимит вызовов”

Причина: rate limit.  
Решение: подождать `retry_after_seconds`, проверить `/limits`.

### Ошибка 3: “Команда /tool не сработала”

Причина: неверный JSON или не-whitelist инструмент.  
Решение: сверить синтаксис и список `/tools`.

### Ошибка 4: “Повторно применилось изменение”

Причина: нет `request_id` в mutating-команде.  
Решение: всегда указывать `request_id`.

---

## 11) Рекомендуемый безопасный режим работы

1. Всегда начинайте с чтения: `/tools`, `/market`, `/riskcheck`.
2. Для любых изменений используйте `request_id`.
3. Не подтверждайте high/critical, пока не прочитали preview.
4. Для сложных задач используйте runbook/planner вместо ручной последовательности.
5. При сомнениях выбирайте `auto_profile=conservative`.

---

## 12) Мини-шпаргалка команд

```text
/tools
/limits
/pending
/cancel
ПОДТВЕРЖДАЮ

/market BTCUSDT
/riskcheck

/runbook help
/runbook planner paper_auto_validation exp_123 30 auto_apply=true auto_profile=balanced

/tool get_trade_history {"limit":10}
/tool update_risk_settings {"request_id":"req_1742","updates":{"base_order_usd":20}}
```

---

## 13) Куда смотреть для диагностики

- История чата: `GET /api/ai/chat/history`
- Список инструментов и pending: `GET /api/ai/chat/tools`
- Лимиты: `GET /api/ai/chat/limits`
- Журнал вызовов: `GET /api/ai/chat/tool_execution_log`
- Логи:
  - `logs/tool_execution_log.jsonl`
  - `logs/tool_execution.log`

---

Если нужно, могу сделать вторую версию этого гайда в формате “операторская инструкция” (чек-листы на каждый день + аварийный playbook на 1 страницу).
