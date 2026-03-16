# Операторская инструкция (1 страница)

## Цель

Быстро и безопасно управлять главным AI-агентом без лишней теории.

---

## 1) Старт смены (2–3 минуты)

1. Проверить доступность инструментов:
   - `/tools`
2. Проверить лимиты:
   - `/limits`
3. Проверить, нет ли зависшего подтверждения:
   - `/pending`
4. Получить рыночный контекст:
   - `/market BTCUSDT`
5. Проверить качество последних сделок:
   - `/riskcheck`

Если есть pending из прошлой сессии — сначала `/cancel`.

---

## 2) Базовый рабочий цикл

1. Сформулировать задачу коротко:
   - что нужно;
   - по какому символу;
   - ограничения по риску.
2. Дать агенту собрать данные (он сам вызовет tools/tool_chain).
3. Проверить итог:
   - вывод;
   - ключевые факты;
   - предложенное действие.
4. Если действие изменяет состояние — работать через `request_id`.

Шаблон запроса:

```text
Проанализируй <SYMBOL>, дай сценарий long/short и безопасный следующий шаг с риском не выше <X>%.
```

---

## 3) Изменения риска и критичные действия

## 3.1 Перед изменением

- Убедиться, что понимаете impact.
- Проверить текущий контекст (`/market`, `/riskcheck`, при необходимости `/pending`).

## 3.2 Выполнение

1. Запуск команды через `/tool ...` с `request_id`.
2. Прочитать preview.
3. Подтвердить: `ПОДТВЕРЖДАЮ`.
4. Если передумали: `/cancel`.

Пример:

```text
/tool update_risk_settings {"request_id":"req_1742123555123_risk","updates":{"base_order_usd":20}}
```

---

## 4) Runbook-режимы (когда нужно быстрее и безопаснее)

- Справка:
  - `/runbook help`
- Полуавтоматический валидатор:
  - `/runbook planner paper_auto_validation exp_123 30 auto_apply=true auto_profile=balanced`
- Реакция на инцидент:
  - `/runbook incident_response BTCUSDT`
- Контроль кампаний:
  - `/runbook campaign_watchdog`

Профили auto-apply:

- `conservative` — максимально безопасно;
- `balanced` — стандартный режим;
- `aggressive` — быстрее, выше риск.

Если не уверены — всегда `conservative`.

---

## 5) Аварийный playbook

## Сигналы аварии

- серия убытков подряд;
- резкий рост drawdown;
- нестабильное поведение исполнения;
- критические alerts planner policy.

## Порядок действий

1. Зафиксировать контекст:
   - `/market <SYMBOL>`
   - `/riskcheck`
2. Запустить диагностику:
   - `/runbook incident_response <SYMBOL>`
3. При необходимости остановить риск:
   - `/tool stop_bot {}`
   - или `/tool emergency_stop_all {}`
4. Подтвердить только после проверки preview:
   - `ПОДТВЕРЖДАЮ`

---

## 6) Частые проблемы и быстрые решения

- **Не выполняется действие**
  - `/pending` → `ПОДТВЕРЖДАЮ` или `/cancel`.
- **Превышен лимит**
  - проверить `/limits`, подождать `retry_after_seconds`.
- **/tool не сработал**
  - проверить JSON и имя инструмента из `/tools`.
- **Повторное изменение**
  - всегда использовать новый `request_id`.

---

## 7) Мини-шпаргалка команд

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
/runbook incident_response BTCUSDT

/tool get_trade_history {"limit":10}
/tool update_risk_settings {"request_id":"req_1742","updates":{"base_order_usd":20}}
```

---

## 8) Что нельзя делать

- Не подтверждать high/critical действие, не прочитав preview.
- Не запускать mutating-команды без `request_id`.
- Не игнорировать `/pending` перед новой критичной операцией.
- Не использовать `aggressive` профиль без понятной причины.
