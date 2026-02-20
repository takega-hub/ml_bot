# Деплой бота на сервер (обновление через Git)

При обновлении через `git pull` **не попадают** в репозиторий следующие файлы (см. `.gitignore`). Их нужно создавать/копировать на сервер вручную или хранить отдельно.

---

## Обязательные файлы (без них бот не запустится или будет с дефолтами)

| Файл | Где лежит | Назначение |
|------|-----------|------------|
| **`.env`** | Корень проекта | API-ключи и секреты: `BYBIT_API_KEY`, `BYBIT_API_SECRET`, `TELEGRAM_TOKEN`, при необходимости `TRADING_SYMBOLS`, `PRIMARY_SYMBOL`, `LEVERAGE` и др. Без `.env` бот не подключится к бирже и Telegram. |

---

## Файлы настроек (если нет — используются значения по умолчанию)

| Файл | Где лежит | Назначение |
|------|-----------|------------|
| **`ml_settings.json`** | Корень проекта | ML-стратегия: `confidence_threshold`, `use_mtf_strategy`, пороги 1h/15m, **`use_dynamic_ensemble_weights`**, **`trend_weights`** / **`flat_weights`**, `pullback_enabled` и др. При отсутствии — дефолты из `bot/config.py`. |
| **`risk_settings.json`** | Корень проекта | Риск: `margin_pct_balance`, `base_order_usd`, стоп-лосс, тейк-профит, трейлинг, безубыток и т.д. При отсутствии — дефолты. |
| **`symbol_ml_settings.json`** | Корень проекта (или `config/`, см. `_get_ml_settings_file`) | Выбранные модели по символам (пути к .pkl). Если нет — автопоиск по шаблону в `ml_models/`. |

Путь к `risk_settings.json` и `symbol_ml_settings.json`: **корень проекта** (родитель каталога `bot/`), т.е. там же, где `run_bot.py`.

---

## Модели и состояние

| Файл/каталог | Назначение |
|--------------|------------|
| **`ml_models/*.pkl`** | Обученные модели. В git не коммитятся. Нужно копировать на сервер отдельно (архив/скопировать папку после обучения). |
| **`runtime_state.json`** | Состояние бота (в т.ч. выбранные модели по символам). Не в git. Создаётся при работе. |

---

## Чеклист перед/после деплоя

1. **Перед первым запуском на сервере**
   - [ ] Создать/скопировать `.env` с ключами Bybit и Telegram.
   - [ ] При необходимости скопировать `ml_settings.json` и `risk_settings.json` (или создать из примеров ниже).
   - [ ] Скопировать папку `ml_models/` с нужными `.pkl` (или обучить на сервере).

2. **После `git pull`**
   - [ ] Проверить, что `.env` на месте (git его не перезаписывает).
   - [ ] Если добавлялись новые поля в `ml_settings.json` / `risk_settings.json` — при необходимости обновить свои копии на сервере (или дать боту создать файлы с дефолтами через Telegram/код).
   - [ ] Убедиться, что в `ml_models/` есть актуальные модели для торгуемых пар.

3. **Проверка работы улучшений**
   - При старте бот пишет в лог блок **`[DEPLOY] Server config`** с флагами: MTF, динамические веса ансамбля, pullback, ATR-адаптивный порог и т.д.
   - Смотреть `logs/bot.log` после запуска: секция `[DEPLOY]` покажет, какие опции реально включены и какие файлы найдены.

---

## Пример минимального `ml_settings.json`

```json
{
  "confidence_threshold": 0.35,
  "min_confidence_for_trade": 0.5,
  "use_mtf_strategy": false,
  "mtf_confidence_threshold_1h": 0.5,
  "mtf_confidence_threshold_15m": 0.35,
  "mtf_alignment_mode": "strict",
  "mtf_require_alignment": true,
  "pullback_enabled": true,
  "use_dynamic_ensemble_weights": false,
  "adx_trend_threshold": 25,
  "adx_flat_threshold": 20,
  "trend_weights": null,
  "flat_weights": null
}
```

Для включения динамических весов ансамбля задайте `"use_dynamic_ensemble_weights": true` и укажите `trend_weights` / `flat_weights` (см. STRATEGY_IMPROVEMENTS_ROADMAP.md, пункт 10).

---

## Где смотреть логи на сервере

- Основной лог: `logs/bot.log`
- Сделки: `logs/trades.log`
- Сигналы: `logs/signals.log`
- Ошибки: `logs/errors.log`

После старта в `logs/bot.log` ищите строки `[DEPLOY]` — по ним видно, какие улучшения активны и какие конфиги подгружены.
