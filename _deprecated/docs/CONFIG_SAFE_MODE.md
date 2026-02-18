# Безопасный режим - конфигурация без rate limit ошибок

## Проблема с pybit и rate limit

Обнаружен баг в библиотеке `pybit` версии 5.6.x:
- При получении ошибки rate limit (10006), pybit пытается прочитать заголовок `X-Bapi-Limit-Reset-Timestamp`
- Иногда Bybit НЕ возвращает этот заголовок, что вызывает `KeyError` и краш бота
- Ошибка: `KeyError: 'x-bapi-limit-reset-timestamp'`

## Решения

### 1. Обновить pybit (рекомендуется)

```bash
# Обновляем библиотеку
pip install --upgrade pybit

# Или переустанавливаем зависимости
pip install -r requirements.txt --upgrade
```

### 2. Настройки .env для минимального rate limit

Добавьте/измените в вашем `.env` файле:

```bash
# ===== КРИТИЧЕСКИ ВАЖНО =====
# Торгуйте максимум 1-2 парами!
ACTIVE_SYMBOLS=BTCUSDT
# Или: ACTIVE_SYMBOLS=BTCUSDT,ETHUSDT (не более 2!)

# Увеличенный интервал опроса (4 минуты)
LIVE_POLL_SECONDS=240

# Остальные настройки
PRIMARY_SYMBOL=BTCUSDT
TIMEFRAME=15m
LEVERAGE=10

# ML стратегия
ML_CONFIDENCE_THRESHOLD=0.6
ML_MIN_SIGNAL_STRENGTH=среднее

# Risk management
BASE_ORDER_USD=50.0
STOP_LOSS_PCT=1.0
TAKE_PROFIT_PCT=2.0
```

### 3. Экстремальный режим (если проблемы продолжаются)

Для максимальной стабильности:

```bash
# Только ОДНА пара
ACTIVE_SYMBOLS=BTCUSDT

# Интервал 5 минут
LIVE_POLL_SECONDS=300
```

## Сравнение нагрузки на API

| Конфигурация | Запросов/минуту | Риск rate limit |
|--------------|-----------------|-----------------|
| **4 пары, 60 сек** | ~20 | ❌ Очень высокий |
| **2 пары, 120 сек** | ~5-6 | ⚠️ Средний |
| **2 пары, 240 сек** | ~3 | ✅ Низкий |
| **1 пара, 240 сек** | ~1-2 | ✅ Очень низкий |

## Пошаговая инструкция по исправлению

### Шаг 1: Обновите pybit

```bash
# Windows (в директории проекта)
.\venv\Scripts\Activate.ps1
pip install --upgrade pybit
```

```bash
# Linux
source venv/bin/activate
pip install --upgrade pybit
```

### Шаг 2: Настройте .env

Откройте файл `.env` и измените:

```bash
ACTIVE_SYMBOLS=BTCUSDT          # Только 1 пара!
LIVE_POLL_SECONDS=240           # 4 минуты
```

### Шаг 3: Перезапустите бота

```bash
# Windows
.\start_bot.bat

# Или вручную
python run_bot.py
```

### Шаг 4: Проверьте логи

После запуска следите за логами 10-15 минут:

```bash
# Windows PowerShell
Get-Content logs\bot.log -Wait -Tail 20

# Linux
tail -f logs/bot.log
```

**Должны увидеть:**
- ✅ `Starting Signal Processing Loop...`
- ✅ `Starting Position Monitoring Loop...`
- ✅ **НЕТ** ошибок `10006` или `x-bapi-limit-reset-timestamp`

## Временные метки (UTC vs MSK)

⚠️ **ВАЖНО**: Время в логах бота в формате **UTC** (Всемирное время)
- MSK (Москва) = UTC + 3 часа
- Пример: `08:10 UTC` = `11:10 MSK`

## Отладка

### Если ошибки 10006 продолжаются:

1. **Увеличьте интервал до 5 минут:**
   ```bash
   LIVE_POLL_SECONDS=300
   ```

2. **Торгуйте только 1 парой:**
   ```bash
   ACTIVE_SYMBOLS=BTCUSDT
   ```

3. **Проверьте, нет ли других процессов:**
   - Убедитесь, что запущен только ОДИН экземпляр бота
   - Проверьте, не используют ли другие скрипты те же API ключи

### Если бот крашится с `KeyError: 'x-bapi-limit-reset-timestamp'`:

1. **Обновите pybit:**
   ```bash
   pip install pybit --upgrade
   ```

2. **Проверьте версию:**
   ```bash
   pip show pybit
   # Должна быть >= 5.7.0
   ```

3. **Если обновление не помогло:**
   - Используйте экстремальный режим (1 пара, 300 сек)
   - Это предотвратит появление ошибки 10006

## Итоговая рекомендация

**Для стабильной работы 24/7:**

```bash
# В .env файле:
ACTIVE_SYMBOLS=BTCUSDT,ETHUSDT  # Максимум 2 пары
LIVE_POLL_SECONDS=240           # 4 минуты
```

Это обеспечит:
- ✅ Минимальная нагрузка на API (~3 запроса/мин)
- ✅ Нет rate limit ошибок
- ✅ Стабильная работа
- ✅ Достаточно сигналов для торговли

---

**Создано**: 2026-02-02 11:20 MSK (08:20 UTC)
**Статус**: ✅ Протестировано
