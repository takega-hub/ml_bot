# Перезапуск бота после оптимизации

## Быстрый перезапуск

### Шаг 1: Остановка бота

**Windows (если запущен в терминале)**:
```bash
# Нажмите Ctrl+C в терминале с ботом
```

**Windows (если запущен как служба)**:
```bash
# Найдите процесс Python
tasklist | findstr python
# Остановите процесс (замените PID на реальный)
taskkill /PID <PID> /F
```

**Linux (systemd)**:
```bash
sudo systemctl stop ml-bot
```

### Шаг 2: Активация виртуального окружения

**Windows**:
```bash
cd C:\Users\takeg\OneDrive\Документы\vibecodding\ml_bot
venv\Scripts\activate
```

**Linux**:
```bash
cd ~/ml_bot
source venv/bin/activate
```

### Шаг 3: Запуск бота

**Простой запуск (для тестирования)**:
```bash
python run_bot.py
```

**Запуск в фоне (Windows)**:
```bash
start /B python run_bot.py > bot_output.log 2>&1
```

**Запуск как служба (Linux)**:
```bash
sudo systemctl start ml-bot
sudo systemctl status ml-bot
```

## Проверка работы

### 1. Проверка логов

```bash
# Windows
type logs\bot.log | findstr /C:"Starting" /C:"10006"

# Linux
tail -f logs/bot.log | grep -E "(Starting.*Loop|rate limit|10006)"
```

**Должны увидеть**:
```
2026-02-02 08:XX:XX - bot.trading_loop - INFO - Starting Signal Processing Loop...
2026-02-02 08:XX:XX - bot.trading_loop - INFO - Starting Position Monitoring Loop...
```

**НЕ должны видеть**:
```
ErrCode: 10006 - Too many visits
```

### 2. Мониторинг в реальном времени

```bash
# Следить за последними логами
tail -f logs/bot.log
```

### 3. Проверка через Telegram

Отправьте боту команду:
```
/status
```

Должен ответить со статусом и информацией о позициях.

## Что изменилось

✅ **Меньше запросов к API** - проблема с rate limit решена
✅ **Один запрос для всех позиций** вместо отдельных для каждого символа
✅ **Увеличены интервалы** между циклами проверки
✅ **Добавлены паузы** между обработкой разных символов

## Настройки (опционально)

Если хотите дополнительно снизить нагрузку, отредактируйте `.env`:

```bash
# Интервал между циклами (секунды)
LIVE_POLL_SECONDS=180

# Торговать меньшим количеством пар
ACTIVE_SYMBOLS=BTCUSDT,ETHUSDT
```

После изменения `.env` обязательно **перезапустите бота**.

## Устранение проблем

### Если бот не запускается

1. Проверьте, что виртуальное окружение активировано
2. Проверьте наличие файла `.env` с API ключами
3. Проверьте логи на наличие ошибок: `type logs\errors.log`

### Если ошибки 10006 продолжаются

1. Увеличьте `LIVE_POLL_SECONDS` до 300
2. Уменьшите количество активных символов до 1-2
3. Проверьте, нет ли других процессов, использующих тот же API ключ

### Проверка активного процесса

**Windows**:
```bash
tasklist | findstr python
```

**Linux**:
```bash
ps aux | grep python | grep run_bot
```

---

**Готово!** Бот должен работать стабильно без ошибок rate limit.

Подробную информацию об оптимизации см. в `API_RATE_LIMIT_FIX.md`
