# РУЧНОЕ РАЗВЕРТЫВАНИЕ ИСПРАВЛЕНИЯ ЗАВИСАНИЯ

## 🔧 ИНСТРУКЦИЯ ДЛЯ СЕРВЕРА LINUX

Так как Git на Windows машине имеет проблемы с кодировкой, применим исправление вручную на сервере.

### 1. Подключитесь к серверу

```bash
ssh root@s3fe42482.fastvps-server.com
cd /opt/ml_bot
```

### 2. Остановите бота

```bash
sudo systemctl stop ml-bot
```

### 3. Создайте резервную копию

```bash
cp bot/trading_loop.py bot/trading_loop.py.backup
```

### 4. Примените исправление

Откройте файл для редактирования:

```bash
nano bot/trading_loop.py
```

**Найдите строки 276-284** (около строки 280):

```python
# Генерация сигнала
try:
    signal = strategy.generate_signal(
        row=row,
        df=df.iloc[:-1] if len(df) >= 2 else df,  # Используем все данные кроме последней незакрытой свечи
        has_position=has_pos,
        current_price=current_price,  # Используем текущую цену из последней свечи
        leverage=self.settings.leverage
    )
except Exception as e:
    logger.error(f"Error generating signal for {symbol}: {e}")
    return
```

**Замените их на:**

```python
# Генерация сигнала
# КРИТИЧНО: generate_signal() выполняет долгие синхронные операции (feature engineering, model.predict)
# Оборачиваем в to_thread() чтобы не блокировать event loop
try:
    logger.debug(f"[{symbol}] Calling strategy.generate_signal() in thread...")
    signal = await asyncio.to_thread(
        strategy.generate_signal,
        row=row,
        df=df.iloc[:-1] if len(df) >= 2 else df,  # Используем все данные кроме последней незакрытой свечи
        has_position=has_pos,
        current_price=current_price,  # Используем текущую цену из последней свечи
        leverage=self.settings.leverage
    )
    logger.debug(f"[{symbol}] strategy.generate_signal() completed")
except Exception as e:
    logger.error(f"Error generating signal for {symbol}: {e}")
    return
```

**Сохраните файл:**
- Нажмите `Ctrl+O` (WriteOut)
- Нажмите `Enter` (подтвердить имя файла)
- Нажмите `Ctrl+X` (Exit)

### 5. Проверьте синтаксис Python

```bash
cd /opt/ml_bot
source venv/bin/activate
python -m py_compile bot/trading_loop.py
```

Если нет ошибок - все ОК, продолжайте. Если есть ошибки - восстановите из backup:

```bash
cp bot/trading_loop.py.backup bot/trading_loop.py
```

### 6. Перезапустите бота

```bash
sudo systemctl start ml-bot
```

### 7. Следите за логами

```bash
tail -f /opt/ml_bot/logs/bot.log
```

**Ожидаемый вывод (ПРАВИЛЬНЫЙ):**

```
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - Starting Signal Processing Loop...
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - Starting Position Monitoring Loop...
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - Position Monitoring Loop: About to sleep for 10 seconds...
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - [ETHUSDT] 🔄 Loading model...
2026-02-02 XX:XX:XX - bot.trading_loop - DEBUG - [ETHUSDT] Calling strategy.generate_signal() in thread...
2026-02-02 XX:XX:XX - bot.trading_loop - DEBUG - [ETHUSDT] strategy.generate_signal() completed
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - [ETHUSDT] Signal: HOLD | ...
2026-02-02 XX:XX:XX - bot.trading_loop - DEBUG - [ETHUSDT] Signal processing completed, returning from process_symbol
2026-02-02 XX:XX:XX - bot.trading_loop - DEBUG - Signal Processing Loop: Completed processing ETHUSDT
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - [SOLUSDT] 🔄 Loading model...
... (обработка следующих символов) ...
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - Position Monitoring Loop: Sleep completed, continuing...
```

**Если бот ЗАВИСАЕТ** (логи останавливаются после "Signal: HOLD"):

```
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - [ETHUSDT] Signal: HOLD | ...
[НИЧЕГО БОЛЬШЕ - это плохо, исправление не применено]
```

Вернитесь к шагу 4 и проверьте, что изменения применены правильно.

### 8. Проверьте работу всех символов

```bash
# Должны увидеть сигналы для всех 4 символов
grep "Signal:" /opt/ml_bot/logs/bot.log | tail -20
```

Ожидается:
- ETHUSDT: Signal: ...
- SOLUSDT: Signal: ...
- XRPUSDT: Signal: ...
- BTCUSDT: Signal: ...

### 9. Проверьте Position Monitoring Loop

```bash
grep "Position Monitoring Loop" /opt/ml_bot/logs/bot.log | tail -10
```

Должны увидеть циклы "About to sleep" → "Sleep completed" → "About to sleep"

### 10. Удалите backup (если все работает)

```bash
cd /opt/ml_bot
rm bot/trading_loop.py.backup
```

## 🎯 ЧТО БЫЛО ИСПРАВЛЕНО

**Проблема:** `strategy.generate_signal()` выполнял тяжелые синхронные операции (feature engineering, ML inference) прямо в asyncio event loop, что блокировало его на ~200-700ms.

**Решение:** Обернули вызов в `await asyncio.to_thread()`, который выполняет код в отдельном потоке, не блокируя event loop.

**Результат:** Бот теперь обрабатывает все символы без зависания, `_position_monitoring_loop()` работает корректно.

## 📚 ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ

См. файл `FIX_FREEZE_ISSUE.md` для подробного объяснения проблемы и решения.

## ❓ TROUBLESHOOTING

### Проблема: Syntax Error после правки

```bash
# Восстановите из backup
cp bot/trading_loop.py.backup bot/trading_loop.py

# Проверьте отступы (должны быть ПРОБЕЛЫ, не табы)
# Используйте nano, а не vi/vim если не уверены
```

### Проблема: Бот все еще зависает

```bash
# Убедитесь, что изменения применены
grep -A 10 "asyncio.to_thread" bot/trading_loop.py

# Должно показать:
#     signal = await asyncio.to_thread(
#         strategy.generate_signal,
#         ...

# Проверьте логи на ошибки
tail -100 /opt/ml_bot/logs/errors.log
```

### Проблема: ImportError или ModuleNotFoundError

```bash
# Убедитесь, что venv активирован
cd /opt/ml_bot
source venv/bin/activate

# Переустановите зависимости
pip install -r requirements.txt
```
