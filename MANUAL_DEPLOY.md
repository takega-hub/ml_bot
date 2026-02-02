# РУЧНОЕ РАЗВЕРТЫВАНИЕ (минуя Git)

## Проблема
PowerShell на Windows имеет проблемы с кодировкой, Git не работает корректно.

## Решение: Прямое копирование файла

### Вариант 1: Через созданный скрипт

1. Запустите `copy_to_server.bat` (двойной клик)
2. Введите пароль root
3. Дождитесь сообщения "SUCCESS!"
4. На сервере выполните:
   ```bash
   cd /opt/ml_bot
   find . -name "*.pyc" -delete
   sudo systemctl restart ml-bot
   tail -f /opt/ml_bot/logs/bot.log
   ```

### Вариант 2: Через WinSCP (GUI)

1. Откройте WinSCP
2. Подключитесь к: `s3fe42482.fastvps-server.com`
3. Логин: `root`
4. Перейдите локально: `C:\Users\takeg\OneDrive\Документы\vibecodding\ml_bot\bot\`
5. Перейдите удаленно: `/opt/ml_bot/bot/`
6. Перетащите `trading_loop.py` из левой панели в правую
7. Подтвердите замену
8. На сервере:
   ```bash
   cd /opt/ml_bot
   find . -name "*.pyc" -delete
   sudo systemctl restart ml-bot
   tail -f /opt/ml_bot/logs/bot.log
   ```

### Вариант 3: Через командную строку (CMD, не PowerShell!)

Откройте **cmd.exe** (не PowerShell):

```cmd
cd C:\Users\takeg\OneDrive\Документы\vibecodding\ml_bot
scp bot\trading_loop.py root@s3fe42482.fastvps-server.com:/opt/ml_bot/bot/trading_loop.py
```

Введите пароль root.

Затем на сервере:
```bash
cd /opt/ml_bot
find . -name "*.pyc" -delete
sudo systemctl restart ml-bot
tail -f /opt/ml_bot/logs/bot.log
```

## Проверка после копирования

На сервере выполните:

```bash
# Проверьте количество строк (должно быть ~908)
wc -l /opt/ml_bot/bot/trading_loop.py

# Проверьте дату модификации (должна быть сегодняшняя)
ls -lh /opt/ml_bot/bot/trading_loop.py

# Проверьте ключевую строку (должна содержать logger.info с эмодзи)
grep -n "🚀 START process_symbol" /opt/ml_bot/bot/trading_loop.py

# Должно показать:
# 178:            logger.info(f"[{symbol}] 🚀 START process_symbol()")
```

Если видите эту строку - файл скопирован правильно!

## Что должно быть в логах после перезапуска

```
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - 🔄 Signal Processing Loop: Processing 4 symbols...
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - 🎯 Signal Processing Loop: Starting to process ETHUSDT
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - [ETHUSDT] 🚀 START process_symbol()
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - [ETHUSDT] Checking cooldown...
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - [ETHUSDT] No cooldown, continuing...
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - [ETHUSDT] 📊 Fetching kline data...
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - [ETHUSDT] ✅ Kline data received: 200 candles
2026-02-02 XX:XX:XX - bot.trading_loop - INFO - [ETHUSDT] 🔄 Loading model: ...
... и так далее для всех символов ...
```

Если видите эти эмодзи - новая версия файла работает!
