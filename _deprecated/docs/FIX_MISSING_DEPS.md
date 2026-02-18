# 🔧 Исправление отсутствующих зависимостей

## Проблема: ModuleNotFoundError

Если при запуске бота возникает ошибка `ModuleNotFoundError: No module named 'X'`, это означает что зависимость не установлена.

## Решение

### На сервере выполните:

```bash
cd /opt/ml_bot
source venv/bin/activate

# Обновите requirements.txt (если нужно)
git pull origin main

# Установите все зависимости
pip install -r requirements.txt

# Или установите конкретную зависимость
pip install pandas-ta
```

### Проверка установки

```bash
python3 -c "import pandas_ta; print('✅ pandas_ta установлен')"
```

## Обновление requirements.txt

Если зависимость отсутствует в `requirements.txt`, добавьте её:

```bash
# На локальной машине
echo "pandas-ta>=0.3.14b0" >> requirements.txt
git add requirements.txt
git commit -m "Add missing pandas-ta dependency"
git push origin main

# На сервере
git pull origin main
pip install -r requirements.txt
```

## Частые отсутствующие зависимости

Если возникают другие ошибки `ModuleNotFoundError`, установите:

```bash
pip install pandas-ta
pip install ta-lib  # Если используется
pip install yfinance  # Если используется
```

## После установки зависимостей

```bash
# Перезапустите сервис
sudo systemctl restart ml-bot

# Проверьте статус
sudo systemctl status ml-bot

# Проверьте логи
tail -f /opt/ml_bot/logs/bot.log
```
