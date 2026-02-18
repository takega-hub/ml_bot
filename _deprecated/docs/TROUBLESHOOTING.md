# 🔧 Устранение проблем с запуском бота

## Проблема: Сервис падает с ошибкой (status=1/FAILURE)

### Шаг 1: Проверьте логи ошибок

```bash
# Логи systemd
sudo journalctl -u ml-bot -n 100 --no-pager

# Логи бота
cat /opt/ml_bot/logs/errors.log
tail -f /opt/ml_bot/logs/bot.log
```

### Шаг 2: Запустите бота вручную для диагностики

```bash
cd /opt/ml_bot
source venv/bin/activate
python3 run_bot.py
```

Это покажет реальную ошибку, которая не видна в systemd логах.

### Шаг 3: Частые причины ошибок

#### 1. Отсутствует .env файл

```bash
# Проверьте наличие .env
ls -la /opt/ml_bot/.env

# Если файла нет, создайте его
nano /opt/ml_bot/.env
```

Добавьте необходимые переменные:
```
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
TELEGRAM_TOKEN=your_token
ALLOWED_USER_ID=your_id
```

#### 2. Не установлены зависимости

```bash
cd /opt/ml_bot
source venv/bin/activate
pip install -r requirements.txt
```

#### 3. Неправильный путь к Python

Проверьте путь в service файле:

```bash
# Проверьте что Python существует
/opt/ml_bot/venv/bin/python3 --version

# Если путь неправильный, обновите service файл
sudo nano /etc/systemd/system/ml-bot.service
```

#### 4. Отсутствуют директории

```bash
cd /opt/ml_bot
mkdir -p logs ml_models ml_data backtest_reports backtest_plots
```

#### 5. Проблемы с правами доступа

```bash
# Проверьте права
ls -la /opt/ml_bot

# Если нужно, измените владельца
sudo chown -R root:root /opt/ml_bot
chmod +x /opt/ml_bot/run_bot.py
```

#### 6. Отсутствует Telegram токен

Проверьте .env файл:
```bash
grep TELEGRAM_TOKEN /opt/ml_bot/.env
```

Если токена нет, добавьте его в .env

### Шаг 4: Проверка всех зависимостей

```bash
cd /opt/ml_bot
source venv/bin/activate

# Проверьте основные библиотеки
python3 -c "import telegram; print('telegram OK')"
python3 -c "import pandas; print('pandas OK')"
python3 -c "import numpy; print('numpy OK')"
python3 -c "import sklearn; print('sklearn OK')"
python3 -c "from pybit import HTTP; print('pybit OK')"
```

### Шаг 5: Тест импорта основного модуля

```bash
cd /opt/ml_bot
source venv/bin/activate
python3 -c "from bot.config import load_settings; print('Config OK')"
```

### Шаг 6: Обновление service файла с дополнительной диагностикой

Если нужно больше информации, обновите service файл:

```bash
sudo nano /etc/systemd/system/ml-bot.service
```

Добавьте в секцию `[Service]`:

```ini
# Для отладки - запуск через bash
# ExecStart=/bin/bash -c 'cd /opt/ml_bot && source venv/bin/activate && python3 run_bot.py'

# Или добавьте переменные окружения для отладки
Environment="PYTHONUNBUFFERED=1"
Environment="PYTHONPATH=/opt/ml_bot"
```

Затем:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ml-bot
```

## Диагностический скрипт

Создайте файл `check_bot.sh`:

```bash
#!/bin/bash
echo "=== Проверка окружения ==="
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo ""
echo "=== Проверка директорий ==="
ls -la /opt/ml_bot/ | head -20
echo ""
echo "=== Проверка .env ==="
if [ -f /opt/ml_bot/.env ]; then
    echo ".env существует"
    grep -v "SECRET\|KEY\|TOKEN" /opt/ml_bot/.env | head -5
else
    echo "❌ .env не найден!"
fi
echo ""
echo "=== Проверка venv ==="
if [ -f /opt/ml_bot/venv/bin/python3 ]; then
    echo "venv Python: $(/opt/ml_bot/venv/bin/python3 --version)"
else
    echo "❌ venv не найден!"
fi
echo ""
echo "=== Проверка зависимостей ==="
cd /opt/ml_bot
source venv/bin/activate
python3 -c "import sys; print('Python path:', sys.executable)"
python3 -c "import telegram, pandas, numpy, sklearn; print('✅ Основные зависимости OK')" 2>&1
echo ""
echo "=== Попытка импорта бота ==="
python3 -c "from bot.config import load_settings; s=load_settings(); print('✅ Config OK')" 2>&1
```

Запустите:
```bash
chmod +x check_bot.sh
./check_bot.sh
```

## После исправления

1. Перезагрузите systemd:
```bash
sudo systemctl daemon-reload
```

2. Перезапустите сервис:
```bash
sudo systemctl restart ml-bot
```

3. Проверьте статус:
```bash
sudo systemctl status ml-bot
```

4. Следите за логами:
```bash
tail -f /opt/ml_bot/logs/bot.log
```
