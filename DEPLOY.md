# 🚀 Инструкция по деплою ML Trading Bot

## Подготовка репозитория

### 1. Инициализация Git (если еще не сделано)

```bash
git init
git add .
git commit -m "Initial commit"
```

### 2. Создание удаленного репозитория

Создайте репозиторий на GitHub/GitLab/Bitbucket и добавьте remote:

```bash
git remote add origin https://github.com/yourusername/ml_bot.git
git branch -M main
git push -u origin main
```

## Деплой на сервер

### Вариант 1: Автоматический деплой через скрипт

1. **Клонируйте репозиторий на сервер:**

```bash
git clone https://github.com/yourusername/ml_bot.git
cd ml_bot
```

2. **Настройте переменные окружения:**

```bash
cp .env.example .env  # Если есть пример
nano .env  # Отредактируйте файл
```

Добавьте в `.env`:
```
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
TELEGRAM_TOKEN=your_telegram_token
ALLOWED_USER_ID=your_user_id
```

3. **Создайте виртуальное окружение:**

```bash
python3 -m venv venv
source venv/bin/activate
```

4. **Запустите скрипт деплоя:**

```bash
chmod +x deploy.sh
./deploy.sh
```

### Вариант 2: Ручной деплой

1. **Клонируйте/обновите репозиторий:**

```bash
cd /path/to/ml_bot
git pull origin main
```

2. **Активируйте виртуальное окружение:**

```bash
source venv/bin/activate
```

3. **Установите/обновите зависимости:**

```bash
pip install -r requirements.txt
```

4. **Создайте необходимые директории:**

```bash
mkdir -p logs ml_models ml_data backtest_reports backtest_plots
```

## Настройка автозапуска (systemd)

### Создание service файла

Создайте файл `/etc/systemd/system/ml-bot.service`:

```ini
[Unit]
Description=ML Trading Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/ml_bot
Environment="PATH=/path/to/ml_bot/venv/bin"
ExecStart=/path/to/ml_bot/venv/bin/python3 run_bot.py
Restart=always
RestartSec=10
StandardOutput=append:/path/to/ml_bot/logs/bot.log
StandardError=append:/path/to/ml_bot/logs/errors.log

[Install]
WantedBy=multi-user.target
```

### Управление сервисом

```bash
# Перезагрузить конфигурацию systemd
sudo systemctl daemon-reload

# Запустить бота
sudo systemctl start ml-bot

# Остановить бота
sudo systemctl stop ml-bot

# Включить автозапуск
sudo systemctl enable ml-bot

# Проверить статус
sudo systemctl status ml-bot

# Просмотр логов
sudo journalctl -u ml-bot -f
```

## Настройка автоматического обновления (опционально)

### Webhook для автоматического деплоя

Создайте скрипт `webhook.sh`:

```bash
#!/bin/bash
cd /path/to/ml_bot
git pull origin main
./deploy.sh
sudo systemctl restart ml-bot
```

Настройте webhook в вашем Git репозитории, который будет вызывать этот скрипт при push.

### Cron для периодического обновления

Добавьте в crontab:

```bash
crontab -e
```

Добавьте строку (обновление каждый день в 3:00):

```
0 3 * * * cd /path/to/ml_bot && git pull origin main && ./deploy.sh && sudo systemctl restart ml-bot
```

## Проверка деплоя

1. **Проверьте что бот запущен:**

```bash
ps aux | grep run_bot.py
```

2. **Проверьте логи:**

```bash
tail -f logs/bot.log
```

3. **Проверьте Telegram бота:**

Отправьте команду `/start` боту в Telegram.

## Обновление бота

### Быстрое обновление:

```bash
cd /path/to/ml_bot
git pull origin main
./deploy.sh
sudo systemctl restart ml-bot
```

### Обновление с проверкой изменений:

```bash
cd /path/to/ml_bot
git fetch origin
git log HEAD..origin/main  # Просмотр изменений
git pull origin main
./deploy.sh
sudo systemctl restart ml-bot
```

## Резервное копирование

### Важные файлы для бэкапа:

- `.env` - переменные окружения
- `ml_models/` - обученные модели
- `ml_data/` - исторические данные
- `logs/` - логи работы
- `runtime_state.json` - состояние бота
- `symbol_ml_settings.json` - настройки моделей
- `risk_settings.json` - настройки риска

### Скрипт бэкапа:

```bash
#!/bin/bash
BACKUP_DIR="/backup/ml_bot"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

tar -czf $BACKUP_DIR/ml_bot_backup_$DATE.tar.gz \
    .env \
    ml_models/ \
    ml_data/ \
    logs/ \
    *.json
```

## Устранение проблем

### Бот не запускается:

1. Проверьте логи: `tail -f logs/errors.log`
2. Проверьте .env файл: `cat .env`
3. Проверьте зависимости: `pip list`
4. Проверьте Python версию: `python3 --version` (нужна 3.8+)

### Ошибки при деплое:

1. Убедитесь что все зависимости в `requirements.txt`
2. Проверьте права доступа: `chmod +x deploy.sh`
3. Проверьте виртуальное окружение: `which python3`

### Проблемы с Git:

```bash
# Если конфликты при pull
git stash
git pull origin main
git stash pop

# Если нужно сбросить изменения
git reset --hard origin/main
```

## Безопасность

⚠️ **Важно:**

1. **Никогда не коммитьте `.env` файл** - он уже в `.gitignore`
2. **Используйте SSH ключи** для доступа к серверу
3. **Ограничьте доступ** к директории проекта: `chmod 700 /path/to/ml_bot`
4. **Регулярно обновляйте зависимости**: `pip install --upgrade -r requirements.txt`
5. **Используйте firewall** для защиты сервера

## Дополнительные ресурсы

- [Документация Python Telegram Bot](https://python-telegram-bot.org/)
- [Systemd руководство](https://www.freedesktop.org/software/systemd/man/systemd.service.html)
- [Git документация](https://git-scm.com/doc)
