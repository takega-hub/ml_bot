# 🔧 Настройка systemd Service для ML Trading Bot

## Быстрая настройка

### Вариант 1: Автоматическая настройка (рекомендуется)

```bash
cd /opt/ml_bot
chmod +x setup_systemd.sh
./setup_systemd.sh
```

Скрипт автоматически:
- Определит пути к проекту
- Создаст service файл с правильными путями
- Перезагрузит systemd daemon

### Вариант 2: Ручная настройка

1. **Создайте service файл:**

```bash
sudo nano /etc/systemd/system/ml-bot.service
```

2. **Вставьте следующее содержимое** (замените пути на ваши):

```ini
[Unit]
Description=ML Trading Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ml_bot
Environment="PATH=/opt/ml_bot/venv/bin"
ExecStart=/opt/ml_bot/venv/bin/python3 /opt/ml_bot/run_bot.py
Restart=always
RestartSec=10
StandardOutput=append:/opt/ml_bot/logs/bot.log
StandardError=append:/opt/ml_bot/logs/errors.log

[Install]
WantedBy=multi-user.target
```

3. **Сохраните файл** (Ctrl+O, Enter, Ctrl+X в nano)

4. **Перезагрузите systemd:**

```bash
sudo systemctl daemon-reload
```

## Управление сервисом

### Запуск

```bash
sudo systemctl start ml-bot
```

### Остановка

```bash
sudo systemctl stop ml-bot
```

### Перезапуск

```bash
sudo systemctl restart ml-bot
```

### Включить автозапуск при загрузке системы

```bash
sudo systemctl enable ml-bot
```

### Отключить автозапуск

```bash
sudo systemctl disable ml-bot
```

### Проверка статуса

```bash
sudo systemctl status ml-bot
```

### Просмотр логов

```bash
# Логи systemd
sudo journalctl -u ml-bot -f

# Логи бота (из файлов)
tail -f /opt/ml_bot/logs/bot.log
tail -f /opt/ml_bot/logs/errors.log
```

## Проверка работы

1. **Запустите сервис:**

```bash
sudo systemctl start ml-bot
```

2. **Проверьте статус:**

```bash
sudo systemctl status ml-bot
```

Должно показать `active (running)`

3. **Проверьте логи:**

```bash
tail -f /opt/ml_bot/logs/bot.log
```

4. **Проверьте Telegram бота:**

Отправьте `/start` боту в Telegram

## Устранение проблем

### Сервис не запускается

1. **Проверьте логи:**

```bash
sudo journalctl -u ml-bot -n 50
```

2. **Проверьте права доступа:**

```bash
# Убедитесь что пользователь имеет доступ к директории
ls -la /opt/ml_bot

# Если нужно, измените владельца
sudo chown -R root:root /opt/ml_bot
```

3. **Проверьте виртуальное окружение:**

```bash
/opt/ml_bot/venv/bin/python3 --version
```

4. **Проверьте .env файл:**

```bash
ls -la /opt/ml_bot/.env
```

### Ошибка "Unit ml-bot.service not found"

Это означает что service файл не создан или не загружен:

```bash
# Проверьте что файл существует
ls -la /etc/systemd/system/ml-bot.service

# Если файла нет, создайте его (см. выше)

# Перезагрузите systemd
sudo systemctl daemon-reload
```

### Сервис падает сразу после запуска

1. **Проверьте логи ошибок:**

```bash
sudo journalctl -u ml-bot -n 100 --no-pager
cat /opt/ml_bot/logs/errors.log
```

2. **Попробуйте запустить вручную:**

```bash
cd /opt/ml_bot
source venv/bin/activate
python3 run_bot.py
```

Это покажет ошибки, которые не видны в systemd логах.

### Проблемы с путями

Если пути в service файле неправильные:

1. **Найдите правильные пути:**

```bash
# Рабочая директория
pwd

# Python из venv
which python3  # после активации venv
# или
/opt/ml_bot/venv/bin/python3 --version
```

2. **Обновите service файл:**

```bash
sudo nano /etc/systemd/system/ml-bot.service
```

3. **Перезагрузите:**

```bash
sudo systemctl daemon-reload
sudo systemctl restart ml-bot
```

## Дополнительные настройки

### Ограничение ресурсов

Добавьте в секцию `[Service]`:

```ini
MemoryLimit=2G
CPUQuota=50%
```

### Запуск от другого пользователя

Если нужно запускать не от root:

```ini
User=your_username
Group=your_group
```

И убедитесь что пользователь имеет доступ:

```bash
sudo chown -R your_username:your_group /opt/ml_bot
```

### Переменные окружения

Если нужно добавить переменные окружения:

```ini
Environment="PYTHONPATH=/opt/ml_bot"
Environment="CUSTOM_VAR=value"
```

## Полезные команды

```bash
# Просмотр всех логов
sudo journalctl -u ml-bot

# Логи за последний час
sudo journalctl -u ml-bot --since "1 hour ago"

# Логи с определенной даты
sudo journalctl -u ml-bot --since "2024-01-01"

# Очистка старых логов
sudo journalctl --vacuum-time=7d
```
