# ⚡ Быстрый старт деплоя через Git

## Шаг 1: Инициализация Git (локально)

```bash
# Если Git еще не инициализирован
git init
git add .
git commit -m "Initial commit"
```

## Шаг 2: Создание удаленного репозитория

1. Создайте репозиторий на GitHub/GitLab
2. Скопируйте URL репозитория

## Шаг 3: Подключение к удаленному репозиторию

```bash
git remote add origin https://github.com/yourusername/ml_bot.git
git branch -M main
git push -u origin main
```

## Шаг 4: Деплой на сервер

### На сервере выполните:

```bash
# Клонируйте репозиторий
git clone https://github.com/yourusername/ml_bot.git
cd ml_bot

# Создайте .env файл
nano .env
# Добавьте все необходимые переменные окружения

# Создайте виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Запустите деплой
chmod +x deploy.sh
./deploy.sh
```

## Шаг 5: Настройка автозапуска (опционально)

```bash
# Скопируйте пример service файла
sudo cp ml-bot.service.example /etc/systemd/system/ml-bot.service

# Отредактируйте пути
sudo nano /etc/systemd/system/ml-bot.service

# Активируйте сервис
sudo systemctl daemon-reload
sudo systemctl enable ml-bot
sudo systemctl start ml-bot
```

## Обновление бота

### На сервере:

```bash
cd /path/to/ml_bot
./update.sh
```

Или вручную:

```bash
git pull origin main
./deploy.sh
sudo systemctl restart ml-bot
```

## Проверка работы

```bash
# Проверка статуса
sudo systemctl status ml-bot

# Просмотр логов
tail -f logs/bot.log

# Проверка в Telegram
# Отправьте /start боту
```

---

📚 **Подробные инструкции:**
- [GIT_SETUP.md](GIT_SETUP.md) - Настройка Git
- [DEPLOY.md](DEPLOY.md) - Полная инструкция по деплою
