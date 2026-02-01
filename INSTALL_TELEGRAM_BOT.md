# Установка зависимостей для Telegram бота

## Быстрая установка

Выполните в терминале (убедитесь, что виртуальное окружение активировано):

```bash
pip install python-telegram-bot
```

Или установите все зависимости из requirements.txt:

```bash
pip install -r requirements.txt
```

## Проверка установки

После установки проверьте:

```bash
python -c "import telegram; print(telegram.__version__)"
```

Если вы видите версию (например, 20.x), значит библиотека установлена правильно.

## Запуск бота

После установки запустите:

```bash
python run_bot.py
```

**Важно**: Убедитесь, что в файле `.env` указаны:
- `TELEGRAM_TOKEN` - токен вашего бота от @BotFather
- `ALLOWED_USER_ID` - ваш Telegram User ID (можно узнать у @userinfobot)
