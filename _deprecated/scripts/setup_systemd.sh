#!/bin/bash
# Скрипт для автоматической настройки systemd service

set -e

echo "🔧 Настройка systemd service для ML Trading Bot..."

# Определяем пути автоматически
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="$SCRIPT_DIR"
VENV_PYTHON="$WORKING_DIR/venv/bin/python3"
CURRENT_USER=$(whoami)

# Проверяем что мы в правильной директории
if [ ! -f "$WORKING_DIR/run_bot.py" ]; then
    echo "❌ Ошибка: run_bot.py не найден в $WORKING_DIR"
    exit 1
fi

# Проверяем виртуальное окружение
if [ ! -f "$VENV_PYTHON" ]; then
    echo "⚠️  Виртуальное окружение не найдено в $WORKING_DIR/venv"
    echo "   Создайте его: python3 -m venv venv"
    exit 1
fi

# Создаем service файл
SERVICE_FILE="/etc/systemd/system/ml-bot.service"

echo "📝 Создание service файла: $SERVICE_FILE"
echo "   Working Directory: $WORKING_DIR"
echo "   User: $CURRENT_USER"
echo "   Python: $VENV_PYTHON"

sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=ML Trading Bot
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$WORKING_DIR
Environment="PATH=$WORKING_DIR/venv/bin"
ExecStart=$VENV_PYTHON $WORKING_DIR/run_bot.py
Restart=always
RestartSec=10
StandardOutput=append:$WORKING_DIR/logs/bot.log
StandardError=append:$WORKING_DIR/logs/errors.log

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Service файл создан"

# Перезагружаем systemd
echo "🔄 Перезагрузка systemd daemon..."
sudo systemctl daemon-reload

echo "✅ Настройка завершена!"
echo ""
echo "💡 Следующие команды:"
echo "   sudo systemctl start ml-bot      # Запустить бота"
echo "   sudo systemctl enable ml-bot     # Включить автозапуск"
echo "   sudo systemctl status ml-bot     # Проверить статус"
echo "   sudo journalctl -u ml-bot -f     # Просмотр логов"
