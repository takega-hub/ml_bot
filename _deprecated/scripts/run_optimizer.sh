#!/bin/bash
# Скрипт-обертка для запуска оптимизации стратегий (Linux)

# Переходим в директорию скрипта
cd "$(dirname "$0")"

# Активируем виртуальное окружение (если есть)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Запускаем оптимизацию
python auto_strategy_optimizer.py "$@"
