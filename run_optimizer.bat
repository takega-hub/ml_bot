@echo off
REM Скрипт-обертка для запуска оптимизации стратегий (Windows)

REM Переходим в директорию скрипта
cd /d "%~dp0"

REM Активируем виртуальное окружение (если есть)
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Запускаем оптимизацию
python auto_strategy_optimizer.py %*
