@echo off
REM Скрипт для запуска ML Trading Bot
REM Убедитесь, что вы находитесь в правильной директории

echo =====================================
echo   ML Trading Bot - Starting...
echo =====================================
echo.

REM Активируем виртуальное окружение
call venv\Scripts\activate.bat

REM Запускаем бота
python run_bot.py

REM Если бот остановился, ждем нажатия клавиши
pause
