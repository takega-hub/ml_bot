@echo off
echo ======================================
echo Copying trading_loop.py to server
echo ======================================
echo.

set LOCAL_FILE=bot\trading_loop.py
set SERVER=root@s3fe42482.fastvps-server.com
set REMOTE_PATH=/opt/ml_bot/bot/trading_loop.py

echo Local file: %LOCAL_FILE%
echo Server: %SERVER%
echo Remote path: %REMOTE_PATH%
echo.

echo Copying file...
scp "%LOCAL_FILE%" "%SERVER%:%REMOTE_PATH%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================
    echo SUCCESS! File copied to server.
    echo ======================================
    echo.
    echo Now run on server:
    echo   cd /opt/ml_bot
    echo   find . -name "*.pyc" -delete
    echo   sudo systemctl restart ml-bot
    echo   tail -f /opt/ml_bot/logs/bot.log
) else (
    echo.
    echo ======================================
    echo ERROR! Failed to copy file.
    echo ======================================
    echo Make sure you have:
    echo 1. SSH access configured
    echo 2. scp command available (install Git for Windows)
)

echo.
pause
