@echo off
REM Batch script to push changes to Git repository

echo Adding changes to Git...
git add bot/trading_loop.py
git add FIX_FREEZE_ISSUE.md
git add push_changes.bat

echo.
echo Committing changes...
git commit -m "Fix: Wrap generate_signal() in asyncio.to_thread() to prevent event loop blocking"

echo.
echo Pushing to origin main...
git push origin main

echo.
echo Done! Changes pushed successfully.
pause
