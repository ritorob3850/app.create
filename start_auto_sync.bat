@echo off
title GitHub Auto-Sync [app.create]
color 0A
echo =========================================================
echo    GITHUB AUTO-SYNC WATCHER
echo    Keep this window open while you work.
echo    Any changes saved in app.create will auto-push to GitHub!
echo =========================================================
echo.
cd /d "%~dp0"
python git_auto_sync.py
pause
