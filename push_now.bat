@echo off
title Instant GitHub Push [app.create]
color 0B
echo =========================================================
echo    1-CLICK GITHUB PUSH
echo =========================================================
cd /d "%~dp0"

echo Staging all changed files...
git add -A

set TIMESTAMP=%DATE% %TIME%
echo Committing changes...
git commit -m "Quick update: %TIMESTAMP%"

echo Pushing to GitHub...
git push origin master

echo.
echo =========================================================
echo  DONE! Changes pushed to GitHub.
echo =========================================================
timeout /t 5
