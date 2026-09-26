@echo off
setlocal
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
cd /d "%~dp0"

set "PY="
py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
if not defined PY ( py -3 -c "import sys" >nul 2>&1 && set "PY=py -3" )
if not defined PY ( python -c "import sys" >nul 2>&1 && set "PY=python" )
if not defined PY (
  echo [ERROR] Python not found on PATH.
  pause
  exit /b 1
)

echo ============================================================
echo   Step 4 - install Ollama, models, datasets, then test runs
echo   Takes 1-3 hours (about 15 GB download). Keep this window open.
echo   Log: logs\prepare_all.log
echo ============================================================
%PY% -u tools\prepare_all.py
echo.
echo Finished. Tell Claude it is done.
pause
