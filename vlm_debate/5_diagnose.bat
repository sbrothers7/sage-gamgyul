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
echo   Step 5 - diagnose speed, answer format, Zenodo labels
echo   About 20-30 minutes. Log: logs\diagnose.log
echo ============================================================
%PY% -u tools\diagnose.py
echo.
echo Finished. Tell Claude it is done.
pause
