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
echo   Self-test - no API key, no dataset needed
echo ============================================================
%PY% -m pip install --quiet --disable-pip-version-check pillow
%PY% tools\selftest\make_fake_data.py
if errorlevel 1 ( pause & exit /b 1 )
%PY% tools\selftest\run_selftest.py
if errorlevel 1 (
  echo.
  echo *** FAILED - read the error above ***
  pause
  exit /b 1
)
echo.
echo SUCCESS
pause
