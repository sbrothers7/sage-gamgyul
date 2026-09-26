@echo off
setlocal
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
cd /d "%~dp0"

set "PY="
rem 1) the exact Python that worked in step 4/5, then other known locations
for %%P in ("%LOCALAPPDATA%\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe" "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" "%LOCALAPPDATA%\Microsoft\WindowsApps\python3.exe") do (
  if not defined PY if exist %%P ( %%P -c "import sys" >nul 2>&1 && set PY="%%~P" )
)
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
  if not defined PY if exist "%%~D\python.exe" set PY="%%~D\python.exe"
)
if not defined PY ( py -3 -c "import sys" >nul 2>&1 && set "PY=py -3" )
if not defined PY ( python -c "import sys" >nul 2>&1 && set "PY=python" )
if defined PY goto have_py
if not exist logs mkdir logs
set PATH > logs\pyfind.txt 2>&1
where python >> logs\pyfind.txt 2>&1
where py >> logs\pyfind.txt 2>&1
dir "%LOCALAPPDATA%\Microsoft\WindowsApps\*python*" >> logs\pyfind.txt 2>&1
echo [ERROR] Python not found. Details saved to logs\pyfind.txt - tell Claude.
pause
exit /b 1
:have_py
echo Using Python: %PY%

set "OL=%LOCALAPPDATA%\Programs\Ollama"
if not exist logs mkdir logs

echo ============================================================
echo   Step 8 - MAIN EXPERIMENT (about 7-9 hours, run overnight)
echo   200 images x (anti-sycophancy ON, then OFF)
echo   Keep the laptop PLUGGED IN. Do NOT close this window
echo   or the small "ollama-serve" window. Sleep is blocked
echo   automatically while it runs.
echo ============================================================
call :stop
call :serve_min main
%PY% -u tools\gpu_check.py main
if errorlevel 1 (
  echo GPU is not being used - stopping. Tell Claude.
  pause
  exit /b 1
)
%PY% -u tools\run_main.py --dataset data/citrusA_mendeley_4cls.json --n 200
echo.
echo Finished. Tell Claude it is done.
pause
exit /b 0

:serve_min
setlocal
set "PATH=%SystemRoot%\system32;%SystemRoot%;%SystemRoot%\System32\Wbem;%OL%"
start "ollama-serve" /min cmd /c "%~dp0tools\serve.cmd" %1
endlocal
timeout /t 12 /nobreak >nul
exit /b 0

:stop
taskkill /f /im "ollama app.exe" >nul 2>&1
taskkill /f /im ollama.exe >nul 2>&1
timeout /t 3 /nobreak >nul
exit /b 0
