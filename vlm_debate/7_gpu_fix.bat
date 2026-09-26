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
if exist logs\gpu_fix.log del logs\gpu_fix.log

echo ============================================================
echo   Step 7 - fix Ollama GPU crash (0xc0000005), then pilot
echo   A small "ollama-serve" window may open. Do NOT close it.
echo   If Windows asks for permission, click Yes.
echo ============================================================
%PY% -u tools\path_scan.py

echo [A] Ollama with a minimal PATH
call :stop
call :serve_min minpath
%PY% -u tools\gpu_check.py minpath
if not errorlevel 1 goto ok_min

echo [B] reinstall Visual C++ runtime
call :stop
winget install -e --id Microsoft.VCRedist.2015+.x64 --silent --force --accept-package-agreements --accept-source-agreements > logs\vcredist.log 2>&1
call :serve_min vcredist
%PY% -u tools\gpu_check.py vcredist
if not errorlevel 1 goto ok_min

echo [C] install older Ollama 0.24.0 (models are kept)
call :stop
curl.exe -L -s -o "%TEMP%\OllamaSetup_0240.exe" https://github.com/ollama/ollama/releases/download/v0.24.0/OllamaSetup.exe
"%TEMP%\OllamaSetup_0240.exe" /VERYSILENT /NORESTART /SUPPRESSMSGBOXES
timeout /t 10 /nobreak >nul
call :stop
call :serve_min old0240
%PY% -u tools\gpu_check.py old0240
if not errorlevel 1 goto ok_min

echo.
echo GPU could not be enabled. Tell Claude it is done.
pause
exit /b 1

:ok_min
echo minpath> logs\ollama_mode.txt
echo.
echo GPU OK - model selection and pilot run (about 20-40 min)
%PY% -u tools\finalize.py
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
