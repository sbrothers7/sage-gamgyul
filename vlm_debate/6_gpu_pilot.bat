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
if exist logs\gpu_fix.log del logs\gpu_fix.log

echo ============================================================
echo   Step 6 - make Ollama use the GPU, pick models, 10-image pilot
echo   A small "ollama-serve" window may open. Do NOT close it.
echo ============================================================
echo [1] current state
%PY% -u tools\gpu_check.py before
if not errorlevel 1 goto gpu_ok

echo [2] restart Ollama as a normal app
call :stop
if exist "%OL%\ollama app.exe" (
  start "" "%OL%\ollama app.exe"
) else (
  start "ollama-serve" /min cmd /c "%~dp0tools\serve.cmd" app
)
timeout /t 15 /nobreak >nul
%PY% -u tools\gpu_check.py app
if not errorlevel 1 goto gpu_ok

echo [3] try CUDA v13 library
call :stop
set "OLLAMA_LLM_LIBRARY=cuda_v13"
start "ollama-serve" /min cmd /c "%~dp0tools\serve.cmd" v13
timeout /t 10 /nobreak >nul
%PY% -u tools\gpu_check.py v13
if not errorlevel 1 goto gpu_ok

echo [4] try CUDA v12 library
call :stop
set "OLLAMA_LLM_LIBRARY=cuda_v12"
start "ollama-serve" /min cmd /c "%~dp0tools\serve.cmd" v12
timeout /t 10 /nobreak >nul
%PY% -u tools\gpu_check.py v12
if not errorlevel 1 goto gpu_ok

echo [5] debug log
call :stop
set "OLLAMA_LLM_LIBRARY="
set "OLLAMA_DEBUG=1"
start "ollama-serve" /min cmd /c "%~dp0tools\serve.cmd" debug
timeout /t 10 /nobreak >nul
%PY% -u tools\gpu_check.py debug
if not errorlevel 1 goto gpu_ok
echo.
echo GPU could not be enabled. Tell Claude it is done.
pause
exit /b 1

:gpu_ok
echo.
echo GPU OK - model selection and pilot run (about 20-40 min)
%PY% -u tools\finalize.py
echo.
echo Finished. Tell Claude it is done.
pause
exit /b 0

:stop
taskkill /f /im "ollama app.exe" >nul 2>&1
taskkill /f /im ollama.exe >nul 2>&1
timeout /t 3 /nobreak >nul
exit /b 0
