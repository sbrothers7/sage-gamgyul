@echo off
rem started in its own window by 6_fix_and_pilot.bat
set "OLLAMA_HOST=127.0.0.1:11434"
"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve > "%~dp0..\logs\serve_%1.log" 2>&1
