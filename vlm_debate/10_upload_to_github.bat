@echo off
setlocal
chcp 65001 >nul 2>&1
cd /d "%~dp0"
set "REPO=https://github.com/sbrothers7/CitrusImageTraining.git"
set "WORK=C:\citrus-git"
set "DST=C:\citrus-git\CitrusImageTraining"
set "BR=vlm-debate"
if not exist logs mkdir logs
set "LOG=%~dp0logs\git_upload.log"
echo ===== %date% %time% ===== > "%LOG%"

echo ============================================================
echo   Upload this project to GitHub (branch: vlm-debate)
echo   A GitHub login window may open. Please sign in.
echo ============================================================

set "GIT="
where git >nul 2>&1 && set "GIT=git"
if defined GIT goto have_git
if exist "%ProgramFiles%\Git\cmd\git.exe" set "GIT=%ProgramFiles%\Git\cmd\git.exe"
if defined GIT goto have_git
echo [1] installing Git ...
winget install -e --id Git.Git --silent --accept-package-agreements --accept-source-agreements >> "%LOG%" 2>&1
if exist "%ProgramFiles%\Git\cmd\git.exe" set "GIT=%ProgramFiles%\Git\cmd\git.exe"
if defined GIT goto have_git
echo [ERROR] Git could not be installed. Tell Claude.
pause
exit /b 1
:have_git
echo Using git: %GIT% >> "%LOG%"
"%GIT%" --version >> "%LOG%" 2>&1

echo [2] get the repository
if not exist "%WORK%" mkdir "%WORK%"
if exist "%DST%\.git" goto have_repo
"%GIT%" clone %REPO% "%DST%" >> "%LOG%" 2>&1
if not exist "%DST%\.git" goto fail
:have_repo
cd /d "%DST%"
"%GIT%" fetch origin >> "%LOG%" 2>&1
"%GIT%" checkout %BR% >> "%LOG%" 2>&1
if errorlevel 1 "%GIT%" checkout -b %BR% origin/main >> "%LOG%" 2>&1
"%GIT%" pull origin %BR% >> "%LOG%" 2>&1

echo [3] copy files into vlm_debate
robocopy "%~dp0." "%DST%\vlm_debate" /E /XD logs _remote __pycache__ fake_data mock_out notes paper .git /XF .env HANDOFF.md *.pyc gitignore_vlm.txt /NFL /NDL /NJH /NJS >> "%LOG%" 2>&1
copy /y "%~dp0tools\gitignore_vlm.txt" "%DST%\vlm_debate\.gitignore" >> "%LOG%" 2>&1

echo [4] commit
"%GIT%" config user.name >nul 2>&1
if errorlevel 1 "%GIT%" config user.name "peter"
"%GIT%" config user.email >nul 2>&1
if errorlevel 1 "%GIT%" config user.email "peter9167@naver.com"
"%GIT%" add -A vlm_debate >> "%LOG%" 2>&1
"%GIT%" status --short >> "%LOG%" 2>&1
"%GIT%" commit -m "Add VLM multi-agent debate experiment (VIDA+PANDA, anti-sycophancy OFF/ON/HARD)" >> "%LOG%" 2>&1

echo [5] push (sign in if a window opens)
"%GIT%" push -u origin %BR% >> "%LOG%" 2>&1
if errorlevel 1 goto fail
"%GIT%" log --oneline -3 >> "%LOG%" 2>&1
echo.
echo SUCCESS - uploaded to branch %BR%. Tell Claude it is done.
echo PUSH_OK >> "%LOG%"
pause
exit /b 0
:fail
echo.
echo FAILED - details in logs\git_upload.log. Tell Claude.
echo PUSH_FAILED >> "%LOG%"
pause
exit /b 1
