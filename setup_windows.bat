@echo off
cd /d "%~dp0"
title 감귤 병해 AI 연구 환경 설치

echo ============================================================
echo   감귤 병해 AI 연구 환경 설치
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 goto NOPYTHON

for /f "tokens=*" %%i in ('python --version') do set PYVER=%%i
echo   파이썬 확인됨: %PYVER%
echo.

if exist ".venv\Scripts\activate.bat" goto HAVEVENV

echo   가상환경을 만드는 중...
python -m venv .venv
if errorlevel 1 goto VENVFAIL
goto ACTIVATE

:HAVEVENV
echo   가상환경이 이미 있습니다. 건너뜁니다.

:ACTIVATE
call .venv\Scripts\activate.bat
echo.
echo   pip 업그레이드 중...
python -m pip install --upgrade pip --quiet
echo.
echo ------------------------------------------------------------
echo   NVIDIA 그래픽카드가 있습니까?
echo.
echo      1 = 예       GPU 버전 설치. 학습이 훨씬 빠릅니다.
echo      2 = 아니오   CPU 버전 설치. 모르겠으면 2번을 고르세요.
echo.
set /p GPUCHOICE=   번호 입력 : 

if "%GPUCHOICE%"=="1" goto GPU
goto CPU

:GPU
echo.
echo   GPU 버전 PyTorch 설치 중... 시간이 오래 걸립니다.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto PIPFAIL
goto REST

:CPU
echo.
echo   CPU 버전 PyTorch 설치 중... 시간이 오래 걸립니다.
pip install torch torchvision
if errorlevel 1 goto PIPFAIL

:REST
echo.
echo   나머지 패키지 설치 중...
pip install timm numpy pandas scikit-learn pillow opencv-python matplotlib PyYAML
if errorlevel 1 goto PIPFAIL

echo.
echo ============================================================
echo   설치 확인
echo ============================================================
python -c "import torch,timm,cv2,sklearn,yaml;print('  PyTorch  :',torch.__version__);print('  GPU 사용 :',torch.cuda.is_available());print('  timm     :',timm.__version__);print('  OpenCV   :',cv2.__version__)"
if errorlevel 1 goto VERIFYFAIL

echo.
echo ============================================================
echo   설치가 끝났습니다.
echo.
echo   다음부터는 start.bat 을 더블클릭해서 연구를 시작하세요.
echo ============================================================
echo.
pause
exit /b 0

:NOPYTHON
echo   [오류] 파이썬이 설치되어 있지 않습니다.
echo.
echo   1. https://www.python.org/downloads/ 접속
echo   2. Python 3.10 이상 다운로드
echo   3. 설치 화면에서 "Add Python to PATH" 를 반드시 체크
echo   4. 설치 후 이 파일을 다시 실행
echo.
pause
exit /b 1

:VENVFAIL
echo.
echo   [오류] 가상환경 생성에 실패했습니다.
echo   폴더 경로에 한글이나 공백이 많으면 문제가 될 수 있습니다.
echo.
pause
exit /b 1

:PIPFAIL
echo.
echo   [오류] 패키지 설치에 실패했습니다.
echo   인터넷 연결과 학교 방화벽을 확인하세요.
echo.
pause
exit /b 1

:VERIFYFAIL
echo.
echo   [오류] 설치는 됐지만 불러오기에 실패했습니다.
echo.
pause
exit /b 1
