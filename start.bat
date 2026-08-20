@echo off
cd /d "%~dp0"
title 감귤 병해 AI 연구

if not exist ".venv\Scripts\activate.bat" goto NOVENV

call .venv\Scripts\activate.bat

echo ============================================================
echo   감귤 병해 영상분류 AI - 일반화 성능 분석
echo ============================================================
echo.
echo   실행 순서 - 아래 명령을 하나씩 입력하세요
echo.
echo     python run_1_prepare.py     1. 데이터 준비 및 점검
echo     python run_2_train.py       2. 모델 학습 - 원 논문 재현
echo     python run_3_evaluate.py    3. 일반화 성능 검증 - 핵심
echo     python run_4_gradcam.py     4. 판단 근거 시각화
echo     python run_5_report.py      5. 그림과 표 생성
echo.
echo   데이터 없이 코드만 시험해 보려면
echo.
echo     python make_test_data.py            가짜 데이터 만들기
echo     python make_test_data.py --clean    가짜 데이터 지우기
echo.
echo ============================================================
echo.

cmd /k
exit /b 0

:NOVENV
echo.
echo   [오류] 연구 환경이 아직 설치되지 않았습니다.
echo.
echo   먼저 setup_windows.bat 을 더블클릭해서 설치하세요.
echo.
pause
exit /b 1
