@echo off
chcp 65001 >nul
title Install Tag2Table Dependencies

echo ========================================
echo   Install Tag2Table Dependencies
echo ========================================
echo.

cd /d "%~dp0"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found
    echo Please install Python and add to PATH
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1] Upgrading pip...
python -m pip install --upgrade pip --no-warn-script-location

echo.
echo [2] Installing packages from requirements.txt...
python -m pip install -r requirements.txt --no-warn-script-location
echo.
echo [3] Core deps: gradio, mediapipe==0.10.9, opencv-python, gallery-dl

echo.
echo ========================================
if errorlevel 1 (
    echo [FAILED] Install failed
    echo.
    echo Try with mirror:
    echo   python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt --no-warn-script-location
) else (
    echo [OK] Installation completed
)
echo ========================================
pause
