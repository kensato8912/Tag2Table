@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [INFO] Starting Tag2Table Gradio Web UI...
echo [INFO] Open http://localhost:7860 in your browser
echo.

python web_ui.py

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start. Make sure dependencies are installed:
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)
