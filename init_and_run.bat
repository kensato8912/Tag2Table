@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

:: 1. Create venv if not exists
if not exist ".venv" (
    echo [System] Creating Python virtual environment...
    python -m venv .venv
)

:: 2. Activate virtual environment
call .venv\Scripts\activate

:: 3. Install dependencies
echo [System] Checking dependencies (MediaPipe, OpenCV, Gradio)...
pip install --upgrade pip
pip install -r requirements.txt

:: 4. Launch application
echo [System] Starting cropper engine...
python web_ui.py

if errorlevel 1 (
    echo [ERROR] Application exited with error.
)
pause