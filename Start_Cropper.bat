@echo off
setlocal
:: 1. Set project path
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

:: 2. StabilityMatrix Python path (adjust for your system)
:: Usually at Data\Packages\stable-diffusion-webui\venv\Scripts\python.exe
set "PY_PATH=C:\StabilityMatrix\Data\Packages\stable-diffusion-webui\venv\Scripts\python.exe"

echo [System] Starting with StabilityMatrix environment...
"%PY_PATH%" web_ui.py

pause
