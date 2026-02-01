@echo off
chcp 65001 >nul
title Tag2Table - AI Tag Translator

echo ========================================
echo   Tag2Table - AI Tag Translator
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python and add to PATH
    pause
    exit /b 1
)

REM Check and install dependencies
python -c "import openai" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing openai...
    python -m pip install openai
)
python -c "from google import genai" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing google-genai...
    python -m pip install google-genai
)
if errorlevel 1 (
    echo [TIP] If install failed, run the dependency installer batch file
)
echo.

echo Starting...
echo.

python -m src.main

echo.
echo ========================================
pause
