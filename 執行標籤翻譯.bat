@echo off
chcp 65001 >nul
title Tag2Table - AI 標籤統計與翻譯工具

echo ========================================
echo   Tag2Table - AI 標籤統計與翻譯工具
echo ========================================
echo.

REM 切換到腳本所在目錄
cd /d "%~dp0"

REM 檢查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 找不到 Python，請確認已安裝並加入 PATH
    pause
    exit /b 1
)

REM 檢查是否有安裝所需套件
python -c "import openai" >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在安裝 openai...
    python -m pip install openai
)
python -c "from google import genai" >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在安裝 google-genai...
    python -m pip install google-genai
)
if errorlevel 1 (
    echo [建議] 若安裝失敗，請執行「安裝依賴.bat」
)
echo.

echo 啟動程式...
echo.

python main.py

echo.
echo ========================================
pause
