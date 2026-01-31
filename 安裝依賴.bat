@echo off
chcp 65001 >nul
title 安裝依賴套件

echo ========================================
echo   安裝 Tag2Table 依賴套件
echo ========================================
echo.

cd /d "%~dp0"

REM 檢查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 找不到 Python
    echo 請先安裝 Python 並加入系統 PATH
    echo 下載: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1] 使用 python -m pip 安裝...
echo.
python -m pip install --upgrade pip
python -m pip install openai
python -m pip install google-genai
python -m pip install Pillow
python -m pip install pyperclip

echo.
echo ========================================
if errorlevel 1 (
    echo [失敗] 安裝過程出現錯誤
    echo.
    echo 請嘗試手動執行：
    echo   python -m pip install openai
    echo   python -m pip install google-genai
    echo   python -m pip install Pillow
    echo   python -m pip install pyperclip
    echo.
    echo 若仍有問題，可檢查網路連線或使用鏡像：
    echo   python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple google-generativeai
) else (
    echo [完成] 套件安裝成功！
)
echo ========================================
pause
