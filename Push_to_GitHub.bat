@echo off
chcp 65001 >nul
title Manual Push to GitHub

cd /d "%~dp0"

echo ========================================
echo   Manual Git Commands (First Push)
echo ========================================
echo.

git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found
    pause
    exit /b 1
)

echo [1] git init
git init
echo.

REM Check Git identity (required for commit)
git config --global user.name >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git user.name not set. Please run Set_Git_Identity.bat first.
    pause
    exit /b 1
)
git config --global user.email >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git user.email not set. Please run Set_Git_Identity.bat first.
    pause
    exit /b 1
)

echo [2] git add .
git add .
echo.
echo Files to be committed:
git status
echo.

echo [3] git commit -m "initial version of Tag2Table"
git commit -m "initial version of Tag2Table"
if errorlevel 1 (
    echo.
    echo [ERROR] Commit failed.
    echo If you see "nothing to commit", check that there are files not ignored by .gitignore
    echo Example: *.py README.md requirements.txt *.bat
    pause
    exit /b 1
)
echo [OK] First commit created
echo.

echo [4] git branch -M main
git branch -M main
echo.

echo [5] git remote add origin https://github.com/kensato8912/Tag2Table.git
git remote add origin https://github.com/kensato8912/Tag2Table.git 2>nul
if errorlevel 1 (
    echo [INFO] remote origin already exists, skip
)
echo.

echo [6] git push -u origin main
git push -u origin main
if errorlevel 1 (
    echo.
    echo [ERROR] Push failed. Please check:
    echo   1. Tag2Table repo exists on GitHub
    echo   2. You are logged in or have credential set
    pause
    exit /b 1
)
echo.
echo [OK] Pushed to https://github.com/kensato8912/Tag2Table
echo ========================================
pause
