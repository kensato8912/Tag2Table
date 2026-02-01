@echo off
chcp 65001 >nul
title Set Git Identity

cd /d "%~dp0"

echo ========================================
echo   Git Identity Setup (required before first commit)
echo ========================================
echo.

git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found
    pause
    exit /b 1
)

echo Setting user.name to kensato8912...
git config --global user.name "kensato8912"
echo.

set USER_EMAIL=kensato89@gmail.com
set /p USER_EMAIL="Enter your GitHub email (Enter = kensato89@gmail.com): "
if "%USER_EMAIL%"=="" set USER_EMAIL=kensato89@gmail.com
git config --global user.email "%USER_EMAIL%"
echo [OK] user.email set
echo.

echo Current config:
git config --global user.name
git config --global user.email
echo.
echo ========================================
echo Done. You can now run Push_to_GitHub.bat
echo ========================================
pause
