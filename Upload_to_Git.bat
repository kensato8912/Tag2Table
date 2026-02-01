@echo off
chcp 65001 >nul
title Upload to Git

cd /d "%~dp0"

echo ========================================
echo   Upload Project to Git (GitHub / GitLab)
echo ========================================
echo.

REM Check if Git is available
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found. Install: https://git-scm.com/download/win
    pause
    exit /b 1
)

REM Initialize if not yet done
if not exist ".git" (
    echo [1] Initializing Git...
    git init
    echo.
)

REM Add all files (.gitignore excludes sensitive files)
echo [2] Adding files...
git add .
echo.

REM Show status
echo [3] File status:
git status
echo.

set /p COMMIT_MSG="Enter commit message (Enter for default): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Update Tag2Table

echo.
echo [4] Committing: %COMMIT_MSG%
git commit -m "%COMMIT_MSG%"
if errorlevel 1 (
    echo.
    echo [TIP] No changes will skip commit, can ignore
)
echo.

REM Set remote if not configured
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo [5] Setting remote origin...
    git remote add origin https://github.com/kensato8912/Tag2Table.git
)
echo.
echo [6] Push to main...
git branch -M main
set /p PUSH="Push to GitHub now? (y/n): "
if /i "%PUSH%"=="y" (
    git push -u origin main
    if errorlevel 1 (
        echo [TIP] Ensure Tag2Table repo exists on GitHub and you have push access
    ) else (
        echo [OK] Pushed to https://github.com/kensato8912/Tag2Table
    )
)
echo.
pause
