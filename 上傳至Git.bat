@echo off
chcp 65001 >nul
title 上傳至 Git

cd /d "%~dp0"

echo ========================================
echo   上傳專案至 Git (GitHub / GitLab)
echo ========================================
echo.

REM 檢查 Git 是否可用
git --version >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 找不到 Git，請先安裝：https://git-scm.com/download/win
    pause
    exit /b 1
)

REM 若尚未初始化，則初始化
if not exist ".git" (
    echo [1] 初始化 Git...
    git init
    echo.
)

REM 加入所有檔案（.gitignore 會排除敏感檔）
echo [2] 加入檔案...
git add .
echo.

REM 顯示狀態
echo [3] 檔案狀態：
git status
echo.

set /p COMMIT_MSG="請輸入 commit 訊息 (Enter 使用預設)："
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Update Tag2Table

echo.
echo [4] 提交：%COMMIT_MSG%
git commit -m "%COMMIT_MSG%"
if errorlevel 1 (
    echo.
    echo [提示] 若無變更則不會產生 commit，可略過
)
echo.

REM 若尚未設定遠端，則設定
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo [5] 設定遠端 origin...
    git remote add origin https://github.com/kensato8912/Tag2Table.git
)
echo.
echo [6] 推送至 main...
git branch -M main
set /p PUSH="要立即推送到 GitHub 嗎？(y/n)："
if /i "%PUSH%"=="y" (
    git push -u origin main
    if errorlevel 1 (
        echo [提示] 請確認 GitHub 已建立 Tag2Table repo，且有 push 權限
    ) else (
        echo [完成] 已推送到 https://github.com/kensato8912/Tag2Table
    )
)
echo.
pause
