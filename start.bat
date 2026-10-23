@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==========================================
echo   离线视频分析实验室
echo   打开: http://localhost:3200/
echo   停止: 按 Ctrl+C
echo ==========================================
echo.
node server.mjs
echo.
echo 服务已停止。
pause
