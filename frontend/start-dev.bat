@echo off
chcp 65001 >nul
echo ========================================
echo   脑肿瘤可视化系统 - 启动脚本
echo ========================================

cd /d "%~dp0"

REM 检查 node_modules
if not exist "node_modules" (
    echo [1/3] 正在安装依赖...
    call npm install --registry=https://registry.npmmirror.com
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络或手动运行 npm install
        pause
        exit /b 1
    )
) else (
    echo [1/3] 依赖已存在，跳过安装
)

echo [2/3] 启动 Vite 开发服务器...
echo       启动后请在浏览器打开 http://localhost:5173/
echo       按 Ctrl+C 可停止服务
echo.

REM 启动 dev 服务器
call npm run dev

pause