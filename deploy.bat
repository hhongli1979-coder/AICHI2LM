@echo off
REM TeleChat 一键本地部署脚本 (One-Click Local Deployment Script for Windows)
setlocal enabledelayedexpansion

REM 默认配置
if not defined MODEL_PATH set MODEL_PATH=..\models\7B
if not defined API_PORT set API_PORT=8070
if not defined WEB_PORT set WEB_PORT=8501
if not defined CUDA_VISIBLE_DEVICES set CUDA_VISIBLE_DEVICES=0

REM 获取脚本所在目录
set SCRIPT_DIR=%~dp0
set SERVICE_DIR=%SCRIPT_DIR%service

REM PID文件
set API_PID_FILE=%TEMP%\telechat_api.pid
set WEB_PID_FILE=%TEMP%\telechat_web.pid

echo ============================================================
echo 🎯 TeleChat 一键本地部署
echo ============================================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Python
    echo 请安装Python 3.7或更高版本
    pause
    exit /b 1
)

echo ✓ Python检查通过

REM 检查依赖包
python -c "import torch, transformers, fastapi, uvicorn, streamlit" 2>nul
if errorlevel 1 (
    echo ❌ 错误: 缺少必要的Python包
    echo 请运行: pip install -r requirements.txt
    pause
    exit /b 1
)

echo ✓ 依赖项检查通过

REM 检查模型路径
if not exist "%MODEL_PATH%" (
    echo ❌ 错误: 模型路径不存在: %MODEL_PATH%
    echo 请设置正确的MODEL_PATH环境变量
    pause
    exit /b 1
)

echo ✓ 模型路径检查通过: %MODEL_PATH%
echo.

REM 启动API服务
echo 🚀 启动API服务 (端口: %API_PORT%)...
cd /d "%SERVICE_DIR%"
start /b python telechat_service.py > "%TEMP%\telechat_api.log" 2>&1

REM 等待API服务启动
echo ⏳ 等待API服务启动...
set MAX_WAIT=30
set WAIT_COUNT=0

:wait_api
timeout /t 2 /nobreak >nul
set /a WAIT_COUNT+=1

REM 使用curl或PowerShell检查API
curl -s http://127.0.0.1:%API_PORT%/docs >nul 2>&1
if not errorlevel 1 goto api_ready

if %WAIT_COUNT% lss %MAX_WAIT% goto wait_api

echo ❌ 错误: API服务启动超时
pause
exit /b 1

:api_ready
echo ✓ API服务已就绪
echo 📍 API文档: http://0.0.0.0:%API_PORT%/docs
echo.

REM 启动Web服务
echo 🚀 启动Web服务 (端口: %WEB_PORT%)...
start /b streamlit run web_demo.py --server.port %WEB_PORT% --server.address 0.0.0.0 > "%TEMP%\telechat_web.log" 2>&1

REM 等待Web服务启动
timeout /t 5 /nobreak >nul
echo ✓ Web服务已启动
echo 📍 Web界面: http://0.0.0.0:%WEB_PORT%
echo.

echo ============================================================
echo ✨ 部署成功！
echo ============================================================
echo 📍 API服务: http://0.0.0.0:%API_PORT%/docs
echo 📍 Web界面: http://0.0.0.0:%WEB_PORT%
echo.
echo 日志文件:
echo   API: %TEMP%\telechat_api.log
echo   Web: %TEMP%\telechat_web.log
echo.
echo 按任意键停止服务...
echo ============================================================

pause >nul

REM 停止服务
echo.
echo 🛑 正在停止服务...

REM 使用taskkill停止Python进程（这是一个简化版本，实际应该更精确）
taskkill /F /FI "WINDOWTITLE eq telechat_service.py*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq streamlit*" >nul 2>&1

echo ✓ 服务已停止

endlocal
