@echo off
REM ============================================================================
REM Crypto Lab - Docker Helper Script (Windows)
REM ============================================================================
REM Usage:
REM   docker-run.bat web        - Start web UI only
REM   docker-run.bat bot        - Start trading bot only
REM   docker-run.bat all        - Start all services
REM   docker-run.bat stop       - Stop all services
REM   docker-run.bat logs       - View logs
REM   docker-run.bat build      - Rebuild images
REM ============================================================================

setlocal enabledelayedexpansion

REM Check if .env file exists
if not exist .env (
    echo [ERROR] .env file not found!
    echo.
    echo Please create .env file with your configuration:
    echo   copy .env.example .env
    echo   notepad .env
    echo.
    echo Required variables:
    echo   UPBIT_ACCESS_KEY=your_access_key
    echo   UPBIT_SECRET_KEY=your_secret_key
    exit /b 1
)

REM Parse command
set COMMAND=%1
if "%COMMAND%"=="" (
    set COMMAND=help
)

if "%COMMAND%"=="web" goto start_web
if "%COMMAND%"=="bot" goto start_bot
if "%COMMAND%"=="all" goto start_all
if "%COMMAND%"=="stop" goto stop_all
if "%COMMAND%"=="logs" goto show_logs
if "%COMMAND%"=="build" goto rebuild
if "%COMMAND%"=="help" goto show_help
goto show_help

:start_web
echo [INFO] Starting Web UI...
docker-compose up -d web-ui
echo.
echo Web UI is starting at http://localhost:8501
echo View logs: docker-compose logs -f web-ui
goto end

:start_bot
echo [WARNING] Starting LIVE TRADING BOT!
echo [WARNING] This will use REAL MONEY on Upbit!
echo.
set /p CONFIRM="Are you sure? Type 'YES' to confirm: "
if not "%CONFIRM%"=="YES" (
    echo [INFO] Cancelled.
    goto end
)
echo [INFO] Starting Trading Bot...
docker-compose up -d trading-bot
echo.
echo Trading Bot is running in background.
echo View logs: docker-compose logs -f trading-bot
goto end

:start_all
echo [INFO] Starting all services...
docker-compose up -d
echo.
echo All services started.
echo Web UI: http://localhost:8501
echo View logs: docker-compose logs -f
goto end

:stop_all
echo [INFO] Stopping all services...
docker-compose down
echo [INFO] All services stopped.
goto end

:show_logs
set SERVICE=%2
if "%SERVICE%"=="" (
    docker-compose logs -f
) else (
    docker-compose logs -f %SERVICE%
)
goto end

:rebuild
echo [INFO] Rebuilding Docker images...
docker-compose build --no-cache
echo [INFO] Build complete.
goto end

:show_help
echo Usage: docker-run.bat [COMMAND]
echo.
echo Commands:
echo   web       - Start web UI only (http://localhost:8501)
echo   bot       - Start trading bot (LIVE TRADING - use with caution!)
echo   all       - Start all services
echo   stop      - Stop all services
echo   logs      - View logs (add service name: logs web-ui)
echo   build     - Rebuild Docker images
echo   help      - Show this help message
echo.
echo Examples:
echo   docker-run.bat web
echo   docker-run.bat bot
echo   docker-run.bat logs web-ui
echo   docker-run.bat stop
goto end

:end
endlocal
