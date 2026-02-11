@echo off
REM ============================================================================
REM Crypto Lab - Docker Development Helper Script (Windows)
REM ============================================================================
REM Development mode: Mounts source code for live reloading without rebuild
REM Usage:
REM   docker-run-dev.bat web        - Start web UI in dev mode (auto-reload)
REM   docker-run-dev.bat bot        - Start trading bot in dev mode
REM   docker-run-dev.bat all        - Start all services in dev mode
REM   docker-run-dev.bat stop       - Stop all services
REM   docker-run-dev.bat logs       - View logs
REM   docker-run-dev.bat restart    - Restart services (apply code changes)
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
if "%COMMAND%"=="restart" goto restart_service
if "%COMMAND%"=="help" goto show_help
goto show_help

:start_web
echo [INFO] Starting Web UI in DEVELOPMENT MODE...
echo [INFO] Code changes will auto-reload (no rebuild needed)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d web-ui
echo.
echo Web UI is starting at http://localhost:8501
echo Code changes will auto-reload automatically!
echo View logs: docker-run-dev.bat logs web-ui
goto end

:start_bot
echo [WARNING] Starting Trading Bot in DEVELOPMENT MODE!
echo [WARNING] This will use REAL MONEY on Upbit!
echo.
set /p CONFIRM="Are you sure? Type 'YES' to confirm: "
if not "%CONFIRM%"=="YES" (
    echo [INFO] Cancelled.
    goto end
)
echo [INFO] Starting Trading Bot in dev mode...
echo [INFO] Code is mounted - restart to apply changes
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d trading-bot
echo.
echo Trading Bot is running in background.
echo Restart after code changes: docker-run-dev.bat restart trading-bot
echo View logs: docker-run-dev.bat logs trading-bot
goto end

:start_all
echo [INFO] Starting all services in DEVELOPMENT MODE...
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
echo.
echo All services started in dev mode.
echo Web UI: http://localhost:8501 (auto-reload enabled)
echo View logs: docker-run-dev.bat logs
goto end

:stop_all
echo [INFO] Stopping all services...
docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
echo [INFO] All services stopped.
goto end

:show_logs
set SERVICE=%2
if "%SERVICE%"=="" (
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f
) else (
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f %SERVICE%
)
goto end

:restart_service
set SERVICE=%2
if "%SERVICE%"=="" (
    echo [INFO] Restarting all services...
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml restart
) else (
    echo [INFO] Restarting %SERVICE%...
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml restart %SERVICE%
)
echo [INFO] Restart complete.
goto end

:show_help
echo Usage: docker-run-dev.bat [COMMAND]
echo.
echo DEVELOPMENT MODE - Source code is mounted for live changes
echo.
echo Commands:
echo   web       - Start web UI (auto-reload on code changes)
echo   bot       - Start trading bot (restart to apply changes)
echo   all       - Start all services
echo   stop      - Stop all services
echo   logs      - View logs (add service name: logs web-ui)
echo   restart   - Restart service (add service name: restart trading-bot)
echo   help      - Show this help message
echo.
echo Examples:
echo   docker-run-dev.bat web
echo   docker-run-dev.bat restart web-ui
echo   docker-run-dev.bat logs web-ui
echo   docker-run-dev.bat stop
echo.
echo Benefits:
echo   - NO REBUILD needed for code changes
echo   - Streamlit auto-reloads on save
echo   - Faster development iteration
goto end

:end
endlocal
