#!/bin/bash
# ============================================================================
# Crypto Lab - Docker Development Helper Script (Linux/Mac)
# ============================================================================
# Development mode: Mounts source code for live reloading without rebuild
# Usage:
#   ./docker-run-dev.sh web        - Start web UI in dev mode (auto-reload)
#   ./docker-run-dev.sh bot        - Start trading bot in dev mode
#   ./docker-run-dev.sh all        - Start all services in dev mode
#   ./docker-run-dev.sh stop       - Stop all services
#   ./docker-run-dev.sh logs       - View logs
#   ./docker-run-dev.sh restart    - Restart services (apply code changes)
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}[ERROR] .env file not found!${NC}"
    echo ""
    echo "Please create .env file with your configuration:"
    echo "  cp .env.example .env"
    echo "  nano .env"
    echo ""
    exit 1
fi

# Parse command
COMMAND=${1:-help}

case "$COMMAND" in
    web)
        echo -e "${BLUE}[INFO] Starting Web UI in DEVELOPMENT MODE...${NC}"
        echo -e "${BLUE}[INFO] Code changes will auto-reload (no rebuild needed)${NC}"
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d web-ui
        echo ""
        echo -e "${GREEN}Web UI is starting at http://localhost:8501${NC}"
        echo "Code changes will auto-reload automatically!"
        echo "View logs: ./docker-run-dev.sh logs web-ui"
        ;;

    bot)
        echo -e "${YELLOW}[WARNING] Starting Trading Bot in DEVELOPMENT MODE!${NC}"
        echo -e "${YELLOW}[WARNING] This will use REAL MONEY on Upbit!${NC}"
        echo ""
        read -p "Are you sure? Type 'YES' to confirm: " CONFIRM
        if [ "$CONFIRM" != "YES" ]; then
            echo -e "${BLUE}[INFO] Cancelled.${NC}"
            exit 0
        fi
        echo -e "${BLUE}[INFO] Starting Trading Bot in dev mode...${NC}"
        echo -e "${BLUE}[INFO] Code is mounted - restart to apply changes${NC}"
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d trading-bot
        echo ""
        echo -e "${GREEN}Trading Bot is running in background.${NC}"
        echo "Restart after code changes: ./docker-run-dev.sh restart trading-bot"
        echo "View logs: ./docker-run-dev.sh logs trading-bot"
        ;;

    all)
        echo -e "${BLUE}[INFO] Starting all services in DEVELOPMENT MODE...${NC}"
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
        echo ""
        echo -e "${GREEN}All services started in dev mode.${NC}"
        echo "Web UI: http://localhost:8501 (auto-reload enabled)"
        echo "View logs: ./docker-run-dev.sh logs"
        ;;

    stop)
        echo -e "${BLUE}[INFO] Stopping all services...${NC}"
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
        echo -e "${GREEN}[INFO] All services stopped.${NC}"
        ;;

    logs)
        SERVICE=${2:-}
        if [ -z "$SERVICE" ]; then
            docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f
        else
            docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f "$SERVICE"
        fi
        ;;

    restart)
        SERVICE=${2:-}
        if [ -z "$SERVICE" ]; then
            echo -e "${BLUE}[INFO] Restarting all services...${NC}"
            docker-compose -f docker-compose.yml -f docker-compose.dev.yml restart
        else
            echo -e "${BLUE}[INFO] Restarting $SERVICE...${NC}"
            docker-compose -f docker-compose.yml -f docker-compose.dev.yml restart "$SERVICE"
        fi
        echo -e "${GREEN}[INFO] Restart complete.${NC}"
        ;;

    help|*)
        echo "Usage: ./docker-run-dev.sh [COMMAND]"
        echo ""
        echo "DEVELOPMENT MODE - Source code is mounted for live changes"
        echo ""
        echo "Commands:"
        echo "  web       - Start web UI (auto-reload on code changes)"
        echo "  bot       - Start trading bot (restart to apply changes)"
        echo "  all       - Start all services"
        echo "  stop      - Stop all services"
        echo "  logs      - View logs (add service name: logs web-ui)"
        echo "  restart   - Restart service (add service name: restart trading-bot)"
        echo "  help      - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./docker-run-dev.sh web"
        echo "  ./docker-run-dev.sh restart web-ui"
        echo "  ./docker-run-dev.sh logs web-ui"
        echo "  ./docker-run-dev.sh stop"
        echo ""
        echo "Benefits:"
        echo "  - NO REBUILD needed for code changes"
        echo "  - Streamlit auto-reloads on save"
        echo "  - Faster development iteration"
        ;;
esac
