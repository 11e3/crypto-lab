#!/bin/bash
# ============================================================================
# Crypto Lab - Docker Helper Script (Linux/Mac)
# ============================================================================
# Usage:
#   ./docker-run.sh web        - Start web UI only
#   ./docker-run.sh bot        - Start trading bot only
#   ./docker-run.sh all        - Start all services
#   ./docker-run.sh stop       - Stop all services
#   ./docker-run.sh logs       - View logs
#   ./docker-run.sh build      - Rebuild images
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
    echo "Required variables:"
    echo "  UPBIT_ACCESS_KEY=your_access_key"
    echo "  UPBIT_SECRET_KEY=your_secret_key"
    exit 1
fi

# Parse command
COMMAND=${1:-help}

case "$COMMAND" in
    web)
        echo -e "${BLUE}[INFO] Starting Web UI...${NC}"
        docker-compose up -d web-ui
        echo ""
        echo -e "${GREEN}Web UI is starting at http://localhost:8501${NC}"
        echo "View logs: docker-compose logs -f web-ui"
        ;;

    bot)
        echo -e "${YELLOW}[WARNING] Starting LIVE TRADING BOT!${NC}"
        echo -e "${YELLOW}[WARNING] This will use REAL MONEY on Upbit!${NC}"
        echo ""
        read -p "Are you sure? Type 'YES' to confirm: " CONFIRM
        if [ "$CONFIRM" != "YES" ]; then
            echo -e "${BLUE}[INFO] Cancelled.${NC}"
            exit 0
        fi
        echo -e "${BLUE}[INFO] Starting Trading Bot...${NC}"
        docker-compose up -d trading-bot
        echo ""
        echo -e "${GREEN}Trading Bot is running in background.${NC}"
        echo "View logs: docker-compose logs -f trading-bot"
        ;;

    all)
        echo -e "${BLUE}[INFO] Starting all services...${NC}"
        docker-compose up -d
        echo ""
        echo -e "${GREEN}All services started.${NC}"
        echo "Web UI: http://localhost:8501"
        echo "View logs: docker-compose logs -f"
        ;;

    stop)
        echo -e "${BLUE}[INFO] Stopping all services...${NC}"
        docker-compose down
        echo -e "${GREEN}[INFO] All services stopped.${NC}"
        ;;

    logs)
        SERVICE=${2:-}
        if [ -z "$SERVICE" ]; then
            docker-compose logs -f
        else
            docker-compose logs -f "$SERVICE"
        fi
        ;;

    build)
        echo -e "${BLUE}[INFO] Rebuilding Docker images...${NC}"
        docker-compose build --no-cache
        echo -e "${GREEN}[INFO] Build complete.${NC}"
        ;;

    help|*)
        echo "Usage: ./docker-run.sh [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  web       - Start web UI only (http://localhost:8501)"
        echo "  bot       - Start trading bot (LIVE TRADING - use with caution!)"
        echo "  all       - Start all services"
        echo "  stop      - Stop all services"
        echo "  logs      - View logs (add service name: logs web-ui)"
        echo "  build     - Rebuild Docker images"
        echo "  help      - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./docker-run.sh web"
        echo "  ./docker-run.sh bot"
        echo "  ./docker-run.sh logs web-ui"
        echo "  ./docker-run.sh stop"
        ;;
esac
