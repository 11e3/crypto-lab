# Docker Deployment Guide for Crypto Lab

Production-grade Docker setup for cryptocurrency trading system with Python 3.12.7 and automatic monkeypatch support.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Usage](#usage)
- [Services](#services)
- [Security Best Practices](#security-best-practices)
- [Monitoring and Logging](#monitoring-and-logging)
- [Troubleshooting](#troubleshooting)
- [Production Deployment](#production-deployment)

---

## Quick Start

### 1. Setup Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your Upbit API keys
# Windows: notepad .env
# Linux/Mac: nano .env
```

**Minimum required in `.env`:**
```env
UPBIT_ACCESS_KEY=your_access_key_here
UPBIT_SECRET_KEY=your_secret_key_here
```

### 2. Start Services

**Windows:**
```cmd
docker-run.bat web       # Web UI only
docker-run.bat bot       # Trading bot (LIVE TRADING!)
docker-run.bat all       # All services
```

**Linux/Mac:**
```bash
chmod +x docker-run.sh
./docker-run.sh web      # Web UI only
./docker-run.sh bot      # Trading bot (LIVE TRADING!)
./docker-run.sh all      # All services
```

### 3. Access Web UI

Open browser: **http://localhost:8501**

---

## Architecture

### Multi-Stage Docker Build

```
┌─────────────────────────────────────────────────────┐
│ Stage 1: Builder                                    │
│ - Python 3.12.7-slim                                │
│ - Install build dependencies                        │
│ - Install uv package manager                        │
│ - Create .venv with all dependencies                │
└─────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: Production                                 │
│ - Python 3.12.7-slim (minimal)                      │
│ - Copy .venv from builder                           │
│ - Copy application code                             │
│ - Apply monkeypatch (src/_monkeypatch.py)           │
│ - Run as non-root user (botuser)                    │
└─────────────────────────────────────────────────────┘
```

**Benefits:**
- ✅ Small image size (~200MB vs ~1GB)
- ✅ Fast builds with layer caching
- ✅ Secure (non-root user, minimal dependencies)
- ✅ Python 3.12.7 type annotation issues resolved

### Service Architecture

```
┌─────────────────────────────────────────────────────┐
│                Docker Compose Stack                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │   Web UI    │  │ Trading Bot  │  │   Data    │ │
│  │             │  │              │  │ Collector │ │
│  │  Streamlit  │  │  Live Trade  │  │ (Optional)│ │
│  │  Port 8501  │  │   24/7 Run   │  │  On-Demand│ │
│  └─────────────┘  └──────────────┘  └───────────┘ │
│         │                 │                 │       │
│         └─────────────────┴─────────────────┘       │
│                           │                         │
│                  ┌────────▼─────────┐               │
│                  │  Shared Volumes  │               │
│                  │  - ./data        │               │
│                  │  - ./logs        │               │
│                  └──────────────────┘               │
└─────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Software

- **Docker**: 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: 2.0+ (included with Docker Desktop)

### Verify Installation

```bash
docker --version
# Docker version 20.10.0+

docker-compose --version
# Docker Compose version 2.0.0+
```

### Upbit API Keys

1. Go to [Upbit Open API](https://upbit.com/mypage/open_api_management)
2. Create API key with appropriate permissions:
   - **For Web UI (Backtesting)**: Read-only access
   - **For Trading Bot**: Full access (trade + read)

⚠️ **SECURITY WARNING**: Never commit API keys to Git!

---

## Configuration

### Environment Variables

Create `.env` file (see `.env.example` for full list):

```env
# ============================================
# Required: Upbit API Credentials
# ============================================
UPBIT_ACCESS_KEY=your_access_key
UPBIT_SECRET_KEY=your_secret_key

# ============================================
# Optional: Telegram Notifications
# ============================================
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TELEGRAM_ENABLED=true

# ============================================
# Optional: Trading Configuration
# ============================================
TRADING_TICKERS=KRW-BTC,KRW-ETH,KRW-XRP
TRADING_FEE_RATE=0.0005
TRADING_MAX_SLOTS=3
TRADING_MIN_ORDER_AMOUNT=5000.0

# ============================================
# Optional: Strategy Configuration
# ============================================
STRATEGY_NAME=VanillaVBO
STRATEGY_SMA_PERIOD=5
STRATEGY_TREND_SMA_PERIOD=10
STRATEGY_SHORT_NOISE_PERIOD=5
STRATEGY_LONG_NOISE_PERIOD=10

# ============================================
# Optional: Bot Configuration
# ============================================
BOT_DAILY_RESET_HOUR=9
BOT_DAILY_RESET_MINUTE=0

# ============================================
# Optional: Logging
# ============================================
LOG_LEVEL=INFO
WEB_SERVER_PORT=8501
```

---

## Usage

### Helper Scripts (Recommended)

**Windows:**
```cmd
docker-run.bat web       # Start web UI
docker-run.bat bot       # Start trading bot (with safety prompt)
docker-run.bat all       # Start all services
docker-run.bat stop      # Stop all services
docker-run.bat logs      # View logs
docker-run.bat build     # Rebuild images
```

**Linux/Mac:**
```bash
./docker-run.sh web      # Start web UI
./docker-run.sh bot      # Start trading bot (with safety prompt)
./docker-run.sh all      # Start all services
./docker-run.sh stop     # Stop all services
./docker-run.sh logs     # View logs
./docker-run.sh build    # Rebuild images
```

### Manual Docker Compose Commands

```bash
# Start specific service
docker-compose up -d web-ui

# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs (follow mode)
docker-compose logs -f web-ui
docker-compose logs -f trading-bot

# View logs (last 100 lines)
docker-compose logs --tail=100 web-ui

# Restart a service
docker-compose restart web-ui

# Rebuild and restart
docker-compose up -d --build web-ui

# Remove all containers and volumes
docker-compose down -v
```

---

## Services

### 1. Web UI (`web-ui`)

**Purpose**: Streamlit dashboard for backtesting, visualization, and strategy development.

**Access**: http://localhost:8501

**Start Command**:
```bash
docker-compose up -d web-ui
```

**Configuration**:
- Port: `8501` (configurable via `WEB_SERVER_PORT`)
- Restart policy: `unless-stopped`
- Volumes:
  - `./data:/app/data` - Data cache (persists backtests)
  - `./logs:/app/logs` - Application logs

**Use Cases**:
- Run backtests on historical data
- Visualize strategy performance
- Optimize strategy parameters
- Compare multiple strategies

**Security**:
- Recommended: Use **read-only** API keys for backtesting
- No real trading occurs in web UI

---

### 2. Trading Bot (`trading-bot`)

**Purpose**: 24/7 live trading bot executing strategies on Upbit.

⚠️ **DANGER**: This service trades with **REAL MONEY**!

**Start Command**:
```bash
docker-compose up -d trading-bot
# OR use helper script with safety prompt:
./docker-run.sh bot  # Linux/Mac
docker-run.bat bot   # Windows
```

**Configuration**:
- Restart policy: `unless-stopped` (auto-restart on failure)
- Volumes:
  - `./logs:/app/logs` - **CRITICAL** for audit trail
  - `./data:/app/data` - Data cache

**Monitoring**:
```bash
# View real-time logs
docker-compose logs -f trading-bot

# Check bot status
docker-compose ps trading-bot

# Check health
docker inspect crypto-quant-trading-bot | grep Health
```

**Safety Features**:
- Health checks every 60 seconds
- Automatic restart on crash
- Comprehensive logging (max 50MB x 10 files)
- Security: Runs with minimal capabilities

**Telegram Integration** (Recommended):
```env
TELEGRAM_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321
TELEGRAM_ENABLED=true
```

Get notifications for:
- Trade executions
- Errors and exceptions
- Daily performance reports

---

### 3. Data Collector (`data-collector`)

**Purpose**: Collect historical OHLCV data from Upbit for backtesting.

**Start Command**:
```bash
# One-time run
docker-compose run --rm data-collector

# Or with profile
docker-compose --profile tools up data-collector
```

**Configuration**:
- Restart policy: `no` (manual runs only)
- Volumes:
  - `./data:/app/data` - Store collected data
  - `./scripts:/app/scripts:ro` - Mount scripts read-only

**Usage**:
```bash
# Collect 30-minute candle data
docker-compose run --rm data-collector

# Collect with custom script
docker-compose run --rm data-collector python scripts/collect_custom_data.py
```

---

## Security Best Practices

### 1. API Key Management

**DO:**
- ✅ Use `.env` file (gitignored)
- ✅ Use read-only keys for web UI
- ✅ Use separate keys for bot and web UI
- ✅ Rotate keys regularly
- ✅ Set IP whitelist on Upbit

**DON'T:**
- ❌ Commit keys to Git
- ❌ Share keys in public
- ❌ Use production keys for testing
- ❌ Grant unnecessary permissions

### 2. Container Security

**Built-in protections:**
- ✅ Non-root user (`botuser`, UID 1000)
- ✅ Minimal base image (Python 3.12.7-slim)
- ✅ Dropped capabilities (`cap_drop: ALL`)
- ✅ Security opt: `no-new-privileges`
- ✅ Read-only file system where possible

**Additional hardening:**
```yaml
# Add to docker-compose.yml for extra security
security_opt:
  - no-new-privileges:true
  - seccomp=unconfined  # Only if needed
read_only: true  # Only if app supports
tmpfs:
  - /tmp:rw,noexec,nosuid,size=100m
```

### 3. Network Security

**Production deployment:**
```bash
# Use firewall to restrict access
sudo ufw allow 8501/tcp  # Web UI (only from trusted IPs)
sudo ufw deny incoming
sudo ufw enable

# Or use reverse proxy (nginx)
# See: Production Deployment section
```

### 4. Logging and Audit Trail

**CRITICAL for trading bot:**
- All trades logged to `./logs/` directory
- Logs rotated (50MB x 10 files for bot)
- JSON format for easy parsing

**Monitor logs regularly:**
```bash
# Trading bot logs
tail -f logs/upbit_bot.log

# Web UI logs
tail -f logs/web_ui.log

# Error logs only
docker-compose logs trading-bot | grep ERROR
```

---

## Monitoring and Logging

### View Logs

```bash
# Real-time logs (all services)
docker-compose logs -f

# Specific service
docker-compose logs -f web-ui
docker-compose logs -f trading-bot

# Last N lines
docker-compose logs --tail=100 trading-bot

# Filter by time
docker-compose logs --since 2024-01-01T00:00:00 trading-bot
docker-compose logs --since 1h trading-bot
```

### Log Files

Logs are stored in `./logs/` directory:

```
logs/
├── web_ui.log              # Web UI logs
├── upbit_bot.log           # Trading bot logs (AUDIT TRAIL)
├── data_collector.log      # Data collection logs
└── error.log               # Error logs only
```

### Health Checks

```bash
# Check container health
docker inspect crypto-quant-trading-bot | grep -A 10 Health

# Or use docker-compose
docker-compose ps
```

**Expected output:**
```
NAME                         STATUS              PORTS
crypto-quant-web-ui          Up (healthy)        0.0.0.0:8501->8501/tcp
crypto-quant-trading-bot     Up (healthy)
```

### Resource Monitoring

```bash
# Resource usage (CPU, Memory)
docker stats crypto-quant-trading-bot

# Disk usage
docker system df
```

---

## Troubleshooting

### Common Issues

#### 1. "Address already in use" (Port 8501)

**Problem**: Another process is using port 8501.

**Solution**:
```bash
# Windows: Find process
netstat -ano | findstr :8501
taskkill /PID <process_id> /F

# Linux/Mac: Find and kill
lsof -ti:8501 | xargs kill -9

# Or change port in .env
WEB_SERVER_PORT=8502
```

#### 2. Monkeypatch not applied

**Problem**: Import errors like `NameError: name 'Callable' is not defined`.

**Solution**:
Check if monkeypatch is imported first in `src/__init__.py`:
```python
# This MUST be first import
import src._monkeypatch  # noqa: F401
```

Rebuild Docker image:
```bash
docker-compose build --no-cache web-ui
```

#### 3. API Key errors

**Problem**: `UpbitApiError: Invalid API key`.

**Solution**:
1. Verify API keys in `.env`
2. Check key permissions on Upbit
3. Restart containers to reload env vars:
```bash
docker-compose restart web-ui
```

#### 4. Container keeps restarting

**Problem**: Container enters restart loop.

**Solution**:
```bash
# View logs
docker-compose logs trading-bot

# Check last exit code
docker inspect crypto-quant-trading-bot | grep -A 5 State

# Disable auto-restart temporarily
docker-compose up --no-recreate trading-bot
```

#### 5. Out of disk space

**Problem**: Docker images/logs filling disk.

**Solution**:
```bash
# Clean up unused images/containers
docker system prune -a

# Remove old logs
rm -rf logs/*.log.old

# Limit log size in docker-compose.yml
logging:
  options:
    max-size: "10m"
    max-file: "3"
```

### Debug Mode

Run container in debug mode:

```bash
# Override entrypoint for debugging
docker-compose run --rm --entrypoint /bin/bash web-ui

# Inside container
python -c "from src.config.settings import Settings; print('OK')"
python -m src.web.app
```

---

## Production Deployment

### AWS EC2 / GCP Compute Engine

**1. Setup server:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

**2. Deploy application:**
```bash
# Clone repository
git clone <your-repo-url>
cd crypto-lab

# Setup environment
cp .env.example .env
nano .env  # Add your API keys

# Start services
docker-compose up -d

# Enable firewall
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 8501/tcp # Web UI (optional)
sudo ufw enable
```

**3. Setup reverse proxy (nginx):**
```bash
sudo apt-get install nginx

# Create nginx config
sudo nano /etc/nginx/sites-available/crypto-quant

# Add:
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/crypto-quant /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

**4. Setup SSL (Let's Encrypt):**
```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Auto-start on boot

Create systemd service:

```bash
sudo nano /etc/systemd/system/crypto-quant.service
```

```ini
[Unit]
Description=Crypto Lab Trading System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/crypto-lab
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable crypto-quant.service
sudo systemctl start crypto-quant.service
```

### Monitoring (Production)

**Setup cron for log rotation:**
```bash
crontab -e

# Add:
0 0 * * * find /home/ubuntu/crypto-lab/logs -name "*.log" -mtime +7 -delete
```

**Setup alerts:**
```bash
# Create alert script
cat > /home/ubuntu/check-bot.sh << 'EOF'
#!/bin/bash
if ! docker ps | grep -q crypto-quant-trading-bot; then
    # Send alert (email, telegram, etc.)
    curl -X POST "https://api.telegram.org/bot$TELEGRAM_TOKEN/sendMessage" \
        -d chat_id="$TELEGRAM_CHAT_ID" \
        -d text="⚠️ Trading bot is DOWN!"
fi
EOF

chmod +x /home/ubuntu/check-bot.sh

# Run every 5 minutes
crontab -e
# Add:
*/5 * * * * /home/ubuntu/check-bot.sh
```

---

## Advanced Configuration

### Multi-stage deployment (staging + production)

Create separate compose files:

```bash
# docker-compose.staging.yml
services:
  web-ui:
    extends:
      file: docker-compose.yml
      service: web-ui
    environment:
      - UPBIT_ACCESS_KEY=${STAGING_ACCESS_KEY}

# Deploy
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d
```

### Custom Python packages

Add to `pyproject.toml` dependencies, then rebuild:
```bash
docker-compose build --no-cache
```

---

## FAQ

**Q: Can I run without Docker?**
A: Yes, use `uv sync --all-extras` and run locally. Docker is optional but recommended for production.

**Q: How much disk space do I need?**
A: ~5GB (Docker image ~200MB + logs/data ~1-2GB + buffer)

**Q: Can I use with other exchanges?**
A: Currently Upbit only. Extend `src/exchange/` for other exchanges.

**Q: Is it safe to run trading bot 24/7?**
A: Yes, but:
- ✅ Use read-only keys for testing first
- ✅ Start with small capital
- ✅ Monitor logs daily
- ✅ Setup Telegram alerts
- ✅ Test strategy thoroughly via backtest

**Q: How do I update the application?**
A:
```bash
git pull
docker-compose build --no-cache
docker-compose up -d
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: [README.md](README.md)
- **CLAUDE.md**: [Development Guide](CLAUDE.md)

---

**Last Updated**: 2026-01-16
**Python Version**: 3.12.7
**Docker Version**: 20.10+
