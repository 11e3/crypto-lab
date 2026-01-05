# 🚨 Emergency: Bot Auto-Start Issue

## Immediate Actions

### 1. Stop the Bot NOW
```bash
# Find and kill Python processes
# Windows PowerShell:
Get-Process python* | Stop-Process -Force

# Or press Ctrl+C in the terminal where bot is running
```

### 2. Check What Command Was Run
The bot should ONLY start when you explicitly run:
```bash
upbit-quant run-bot
```

**DO NOT** run:
- `python -m src.main` (might start bot)
- `python src/main.py` (might start bot)
- `python -m src.execution.bot_facade` (will start bot!)

## Root Cause Analysis

### Issue
The bot started automatically, possibly due to:
1. Running `python -m src.execution.bot_facade` directly
2. Importing modules that have `if __name__ == "__main__"` blocks
3. Accidental execution of bot code during testing

### Files with Auto-Start Code
- `src/execution/bot_facade.py` - Line 485: `bot.run()` in `main()`
- `src/execution/bot.py` - Line 473: `bot.run()` in `main()`
- `src/main.py` - Safe, only calls CLI

## Fixes Needed

### 1. Add Safety Check to Bot Facade
Prevent accidental execution.

### 2. Update Error Detection Plan
Add warning about not running bot modules directly.

### 3. Add Environment Check
Require explicit flag to run bot in non-production.

## Prevention

### Safe Commands
✅ **SAFE** - These won't start the bot:
```bash
uv run pytest ...
uv run ruff check ...
uv run mypy src ...
python -c "from src.cli.main import main"
```

### Dangerous Commands
❌ **DANGEROUS** - These WILL start the bot:
```bash
python -m src.execution.bot_facade
python src/execution/bot_facade.py
python -m src.execution.bot
python src/execution/bot.py
```

### Safe Way to Test Bot
✅ **SAFE** - Use CLI:
```bash
upbit-quant run-bot --dry-run  # (when implemented)
```

## Immediate Fix

Add safety check to prevent accidental bot starts.
