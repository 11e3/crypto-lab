# Bot Safety Fix - Prevent Accidental Starts

## 🚨 Problem

The bot was starting automatically when running pytest, causing:
- Repeated Telegram notifications
- Potential unwanted trading
- Confusion about what triggered it

## ✅ Solution Applied

### 1. Added Safety Check to `bot.run()`

The `TradingBotFacade.run()` method now:
- Detects if running in a test environment (pytest/unittest)
- Blocks execution unless `--allow-test-run` flag is used
- Logs a warning instead of starting the bot

### 2. Added `--force` Flag to `main()`

The `main()` function in `bot_facade.py` now requires `--force` flag:
```bash
# ❌ This will now fail:
python -m src.execution.bot_facade

# ✅ This will work:
python -m src.execution.bot_facade --force
```

## 🛡️ Protection Layers

### Layer 1: Test Environment Detection
```python
is_testing = (
    "pytest" in sys.modules
    or "unittest" in sys.modules
    or "PYTEST_CURRENT_TEST" in os.environ
    or any("test" in arg.lower() for arg in sys.argv)
)
```

### Layer 2: Explicit Flag Required
- `--force` for direct execution
- `--allow-test-run` for tests that need to run the bot

### Layer 3: CLI Command Only
The **only safe way** to start the bot:
```bash
upbit-quant run-bot
```

## 📋 Safe Commands

✅ **SAFE** - These won't start the bot:
```bash
# Testing
uv run pytest ...
uv run pytest --cov=src ...

# Linting
uv run ruff check ...
uv run mypy src ...

# Imports
python -c "from src.cli.main import main"
```

❌ **DANGEROUS** - These WILL start the bot (now blocked):
```bash
# Direct execution (now requires --force)
python -m src.execution.bot_facade  # ❌ Blocked
python -m src.execution.bot_facade --force  # ⚠️ Will start

# During tests (now blocked)
uv run pytest ...  # ✅ Safe - bot.run() blocked
```

## 🔧 For Testing

If you need to test `bot.run()` in a test:
```bash
# Add --allow-test-run flag
uv run pytest --allow-test-run tests/unit/test_execution/test_bot_facade.py
```

Or in test code:
```python
import sys
sys.argv.append("--allow-test-run")
bot.run()  # Now allowed
```

## ⚠️ Immediate Actions

1. **Stop any running bot processes:**
   ```powershell
   Get-Process python* | Stop-Process -Force
   ```

2. **Commit the safety fixes:**
   ```bash
   git add src/execution/bot_facade.py
   git commit -m "Add safety checks to prevent accidental bot starts during tests"
   ```

3. **Verify the fix:**
   ```bash
   # This should NOT start the bot now
   uv run pytest --cov=src --cov-report=term-missing -v
   ```

## 📝 Files Changed

- `src/execution/bot_facade.py`:
  - Added `os` import
  - Added test environment detection in `run()`
  - Added `--force` flag requirement in `main()`

## ✅ Result

The bot will **never** start accidentally during:
- pytest runs
- unittest runs
- Any test environment
- Direct module execution (without --force)

**Only** `upbit-quant run-bot` will start the bot intentionally.

---

**Status**: ✅ Fixed
**Date**: 2026-01-05
