# Crypto Quant System — Claude Code Guide

## Project Overview

Crypto trading platform for Upbit exchange (KRW pairs). Backtesting, live trading, portfolio optimization, Streamlit dashboard.

**Stack**: Python 3.13 | uv | Ruff | MyPy (strict) | Pytest | Streamlit | pandas/numpy/scipy

## Module Map

| Module | LOC | Purpose |
|--------|-----|---------|
| `src/backtester/` | 8.6K | Dual engines: VectorizedBacktestEngine + EventDrivenBacktestEngine |
| `src/web/` | 6.2K | Streamlit multi-page app (backtest, optimization, analysis, monitor, data_collect) |
| `src/strategies/` | 4.3K | VBO (main), MeanReversion, Momentum, ORB — composable Condition pattern |
| `src/execution/` | 3.3K | Live bot, trade executors, event bus, position management |
| `src/data/` | 2.3K | Upbit data collection, caching, indicators |
| `src/risk/` | 1.1K | Portfolio optimization (MPT, risk parity), position sizing, VaR/CVaR |
| `src/exchange/` | 0.8K | Upbit API wrapper (REST), order management |
| `src/config/` | 0.7K | YAML config loader, pydantic settings |

## Windows/Environment Gotchas

- **`uv run` broken**: Fails with "canonicalize script path" — use `python -m pytest` directly
- **PowerShell**: No `&&` support — use `;` or use git bash
- **Test runner**: `python -m pytest --cov=src --cov-report=term-missing --cov-fail-under=80`
- **Ruff/MyPy**: Run directly (`ruff check src/`, `mypy src/`), not through uv

## Quality Standards

- **Coverage**: 80% minimum (currently ~89.88%, 1665 tests)
- **MyPy**: 14 known pre-existing errors in web/monitoring/storage (`dict` annotations) — only flag NEW errors
- **Ruff**: 0 violations required
- **Conventional commits**: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `perf:`

## Architecture Patterns

### Strategy Composition
Strategies use composable `Condition` objects (ABC). Shared conditions live in `src/strategies/common_conditions.py`. Each strategy module re-exports what it needs for backward compatibility.

```
Strategy → [EntryCondition, ExitCondition] → evaluate(current, history, indicators) → bool
```

### Dual Backtest Engines
- **Vectorized**: Fast parameter sweeps, portfolio-level signals
- **Event-driven**: Realistic order simulation, slippage/fees
- They produce different results by design (whipsaw counting, equity curve lengths)

### Key Constants
- `ANNUALIZATION_FACTOR = 365` (crypto, not 252)
- `RISK_FREE_RATE` from config
- Fee model: Upbit 0.05% per trade

### Data Format
- `data_loader` expects flat `{ticker}_{interval}.parquet` in `RAW_DATA_DIR`
- `exit_price_base` column convention for strategies with non-close exit prices (e.g., VBOV1 exits at open)

## Common Pitfalls

### Patch Targets in Tests
- Bot components: patch `bot_factory` not `bot_init` (ExchangeFactory, get_config, get_notifier)
- Signal handler: logger patches go to `signal_data` / `signal_metrics` (after SRP split)

### Exception Handling
- Exchange modules use `_UPBIT_ERRORS` tuple constant — add new types there, not inline
- 11 intentional broad `except Exception` remain (event_bus, notification_handler, bot main loop)

### Streamlit Specifics
- Render `col3` (asset_selector) before `col1` (trading_config) so `session_state` is populated
- Dynamic widget keys (`key=f"max_slots_{count}"`) to force re-render on state change
- `@st.cache_resource` for registry — import `get_cached_registry` from `strategy_selector`

### Position Sizing
- `_volatility_based_sizing()` uses baseline normalization (0.02) — was a no-op before 2026-02-11 fix
- `_inverse_volatility_sizing()` is the portfolio-level equivalent

## Workflow Rules

### After Every Code Change
Run `/quality-gate` or manually:
```bash
ruff format src/ tests/
ruff check --fix src/ tests/
mypy src/
python -m pytest --cov=src --cov-report=term-missing --cov-fail-under=80
```

### Before Committing
- Verify all 4 quality checks pass
- Stage specific files (not `git add .`) to avoid committing .env or data files
- Use conventional commit messages

### For New Features
1. Use `/feature-dev` for architecture guidance
2. Write tests first (TDD)
3. Implement, then `/quality-gate`
4. `/code-review` for self-review before push

### Periodic Maintenance
- `/weekly-review` — code simplification, dead code, coverage gaps
- `/claude-md-improver` — keep this file current
- Code Simplifier plugin — deep analysis after major changes

## File Protection

- `.env` — contains Upbit API keys, NEVER read or edit
- `src/_monkeypatch.py` — has `if patched_content != content:` guards, don't remove them
- Lock files, `*.parquet` data — don't commit

## Test Structure

```
tests/
├── unit/           # Fast, isolated (mock external dependencies)
│   ├── test_backtester/
│   ├── test_strategies/
│   ├── test_execution/
│   ├── test_exchange/
│   ├── test_risk/
│   ├── test_web/
│   └── ...
├── integration/    # Multi-component (9 test files)
└── fixtures/       # Shared test data
```

## Quick Reference

```bash
# Run tests
python -m pytest -q                                    # fast run
python -m pytest --cov=src --cov-fail-under=80 -q      # with coverage
python -m pytest tests/unit/test_strategies/ -v         # specific module

# Lint & format
ruff format src/ tests/
ruff check --fix src/ tests/
mypy src/

# Run dashboard
streamlit run src/web/app.py

# Git
git status -u
git add <specific files>
git commit -m "feat: description"
git push
```
