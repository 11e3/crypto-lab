# Crypto Lab — Claude Code Guide

## Project Overview

Crypto backtesting & analysis platform for Upbit exchange (KRW pairs). Backtesting, portfolio optimization, Streamlit dashboard. Live trading is in a separate repo (dev/crypto-bot).

**Stack**: Python 3.13 | uv | Ruff | MyPy (strict) | Pytest | Streamlit | pandas/numpy/scipy

## Module Map

| Module | Purpose |
|--------|---------|
| `src/backtester/` | Dual engines: VectorizedBacktestEngine + EventDrivenBacktestEngine |
| `src/web/` | Streamlit multi-page app (backtest, optimization, analysis, monitor, data_collect) |
| `src/strategies/` | VBO (main), MeanReversion, Momentum, ORB — composable Condition pattern |
| `src/orders/` | Advanced order types (stop loss, take profit, trailing stop) for backtester |
| `src/data/` | Upbit data collection, caching, indicators |
| `src/risk/` | Portfolio optimization (MPT, risk parity), position sizing, VaR/CVaR |
| `src/config/` | YAML config loader, pydantic settings, centralized constants |

## Windows/Environment Gotchas

- **PowerShell**: No `&&` support — use `;` or use git bash
- **Test runner**: `uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=80`
- **Ruff/MyPy**: `uv run ruff check src/`, `uv run mypy src/`

## Quality Standards

- **Coverage**: 80% minimum (currently ~88.30%, 1249 tests)
- **MyPy**: 0 errors — keep it clean
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
- `parquet_filename()` in `src/config/constants.py` — centralized file naming convention
- `_MAX_CAGR_PCT = 99999.0` in `src/utils/metrics_core.py` — overflow cap
- `_MIN_VOLATILITY = 1e-8` in `src/risk/portfolio_methods.py` — zero-division guard

### Data Format
- Use `parquet_filename(ticker, interval)` from `src.config` for file naming (not inline f-strings)
- `data_loader` expects flat `{ticker}_{interval}.parquet` in `RAW_DATA_DIR`
- `exit_price_base` column convention for strategies with non-close exit prices (e.g., VBOV1 exits at open)

## Common Pitfalls

### Streamlit Specifics
- Render `col3` (asset_selector) before `col1` (trading_config) so `session_state` is populated
- Dynamic widget keys (`key=f"max_slots_{count}"`) to force re-render on state change
- `@st.cache_resource` for registry — import `get_cached_registry` from `strategy_selector`

### Position Sizing
- `_volatility_based_sizing()` uses baseline normalization (0.02) — was a no-op before 2026-02-11 fix
- `_inverse_volatility_sizing()` is the portfolio-level equivalent

### Thread Safety
- `get_cache()` uses double-checked locking with `threading.Lock()` (same as `get_config()`)
- Global singletons must use this pattern to be safe in Streamlit's multi-threaded environment

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

### Strategy Research (Binance Futures)
바이낸스 선물 전략은 `research/` 폴더에서 스크립트로 검증. `src/strategies/`와 별개 — 실거래는 crypto-bot 레포로 이관.

**구조**:
```
research/
├── fetch_data.py       # 바이낸스 선물 데이터 수집 (ccxt)
├── dual_momentum.py    # 단일 파일, pandas 백테스트 + 시각화
├── data/               # OHLCV parquet (.gitignore)
└── notebooks/          # 선택
```

**워크플로우**:
1. `research/` 스크립트에 `backtest()` + `plot_equity()` + `parameter_sweep()` 구현
2. 핵심 메트릭 확인: Sharpe ≥ 1.0, MDD ≤ 30%, capacity 분석
3. 통과 → crypto-bot 레포에 실거래 구현 (이 레포의 src/strategies/와 무관)
4. 탈락 → 스크립트에 결과 코멘트 남기고 보관

**규칙**:
- `research/` 코드는 quality-gate 대상 아님 (ruff/mypy/coverage 제외)
- `research/data/`는 `.gitignore` 등록, 커밋 금지
- 비용 모델: taker 0.05% + slippage 0.03% = 0.08% per leg
- `src/` 하위의 업비트 현물 전략과 혼합하지 않음

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
│   ├── test_orders/
│   ├── test_risk/
│   ├── test_web/
│   └── ...
├── integration/    # Multi-component (5 test files)
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
