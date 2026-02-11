---
name: quality-gate
description: Run full quality checks — ruff format, lint, mypy, pytest with coverage
---

Run the complete quality gate for crypto-lab. Execute each step sequentially and report results.

**IMPORTANT**: Use `python -m pytest` (not `uv run pytest`) — uv has path issues on Windows.

## Steps

1. **Ruff Format**
   ```bash
   ruff format src/ tests/
   ```
   Report: number of files reformatted (0 = clean)

2. **Ruff Lint**
   ```bash
   ruff check --fix src/ tests/
   ```
   Report: number of fixable/unfixable errors. If unfixable errors exist, show them.

3. **MyPy**
   ```bash
   mypy src/
   ```
   Report: pass/fail. Note: 14 pre-existing errors in web/monitoring/storage (bare `dict` annotations) are known — only flag NEW errors.

4. **Pytest + Coverage**
   ```bash
   python -m pytest --cov=src --cov-report=term-missing --cov-fail-under=80 -q
   ```
   Report: total passed/failed, coverage percentage.

## Output Format

```
Quality Gate Results:
  Ruff Format:  ✓ clean / ✗ N files reformatted
  Ruff Lint:    ✓ clean / ✗ N errors
  MyPy:         ✓ clean / ✗ N new errors
  Pytest:       ✓ N passed, coverage XX.XX%

  Overall: ✓ PASS / ✗ FAIL
```

If any step fails critically, stop and report the failure. Do not proceed to commit.
