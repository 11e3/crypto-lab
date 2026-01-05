# Error Detection Plan

## 🎯 Objective
Systematically identify and fix all errors in the codebase to ensure CI passes and code quality is maintained.

## 📋 Detection Strategy

### Phase 1: Automated Checks (Quick Wins)

#### 1.1 Run Tests Locally
```bash
# Run all tests
uv run pytest --cov=src --cov-report=term-missing -v

# Run specific test files that were failing
uv run pytest tests/unit/test_cli/ -v

# Run with Python 3.10 specifically (if available)
python3.10 -m pytest tests/unit/test_cli/ -v
```

**What to look for:**
- Test failures
- Import errors
- Attribute errors
- Type errors
- Coverage gaps

#### 1.2 Linting
```bash
# Run Ruff linter
uv run ruff check . --output-format=concise

# Run Ruff formatter check
uv run ruff format --check .

# Fix auto-fixable issues
uv run ruff check . --fix
uv run ruff format .
```

**What to look for:**
- Unused imports
- Syntax errors
- Style violations
- Import order issues

#### 1.3 Type Checking
```bash
# Run MyPy
uv run mypy src --show-error-codes

# Check specific modules
uv run mypy src/cli/ src/backtester/ src/strategies/
```

**What to look for:**
- Type mismatches
- Missing type hints
- Incorrect type annotations
- Import errors

### Phase 2: Code Review (Manual Inspection)

#### 2.1 Review Recent Changes
```bash
# Check git diff for recent changes
git diff HEAD~5 HEAD -- src/ tests/

# Check for common patterns
git log --oneline -10
```

**What to look for:**
- Breaking changes
- Incomplete refactoring
- Missing updates
- Inconsistent patterns

#### 2.2 Check Import Statements
```bash
# Find all imports
grep -r "^import\|^from" src/ --include="*.py" | head -50

# Check for circular imports
uv run python -c "import src.cli.main; import src.cli.commands.backtest"
```

**What to look for:**
- Circular imports
- Missing imports
- Incorrect import paths
- Unused imports

#### 2.3 Check Function Signatures
```bash
# Find function definitions
grep -r "^def \|^async def " src/ --include="*.py" | grep -v "__pycache__"

# Check for mismatched signatures
grep -r "def.*->" src/ --include="*.py"
```

**What to look for:**
- Parameter mismatches
- Return type issues
- Missing parameters
- Incorrect defaults

### Phase 3: Runtime Checks

#### 3.1 Test CLI Commands
```bash
# Test each CLI command
uv run upbit-quant --help
uv run upbit-quant collect --help
uv run upbit-quant backtest --help
uv run upbit-quant run-bot --help

# Test with minimal inputs (dry run)
uv run upbit-quant collect --tickers KRW-BTC --intervals day --full-refresh
```

**What to look for:**
- Command execution errors
- Missing dependencies
- Configuration errors
- Runtime exceptions

#### 3.2 Test Imports
```bash
# Test critical imports
uv run python -c "from src.cli.commands.backtest import backtest; print('OK')"
uv run python -c "from src.cli.commands.collect import collect; print('OK')"
uv run python -c "from src.cli.commands.run_bot import run_bot; print('OK')"
uv run python -c "from src.cli.main import main, cli; print('OK')"
```

**What to look for:**
- Import errors
- Module not found
- Circular dependencies
- Missing dependencies

### Phase 4: CI Simulation

#### 4.1 Run CI Commands Locally
```bash
# Simulate CI test job
uv run pytest --cov=src --cov-report=xml --cov-report=term-missing --cov-fail-under=80

# Simulate CI lint job
uv run ruff check .
uv run ruff format --check .
uv run mypy src

# Simulate CI security job
uv pip install bandit[toml]
uv run bandit -r src/ -f json -o bandit-report.json
```

**What to look for:**
- CI-specific failures
- Environment differences
- Missing dependencies
- Configuration issues

### Phase 5: Pattern-Based Search

#### 5.1 Common Error Patterns
```bash
# Find TODO/FIXME comments
grep -r "TODO\|FIXME\|XXX\|HACK" src/ --include="*.py"

# Find exception handling
grep -r "except\|raise\|assert" src/ --include="*.py" | head -30

# Find potential None issues
grep -r "\.get(\|if.*is None\|if.*== None" src/ --include="*.py" | head -20
```

**What to look for:**
- Incomplete implementations
- Error handling gaps
- Potential None errors
- Assertion failures

#### 5.2 Check Test Coverage Gaps
```bash
# Generate coverage report
uv run pytest --cov=src --cov-report=html

# Check uncovered lines
uv run pytest --cov=src --cov-report=term-missing | grep -E "^\s+[0-9]+\s+[0-9]+\s+[0-9]+\s+[0-9]+%"
```

**What to look for:**
- Uncovered critical paths
- Missing test cases
- Error paths not tested
- Edge cases

### Phase 6: Dependency Check

#### 6.1 Verify Dependencies
```bash
# Check installed packages
uv pip list

# Check for version conflicts
uv pip check

# Verify lock file
uv sync --check
```

**What to look for:**
- Missing dependencies
- Version conflicts
- Outdated packages
- Incompatible versions

## 🔍 Specific Areas to Check

### CLI Commands
- [ ] `backtest` command imports and usage
- [ ] `collect` command imports and usage
- [ ] `run_bot` command imports and usage
- [ ] `main` CLI group structure

### Test Files
- [ ] Patch decorators use correct paths
- [ ] Mock objects are properly configured
- [ ] Test fixtures are correct
- [ ] Assertions are valid

### Import Paths
- [ ] All imports use correct paths
- [ ] No circular dependencies
- [ ] Relative vs absolute imports
- [ ] `__init__.py` files are correct

### Type Hints
- [ ] All functions have type hints
- [ ] Type hints match actual types
- [ ] Generic types are correct
- [ ] Optional types are marked

## 🛠️ Tools to Use

1. **pytest**: Test execution and coverage
2. **Ruff**: Linting and formatting
3. **MyPy**: Type checking
4. **Bandit**: Security scanning
5. **grep/ripgrep**: Pattern searching
6. **git**: Change tracking
7. **uv**: Dependency management

## 📊 Error Categories

### Critical (Block CI)
- Test failures
- Import errors
- Syntax errors
- Type errors

### High Priority (Should Fix)
- Linting errors
- Type warnings
- Coverage gaps
- Security issues

### Medium Priority (Nice to Have)
- Code style issues
- Documentation gaps
- Performance warnings
- Deprecation warnings

## ✅ Success Criteria

- [ ] All tests pass locally
- [ ] All linting checks pass
- [ ] Type checking passes (or acceptable warnings)
- [ ] CI passes on all Python versions
- [ ] No critical security issues
- [ ] Coverage above 80%

## 🚀 Quick Start

Run this command to check everything at once:

```bash
# Quick error check
echo "=== Running Tests ===" && \
uv run pytest --cov=src --cov-report=term-missing -q && \
echo "=== Running Linter ===" && \
uv run ruff check . && \
echo "=== Running Type Check ===" && \
uv run mypy src --no-error-summary && \
echo "✅ All checks passed!"
```

## 📝 Next Steps After Finding Errors

1. **Categorize**: Group errors by type and severity
2. **Prioritize**: Fix critical errors first
3. **Fix**: Address errors one by one
4. **Verify**: Re-run checks after each fix
5. **Document**: Note any patterns or recurring issues

---

**Last Updated**: 2026-01-XX
**Status**: Active
