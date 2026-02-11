---
name: weekly-review
description: Weekly code quality review — simplification opportunities, dead code, coverage gaps
disable-model-invocation: true
---

Perform a weekly code quality review of crypto-quant-system. This is a READ-ONLY analysis — do NOT implement changes automatically.

## Review Steps

### 1. Code Simplification Scan
Use the code-simplifier plugin to analyze `src/` for:
- Duplicate code across modules
- Dead code (unreachable branches, unused imports/functions)
- Overly complex methods (>50 lines)
- Mathematical or logical no-ops

### 2. Test Coverage Check
```bash
python -m pytest --cov=src --cov-report=term-missing -q
```
- Flag any module below 80% coverage
- Identify untested public methods
- Note coverage trend (compare with previous ~89.88%)

### 3. Static Analysis
```bash
ruff check src/ tests/
mypy src/
```
- Report any new violations
- Compare with known baseline (14 mypy pre-existing)

### 4. Dependency Health
- Check for known vulnerable dependencies
- Flag any pinned versions that are significantly outdated

## Output Format

```markdown
## Weekly Review Report — [date]

### Code Quality
- Simplification opportunities: N items (H high, M medium, L low)
- Dead code found: [list if any]
- Complex methods (>50 lines): [list if any]

### Test Health
- Total tests: N (trend: ↑/↓/→ from previous)
- Coverage: XX.XX% (trend: ↑/↓/→)
- Modules below 80%: [list]

### Static Analysis
- Ruff violations: N (N new)
- MyPy errors: N (N new beyond 14 known)

### Action Items
1. [HIGH] ...
2. [MEDIUM] ...
3. [LOW] ...
```

Present the report and wait for user to decide which items to implement.
