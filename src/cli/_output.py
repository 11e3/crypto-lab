"""CLI output saving utilities.

Provides a context manager that tees stdout to both terminal and a results file.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator


class _Tee:
    """Writes to both the original stdout and a file simultaneously."""

    def __init__(self, file: object, stdout: object) -> None:
        self._file = file
        self._stdout = stdout

    def write(self, data: str) -> int:
        self._stdout.write(data)  # type: ignore[attr-defined]
        return self._file.write(data)  # type: ignore[attr-defined]

    def flush(self) -> None:
        self._stdout.flush()  # type: ignore[attr-defined]
        self._file.flush()  # type: ignore[attr-defined]


@contextmanager
def save_output(
    command: str,
    strategy: str,
    enabled: bool = True,
) -> Generator[Path, None, None]:
    """Tee stdout to results/{command}_{strategy}_{timestamp}.txt.

    Args:
        command: CLI subcommand name (backtest, optimize, wfa, analyze)
        strategy: Strategy name for the filename
        enabled: If False, skips file saving (--no-save flag)

    Yields:
        Path to the result file (may not exist if enabled=False)
    """
    results_dir = Path("results")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"{command}_{strategy}_{ts}.txt"

    if not enabled:
        yield path
        return

    results_dir.mkdir(exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        old_stdout = sys.stdout
        sys.stdout = _Tee(f, old_stdout)  # type: ignore[assignment]
        try:
            yield path
        finally:
            sys.stdout = old_stdout

    print(f"\n  Saved: {path}")


__all__ = ["save_output"]
