"""
HTML report CSS styles.

Loads CSS from an external file for the HTML backtest report.
"""

import functools
from pathlib import Path

_CSS_PATH = Path(__file__).parent / "report.css"


@functools.lru_cache(maxsize=1)
def get_report_css() -> str:
    """Return CSS styles for the HTML report."""
    return _CSS_PATH.read_text(encoding="utf-8")
