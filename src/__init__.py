"""
Upbit Quant System - Automated trading system using volatility breakout strategy.
"""

# Apply monkeypatch for Python 3.12.7+ type annotation fixes
# This must run before any third-party imports
import src._monkeypatch  # noqa: F401
from src.__version__ import __version__

__all__ = ["__version__"]
