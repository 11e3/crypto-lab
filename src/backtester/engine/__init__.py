"""
Backtester engine package.

Contains different backtesting engine implementations:
- EventDrivenBacktestEngine: Clear, debuggable event-driven approach
- VectorizedBacktestEngine: High-performance vectorized approach

Both engines implement BacktestEngineProtocol for interchangeability.
"""

from src.backtester.engine.backtest_runner import run_backtest
from src.backtester.engine.event_driven import EventDrivenBacktestEngine
from src.backtester.engine.protocols import BacktestEngineProtocol
from src.backtester.engine.vectorized import VectorizedBacktestEngine
from src.backtester.models import BacktestConfig, BacktestResult, Trade

# Type alias for engine protocol (use for type hints)
BacktestEngine = BacktestEngineProtocol

# Backward compatibility alias (deprecated)
SimpleBacktestEngine = EventDrivenBacktestEngine

__all__ = [
    # Protocol
    "BacktestEngineProtocol",
    "BacktestEngine",  # Type alias for protocol
    # Implementations
    "EventDrivenBacktestEngine",
    "VectorizedBacktestEngine",
    "SimpleBacktestEngine",  # Deprecated alias
    # Utilities
    "run_backtest",
    # Models
    "BacktestConfig",
    "BacktestResult",
    "Trade",
]
