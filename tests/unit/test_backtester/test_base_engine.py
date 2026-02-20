"""Tests for src/backtester/engine/base_engine.py — BaseBacktestEngine ABC."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.backtester.engine.base_engine import BaseBacktestEngine
from src.backtester.models import BacktestConfig, BacktestResult
from src.strategies.base import Strategy

# ---------------------------------------------------------------------------
# Concrete subclass for testing
# ---------------------------------------------------------------------------


class _ConcreteEngine(BaseBacktestEngine):
    """Minimal concrete engine to test ABC behaviour."""

    def run(
        self,
        strategy: Strategy,
        data_files: dict[str, Path],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> BacktestResult:
        return BacktestResult(strategy_name="test")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaseBacktestEngine:
    def test_cannot_instantiate_abc_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseBacktestEngine()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        engine = _ConcreteEngine()
        assert engine is not None

    def test_default_config_is_backtest_config(self) -> None:
        engine = _ConcreteEngine()
        assert isinstance(engine.config, BacktestConfig)

    def test_custom_config_is_stored(self) -> None:
        config = BacktestConfig(initial_capital=999.0)
        engine = _ConcreteEngine(config)
        assert engine.config.initial_capital == 999.0

    def test_none_config_uses_defaults(self) -> None:
        engine = _ConcreteEngine(None)
        assert engine.config.initial_capital == BacktestConfig().initial_capital

    def test_vectorized_engine_is_subclass(self) -> None:
        from src.backtester.engine.vectorized import VectorizedBacktestEngine

        assert issubclass(VectorizedBacktestEngine, BaseBacktestEngine)

    def test_event_driven_engine_is_subclass(self) -> None:
        from src.backtester.engine.event_driven import EventDrivenBacktestEngine

        assert issubclass(EventDrivenBacktestEngine, BaseBacktestEngine)
