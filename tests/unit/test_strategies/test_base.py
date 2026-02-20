"""
Unit tests for strategy base classes.
"""

from datetime import date

import pandas as pd
import pytest

from src.strategies.base import (
    OHLCV,
    CompositeCondition,
    Condition,
    Position,
    Strategy,
)


# Mock condition for testing
class MockCondition(Condition):
    """Mock condition for testing CompositeCondition."""

    def __init__(self, name: str, return_value: bool = True) -> None:
        super().__init__(name)
        self.return_value = return_value
        self.call_count = 0

    def evaluate(
        self,
        current: OHLCV,
        history: pd.DataFrame,
        indicators: dict[str, float],
    ) -> bool:
        """Return configured value for testing."""
        self.call_count += 1
        return self.return_value


# Mock strategy for testing
class MockStrategy(Strategy):
    """Mock strategy for testing Strategy base class."""

    def required_indicators(self) -> list[str]:
        """Return required indicators."""
        return ["sma", "target"]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators."""
        df = df.copy()
        df["sma"] = df["close"].rolling(window=5).mean()
        df["target"] = df["close"] * 1.1
        return df


class TestOHLCV:
    """Tests for OHLCV dataclass."""

    def test_ohlcv_initialization(self) -> None:
        """Test OHLCV initialization."""
        ohlcv = OHLCV(
            date=date(2024, 1, 1),
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=1000.0,
        )
        assert ohlcv.date == date(2024, 1, 1)
        assert ohlcv.open == 100.0
        assert ohlcv.high == 110.0
        assert ohlcv.low == 95.0
        assert ohlcv.close == 105.0
        assert ohlcv.volume == 1000.0

    def test_ohlcv_range_property(self) -> None:
        """Test OHLCV range property."""
        ohlcv = OHLCV(
            date=date(2024, 1, 1),
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=1000.0,
        )
        assert ohlcv.range == 15.0  # high - low

    def test_ohlcv_body_property(self) -> None:
        """Test OHLCV body property."""
        # Upward candle
        ohlcv_up = OHLCV(
            date=date(2024, 1, 1),
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=1000.0,
        )
        assert ohlcv_up.body == 5.0  # abs(close - open)

        # Downward candle
        ohlcv_down = OHLCV(
            date=date(2024, 1, 1),
            open=105.0,
            high=110.0,
            low=95.0,
            close=100.0,
            volume=1000.0,
        )
        assert ohlcv_down.body == 5.0  # abs(close - open)


class TestCompositeCondition:
    """Tests for CompositeCondition class."""

    def test_composite_condition_initialization(self) -> None:
        """Test CompositeCondition initialization."""
        condition1 = MockCondition("Condition1")
        condition2 = MockCondition("Condition2")
        composite = CompositeCondition([condition1, condition2], operator="AND")
        assert len(composite.conditions) == 2
        assert composite.operator == "AND"

    def test_composite_condition_empty_list(self) -> None:
        """Test CompositeCondition with empty conditions list."""
        composite = CompositeCondition([], operator="AND")
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}
        assert composite.evaluate(current, history, indicators) is True

    def test_composite_condition_and_operator(self) -> None:
        """Test CompositeCondition with AND operator."""
        condition1 = MockCondition("Condition1", return_value=True)
        condition2 = MockCondition("Condition2", return_value=True)
        composite = CompositeCondition([condition1, condition2], operator="AND")

        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert composite.evaluate(current, history, indicators) is True
        assert condition1.call_count == 1
        assert condition2.call_count == 1

    def test_composite_condition_and_operator_false(self) -> None:
        """Test CompositeCondition with AND operator when one condition is False."""
        condition1 = MockCondition("Condition1", return_value=True)
        condition2 = MockCondition("Condition2", return_value=False)
        composite = CompositeCondition([condition1, condition2], operator="AND")

        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert composite.evaluate(current, history, indicators) is False

    def test_composite_condition_or_operator(self) -> None:
        """Test CompositeCondition with OR operator."""
        condition1 = MockCondition("Condition1", return_value=False)
        condition2 = MockCondition("Condition2", return_value=True)
        composite = CompositeCondition([condition1, condition2], operator="OR")

        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert composite.evaluate(current, history, indicators) is True

    def test_composite_condition_or_operator_false(self) -> None:
        """Test CompositeCondition with OR operator when all conditions are False."""
        condition1 = MockCondition("Condition1", return_value=False)
        condition2 = MockCondition("Condition2", return_value=False)
        composite = CompositeCondition([condition1, condition2], operator="OR")

        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert composite.evaluate(current, history, indicators) is False

    def test_composite_condition_invalid_operator(self) -> None:
        """Test CompositeCondition with invalid operator."""
        condition1 = MockCondition("Condition1")
        composite = CompositeCondition([condition1], operator="INVALID")

        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        with pytest.raises(ValueError, match="Unknown operator"):
            composite.evaluate(current, history, indicators)

    def test_composite_condition_add(self) -> None:
        """Test CompositeCondition add method."""
        condition1 = MockCondition("Condition1")
        composite = CompositeCondition([condition1], operator="AND")
        assert len(composite.conditions) == 1

        condition2 = MockCondition("Condition2")
        result = composite.add(condition2)
        assert result is composite  # Should return self for chaining
        assert len(composite.conditions) == 2

    def test_composite_condition_remove(self) -> None:
        """Test CompositeCondition remove method."""
        condition1 = MockCondition("Condition1")
        condition2 = MockCondition("Condition2")
        composite = CompositeCondition([condition1, condition2], operator="AND")
        assert len(composite.conditions) == 2

        result = composite.remove(condition1)
        assert result is composite  # Should return self for chaining
        assert len(composite.conditions) == 1
        assert composite.conditions[0] is condition2

    def test_composite_condition_operator_case_insensitive(self) -> None:
        """Test CompositeCondition operator is case insensitive."""
        condition1 = MockCondition("Condition1", return_value=True)
        condition2 = MockCondition("Condition2", return_value=True)
        composite = CompositeCondition([condition1, condition2], operator="and")  # lowercase

        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert composite.operator == "AND"  # Should be uppercase
        assert composite.evaluate(current, history, indicators) is True

    def test_composite_condition_with_series_returning_condition(self) -> None:
        """Test CompositeCondition with condition returning pd.Series (line 99)."""

        # Create a condition that returns a pd.Series
        class SeriesCondition(Condition):
            def __init__(self, name: str, series_values: list[bool]) -> None:
                super().__init__(name)
                self.series_values = series_values

            def evaluate(
                self,
                current: OHLCV,
                history: pd.DataFrame,
                indicators: dict[str, float],
            ) -> pd.Series:
                return pd.Series(self.series_values)

        condition1 = SeriesCondition("SeriesCondition", [False, True, True])
        composite = CompositeCondition([condition1], operator="AND")

        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        # Should use the last value (True) from the series
        result = composite.evaluate(current, history, indicators)
        assert result is True


class TestStrategy:
    """Tests for Strategy base class."""

    def test_strategy_initialization(self) -> None:
        """Test Strategy initialization."""
        strategy = MockStrategy(name="TestStrategy")
        assert strategy.name == "TestStrategy"
        assert len(strategy.entry_conditions.conditions) == 0
        assert len(strategy.exit_conditions.conditions) == 0

    def test_strategy_default_name(self) -> None:
        """Test Strategy uses class name as default name."""
        strategy = MockStrategy()
        assert strategy.name == "MockStrategy"

    def test_strategy_with_entry_conditions(self) -> None:
        """Test Strategy initialization with entry conditions."""
        condition1 = MockCondition("Condition1")
        condition2 = MockCondition("Condition2")
        strategy = MockStrategy(entry_conditions=[condition1, condition2])

        assert len(strategy.entry_conditions.conditions) == 2
        assert strategy.entry_conditions.operator == "AND"

    def test_strategy_with_exit_conditions(self) -> None:
        """Test Strategy initialization with exit conditions."""
        condition1 = MockCondition("Condition1")
        strategy = MockStrategy(exit_conditions=[condition1])

        assert len(strategy.exit_conditions.conditions) == 1
        assert strategy.exit_conditions.operator == "AND"

    def test_strategy_generate_signals_default(self) -> None:
        """Test Strategy generate_signals default implementation."""
        # Create sample data
        dates = pd.date_range(start="2024-01-01", periods=20, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0] * 20,
                "high": [110.0] * 20,
                "low": [95.0] * 20,
                "close": [105.0] * 20,
                "volume": [1000.0] * 20,
                "sma": [100.0] * 20,
                "target": [110.0] * 20,
                "sma_trend": [98.0] * 20,
                "short_noise": [0.5] * 20,
                "long_noise": [0.6] * 20,
            },
            index=dates,
        )

        strategy = MockStrategy()
        result = strategy.generate_signals(df)

        assert "entry_signal" in result.columns
        assert "exit_signal" in result.columns
        assert result["entry_signal"].dtype == bool
        assert result["exit_signal"].dtype == bool

    def test_strategy_check_entry(self) -> None:
        """Test Strategy check_entry method."""
        condition1 = MockCondition("Condition1", return_value=True)
        condition2 = MockCondition("Condition2", return_value=True)
        strategy = MockStrategy(entry_conditions=[condition1, condition2])

        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {"sma": 100.0, "target": 110.0}

        assert strategy.check_entry(current, history, indicators) is True
        assert condition1.call_count == 1
        assert condition2.call_count == 1

    def test_strategy_check_entry_false(self) -> None:
        """Test Strategy check_entry returns False when conditions not met."""
        condition1 = MockCondition("Condition1", return_value=False)
        strategy = MockStrategy(entry_conditions=[condition1])

        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}

        assert strategy.check_entry(current, history, indicators) is False

    def test_strategy_check_exit(self) -> None:
        """Test Strategy check_exit method."""
        condition1 = MockCondition("Condition1", return_value=True)
        strategy = MockStrategy(exit_conditions=[condition1])

        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        history = pd.DataFrame()
        indicators: dict[str, float] = {}
        position = Position(
            ticker="KRW-BTC",
            amount=0.001,
            entry_price=50000.0,
            entry_date=date(2024, 1, 1),
        )

        assert strategy.check_exit(current, history, indicators, position) is True

    def test_strategy_add_entry_condition(self) -> None:
        """Test Strategy add_entry_condition method."""
        strategy = MockStrategy()
        assert len(strategy.entry_conditions.conditions) == 0

        condition = MockCondition("NewCondition")
        result = strategy.add_entry_condition(condition)

        assert result is strategy  # Should return self for chaining
        assert len(strategy.entry_conditions.conditions) == 1

    def test_strategy_add_exit_condition(self) -> None:
        """Test Strategy add_exit_condition method."""
        strategy = MockStrategy()
        assert len(strategy.exit_conditions.conditions) == 0

        condition = MockCondition("NewCondition")
        result = strategy.add_exit_condition(condition)

        assert result is strategy  # Should return self for chaining
        assert len(strategy.exit_conditions.conditions) == 1

    def test_strategy_remove_entry_condition(self) -> None:
        """Test Strategy remove_entry_condition method."""
        condition = MockCondition("Condition1")
        strategy = MockStrategy(entry_conditions=[condition])
        assert len(strategy.entry_conditions.conditions) == 1

        result = strategy.remove_entry_condition(condition)

        assert result is strategy  # Should return self for chaining
        assert len(strategy.entry_conditions.conditions) == 0

    def test_strategy_remove_exit_condition(self) -> None:
        """Test Strategy remove_exit_condition method."""
        condition = MockCondition("Condition1")
        strategy = MockStrategy(exit_conditions=[condition])
        assert len(strategy.exit_conditions.conditions) == 1

        result = strategy.remove_exit_condition(condition)

        assert result is strategy  # Should return self for chaining
        assert len(strategy.exit_conditions.conditions) == 0

    def test_strategy_repr(self) -> None:
        """Test Strategy __repr__ method."""
        strategy = MockStrategy(name="TestStrategy")
        condition1 = MockCondition("Condition1")
        condition2 = MockCondition("Condition2")
        strategy.add_entry_condition(condition1)
        strategy.add_exit_condition(condition2)

        repr_str = repr(strategy)
        assert "MockStrategy" in repr_str
        assert "TestStrategy" in repr_str
        assert "entry_conditions=1" in repr_str
        assert "exit_conditions=1" in repr_str


class TestStrategyNewMethods:
    """Tests for Strategy ABC methods added during refactoring."""

    def test_exit_price_column_default_is_close(self) -> None:
        """Default exit_price_column returns 'close'."""
        strategy = MockStrategy()
        assert strategy.exit_price_column == "close"

    def test_parameter_schema_default_is_empty(self) -> None:
        """Default parameter_schema returns empty dict."""
        assert MockStrategy.parameter_schema() == {}

    def test_exit_price_column_can_be_overridden(self) -> None:
        """Subclasses can override exit_price_column."""

        class CustomExitStrategy(MockStrategy):
            @property
            def exit_price_column(self) -> str:
                return "exit_price_base"

        strategy = CustomExitStrategy()
        assert strategy.exit_price_column == "exit_price_base"

    def test_parameter_schema_can_be_overridden(self) -> None:
        """Subclasses can override parameter_schema."""

        class ParametricStrategy(MockStrategy):
            @classmethod
            def parameter_schema(cls) -> dict[str, object]:
                return {"k": {"type": "float", "min": 0.1, "max": 0.9}}

        schema = ParametricStrategy.parameter_schema()
        assert "k" in schema

    def test_vbov1_exit_price_column(self) -> None:
        """VBOV1 overrides exit_price_column to 'exit_price_base'."""
        from src.strategies.volatility_breakout.vbo_v1 import VBOV1

        strategy = VBOV1()
        assert strategy.exit_price_column == "exit_price_base"

    def test_vbov1_parameter_schema(self) -> None:
        """VBOV1 exposes noise_ratio, ma_short, btc_ma in schema."""
        from src.strategies.volatility_breakout.vbo_v1 import VBOV1

        schema = VBOV1.parameter_schema()
        assert "noise_ratio" in schema
        assert "ma_short" in schema
        assert "btc_ma" in schema


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_initialization(self) -> None:
        """Test Position initialization."""
        position = Position(
            ticker="KRW-BTC",
            amount=0.001,
            entry_price=50000.0,
            entry_date=date(2024, 1, 1),
        )
        assert position.ticker == "KRW-BTC"
        assert position.amount == 0.001
        assert position.entry_price == 50000.0
        assert position.entry_date == date(2024, 1, 1)

    def test_position_highest_price_defaults_to_none(self) -> None:
        """Position.highest_price defaults to None."""
        position = Position(
            ticker="KRW-BTC",
            amount=0.001,
            entry_price=50000.0,
            entry_date=date(2024, 1, 1),
        )
        assert position.highest_price is None

    def test_position_highest_price_can_be_set(self) -> None:
        """Position.highest_price can be set for trailing stop tracking."""
        position = Position(
            ticker="KRW-BTC",
            amount=0.001,
            entry_price=50000.0,
            entry_date=date(2024, 1, 1),
            highest_price=55000.0,
        )
        assert position.highest_price == 55000.0
