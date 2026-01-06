"""
Unit tests for MomentumStrategy.
"""

import pandas as pd

from src.strategies.momentum import MomentumStrategy, SimpleMomentumStrategy
from src.strategies.momentum.conditions import (
    MACDBearishCondition,
    MACDBullishCondition,
    PriceAboveSMACondition,
    PriceBelowSMACondition,
    RSIOverboughtCondition,
)


class TestMomentumStrategy:
    """Test cases for MomentumStrategy."""

    def test_strategy_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = MomentumStrategy()
        assert strategy.sma_period == 20
        assert strategy.rsi_period == 14
        assert strategy.macd_fast == 12
        assert strategy.macd_slow == 26
        assert strategy.macd_signal == 9
        assert strategy.rsi_oversold == 30.0
        assert strategy.rsi_overbought == 70.0
        assert strategy.name == "MomentumStrategy"

    def test_strategy_initialization_custom_params(self) -> None:
        """Test strategy initialization with custom parameters."""
        strategy = MomentumStrategy(
            name="CustomMomentum",
            sma_period=50,
            rsi_period=21,
            macd_fast=8,
            macd_slow=21,
            macd_signal=5,
            rsi_oversold=25.0,
            rsi_overbought=75.0,
        )
        assert strategy.sma_period == 50
        assert strategy.rsi_period == 21
        assert strategy.macd_fast == 8
        assert strategy.macd_slow == 21
        assert strategy.macd_signal == 5
        assert strategy.rsi_oversold == 25.0
        assert strategy.rsi_overbought == 75.0
        assert strategy.name == "CustomMomentum"

    def test_strategy_initialization_without_default_conditions(self) -> None:
        """Test strategy initialization without default conditions."""
        strategy = MomentumStrategy(use_default_conditions=False)
        assert len(strategy.entry_conditions.conditions) == 0
        assert len(strategy.exit_conditions.conditions) == 0

    def test_strategy_initialization_custom_conditions(self) -> None:
        """Test strategy initialization with custom conditions."""
        custom_entry = [PriceAboveSMACondition(sma_key="sma")]
        custom_exit = [PriceBelowSMACondition(sma_key="sma")]
        strategy = MomentumStrategy(
            entry_conditions=custom_entry,
            exit_conditions=custom_exit,
            use_default_conditions=False,
        )
        assert len(strategy.entry_conditions.conditions) == 1
        assert len(strategy.exit_conditions.conditions) == 1
        assert strategy.entry_conditions.conditions[0].name == "PriceAboveSMA"
        assert strategy.exit_conditions.conditions[0].name == "PriceBelowSMA"

    def test_strategy_default_conditions(self) -> None:
        """Test that default conditions are added."""
        strategy = MomentumStrategy()
        entry_names = [c.name for c in strategy.entry_conditions.conditions]
        exit_names = [c.name for c in strategy.exit_conditions.conditions]

        assert "PriceAboveSMA" in entry_names
        assert "MACDBullish" in entry_names
        assert "PriceBelowSMA" in exit_names
        assert "RSIOverbought" in exit_names
        assert "MACDBearish" in exit_names

    def test_required_indicators(self) -> None:
        """Test required_indicators method."""
        strategy = MomentumStrategy()
        indicators = strategy.required_indicators()
        assert "sma" in indicators
        assert "rsi" in indicators
        assert "macd" in indicators
        assert "macd_signal" in indicators
        assert "macd_histogram" in indicators

    def test_calculate_indicators(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test indicator calculation."""
        strategy = MomentumStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())

        # Check that indicators are added
        assert "sma" in df.columns
        assert "rsi" in df.columns
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_histogram" in df.columns

        # Check that indicators are calculated (not all NaN)
        assert df["sma"].notna().sum() > 0
        assert df["rsi"].notna().sum() > 0
        assert df["macd"].notna().sum() > 0

    def test_calculate_indicators_custom_periods(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test indicator calculation with custom periods."""
        strategy = MomentumStrategy(
            sma_period=10,
            rsi_period=7,
            macd_fast=6,
            macd_slow=13,
            macd_signal=3,
        )
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())

        # Should still calculate all indicators
        assert "sma" in df.columns
        assert "rsi" in df.columns
        assert "macd" in df.columns

    def test_generate_signals(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test signal generation."""
        strategy = MomentumStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())
        df = strategy.generate_signals(df)

        # Check that signals are added
        assert "entry_signal" in df.columns
        assert "exit_signal" in df.columns

        # Check signal types
        assert df["entry_signal"].dtype == bool
        assert df["exit_signal"].dtype == bool

    def test_generate_signals_empty_conditions(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test signal generation with no conditions."""
        strategy = MomentumStrategy(use_default_conditions=False)
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())
        df = strategy.generate_signals(df)

        # All entry signals should be True (no conditions = all pass)
        assert df["entry_signal"].all()
        # All exit signals should be False (no conditions = no exit)
        assert not df["exit_signal"].any()

    def test_generate_signals_price_above_sma(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test signal generation with only price above SMA condition."""
        strategy = MomentumStrategy(
            entry_conditions=[PriceAboveSMACondition(sma_key="sma")],
            use_default_conditions=False,
        )
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())
        df = strategy.generate_signals(df)

        # Entry signal should be True when close > sma
        entry_rows = df[df["entry_signal"]]
        if len(entry_rows) > 0:
            assert (entry_rows["close"] > entry_rows["sma"]).all()

    def test_generate_signals_price_below_sma_exit(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test signal generation with SMA exit condition."""
        strategy = MomentumStrategy(
            entry_conditions=[],
            exit_conditions=[PriceBelowSMACondition(sma_key="sma")],
            use_default_conditions=False,
        )
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())
        df = strategy.generate_signals(df)

        # Exit signal should be True when close < sma
        exit_rows = df[df["exit_signal"]]
        if len(exit_rows) > 0:
            assert (exit_rows["close"] < exit_rows["sma"]).all()

    def test_generate_signals_macd_bullish(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test signal generation with MACD bullish condition."""
        strategy = MomentumStrategy(
            entry_conditions=[MACDBullishCondition()],
            use_default_conditions=False,
        )
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())
        df = strategy.generate_signals(df)

        # Entry signal should be True when MACD > signal
        entry_rows = df[df["entry_signal"]]
        if len(entry_rows) > 0:
            assert (entry_rows["macd"] > entry_rows["macd_signal"]).all()

    def test_generate_signals_rsi_overbought_exit(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test signal generation with RSI overbought exit condition."""
        strategy = MomentumStrategy(
            entry_conditions=[],
            exit_conditions=[RSIOverboughtCondition(overbought_threshold=70.0)],
            use_default_conditions=False,
        )
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())
        df = strategy.generate_signals(df)

        # Exit signal should be True when RSI > 70
        exit_rows = df[df["exit_signal"]]
        if len(exit_rows) > 0:
            assert (exit_rows["rsi"] > 70.0).all()

    def test_generate_signals_multiple_conditions(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test signal generation with multiple entry conditions."""
        strategy = MomentumStrategy(
            entry_conditions=[
                PriceAboveSMACondition(sma_key="sma"),
                MACDBullishCondition(),
            ],
            use_default_conditions=False,
        )
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())
        df = strategy.generate_signals(df)

        entry_rows = df[df["entry_signal"]]
        if len(entry_rows) > 0:
            # All conditions must be met (AND logic)
            assert (entry_rows["close"] > entry_rows["sma"]).all()
            assert (entry_rows["macd"] > entry_rows["macd_signal"]).all()

    def test_rsi_calculation_range(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test that RSI values are in valid range (0-100)."""
        strategy = MomentumStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())

        valid_rsi = df["rsi"][df["rsi"].notna()]
        if len(valid_rsi) > 0:
            assert (valid_rsi >= 0).all()
            assert (valid_rsi <= 100).all()

    def test_macd_calculation(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test MACD calculation."""
        strategy = MomentumStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())

        # MACD should be calculated
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_histogram" in df.columns

        # MACD histogram should equal macd - signal
        valid_rows = df[df["macd"].notna() & df["macd_signal"].notna()]
        if len(valid_rows) > 0:
            calculated_histogram = (
                valid_rows["macd"] - valid_rows["macd_signal"]
            )
            pd.testing.assert_series_equal(
                valid_rows["macd_histogram"],
                calculated_histogram,
                check_names=False,
            )


class TestSimpleMomentumStrategy:
    """Test cases for SimpleMomentumStrategy."""

    def test_simple_momentum_initialization(self) -> None:
        """Test SimpleMomentumStrategy initialization."""
        strategy = SimpleMomentumStrategy()
        assert strategy.name == "SimpleMomentum"
        assert len(strategy.entry_conditions.conditions) == 1
        assert strategy.entry_conditions.conditions[0].name == "PriceAboveSMA"
        assert len(strategy.exit_conditions.conditions) == 1
        assert strategy.exit_conditions.conditions[0].name == "PriceBelowSMA"

    def test_simple_momentum_custom_name(self) -> None:
        """Test SimpleMomentumStrategy with custom name."""
        strategy = SimpleMomentumStrategy(name="MySimpleMomentum")
        assert strategy.name == "MySimpleMomentum"

    def test_simple_momentum_signals(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test SimpleMomentumStrategy signal generation."""
        strategy = SimpleMomentumStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())
        df = strategy.generate_signals(df)

        assert "entry_signal" in df.columns
        assert "exit_signal" in df.columns

        # Entry signals should only depend on price above SMA
        entry_rows = df[df["entry_signal"]]
        if len(entry_rows) > 0:
            assert (entry_rows["close"] > entry_rows["sma"]).all()

        # Exit signals should only depend on price below SMA
        exit_rows = df[df["exit_signal"]]
        if len(exit_rows) > 0:
            assert (exit_rows["close"] < exit_rows["sma"]).all()

    def test_simple_momentum_inheritance(self) -> None:
        """Test that SimpleMomentumStrategy inherits from MomentumStrategy."""
        strategy = SimpleMomentumStrategy()
        assert isinstance(strategy, MomentumStrategy)
        assert strategy.sma_period == 20  # Default from MomentumStrategy
