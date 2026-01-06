"""
Unit tests for MeanReversionStrategy.
"""

import pandas as pd

from src.strategies.mean_reversion import (
    MeanReversionStrategy,
    SimpleMeanReversionStrategy,
)
from src.strategies.mean_reversion.conditions import (
    BollingerLowerBandCondition,
    BollingerUpperBandCondition,
    PriceAboveSMACondition,
    PriceBelowSMACondition,
    RSIOverboughtCondition,
    RSIOversoldCondition,
)


class TestMeanReversionStrategy:
    """Test cases for MeanReversionStrategy."""

    def test_strategy_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = MeanReversionStrategy()
        assert strategy.sma_period == 20
        assert strategy.bb_period == 20
        assert strategy.bb_std == 2.0
        assert strategy.rsi_period == 14
        assert strategy.rsi_oversold == 30.0
        assert strategy.rsi_overbought == 70.0
        assert strategy.name == "MeanReversionStrategy"

    def test_strategy_initialization_custom_params(self) -> None:
        """Test strategy initialization with custom parameters."""
        strategy = MeanReversionStrategy(
            name="CustomMeanReversion",
            sma_period=30,
            bb_period=20,
            bb_std=2.5,
            rsi_period=21,
            rsi_oversold=25.0,
            rsi_overbought=75.0,
        )
        assert strategy.sma_period == 30
        assert strategy.bb_period == 20
        assert strategy.bb_std == 2.5
        assert strategy.rsi_period == 21
        assert strategy.rsi_oversold == 25.0
        assert strategy.rsi_overbought == 75.0
        assert strategy.name == "CustomMeanReversion"

    def test_strategy_initialization_without_default_conditions(self) -> None:
        """Test strategy initialization without default conditions."""
        strategy = MeanReversionStrategy(use_default_conditions=False)
        assert len(strategy.entry_conditions.conditions) == 0
        assert len(strategy.exit_conditions.conditions) == 0

    def test_strategy_initialization_custom_conditions(self) -> None:
        """Test strategy initialization with custom conditions."""
        custom_entry = [BollingerLowerBandCondition()]
        custom_exit = [BollingerUpperBandCondition()]
        strategy = MeanReversionStrategy(
            entry_conditions=custom_entry,
            exit_conditions=custom_exit,
            use_default_conditions=False,
        )
        assert len(strategy.entry_conditions.conditions) == 1
        assert len(strategy.exit_conditions.conditions) == 1
        assert strategy.entry_conditions.conditions[0].name == "BollingerLowerBand"
        assert strategy.exit_conditions.conditions[0].name == "BollingerUpperBand"

    def test_strategy_default_conditions(self) -> None:
        """Test that default conditions are added."""
        strategy = MeanReversionStrategy()
        entry_names = [c.name for c in strategy.entry_conditions.conditions]
        exit_names = [c.name for c in strategy.exit_conditions.conditions]

        assert "PriceBelowSMA" in entry_names
        assert "BollingerLowerBand" in entry_names
        assert "RSIOversold" in entry_names
        assert "PriceAboveSMA" in exit_names
        assert "BollingerUpperBand" in exit_names
        assert "RSIOverbought" in exit_names

    def test_required_indicators(self) -> None:
        """Test required_indicators method."""
        strategy = MeanReversionStrategy()
        indicators = strategy.required_indicators()
        assert "sma" in indicators
        assert "bb_upper" in indicators
        assert "bb_middle" in indicators
        assert "bb_lower" in indicators
        assert "rsi" in indicators

    def test_calculate_indicators(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test indicator calculation."""
        strategy = MeanReversionStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())

        # Check that indicators are added
        assert "sma" in df.columns
        assert "bb_upper" in df.columns
        assert "bb_middle" in df.columns
        assert "bb_lower" in df.columns
        assert "rsi" in df.columns

        # Check that indicators are calculated (not all NaN)
        assert df["sma"].notna().sum() > 0
        assert df["bb_upper"].notna().sum() > 0
        assert df["bb_lower"].notna().sum() > 0
        assert df["rsi"].notna().sum() > 0

    def test_calculate_indicators_custom_periods(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test indicator calculation with custom periods."""
        strategy = MeanReversionStrategy(
            sma_period=30,
            bb_period=20,
            bb_std=2.5,
            rsi_period=21,
        )
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())

        # Should still calculate all indicators
        assert "sma" in df.columns
        assert "bb_upper" in df.columns
        assert "rsi" in df.columns

    def test_generate_signals(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test signal generation."""
        strategy = MeanReversionStrategy()
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
        strategy = MeanReversionStrategy(use_default_conditions=False)
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())
        df = strategy.generate_signals(df)

        # All entry signals should be True (no conditions = all pass)
        assert df["entry_signal"].all()
        # All exit signals should be False (no conditions = no exit)
        assert not df["exit_signal"].any()

    def test_generate_signals_bollinger_lower_band(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test signal generation with only Bollinger lower band condition."""
        strategy = MeanReversionStrategy(
            entry_conditions=[BollingerLowerBandCondition()],
            use_default_conditions=False,
        )
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())
        df = strategy.generate_signals(df)

        # Entry signal should be True when low <= bb_lower
        entry_rows = df[df["entry_signal"]]
        if len(entry_rows) > 0:
            assert (entry_rows["low"] <= entry_rows["bb_lower"]).all()

    def test_generate_signals_bollinger_upper_band_exit(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test signal generation with Bollinger upper band exit condition."""
        strategy = MeanReversionStrategy(
            entry_conditions=[],
            exit_conditions=[BollingerUpperBandCondition()],
            use_default_conditions=False,
        )
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())
        df = strategy.generate_signals(df)

        # Exit signal should be True when high >= bb_upper
        exit_rows = df[df["exit_signal"]]
        if len(exit_rows) > 0:
            assert (exit_rows["high"] >= exit_rows["bb_upper"]).all()

    def test_generate_signals_price_below_sma(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test signal generation with price below SMA condition."""
        strategy = MeanReversionStrategy(
            entry_conditions=[PriceBelowSMACondition(sma_key="sma")],
            use_default_conditions=False,
        )
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())
        df = strategy.generate_signals(df)

        # Entry signal should be True when close < sma
        entry_rows = df[df["entry_signal"]]
        if len(entry_rows) > 0:
            assert (entry_rows["close"] < entry_rows["sma"]).all()

    def test_generate_signals_rsi_oversold(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test signal generation with RSI oversold condition."""
        strategy = MeanReversionStrategy(
            entry_conditions=[RSIOversoldCondition(oversold_threshold=30.0)],
            use_default_conditions=False,
        )
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())
        df = strategy.generate_signals(df)

        # Entry signal should be True when RSI < 30
        entry_rows = df[df["entry_signal"]]
        if len(entry_rows) > 0:
            assert (entry_rows["rsi"] < 30.0).all()

    def test_generate_signals_multiple_conditions(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test signal generation with multiple entry conditions."""
        strategy = MeanReversionStrategy(
            entry_conditions=[
                PriceBelowSMACondition(sma_key="sma"),
                BollingerLowerBandCondition(),
            ],
            use_default_conditions=False,
        )
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())
        df = strategy.generate_signals(df)

        entry_rows = df[df["entry_signal"]]
        if len(entry_rows) > 0:
            # All conditions must be met (AND logic)
            assert (entry_rows["close"] < entry_rows["sma"]).all()
            assert (entry_rows["low"] <= entry_rows["bb_lower"]).all()

    def test_bollinger_bands_calculation(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test Bollinger Bands calculation."""
        strategy = MeanReversionStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())

        # Bollinger Bands should be calculated
        assert "bb_upper" in df.columns
        assert "bb_middle" in df.columns
        assert "bb_lower" in df.columns

        # bb_middle should equal SMA
        valid_rows = df[df["bb_middle"].notna() & df["sma"].notna()]
        if len(valid_rows) > 0:
            pd.testing.assert_series_equal(
                valid_rows["bb_middle"],
                valid_rows["sma"],
                check_names=False,
            )

        # bb_upper should be >= bb_middle >= bb_lower
        valid_rows = df[
            df["bb_upper"].notna()
            & df["bb_middle"].notna()
            & df["bb_lower"].notna()
        ]
        if len(valid_rows) > 0:
            assert (valid_rows["bb_upper"] >= valid_rows["bb_middle"]).all()
            assert (valid_rows["bb_middle"] >= valid_rows["bb_lower"]).all()

    def test_rsi_calculation_range(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test that RSI values are in valid range (0-100)."""
        strategy = MeanReversionStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())

        valid_rsi = df["rsi"][df["rsi"].notna()]
        if len(valid_rsi) > 0:
            assert (valid_rsi >= 0).all()
            assert (valid_rsi <= 100).all()


class TestSimpleMeanReversionStrategy:
    """Test cases for SimpleMeanReversionStrategy."""

    def test_simple_mean_reversion_initialization(self) -> None:
        """Test SimpleMeanReversionStrategy initialization."""
        strategy = SimpleMeanReversionStrategy()
        assert strategy.name == "SimpleMeanReversion"
        assert len(strategy.entry_conditions.conditions) == 1
        assert strategy.entry_conditions.conditions[0].name == "BollingerLowerBand"
        assert len(strategy.exit_conditions.conditions) == 1
        assert strategy.exit_conditions.conditions[0].name == "BollingerUpperBand"

    def test_simple_mean_reversion_custom_name(self) -> None:
        """Test SimpleMeanReversionStrategy with custom name."""
        strategy = SimpleMeanReversionStrategy(name="MySimpleMeanReversion")
        assert strategy.name == "MySimpleMeanReversion"

    def test_simple_mean_reversion_signals(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test SimpleMeanReversionStrategy signal generation."""
        strategy = SimpleMeanReversionStrategy()
        df = strategy.calculate_indicators(sample_ohlcv_data.copy())
        df = strategy.generate_signals(df)

        assert "entry_signal" in df.columns
        assert "exit_signal" in df.columns

        # Entry signals should only depend on Bollinger lower band
        entry_rows = df[df["entry_signal"]]
        if len(entry_rows) > 0:
            assert (entry_rows["low"] <= entry_rows["bb_lower"]).all()

        # Exit signals should only depend on Bollinger upper band
        exit_rows = df[df["exit_signal"]]
        if len(exit_rows) > 0:
            assert (exit_rows["high"] >= exit_rows["bb_upper"]).all()

    def test_simple_mean_reversion_inheritance(self) -> None:
        """Test that SimpleMeanReversionStrategy inherits from MeanReversionStrategy."""
        strategy = SimpleMeanReversionStrategy()
        assert isinstance(strategy, MeanReversionStrategy)
        assert strategy.sma_period == 20  # Default from MeanReversionStrategy
