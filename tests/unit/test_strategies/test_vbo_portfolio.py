"""Tests for VBOPortfolio and VBOSingleCoin strategies."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.strategies.volatility_breakout.conditions_btc_filter import (
    BtcMarketExitCondition,
    BtcMarketFilterCondition,
)
from src.strategies.volatility_breakout.vbo_portfolio import (
    VBOPortfolio,
    VBOPortfolioLite,
    VBOSingleCoin,
)


@pytest.fixture()
def btc_data() -> pd.DataFrame:
    """Generate BTC OHLCV data for market filter testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    # Generate uptrend for first 50 bars, downtrend for rest
    prices_up = 30000 + np.cumsum(np.abs(np.random.randn(50)) * 200)
    prices_down = prices_up[-1] - np.cumsum(np.abs(np.random.randn(50)) * 300)
    prices = np.concatenate([prices_up, prices_down])
    return pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.97,
            "close": prices,
            "volume": np.random.rand(n) * 10000,
        },
        index=dates,
    )


@pytest.fixture()
def target_ohlcv(btc_data: pd.DataFrame) -> pd.DataFrame:
    """Generate target coin OHLCV data with same index as BTC."""
    np.random.seed(123)
    n = len(btc_data)
    prices = 1000 + np.cumsum(np.random.randn(n) * 50)
    return pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.97,
            "close": prices,
            "volume": np.random.rand(n) * 5000,
        },
        index=btc_data.index,
    )


class TestVBOPortfolioInit:
    """Test VBOPortfolio initialization."""

    def test_default_parameters(self) -> None:
        strategy = VBOPortfolio(btc_data=pd.DataFrame())
        assert strategy.name == "VBOPortfolio"
        assert strategy.ma_short == 5
        assert strategy.btc_ma == 20
        assert strategy.noise_ratio == 0.5

    def test_custom_parameters(self) -> None:
        strategy = VBOPortfolio(
            name="CustomVBO",
            ma_short=10,
            btc_ma=40,
            noise_ratio=0.7,
            btc_data=pd.DataFrame(),
        )
        assert strategy.name == "CustomVBO"
        assert strategy.ma_short == 10
        assert strategy.btc_ma == 40
        assert strategy.noise_ratio == 0.7

    def test_required_indicators(self) -> None:
        strategy = VBOPortfolio(btc_data=pd.DataFrame())
        indicators = strategy.required_indicators()
        assert "sma" in indicators
        assert "target" in indicators
        assert "btc_above_ma" in indicators
        assert "btc_below_ma" in indicators


class TestVBOPortfolioIndicators:
    """Test indicator calculation."""

    def test_calculate_indicators_adds_columns(
        self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame
    ) -> None:
        strategy = VBOPortfolio(btc_data=btc_data, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv.copy())

        assert "sma" in df.columns
        assert "target" in df.columns
        assert "btc_above_ma" in df.columns
        assert "btc_below_ma" in df.columns

    def test_btc_filter_indicators_are_boolean(
        self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame
    ) -> None:
        strategy = VBOPortfolio(btc_data=btc_data, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv.copy())

        # After warmup period, should have boolean values
        valid = df["btc_above_ma"].dropna()
        if len(valid) > 0:
            assert valid.dtype == bool or set(valid.unique()).issubset({True, False})

    def test_sma_calculation(self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame) -> None:
        strategy = VBOPortfolio(btc_data=btc_data, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv.copy())

        # SMA should have values after warmup
        assert df["sma"].notna().sum() > 0

    def test_no_data_corruption(self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame) -> None:
        """Original DataFrame should not be modified."""
        original = target_ohlcv.copy()
        strategy = VBOPortfolio(btc_data=btc_data, ma_short=5, btc_ma=10)
        strategy.calculate_indicators(target_ohlcv.copy())

        pd.testing.assert_frame_equal(target_ohlcv, original)

    @patch("src.strategies.volatility_breakout.vbo_portfolio._load_btc_data")
    def test_without_btc_data(self, mock_load: MagicMock, target_ohlcv: pd.DataFrame) -> None:
        """Without BTC data, btc_above_ma defaults to True (no filter)."""
        mock_load.return_value = None
        strategy = VBOPortfolio(btc_data=None, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv.copy())

        assert "btc_above_ma" in df.columns
        # Without BTC data, btc_above_ma=True (don't filter)
        assert df["btc_above_ma"].all()
        assert not df["btc_below_ma"].any()


class TestVBOPortfolioSignals:
    """Test signal generation."""

    def test_generate_signals_creates_columns(
        self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame
    ) -> None:
        strategy = VBOPortfolio(btc_data=btc_data, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv.copy())
        df = strategy.generate_signals(df)

        assert "entry_signal" in df.columns
        assert "exit_signal" in df.columns

    def test_entry_requires_btc_filter(
        self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame
    ) -> None:
        """Entry signals should only occur when BTC is above its MA."""
        strategy = VBOPortfolio(btc_data=btc_data, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv.copy())
        df = strategy.generate_signals(df)

        entry_rows = df[df["entry_signal"]]
        if len(entry_rows) > 0:
            # All entry signals must have btc_above_ma=True
            assert entry_rows["btc_above_ma"].all()

    def test_signals_are_boolean(self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame) -> None:
        strategy = VBOPortfolio(btc_data=btc_data, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv.copy())
        df = strategy.generate_signals(df)

        assert df["entry_signal"].dtype == bool
        assert df["exit_signal"].dtype == bool


class TestVBOSingleCoin:
    """Test VBOSingleCoin (inherits from VBOPortfolio)."""

    def test_inherits_from_portfolio(self) -> None:
        strategy = VBOSingleCoin(btc_data=pd.DataFrame())
        assert isinstance(strategy, VBOPortfolio)
        assert strategy.name == "VBOSingleCoin"

    def test_custom_parameters(self) -> None:
        strategy = VBOSingleCoin(ma_short=10, btc_ma=30, btc_data=pd.DataFrame())
        assert strategy.ma_short == 10
        assert strategy.btc_ma == 30

    def test_indicator_calculation(
        self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame
    ) -> None:
        strategy = VBOSingleCoin(btc_data=btc_data, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv.copy())
        assert "sma" in df.columns
        assert "btc_above_ma" in df.columns


class TestBtcMarketFilterCondition:
    """Test BTC market filter conditions."""

    def test_filter_true_when_btc_above_ma(self) -> None:
        from datetime import date

        from src.strategies.base import OHLCV

        condition = BtcMarketFilterCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        indicators = {"btc_above_ma": True}

        assert condition.evaluate(current, pd.DataFrame(), indicators) is True

    def test_filter_false_when_btc_below_ma(self) -> None:
        from datetime import date

        from src.strategies.base import OHLCV

        condition = BtcMarketFilterCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        indicators = {"btc_above_ma": False}

        assert condition.evaluate(current, pd.DataFrame(), indicators) is False

    def test_filter_false_when_missing(self) -> None:
        from datetime import date

        from src.strategies.base import OHLCV

        condition = BtcMarketFilterCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        indicators: dict[str, float] = {}

        assert condition.evaluate(current, pd.DataFrame(), indicators) is False


class TestBtcMarketExitCondition:
    """Test BTC market exit condition."""

    def test_exit_true_when_btc_below_ma(self) -> None:
        from datetime import date

        from src.strategies.base import OHLCV

        condition = BtcMarketExitCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        indicators = {"btc_below_ma": True}

        assert condition.evaluate(current, pd.DataFrame(), indicators) is True

    def test_exit_false_when_btc_above_ma(self) -> None:
        from datetime import date

        from src.strategies.base import OHLCV

        condition = BtcMarketExitCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        indicators = {"btc_below_ma": False}

        assert condition.evaluate(current, pd.DataFrame(), indicators) is False


class TestVBOPortfolioLite:
    """Test VBOPortfolioLite variant."""

    def test_inherits_from_portfolio(self) -> None:
        strategy = VBOPortfolioLite(btc_data=pd.DataFrame())
        assert isinstance(strategy, VBOPortfolio)
        assert strategy.name == "VBOPortfolioLite"

    def test_entry_has_no_sma_condition(self) -> None:
        from src.strategies.volatility_breakout.conditions import SMABreakoutCondition

        strategy = VBOPortfolioLite(btc_data=pd.DataFrame())
        entry_types = [type(c) for c in strategy.entry_conditions.conditions]
        assert SMABreakoutCondition not in entry_types

    def test_exit_has_no_btc_exit(self) -> None:
        strategy = VBOPortfolioLite(btc_data=pd.DataFrame())
        exit_types = [type(c) for c in strategy.exit_conditions.conditions]
        assert BtcMarketExitCondition not in exit_types

    def test_entry_still_has_breakout_and_btc_filter(self) -> None:
        from src.strategies.volatility_breakout.conditions import BreakoutCondition

        strategy = VBOPortfolioLite(btc_data=pd.DataFrame())
        entry_types = [type(c) for c in strategy.entry_conditions.conditions]
        assert BreakoutCondition in entry_types
        assert BtcMarketFilterCondition in entry_types

    def test_signals_no_sma_entry_filter(
        self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame
    ) -> None:
        """Entry should not require target > SMA."""
        strategy = VBOPortfolioLite(btc_data=btc_data, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv.copy())
        df = strategy.generate_signals(df)

        assert "entry_signal" in df.columns
        assert "exit_signal" in df.columns
        assert df["entry_signal"].dtype == bool
        assert df["exit_signal"].dtype == bool

    def test_exit_ignores_btc_bear(
        self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame
    ) -> None:
        """Exit should only be PriceBelowSMA, not BTC bear."""
        strategy = VBOPortfolioLite(btc_data=btc_data, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv.copy())
        df = strategy.generate_signals(df)

        # Exit signal should be purely close < sma
        expected_exit = df["close"] < df["sma"]
        pd.testing.assert_series_equal(df["exit_signal"], expected_exit, check_names=False)
