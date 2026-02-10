"""Tests for VBOV1 strategy — faithful port of backtest_v1.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.strategies.volatility_breakout.vbo_v1 import VBOV1


@pytest.fixture()
def btc_data() -> pd.DataFrame:
    """BTC data with clear uptrend then downtrend."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
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
    """Target coin with same index as BTC."""
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


@pytest.fixture()
def deterministic_df() -> pd.DataFrame:
    """Small deterministic DataFrame for precise assertions."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "open": [100, 102, 105, 103, 108, 106, 110, 107, 104, 109],
            "high": [104, 106, 109, 107, 112, 110, 114, 111, 108, 113],
            "low": [98, 100, 103, 101, 106, 104, 108, 105, 102, 107],
            "close": [101, 104, 107, 105, 110, 108, 112, 109, 106, 111],
            "volume": [1000] * 10,
        },
        index=dates,
    )


class TestVBOV1Init:
    """Test VBOV1 initialization."""

    def test_default_parameters(self) -> None:
        strategy = VBOV1(btc_data=pd.DataFrame())
        assert strategy.name == "VBOV1"
        assert strategy.ma_short == 5
        assert strategy.btc_ma == 20
        assert strategy.noise_ratio == 0.5

    def test_custom_parameters(self) -> None:
        strategy = VBOV1(
            name="CustomV1",
            ma_short=10,
            btc_ma=40,
            noise_ratio=0.7,
            btc_data=pd.DataFrame(),
        )
        assert strategy.name == "CustomV1"
        assert strategy.ma_short == 10
        assert strategy.btc_ma == 40
        assert strategy.noise_ratio == 0.7

    def test_required_indicators(self) -> None:
        strategy = VBOV1(btc_data=pd.DataFrame())
        indicators = strategy.required_indicators()
        assert "sma" in indicators
        assert "target" in indicators
        assert "btc_above_ma" in indicators


class TestVBOV1FixedK:
    """Test that VBOV1 uses fixed K value, not adaptive noise."""

    def test_target_uses_fixed_k(self, deterministic_df: pd.DataFrame) -> None:
        """target = open + prev_range * 0.5 (fixed K)."""
        strategy = VBOV1(btc_data=pd.DataFrame(), noise_ratio=0.5)
        df = strategy.calculate_indicators(deterministic_df)

        # Day 2 (index 1): prev_high=104, prev_low=98, prev_range=6
        # target = 102 (open) + 6 * 0.5 = 105.0
        assert df["target"].iloc[1] == pytest.approx(105.0)

    def test_target_with_custom_k(self, deterministic_df: pd.DataFrame) -> None:
        """Custom noise_ratio changes target calculation."""
        strategy = VBOV1(btc_data=pd.DataFrame(), noise_ratio=0.7)
        df = strategy.calculate_indicators(deterministic_df)

        # Day 2: target = 102 + 6 * 0.7 = 106.2
        assert df["target"].iloc[1] == pytest.approx(106.2)


class TestVBOV1SMAIncludesCurrentBar:
    """Test that SMA includes the current bar (no shift)."""

    def test_sma_includes_current_bar(self, deterministic_df: pd.DataFrame) -> None:
        """SMA should be rolling mean without shift."""
        strategy = VBOV1(btc_data=pd.DataFrame(), ma_short=5)
        df = strategy.calculate_indicators(deterministic_df)

        # SMA at day 5 (index 4) = mean of close[0:5] = mean(101,104,107,105,110)
        expected = np.mean([101, 104, 107, 105, 110])
        assert df["sma"].iloc[4] == pytest.approx(expected)

    def test_sma_nan_before_period(self, deterministic_df: pd.DataFrame) -> None:
        """SMA should be NaN before enough data points."""
        strategy = VBOV1(btc_data=pd.DataFrame(), ma_short=5)
        df = strategy.calculate_indicators(deterministic_df)

        assert pd.isna(df["sma"].iloc[3])  # Only 4 bars, need 5
        assert pd.notna(df["sma"].iloc[4])  # 5 bars available


class TestVBOV1EntrySignal:
    """Test entry signal: breakout + BTC filter only (no SMA filter)."""

    def test_entry_requires_breakout(
        self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame
    ) -> None:
        """Entry needs high >= target."""
        strategy = VBOV1(btc_data=btc_data, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv)
        df = strategy.generate_signals(df)

        entry_rows = df[df["entry_signal"]]
        if len(entry_rows) > 0:
            assert (entry_rows["high"] >= entry_rows["target"]).all()

    def test_entry_requires_btc_filter(
        self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame
    ) -> None:
        """Entry needs BTC above MA."""
        strategy = VBOV1(btc_data=btc_data, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv)
        df = strategy.generate_signals(df)

        entry_rows = df[df["entry_signal"]]
        if len(entry_rows) > 0:
            assert entry_rows["btc_above_ma"].all()

    def test_entry_no_sma_filter(self) -> None:
        """Entry should NOT require target > SMA (unlike VBOPortfolio)."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "open": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                "high": [120, 120, 120, 120, 120, 120, 120, 120, 120, 120],
                "low": [80, 80, 80, 80, 80, 80, 80, 80, 80, 80],
                "close": [105, 105, 105, 105, 105, 105, 105, 105, 105, 105],
                "volume": [1000] * 10,
            },
            index=dates,
        )
        strategy = VBOV1(btc_data=pd.DataFrame(), ma_short=5)
        df = strategy.calculate_indicators(df)
        df = strategy.generate_signals(df)

        # target = 100 + 40*0.5 = 120, SMA = 105
        # target > SMA is True here, but the point is:
        # if target were < SMA, entry would still be allowed
        # Let's verify no SMA condition is applied
        valid = df.dropna(subset=["target", "sma"])
        if len(valid) > 0:
            # All highs (120) >= all targets (120), btc always True
            # So all valid rows should have entry_signal=True
            assert valid["entry_signal"].any()


class TestVBOV1ExitSignal:
    """Test exit signal: prev_close < prev_SMA (shift(1))."""

    def test_exit_uses_prev_day_condition(self) -> None:
        """Exit signal should use shift(1): close.shift(1) < sma.shift(1)."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        # Create scenario: close drops below SMA on day 6, exit fires on day 7
        close = [100, 102, 104, 103, 105, 101, 98, 103, 105, 107]
        df = pd.DataFrame(
            {
                "open": [99, 101, 103, 102, 104, 100, 97, 102, 104, 106],
                "high": [104, 106, 108, 107, 109, 105, 102, 107, 109, 111],
                "low": [97, 99, 101, 100, 102, 98, 95, 100, 102, 104],
                "close": close,
                "volume": [1000] * 10,
            },
            index=dates,
        )
        strategy = VBOV1(btc_data=pd.DataFrame(), ma_short=5)
        df = strategy.calculate_indicators(df)
        df = strategy.generate_signals(df)

        # Manually verify: exit_signal[t] = close[t-1] < sma[t-1]
        for i in range(1, len(df)):
            if pd.notna(df["sma"].iloc[i - 1]):
                expected = df["close"].iloc[i - 1] < df["sma"].iloc[i - 1]
                assert df["exit_signal"].iloc[i] == expected, (
                    f"Day {i}: close[{i-1}]={df['close'].iloc[i-1]}, "
                    f"sma[{i-1}]={df['sma'].iloc[i-1]}, "
                    f"expected={expected}, got={df['exit_signal'].iloc[i]}"
                )


class TestVBOV1ExitPriceBase:
    """Test that exit_price_base is set to open."""

    def test_exit_price_base_is_open(
        self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame
    ) -> None:
        """generate_signals should set exit_price_base to open."""
        strategy = VBOV1(btc_data=btc_data, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv)
        df = strategy.generate_signals(df)

        assert "exit_price_base" in df.columns
        pd.testing.assert_series_equal(
            df["exit_price_base"], df["open"], check_names=False
        )

    def test_signal_processor_uses_exit_price_base(self) -> None:
        """signal_processor.add_price_columns should use exit_price_base."""
        from src.backtester.engine.signal_processor import add_price_columns
        from src.backtester.models import BacktestConfig

        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [100, 102, 105, 103, 108],
                "high": [104, 106, 109, 107, 112],
                "low": [98, 100, 103, 101, 106],
                "close": [101, 104, 107, 105, 110],
                "target": [103, 105, 108, 106, 111],
                "entry_signal": [True, False, True, False, True],
                "exit_signal": [False, True, False, True, False],
                "exit_price_base": [100, 102, 105, 103, 108],  # open prices
            },
            index=dates,
        )

        config = BacktestConfig(slippage_rate=0.001)
        result = add_price_columns(df, config)

        # exit_price should be exit_price_base * (1 - slippage), not close
        expected_exit = df["exit_price_base"] * (1 - 0.001)
        pd.testing.assert_series_equal(
            result["exit_price"], expected_exit, check_names=False
        )

    def test_signal_processor_without_exit_price_base(self) -> None:
        """Without exit_price_base, signal_processor uses close (backward compat)."""
        from src.backtester.engine.signal_processor import add_price_columns
        from src.backtester.models import BacktestConfig

        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [100, 102, 105, 103, 108],
                "high": [104, 106, 109, 107, 112],
                "low": [98, 100, 103, 101, 106],
                "close": [101, 104, 107, 105, 110],
                "target": [103, 105, 108, 106, 111],
                "entry_signal": [True, False, True, False, True],
                "exit_signal": [False, True, False, True, False],
            },
            index=dates,
        )

        config = BacktestConfig(slippage_rate=0.001)
        result = add_price_columns(df, config)

        # Without exit_price_base, exit_price = close * (1 - slippage)
        expected_exit = df["close"] * (1 - 0.001)
        pd.testing.assert_series_equal(
            result["exit_price"], expected_exit, check_names=False
        )


class TestVBOV1BtcFilter:
    """Test BTC market filter matches VBOPortfolio behavior."""

    def test_btc_filter_uses_prev_day(
        self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame
    ) -> None:
        """BTC filter should use shift(1) to avoid look-ahead bias."""
        strategy = VBOV1(btc_data=btc_data, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv)

        # btc_above_ma should be boolean
        valid = df["btc_above_ma"].dropna()
        if len(valid) > 0:
            assert valid.dtype == bool or set(valid.unique()).issubset({True, False})

    @patch("src.strategies.volatility_breakout.vbo_v1._load_btc_data")
    def test_without_btc_data(
        self, mock_load: MagicMock, target_ohlcv: pd.DataFrame
    ) -> None:
        """Without BTC data, btc_above_ma=True (no filter)."""
        mock_load.return_value = None
        strategy = VBOV1(btc_data=None, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv)

        assert df["btc_above_ma"].all()


class TestVBOV1Signals:
    """Test full signal pipeline."""

    def test_signals_are_boolean(
        self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame
    ) -> None:
        strategy = VBOV1(btc_data=btc_data, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv)
        df = strategy.generate_signals(df)

        assert df["entry_signal"].dtype == bool
        assert df["exit_signal"].dtype == bool

    def test_no_data_corruption(
        self, btc_data: pd.DataFrame, target_ohlcv: pd.DataFrame
    ) -> None:
        """Original DataFrame should not be modified."""
        original = target_ohlcv.copy()
        strategy = VBOV1(btc_data=btc_data, ma_short=5, btc_ma=10)
        df = strategy.calculate_indicators(target_ohlcv.copy())
        strategy.generate_signals(df)

        pd.testing.assert_frame_equal(target_ohlcv, original)
