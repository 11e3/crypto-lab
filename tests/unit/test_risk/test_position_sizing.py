"""
Unit tests for position sizing module.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk.position_sizing import (
    _equal_sizing,
    _fixed_risk_sizing,
    _volatility_based_sizing,
    calculate_multi_asset_position_sizes,
    calculate_position_size,
)


@pytest.fixture
def sample_historical_data() -> pd.DataFrame:
    """Generate sample historical data for testing volatility."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    data = pd.DataFrame(
        {
            "close": np.linspace(100, 110, 100) + np.random.randn(100),
            "open": np.linspace(99, 109, 100) + np.random.randn(100),
            "high": np.linspace(101, 111, 100) + np.random.randn(100),
            "low": np.linspace(98, 108, 100) + np.random.randn(100),
        },
        index=dates,
    )
    return data


def test_equal_sizing() -> None:
    """Test _equal_sizing function."""
    assert _equal_sizing(1000, 2) == 500
    assert _equal_sizing(0, 1) == 0
    assert _equal_sizing(1000, 1) == 1000
    assert _equal_sizing(1000.0, 0) == 0.0  # No slots available = no investment


def test_volatility_based_sizing(sample_historical_data: pd.DataFrame) -> None:
    """Test _volatility_based_sizing function."""
    available_cash = 10000
    available_slots = 2
    lookback_period = 20

    # Sufficient data
    size = _volatility_based_sizing(
        available_cash, available_slots, sample_historical_data, lookback_period
    )
    assert isinstance(size, float)
    assert size > 0

    # Insufficient data
    insufficient_data = sample_historical_data.head(5)
    size_insufficient = _volatility_based_sizing(
        available_cash, available_slots, insufficient_data, lookback_period
    )
    assert size_insufficient == _equal_sizing(available_cash, available_slots)

    # Zero volatility
    zero_vol_data = pd.DataFrame({"close": [100] * 100}, index=sample_historical_data.index)
    size_zero_vol = _volatility_based_sizing(
        available_cash, available_slots, zero_vol_data, lookback_period
    )
    assert size_zero_vol == _equal_sizing(available_cash, available_slots)

    # NaN volatility (e.g., from constant or single value returns)
    nan_vol_data = pd.DataFrame(
        {"close": [100, 100, 100]}, index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    )
    size_nan_vol = _volatility_based_sizing(
        available_cash, available_slots, nan_vol_data, lookback_period=2
    )
    assert size_nan_vol == _equal_sizing(available_cash, available_slots)


def test_fixed_risk_sizing(sample_historical_data: pd.DataFrame) -> None:
    """Test _fixed_risk_sizing function."""
    available_cash = 10000
    available_slots = 2
    current_price = 105
    target_risk_pct = 0.01  # 1%
    lookback_period = 20

    # Sufficient data
    size = _fixed_risk_sizing(
        available_cash,
        available_slots,
        current_price,
        sample_historical_data,
        target_risk_pct,
        lookback_period,
    )
    assert isinstance(size, float)
    assert size > 0
    assert size <= available_cash / available_slots  # Limited by max_per_slot

    # Insufficient data
    insufficient_data = sample_historical_data.head(5)
    size_insufficient = _fixed_risk_sizing(
        available_cash,
        available_slots,
        current_price,
        insufficient_data,
        target_risk_pct,
        lookback_period,
    )
    assert size_insufficient == _equal_sizing(available_cash, available_slots)

    # Zero volatility
    zero_vol_data = pd.DataFrame({"close": [100] * 100}, index=sample_historical_data.index)
    size_zero_vol = _fixed_risk_sizing(
        available_cash,
        available_slots,
        current_price,
        zero_vol_data,
        target_risk_pct,
        lookback_period,
    )
    assert size_zero_vol == _equal_sizing(available_cash, available_slots)

    # Zero current price
    size_zero_price = _fixed_risk_sizing(
        available_cash,
        available_slots,
        0,
        sample_historical_data,
        target_risk_pct,
        lookback_period,
    )
    assert size_zero_price == _equal_sizing(available_cash, available_slots)


def test_inverse_volatility_uses_volatility_based(sample_historical_data: pd.DataFrame) -> None:
    """Test that inverse-volatility method delegates to volatility-based sizing."""
    available_cash = 10000.0
    available_slots = 2
    lookback_period = 20

    # inverse-volatility should produce same result as volatility
    vol_size = calculate_position_size(
        method="volatility",
        available_cash=available_cash,
        available_slots=available_slots,
        ticker="TEST",
        current_price=100.0,
        historical_data=sample_historical_data,
        lookback_period=lookback_period,
    )
    inv_vol_size = calculate_position_size(
        method="inverse-volatility",
        available_cash=available_cash,
        available_slots=available_slots,
        ticker="TEST",
        current_price=100.0,
        historical_data=sample_historical_data,
        lookback_period=lookback_period,
    )
    assert vol_size == inv_vol_size


class TestCalculatePositionSize:
    """Test cases for calculate_position_size function."""

    def test_calculate_position_size_equal(self, sample_historical_data: pd.DataFrame) -> None:
        """Test equal position sizing."""
        size = calculate_position_size(
            method="equal",
            available_cash=1000,
            available_slots=2,
            ticker="TEST",
            current_price=100,
            historical_data=sample_historical_data,
        )
        assert size == 500

    def test_calculate_position_size_volatility(self, sample_historical_data: pd.DataFrame) -> None:
        """Test volatility-based position sizing."""
        size = calculate_position_size(
            method="volatility",
            available_cash=1000,
            available_slots=2,
            ticker="TEST",
            current_price=100,
            historical_data=sample_historical_data,
            lookback_period=20,
        )
        assert isinstance(size, float)
        assert size > 0

    def test_calculate_position_size_fixed_risk(self, sample_historical_data: pd.DataFrame) -> None:
        """Test fixed-risk position sizing."""
        size = calculate_position_size(
            method="fixed-risk",
            available_cash=1000,
            available_slots=2,
            ticker="TEST",
            current_price=100,
            historical_data=sample_historical_data,
            target_risk_pct=0.01,
            lookback_period=20,
        )
        assert isinstance(size, float)
        assert size > 0

    def test_calculate_position_size_inverse_volatility(
        self, sample_historical_data: pd.DataFrame
    ) -> None:
        """Test inverse volatility position sizing."""
        size = calculate_position_size(
            method="inverse-volatility",
            available_cash=1000.0,
            available_slots=2,
            ticker="TEST",
            current_price=100.0,
            historical_data=sample_historical_data,
            lookback_period=20,
        )
        assert isinstance(size, float)
        assert size > 0

    def test_calculate_position_size_unavailable_slots(
        self, sample_historical_data: pd.DataFrame
    ) -> None:
        """Test with no available slots."""
        size = calculate_position_size(
            method="equal",
            available_cash=1000,
            available_slots=0,
            ticker="TEST",
            current_price=100,
            historical_data=sample_historical_data,
        )
        assert size == 0

    def test_calculate_position_size_insufficient_data_fallback(self) -> None:
        """Test fallback to equal sizing when historical data is insufficient."""
        # Create a DataFrame with less than lookback_period data
        insufficient_data = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        )
        size = calculate_position_size(
            method="volatility",  # Requires historical data
            available_cash=1000,
            available_slots=2,
            ticker="TEST",
            current_price=100,
            historical_data=insufficient_data,
            lookback_period=20,
        )
        assert size == 500  # Should fall back to equal sizing

    def test_calculate_position_size_unknown_method(
        self, sample_historical_data: pd.DataFrame
    ) -> None:
        """Test unknown position sizing method fallback."""
        size = calculate_position_size(
            method="unknown_method",
            available_cash=1000,
            available_slots=2,
            ticker="TEST",
            current_price=100,
            historical_data=sample_historical_data,
        )
        assert size == 500  # Should fall back to equal sizing


class TestCalculateMultiAssetPositionSizes:
    """Test cases for calculate_multi_asset_position_sizes function."""

    @pytest.fixture
    def multi_asset_historical_data(
        self, sample_historical_data: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Generate sample historical data for multiple assets."""
        data_btc = sample_historical_data.copy()
        data_btc["close"] = data_btc["close"] * 10  # Different scale
        data_eth = sample_historical_data.copy()
        data_eth["close"] = data_eth["close"] * 5  # Different scale
        return {"BTC": data_btc, "ETH": data_eth}

    def test_multi_asset_equal_sizing(
        self, multi_asset_historical_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test equal sizing for multiple assets."""
        available_cash = 10000
        tickers = ["BTC", "ETH"]
        current_prices = {"BTC": 1000, "ETH": 500}
        sizes = calculate_multi_asset_position_sizes(
            method="equal",
            available_cash=available_cash,
            tickers=tickers,
            current_prices=current_prices,
            historical_data=multi_asset_historical_data,
        )
        assert sizes == {"BTC": 5000, "ETH": 5000}

    def test_multi_asset_volatility_sizing(
        self, multi_asset_historical_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test volatility-based sizing for multiple assets."""
        available_cash = 10000
        tickers = ["BTC", "ETH"]
        current_prices = {"BTC": 1000, "ETH": 500}
        sizes = calculate_multi_asset_position_sizes(
            method="volatility",
            available_cash=available_cash,
            tickers=tickers,
            current_prices=current_prices,
            historical_data=multi_asset_historical_data,
            lookback_period=20,
        )
        assert isinstance(sizes, dict)
        assert "BTC" in sizes and "ETH" in sizes
        assert sizes["BTC"] > 0 and sizes["ETH"] > 0
        assert sum(sizes.values()) == pytest.approx(available_cash)

    def test_multi_asset_fixed_risk_sizing(
        self, multi_asset_historical_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test fixed-risk sizing for multiple assets."""
        available_cash = 10000
        tickers = ["BTC", "ETH"]
        current_prices = {"BTC": 1000, "ETH": 500}
        target_risk_pct = 0.01
        sizes = calculate_multi_asset_position_sizes(
            method="fixed-risk",
            available_cash=available_cash,
            tickers=tickers,
            current_prices=current_prices,
            historical_data=multi_asset_historical_data,
            target_risk_pct=target_risk_pct,
            lookback_period=20,
        )
        assert isinstance(sizes, dict)
        assert "BTC" in sizes and "ETH" in sizes
        assert sizes["BTC"] > 0 and sizes["ETH"] > 0
        assert (
            sum(sizes.values()) <= available_cash + 0.01
        )  # Can be slightly less due to normalization

    def test_multi_asset_inverse_volatility_sizing(
        self, multi_asset_historical_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test inverse volatility sizing for multiple assets."""
        available_cash = 10000
        tickers = ["BTC", "ETH"]
        current_prices = {"BTC": 1000, "ETH": 500}
        sizes = calculate_multi_asset_position_sizes(
            method="inverse-volatility",
            available_cash=available_cash,
            tickers=tickers,
            current_prices=current_prices,
            historical_data=multi_asset_historical_data,
            lookback_period=20,
        )
        assert isinstance(sizes, dict)
        assert "BTC" in sizes and "ETH" in sizes
        assert sizes["BTC"] > 0 and sizes["ETH"] > 0
        assert sum(sizes.values()) == pytest.approx(available_cash)

    def test_multi_asset_insufficient_data_fallback(self) -> None:
        """Test fallback to equal sizing for multi-asset when historical data is insufficient."""
        available_cash = 10000
        tickers = ["BTC", "ETH"]
        current_prices = {"BTC": 1000, "ETH": 500}
        # Insufficient data for both
        insufficient_data = {
            "BTC": pd.DataFrame(
                {"close": [100, 101, 102]},
                index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            ),
            "ETH": pd.DataFrame(
                {"close": [50, 51, 52]},
                index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            ),
        }
        sizes = calculate_multi_asset_position_sizes(
            method="volatility",
            available_cash=available_cash,
            tickers=tickers,
            current_prices=current_prices,
            historical_data=insufficient_data,
            lookback_period=20,
        )
        assert sizes == {"BTC": 5000, "ETH": 5000}  # Fallback to equal

    def test_multi_asset_mixed_data_fixed_risk(
        self, multi_asset_historical_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test fixed-risk with mixed sufficient/insufficient data and zero price."""
        available_cash = 10000
        tickers = ["BTC", "ETH", "XRP"]
        current_prices = {"BTC": 1000, "ETH": 500, "XRP": 0}  # XRP has zero price

        # ETH has insufficient data
        mixed_historical_data = {
            "BTC": multi_asset_historical_data["BTC"],
            "ETH": pd.DataFrame(
                {"close": [50, 51, 52]},
                index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            ),
            "XRP": multi_asset_historical_data["ETH"],  # Use ETH data for XRP for vol calculation
        }

        sizes = calculate_multi_asset_position_sizes(
            method="fixed-risk",
            available_cash=available_cash,
            tickers=tickers,
            current_prices=current_prices,
            historical_data=mixed_historical_data,
            target_risk_pct=0.01,
            lookback_period=20,
        )
        assert isinstance(sizes, dict)
        assert sizes["XRP"] == 0.0  # Zero price means zero position
        # BTC and ETH should get some non-zero allocation, ETH will fallback to equal
        assert sizes["BTC"] > 0
        assert sizes["ETH"] > 0
        assert sum(sizes.values()) <= available_cash + 0.01  # Due to normalization and XRP being 0

    def test_multi_asset_zero_volatility_fixed_risk(
        self, multi_asset_historical_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test fixed-risk sizing with zero volatility for one asset."""
        available_cash = 10000
        tickers = ["BTC", "ETH"]
        current_prices = {"BTC": 1000, "ETH": 500}

        zero_vol_data = pd.DataFrame(
            {"close": [100] * 100}, index=multi_asset_historical_data["BTC"].index
        )

        mixed_historical_data = {
            "BTC": multi_asset_historical_data["BTC"],
            "ETH": zero_vol_data,
        }

        sizes = calculate_multi_asset_position_sizes(
            method="fixed-risk",
            available_cash=available_cash,
            tickers=tickers,
            current_prices=current_prices,
            historical_data=mixed_historical_data,
            target_risk_pct=0.01,
            lookback_period=20,
        )
        assert isinstance(sizes, dict)
        assert sizes["ETH"] < 5000.0  # Should be scaled down from 5000.0 due to normalization
        assert sizes["BTC"] > 0
        assert sum(sizes.values()) == pytest.approx(available_cash)
