"""
Unit tests for trade_simulator module.
"""

import numpy as np
import pytest

from src.backtester.engine.trade_simulator import (
    _check_exit_conditions_intraday,
    _get_current_price,
    calculate_daily_equity,
    track_asset_returns,
)
from src.backtester.engine.trade_simulator_state import initialize_simulation_state
from src.backtester.models import BacktestConfig

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def backtest_config() -> BacktestConfig:
    """Create a sample BacktestConfig."""
    config = BacktestConfig()
    config.initial_capital = 100000.0
    config.fee_rate = 0.001
    config.stop_loss_pct = 0.05
    config.take_profit_pct = 0.10
    return config


@pytest.fixture
def sample_state(backtest_config: BacktestConfig):
    """Create sample simulation state."""
    tickers = ["KRW-BTC", "KRW-ETH"]
    return initialize_simulation_state(
        initial_capital=backtest_config.initial_capital,
        n_tickers=len(tickers),
        n_dates=3,
        tickers=tickers,
    )


@pytest.fixture
def sample_closes() -> np.ndarray:
    """Create sample close prices."""
    return np.array(
        [
            [50000.0, 51000.0, 52000.0],  # KRW-BTC
            [2000.0, 2050.0, 2100.0],  # KRW-ETH
        ]
    )


@pytest.fixture
def sample_valid_data() -> np.ndarray:
    """Create sample valid data mask."""
    return np.array([True, True])


# -------------------------------------------------------------------------
# _check_exit_conditions_intraday Tests
# -------------------------------------------------------------------------


class TestCheckExitConditionsIntraday:
    """Test _check_exit_conditions_intraday function.

    The function uses intraday high/low for accurate stop-loss and take-profit:
    - Stop-loss: triggers if LOW price <= stop price
    - Take-profit: triggers if HIGH price >= target price
    """

    @pytest.mark.parametrize(
        "entry,high,low,stop_loss_pct,take_profit_pct,expected_exit,expected_sl,expected_tp",
        [
            # Normal conditions - no trigger
            (100.0, 102.0, 98.0, 0.05, 0.10, False, False, False),
            # Stop loss hit (low touches stop price)
            (100.0, 102.0, 94.9, 0.05, 0.10, True, True, False),  # low=94.9 < stop=95
            (100.0, 102.0, 94.0, 0.05, 0.10, True, True, False),  # low=94 < stop=95
            # Take profit hit (high exceeds target)
            (100.0, 110.1, 98.0, 0.05, 0.10, True, False, True),  # high=110.1 > target=110
            (100.0, 115.0, 98.0, 0.05, 0.10, True, False, True),  # high=115 > target=110
            # No stop loss configured
            (100.0, 102.0, 94.0, None, 0.10, False, False, False),
            # No take profit configured
            (100.0, 115.0, 98.0, 0.05, None, False, False, False),
            # Edge case: slightly below stop loss (triggers)
            (100.0, 102.0, 94.99, 0.05, 0.10, True, True, False),
            # Edge case: slightly above take profit (triggers)
            (100.0, 110.01, 98.0, 0.05, 0.10, True, False, True),
        ],
    )
    def test_exit_conditions_intraday(
        self,
        entry: float,
        high: float,
        low: float,
        stop_loss_pct: float | None,
        take_profit_pct: float | None,
        expected_exit: bool,
        expected_sl: bool,
        expected_tp: bool,
    ) -> None:
        """Test exit condition detection using intraday high/low prices."""
        config = BacktestConfig()
        config.stop_loss_pct = stop_loss_pct
        config.take_profit_pct = take_profit_pct

        should_exit, is_sl, is_tp = _check_exit_conditions_intraday(
            entry, high, low, config
        )

        assert should_exit == expected_exit
        assert is_sl == expected_sl
        assert is_tp == expected_tp

    def test_stop_loss_priority_over_take_profit(self) -> None:
        """Test that stop loss takes priority when both could trigger."""
        config = BacktestConfig()
        config.stop_loss_pct = 0.05
        config.take_profit_pct = 0.10

        # Entry at 100, stop at 95, target at 110
        # If low=94 (stop hit) and high=111 (target hit), stop loss wins
        # This is because stop loss is checked first in the function
        entry, high, low = 100.0, 111.0, 94.0
        should_exit, is_sl, is_tp = _check_exit_conditions_intraday(
            entry, high, low, config
        )

        assert should_exit is True
        assert is_sl is True
        assert is_tp is False


# -------------------------------------------------------------------------
# _get_current_price Tests
# -------------------------------------------------------------------------


class TestGetCurrentPrice:
    """Test _get_current_price function."""

    @pytest.mark.parametrize(
        "d_idx,price_valid,expected_source",
        [
            (0, True, "current"),  # Use current price
            (1, True, "current"),  # Use current price
            (0, False, "entry"),  # Use entry price when no current
            (1, False, "previous"),  # Use previous price when available
        ],
    )
    def test_get_current_price_variants(
        self,
        d_idx: int,
        price_valid: bool,
        expected_source: str,
        sample_state,
        sample_closes: np.ndarray,
    ) -> None:
        """Test price retrieval with different data availability."""
        sample_state.position_amounts[0] = 1.0
        sample_state.position_entry_prices[0] = 50000.0
        valid_data = np.array([price_valid, True])

        # Modify prices based on validity
        if not price_valid:
            sample_closes[0, d_idx] = np.nan

        price = _get_current_price(sample_state, 0, d_idx, sample_closes, valid_data)

        if expected_source == "current":
            assert price == pytest.approx(sample_closes[0, d_idx])
        elif expected_source == "previous" and d_idx > 0:
            assert price == pytest.approx(sample_closes[0, d_idx - 1])
        elif expected_source == "entry":
            assert price == pytest.approx(50000.0)

    def test_get_current_price_invalid_then_valid(
        self,
        sample_state,
        sample_closes: np.ndarray,
    ) -> None:
        """Test fallback to previous price when current is NaN."""
        sample_state.position_amounts[0] = 1.0
        sample_state.position_entry_prices[0] = 50000.0
        valid_data = np.array([False, True])
        sample_closes[0, 1] = np.nan  # Current price invalid

        # At d_idx=1 with previous price available
        price = _get_current_price(sample_state, 0, 1, sample_closes, valid_data)

        assert price == pytest.approx(50000.0)  # Entry price fallback


# -------------------------------------------------------------------------
# calculate_daily_equity Tests
# -------------------------------------------------------------------------


class TestCalculateDailyEquity:
    """Test calculate_daily_equity function."""

    @pytest.mark.parametrize(
        "btc_position,eth_position,d_idx,expected_positions_value",
        [
            (1.0, 5.0, 0, 60000.0),  # 1*50k + 5*2k
            (0.5, 10.0, 0, 45000.0),  # 0.5*50k + 10*2k
            (0.0, 0.0, 0, 0.0),  # No positions
            (1.0, 0.0, 1, 51000.0),  # Only BTC on day 1
        ],
    )
    def test_calculate_equity_variants(
        self,
        btc_position: float,
        eth_position: float,
        d_idx: int,
        expected_positions_value: float,
        sample_state,
        sample_closes: np.ndarray,
        sample_valid_data: np.ndarray,
        backtest_config: BacktestConfig,
    ) -> None:
        """Test daily equity calculation with different positions."""
        sample_state.position_amounts[0] = btc_position
        sample_state.position_amounts[1] = eth_position
        sample_state.position_entry_prices[0] = 50000.0
        sample_state.position_entry_prices[1] = 2000.0

        calculate_daily_equity(sample_state, d_idx, 2, sample_closes, sample_valid_data)

        expected_equity = backtest_config.initial_capital + expected_positions_value
        assert sample_state.equity_curve[d_idx] == pytest.approx(expected_equity, rel=0.01)

    def test_calculate_equity_invalid_data(
        self,
        sample_state,
        sample_closes: np.ndarray,
        backtest_config: BacktestConfig,
    ) -> None:
        """Test equity with missing/invalid price data."""
        sample_state.position_amounts[0] = 1.0
        sample_state.position_entry_prices[0] = 50000.0
        valid_data = np.array([False, True])

        # Should fallback to entry price when data invalid
        calculate_daily_equity(sample_state, 0, 2, sample_closes, valid_data)

        # Should still have equity
        assert sample_state.equity_curve[0] > 0

    def test_calculate_equity_nan_recovery(
        self,
        sample_state,
        sample_closes: np.ndarray,
        backtest_config: BacktestConfig,
    ) -> None:
        """Test that equity handles NaN correctly."""
        sample_state.equity_curve[0] = 100000.0
        sample_state.cash = 50000.0
        sample_state.position_amounts[0] = 1.0
        sample_state.position_entry_prices[0] = 50000.0

        valid_data = np.array([True, True])

        # Day 1: should calculate properly
        calculate_daily_equity(sample_state, 1, 2, sample_closes, valid_data)

        # Equity should be: 50000 (cash) + 1*51000 (position) = 101000
        assert sample_state.equity_curve[1] == pytest.approx(101000.0, rel=0.01)


# -------------------------------------------------------------------------
# track_asset_returns Tests
# -------------------------------------------------------------------------


class TestTrackAssetReturns:
    """Test track_asset_returns function."""

    @pytest.mark.parametrize(
        "d_idx,btc_price_valid,eth_price_valid,expected_btc_returns,expected_eth_returns",
        [
            (1, True, True, 1, 1),  # Both days have data
            (1, True, False, 1, 0),  # Only BTC has data
            (1, False, True, 0, 1),  # Only ETH has data
            (0, True, True, 0, 0),  # First day - no returns yet
        ],
    )
    def test_track_returns_variants(
        self,
        d_idx: int,
        btc_price_valid: bool,
        eth_price_valid: bool,
        expected_btc_returns: int,
        expected_eth_returns: int,
        sample_state,
        sample_closes: np.ndarray,
    ) -> None:
        """Test asset return tracking with different data availability."""
        tickers = ["KRW-BTC", "KRW-ETH"]
        valid_data = np.array([btc_price_valid, eth_price_valid])

        if not btc_price_valid:
            sample_closes[0, d_idx] = np.nan
        if not eth_price_valid:
            sample_closes[1, d_idx] = np.nan

        # Initialize previous prices
        if d_idx > 0:
            sample_state.previous_closes[0] = 50000.0
            sample_state.previous_closes[1] = 2000.0

        track_asset_returns(sample_state, d_idx, 2, tickers, sample_closes, valid_data)

        assert len(sample_state.asset_returns["KRW-BTC"]) == expected_btc_returns
        assert len(sample_state.asset_returns["KRW-ETH"]) == expected_eth_returns

    def test_track_returns_calculation(
        self,
        sample_state,
        sample_closes: np.ndarray,
    ) -> None:
        """Test correct return calculation."""
        tickers = ["KRW-BTC", "KRW-ETH"]
        valid_data = np.array([True, True])

        # Set up previous prices
        sample_state.previous_closes[0] = 50000.0
        sample_state.previous_closes[1] = 2000.0

        # Day 1: BTC 50000 -> 51000 (2% return), ETH 2000 -> 2050 (2.5% return)
        track_asset_returns(sample_state, 1, 2, tickers, sample_closes, valid_data)

        assert len(sample_state.asset_returns["KRW-BTC"]) == 1
        assert sample_state.asset_returns["KRW-BTC"][0] == pytest.approx(0.02, rel=0.001)

        assert len(sample_state.asset_returns["KRW-ETH"]) == 1
        assert sample_state.asset_returns["KRW-ETH"][0] == pytest.approx(0.025, rel=0.001)

    def test_track_returns_nan_handling(
        self,
        sample_state,
        sample_closes: np.ndarray,
    ) -> None:
        """Test that NaN prices are skipped."""
        tickers = ["KRW-BTC", "KRW-ETH"]
        sample_closes[0, 1] = np.nan  # BTC price is NaN

        sample_state.previous_closes[0] = 50000.0
        sample_state.previous_closes[1] = 2000.0

        valid_data = np.array([True, True])

        track_asset_returns(sample_state, 1, 2, tickers, sample_closes, valid_data)

        assert len(sample_state.asset_returns["KRW-BTC"]) == 0  # NaN skipped
        assert len(sample_state.asset_returns["KRW-ETH"]) == 1
