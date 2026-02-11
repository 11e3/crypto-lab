"""
Pytest configuration and shared fixtures.
"""

import os
import sys
from pathlib import Path

# Set matplotlib to use non-interactive backend for CI/testing
os.environ.setdefault("MPLBACKEND", "Agg")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import date, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.backtester.models import BacktestConfig, BacktestResult, Trade  # noqa: E402
from src.strategies.volatility_breakout import VanillaVBO  # noqa: E402
from tests.fixtures.data.sample_ohlcv import (  # noqa: E402
    generate_multiple_tickers_data,
    generate_ohlcv_data,
    generate_trending_data,
    generate_volatile_data,
)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    return generate_ohlcv_data(periods=100, seed=42)


@pytest.fixture
def trending_ohlcv_data() -> pd.DataFrame:
    """Generate trending OHLCV data for testing."""
    return generate_trending_data(periods=100, seed=42)


@pytest.fixture
def volatile_ohlcv_data() -> pd.DataFrame:
    """Generate volatile OHLCV data for testing."""
    return generate_volatile_data(periods=100, seed=42)


@pytest.fixture
def multiple_tickers_data() -> dict[str, pd.DataFrame]:
    """Generate OHLCV data for multiple tickers."""
    tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
    return generate_multiple_tickers_data(tickers=tickers, periods=100, seed=42)


@pytest.fixture
def test_config_path() -> Path:
    """Get path to test configuration file."""
    return Path(__file__).parent / "fixtures" / "config" / "test_settings.yaml"


@pytest.fixture
def vbo_strategy() -> VanillaVBO:
    """Create a VanillaVBO strategy instance for testing."""
    return VanillaVBO(
        sma_period=4,
        trend_sma_period=8,
        short_noise_period=4,
        long_noise_period=8,
    )


# ============================================================================
# Backtest Fixtures
# ============================================================================


@pytest.fixture
def default_backtest_config() -> BacktestConfig:
    """Create default backtest configuration."""
    return BacktestConfig(
        initial_capital=10_000_000,
        fee_rate=0.0005,
        slippage_rate=0.0002,
        max_slots=5,
    )


@pytest.fixture
def conservative_backtest_config() -> BacktestConfig:
    """Create conservative backtest configuration with stop loss."""
    return BacktestConfig(
        initial_capital=10_000_000,
        fee_rate=0.0005,
        slippage_rate=0.0002,
        max_slots=3,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )


@pytest.fixture
def sample_trades() -> list[Trade]:
    """Create sample trade list for testing."""
    base_date = date(2024, 1, 1)
    return [
        Trade(
            ticker="KRW-BTC",
            entry_date=base_date,
            exit_date=base_date + timedelta(days=1),
            entry_price=50000000.0,
            exit_price=51000000.0,
            amount=0.01,
            pnl=10000.0,
            pnl_pct=0.02,
        ),
        Trade(
            ticker="KRW-ETH",
            entry_date=base_date + timedelta(days=2),
            exit_date=base_date + timedelta(days=3),
            entry_price=3000000.0,
            exit_price=2940000.0,
            amount=0.1,
            pnl=-6000.0,
            pnl_pct=-0.02,
        ),
        Trade(
            ticker="KRW-BTC",
            entry_date=base_date + timedelta(days=4),
            exit_date=base_date + timedelta(days=5),
            entry_price=49000000.0,
            exit_price=50470000.0,
            amount=0.01,
            pnl=14700.0,
            pnl_pct=0.03,
        ),
    ]


@pytest.fixture
def sample_equity_curve() -> np.ndarray:
    """Create sample equity curve for testing."""
    initial = 10_000_000
    returns = [1.0, 1.02, 1.01, 0.98, 1.03, 1.05, 1.04, 1.06, 1.02, 1.08]
    return np.array([initial * r for r in returns])


@pytest.fixture
def sample_backtest_result(
    default_backtest_config: BacktestConfig,
    sample_trades: list[Trade],
    sample_equity_curve: np.ndarray,
) -> BacktestResult:
    """Create sample backtest result for testing."""
    dates = np.array(
        [date(2024, 1, 1) + timedelta(days=i) for i in range(len(sample_equity_curve))]
    )
    return BacktestResult(
        equity_curve=sample_equity_curve,
        dates=dates,
        trades=sample_trades,
        config=default_backtest_config,
        strategy_name="VBO_Test",
        total_return=0.08,
        sharpe_ratio=1.5,
        mdd=-0.05,
        total_trades=3,
        winning_trades=2,
        win_rate=0.667,
    )


# ============================================================================
# Strategy Fixtures
# ============================================================================


@pytest.fixture
def momentum_strategy_params() -> dict[str, Any]:
    """Create momentum strategy parameters."""
    return {
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
    }


@pytest.fixture
def mean_reversion_strategy_params() -> dict[str, Any]:
    """Create mean reversion strategy parameters."""
    return {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
    }


@pytest.fixture
def vbo_strategy_params() -> dict[str, Any]:
    """Create VBO strategy parameters."""
    return {
        "sma_period": 4,
        "trend_sma_period": 8,
        "short_noise_period": 4,
        "long_noise_period": 8,
        "k": 0.5,
    }


# ============================================================================
# Data Fixtures
# ============================================================================


@pytest.fixture
def large_ohlcv_data() -> pd.DataFrame:
    """Generate large OHLCV dataset for stress testing."""
    return generate_ohlcv_data(periods=1000, seed=42)


@pytest.fixture
def downtrend_ohlcv_data() -> pd.DataFrame:
    """Generate downtrending OHLCV data for testing."""
    return generate_trending_data(periods=100, trend=-0.002, seed=42)


@pytest.fixture
def sideways_ohlcv_data() -> pd.DataFrame:
    """Generate sideways/range-bound OHLCV data for testing."""
    return generate_ohlcv_data(periods=100, volatility=0.005, seed=42)


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory with parquet files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_parquet_files(
    temp_data_dir: Path, multiple_tickers_data: dict[str, pd.DataFrame]
) -> dict[str, Path]:
    """Create sample parquet files in temp directory."""
    files = {}
    for ticker, df in multiple_tickers_data.items():
        filepath = temp_data_dir / f"{ticker}_day.parquet"
        df.to_parquet(filepath)
        files[ticker] = filepath
    return files
