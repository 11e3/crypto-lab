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

from collections.abc import Generator  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.exchange.types import (  # noqa: E402
    Balance,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Ticker,
)
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
def mock_exchange() -> Generator["MockExchange", None, None]:  # noqa: F821
    """Create a mock exchange for testing."""
    from tests.fixtures.mock_exchange import MockExchange  # noqa: PLC0415

    yield MockExchange()


@pytest.fixture
def vbo_strategy() -> VanillaVBO:
    """Create a VanillaVBO strategy instance for testing."""
    return VanillaVBO(
        sma_period=4,
        trend_sma_period=8,
        short_noise_period=4,
        long_noise_period=8,
    )


@pytest.fixture
def sample_balance() -> Balance:
    """Create a sample balance for testing."""
    return Balance(
        currency="KRW",
        balance=1_000_000.0,
        locked=0.0,
    )


@pytest.fixture
def sample_order() -> Order:
    """Create a sample order for testing."""
    return Order(
        order_id="test-order-123",
        symbol="KRW-BTC",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        status=OrderStatus.PENDING,
        amount=0.001,
        price=50_000_000.0,
        filled_amount=0.0,
        created_at=datetime.now(),
    )


@pytest.fixture
def sample_ticker() -> Ticker:
    """Create a sample ticker for testing."""
    return Ticker(
        symbol="KRW-BTC",
        price=50_000_000.0,
        volume=1000.0,
        timestamp=datetime.now(),
    )
