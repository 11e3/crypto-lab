"""
Unit tests for SignalHandler.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.execution.event_bus import EventBus, set_event_bus
from src.execution.events import EventType, SignalEvent
from src.execution.signal_handler import SignalHandler
from src.strategies.volatility_breakout import VanillaVBO
from tests.fixtures.mock_exchange import MockExchange


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Provides sample OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    data = {
        "open": [100.0 + i for i in range(20)],
        "high": [105.0 + i for i in range(20)],
        "low": [95.0 + i for i in range(20)],
        "close": [102.0 + i for i in range(20)],
        "volume": [1000.0 + i * 10 for i in range(20)],
    }
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def vbo_strategy() -> VanillaVBO:
    """Provides a VanillaVBO strategy instance."""
    return VanillaVBO(
        sma_period=4,
        trend_sma_period=8,
        short_noise_period=4,
        long_noise_period=8,
    )


@pytest.fixture
def signal_handler(mock_exchange: MockExchange, vbo_strategy: VanillaVBO) -> SignalHandler:
    """Provides a SignalHandler instance."""
    return SignalHandler(
        strategy=vbo_strategy,
        exchange=mock_exchange,
        min_data_points=10,
        publish_events=False,
    )


class TestSignalHandler:
    """Test cases for SignalHandler."""

    def test_initialization(self, mock_exchange: MockExchange, vbo_strategy: VanillaVBO) -> None:
        """Test SignalHandler initialization."""
        handler = SignalHandler(
            strategy=vbo_strategy,
            exchange=mock_exchange,
            min_data_points=10,
            publish_events=False,
        )
        assert handler.strategy == vbo_strategy
        assert handler.exchange == mock_exchange
        assert handler.min_data_points == 10
        assert handler.publish_events is False
        assert handler.event_bus is None

    def test_initialization_with_events(
        self, mock_exchange: MockExchange, vbo_strategy: VanillaVBO
    ) -> None:
        """Test SignalHandler initialization with events enabled."""
        handler = SignalHandler(
            strategy=vbo_strategy,
            exchange=mock_exchange,
            min_data_points=10,
            publish_events=True,
        )
        assert handler.publish_events is True
        assert handler.event_bus is not None

    def test_get_ohlcv_data(
        self, signal_handler: SignalHandler, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test getting OHLCV data."""
        signal_handler.exchange.set_ohlcv_data("KRW-BTC", "day", sample_ohlcv_data)

        df = signal_handler.get_ohlcv_data("KRW-BTC")
        assert df is not None
        assert len(df) >= signal_handler.min_data_points
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_get_ohlcv_data_with_count(
        self, signal_handler: SignalHandler, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test getting OHLCV data with custom count."""
        signal_handler.exchange.set_ohlcv_data("KRW-BTC", "day", sample_ohlcv_data)

        df = signal_handler.get_ohlcv_data("KRW-BTC", count=15)
        assert df is not None
        assert len(df) == 15

    def test_get_ohlcv_data_with_interval(
        self, signal_handler: SignalHandler, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test getting OHLCV data with custom interval."""
        signal_handler.exchange.set_ohlcv_data("KRW-BTC", "minute240", sample_ohlcv_data)

        df = signal_handler.get_ohlcv_data("KRW-BTC", interval="minute240")
        assert df is not None

    def test_get_ohlcv_data_insufficient(self, signal_handler: SignalHandler) -> None:
        """Test getting OHLCV data with insufficient data (covers lines 72-75 warning)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        minimal_data = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [105.0] * 5,
                "low": [95.0] * 5,
                "close": [102.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        signal_handler.exchange.set_ohlcv_data("KRW-BTC", "day", minimal_data)

        with patch("src.execution.signal_data.logger") as mock_logger:
            df = signal_handler.get_ohlcv_data("KRW-BTC")
            assert df is None
            # Verify warning was logged (lines 72-75)
            mock_logger.warning.assert_called_once()
            assert "Insufficient data" in str(mock_logger.warning.call_args[0][0])

    def test_get_ohlcv_data_none_from_exchange(self, signal_handler: SignalHandler) -> None:
        """Test getting OHLCV data when exchange returns None."""
        with patch.object(signal_handler.exchange, "get_ohlcv", return_value=None):
            df = signal_handler.get_ohlcv_data("KRW-BTC")
            assert df is None

    def test_get_ohlcv_data_error(self, signal_handler: SignalHandler) -> None:
        """Test getting OHLCV data with error."""
        with patch.object(
            signal_handler.exchange, "get_ohlcv", side_effect=ConnectionError("API Error")
        ):
            df = signal_handler.get_ohlcv_data("KRW-BTC")
            assert df is None

    def test_check_entry_signal(
        self, signal_handler: SignalHandler, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test checking entry signal."""
        signal_handler.exchange.set_ohlcv_data("KRW-BTC", "day", sample_ohlcv_data)

        entry_signal = signal_handler.check_entry_signal("KRW-BTC", current_price=100.0)
        assert isinstance(entry_signal, bool)

    def test_check_entry_signal_with_target_price(
        self, signal_handler: SignalHandler, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test checking entry signal with target price."""
        signal_handler.exchange.set_ohlcv_data("KRW-BTC", "day", sample_ohlcv_data)

        df = signal_handler.strategy.calculate_indicators(sample_ohlcv_data)
        target_price = float(df.iloc[-2]["target"])

        entry_signal = signal_handler.check_entry_signal(
            "KRW-BTC", current_price=target_price + 10.0, target_price=target_price
        )
        assert isinstance(entry_signal, bool)

    def test_check_entry_signal_target_price_not_reached(
        self, signal_handler: SignalHandler, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test checking entry signal when target price not reached."""
        signal_handler.exchange.set_ohlcv_data("KRW-BTC", "day", sample_ohlcv_data)

        df = signal_handler.strategy.calculate_indicators(sample_ohlcv_data)
        target_price = float(df.iloc[-2]["target"])

        entry_signal = signal_handler.check_entry_signal(
            "KRW-BTC", current_price=target_price - 10.0, target_price=target_price
        )
        assert isinstance(entry_signal, bool)

    def test_check_entry_signal_with_event(
        self, mock_exchange: MockExchange, vbo_strategy: VanillaVBO, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test checking entry signal with event publishing (covers lines 121-129)."""
        event_bus = EventBus()
        set_event_bus(event_bus)
        handler = MagicMock()
        event_bus.subscribe(EventType.ENTRY_SIGNAL, handler)

        signal_handler = SignalHandler(
            strategy=vbo_strategy,
            exchange=mock_exchange,
            min_data_points=10,
            publish_events=True,
        )
        mock_exchange.set_ohlcv_data("KRW-BTC", "day", sample_ohlcv_data)

        # Prepare DataFrame with entry signal set to True
        df = vbo_strategy.calculate_indicators(sample_ohlcv_data)
        df = vbo_strategy.generate_signals(df)
        # Force entry signal to True to ensure event publishing path is executed
        df.loc[df.index[-2], "entry_signal"] = True

        # Patch both get_ohlcv_data and strategy methods to ensure signal stays True
        with (
            patch.object(signal_handler, "get_ohlcv_data", return_value=df),
            patch.object(signal_handler.strategy, "calculate_indicators", return_value=df),
            patch.object(signal_handler.strategy, "generate_signals", return_value=df),
        ):
            entry_signal = signal_handler.check_entry_signal("KRW-BTC", current_price=100.0)
            # Ensure signal is True so event publishing path (lines 121-129) is executed
            assert entry_signal is True
            handler.assert_called_once()
            event = handler.call_args[0][0]
            assert isinstance(event, SignalEvent)
            assert event.event_type == EventType.ENTRY_SIGNAL
            assert event.ticker == "KRW-BTC"
            assert event.price == 100.0

    def test_check_entry_signal_no_data(self, signal_handler: SignalHandler) -> None:
        """Test checking entry signal with no data."""
        entry_signal = signal_handler.check_entry_signal("KRW-BTC", current_price=100.0)
        assert entry_signal is False

    def test_check_entry_signal_insufficient_data_after_calculation(
        self, signal_handler: SignalHandler
    ) -> None:
        """Test checking entry signal when data becomes insufficient after calculation."""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        minimal_data = pd.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
            },
            index=dates,
        )

        with patch.object(signal_handler, "get_ohlcv_data", return_value=minimal_data):
            entry_signal = signal_handler.check_entry_signal("KRW-BTC", current_price=100.0)
            assert entry_signal is False

    def test_check_entry_signal_error(
        self, signal_handler: SignalHandler, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test checking entry signal with error."""
        signal_handler.exchange.set_ohlcv_data("KRW-BTC", "day", sample_ohlcv_data)

        with patch.object(
            signal_handler.strategy, "calculate_indicators", side_effect=ValueError("Error")
        ):
            entry_signal = signal_handler.check_entry_signal("KRW-BTC", current_price=100.0)
            assert entry_signal is False

    def test_check_exit_signal(
        self, signal_handler: SignalHandler, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test checking exit signal."""
        signal_handler.exchange.set_ohlcv_data("KRW-BTC", "day", sample_ohlcv_data)

        exit_signal = signal_handler.check_exit_signal("KRW-BTC")
        assert isinstance(exit_signal, bool)

    def test_check_exit_signal_with_event(
        self, mock_exchange: MockExchange, vbo_strategy: VanillaVBO, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test checking exit signal with event publishing (covers lines 164-173)."""
        event_bus = EventBus()
        set_event_bus(event_bus)
        handler = MagicMock()
        event_bus.subscribe(EventType.EXIT_SIGNAL, handler)

        signal_handler = SignalHandler(
            strategy=vbo_strategy,
            exchange=mock_exchange,
            min_data_points=10,
            publish_events=True,
        )
        mock_exchange.set_ohlcv_data("KRW-BTC", "day", sample_ohlcv_data)
        mock_exchange.set_price("KRW-BTC", 100.0)

        # Prepare DataFrame with exit signal set to True
        df = vbo_strategy.calculate_indicators(sample_ohlcv_data)
        df = vbo_strategy.generate_signals(df)
        # Force exit signal to True to ensure event publishing path is executed
        df.loc[df.index[-2], "exit_signal"] = True

        # Patch both get_ohlcv_data and strategy methods to ensure signal stays True
        with (
            patch.object(signal_handler, "get_ohlcv_data", return_value=df),
            patch.object(signal_handler.strategy, "calculate_indicators", return_value=df),
            patch.object(signal_handler.strategy, "generate_signals", return_value=df),
        ):
            exit_signal = signal_handler.check_exit_signal("KRW-BTC")
            # Ensure signal is True so event publishing path (lines 164-173) is executed
            assert exit_signal is True
            handler.assert_called_once()
            event = handler.call_args[0][0]
            assert isinstance(event, SignalEvent)
            assert event.event_type == EventType.EXIT_SIGNAL
            assert event.ticker == "KRW-BTC"
            assert event.price == 100.0

    def test_check_exit_signal_event_price_error(
        self, mock_exchange: MockExchange, vbo_strategy: VanillaVBO, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test checking exit signal when getting price for event fails (covers lines 174-175)."""
        event_bus = EventBus()
        set_event_bus(event_bus)
        handler = MagicMock()
        event_bus.subscribe(EventType.EXIT_SIGNAL, handler)

        signal_handler = SignalHandler(
            strategy=vbo_strategy,
            exchange=mock_exchange,
            min_data_points=10,
            publish_events=True,
        )
        mock_exchange.set_ohlcv_data("KRW-BTC", "day", sample_ohlcv_data)

        # Prepare DataFrame with exit signal set to True
        df = vbo_strategy.calculate_indicators(sample_ohlcv_data)
        df = vbo_strategy.generate_signals(df)
        # Force exit signal to True to ensure we enter the event publishing block
        df.loc[df.index[-2], "exit_signal"] = True

        # Patch both get_ohlcv_data and strategy methods to ensure signal stays True
        with (
            patch.object(signal_handler, "get_ohlcv_data", return_value=df),
            patch.object(signal_handler.strategy, "calculate_indicators", return_value=df),
            patch.object(signal_handler.strategy, "generate_signals", return_value=df),
            patch.object(
                signal_handler.exchange,
                "get_current_price",
                side_effect=ConnectionError("Price error"),
            ),
            patch("src.execution.signal_handler.logger") as mock_logger,
        ):
            exit_signal = signal_handler.check_exit_signal("KRW-BTC")
            # Signal should still be True even if price fetch fails
            assert exit_signal is True
            # Event should not be published if price fetch fails (lines 174-175)
            handler.assert_not_called()
            # Error should be logged
            mock_logger.error.assert_called_once()
            assert "Error getting price for exit signal event" in str(
                mock_logger.error.call_args[0][0]
            )

    def test_check_exit_signal_no_data(self, signal_handler: SignalHandler) -> None:
        """Test checking exit signal with no data."""
        exit_signal = signal_handler.check_exit_signal("KRW-BTC")
        assert exit_signal is False

    def test_check_exit_signal_insufficient_data_after_calculation(
        self, signal_handler: SignalHandler
    ) -> None:
        """Test checking exit signal when data becomes insufficient after calculation."""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        minimal_data = pd.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
            },
            index=dates,
        )

        with patch.object(signal_handler, "get_ohlcv_data", return_value=minimal_data):
            exit_signal = signal_handler.check_exit_signal("KRW-BTC")
            assert exit_signal is False

    def test_check_exit_signal_error(
        self, signal_handler: SignalHandler, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test checking exit signal with error."""
        signal_handler.exchange.set_ohlcv_data("KRW-BTC", "day", sample_ohlcv_data)

        with patch.object(
            signal_handler.strategy, "calculate_indicators", side_effect=ValueError("Error")
        ):
            exit_signal = signal_handler.check_exit_signal("KRW-BTC")
            assert exit_signal is False

    def test_calculate_metrics(
        self, signal_handler: SignalHandler, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test calculating metrics."""
        signal_handler.exchange.set_ohlcv_data("KRW-BTC", "day", sample_ohlcv_data)

        metrics = signal_handler.calculate_metrics("KRW-BTC", required_period=10)
        assert metrics is not None
        assert "target" in metrics
        assert "k" in metrics
        assert "long_noise" in metrics
        assert "sma" in metrics
        assert "sma_trend" in metrics

        assert isinstance(metrics["target"], float)
        assert isinstance(metrics["k"], float)
        assert isinstance(metrics["long_noise"], float)
        assert isinstance(metrics["sma"], float)
        assert isinstance(metrics["sma_trend"], float)

    def test_calculate_metrics_with_default_period(
        self, signal_handler: SignalHandler, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test calculating metrics with default required_period."""
        signal_handler.exchange.set_ohlcv_data("KRW-BTC", "day", sample_ohlcv_data)

        metrics = signal_handler.calculate_metrics("KRW-BTC", required_period=None)
        assert metrics is not None
        assert "target" in metrics

    def test_calculate_metrics_insufficient_data(self, signal_handler: SignalHandler) -> None:
        """Test calculating metrics with insufficient data (covers lines 206-209 warning)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        minimal_data = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [105.0] * 5,
                "low": [95.0] * 5,
                "close": [102.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        signal_handler.exchange.set_ohlcv_data("KRW-BTC", "day", minimal_data)

        with patch("src.execution.signal_metrics.logger") as mock_logger:
            metrics = signal_handler.calculate_metrics("KRW-BTC", required_period=10)
            assert metrics is None
            # Verify warning was logged (lines 206-209)
            # Note: warning may be called multiple times (from get_ohlcv_data and calculate_metrics)
            assert mock_logger.warning.call_count >= 1
            # Check that the calculate_metrics warning was called
            warning_calls = [str(call[0][0]) for call in mock_logger.warning.call_args_list]
            assert any("need 10" in call and "got" in call for call in warning_calls)

    def test_calculate_metrics_insufficient_data_after_calculation(
        self, signal_handler: SignalHandler
    ) -> None:
        """Test calculating metrics when data becomes insufficient after calculation."""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        minimal_data = pd.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
            },
            index=dates,
        )

        with patch.object(signal_handler, "get_ohlcv_data", return_value=minimal_data):
            metrics = signal_handler.calculate_metrics("KRW-BTC", required_period=10)
            assert metrics is None

    def test_calculate_metrics_len_less_than_2(
        self, signal_handler: SignalHandler, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test calculating metrics when data length is less than 2 after calculation (line 218)."""
        # Create data with exactly 1 row after indicators calculation might reduce it
        dates = pd.date_range("2024-01-01", periods=15, freq="D")
        minimal_data = pd.DataFrame(
            {
                "open": [100.0 + i for i in range(15)],
                "high": [105.0 + i for i in range(15)],
                "low": [95.0 + i for i in range(15)],
                "close": [102.0 + i for i in range(15)],
                "volume": [1000.0 + i * 10 for i in range(15)],
            },
            index=dates,
        )
        signal_handler.exchange.set_ohlcv_data("KRW-BTC", "day", minimal_data)

        # Patch calculate_indicators to return DataFrame with only 1 row
        with patch.object(
            signal_handler.strategy,
            "calculate_indicators",
            return_value=minimal_data.iloc[-1:].copy(),
        ):
            metrics = signal_handler.calculate_metrics("KRW-BTC", required_period=1)
            assert metrics is None

    def test_calculate_metrics_error(self, signal_handler: SignalHandler) -> None:
        """Test calculating metrics with error."""
        with patch.object(signal_handler, "get_ohlcv_data", return_value=None):
            metrics = signal_handler.calculate_metrics("KRW-BTC")
            assert metrics is None

    def test_calculate_metrics_calculation_error(
        self, signal_handler: SignalHandler, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test calculating metrics when calculation fails."""
        signal_handler.exchange.set_ohlcv_data("KRW-BTC", "day", sample_ohlcv_data)

        with patch.object(
            signal_handler.strategy,
            "calculate_indicators",
            side_effect=ValueError("Calculation error"),
        ):
            metrics = signal_handler.calculate_metrics("KRW-BTC", required_period=10)
            assert metrics is None
