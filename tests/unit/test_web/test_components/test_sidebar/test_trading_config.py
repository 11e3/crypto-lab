"""Tests for TradingConfig dataclass."""

from __future__ import annotations

from src.web.components.sidebar.trading_config import TradingConfig


class TestTradingConfig:
    """Tests for TradingConfig data class."""

    def test_basic_creation(self) -> None:
        config = TradingConfig(
            interval="day",
            fee_rate=0.0005,
            slippage_rate=0.001,
            initial_capital=10_000_000.0,
            max_slots=4,
        )
        assert config.interval == "day"
        assert config.fee_rate == 0.0005
        assert config.slippage_rate == 0.001
        assert config.initial_capital == 10_000_000.0
        assert config.max_slots == 4
        assert config.stop_loss_pct is None
        assert config.take_profit_pct is None
        assert config.trailing_stop_pct is None

    def test_with_all_optional_params(self) -> None:
        config = TradingConfig(
            interval="minute60",
            fee_rate=0.0005,
            slippage_rate=0.001,
            initial_capital=5_000_000.0,
            max_slots=2,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            trailing_stop_pct=0.03,
        )
        assert config.stop_loss_pct == 0.05
        assert config.take_profit_pct == 0.10
        assert config.trailing_stop_pct == 0.03

    def test_various_intervals(self) -> None:
        for interval in ["minute1", "minute5", "minute60", "day", "week", "month"]:
            config = TradingConfig(
                interval=interval,  # type: ignore[arg-type]
                fee_rate=0.0005,
                slippage_rate=0.0,
                initial_capital=1_000_000.0,
                max_slots=1,
            )
            assert config.interval == interval

    def test_zero_fees(self) -> None:
        config = TradingConfig(
            interval="day",
            fee_rate=0.0,
            slippage_rate=0.0,
            initial_capital=1_000_000.0,
            max_slots=1,
        )
        assert config.fee_rate == 0.0
        assert config.slippage_rate == 0.0
