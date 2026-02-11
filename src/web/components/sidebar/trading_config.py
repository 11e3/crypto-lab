"""Trading configuration component.

Trading-related settings (interval, fees, slippage, etc.) UI component.
"""

import streamlit as st

from src.data.collector_fetch import Interval
from src.web.config import WebAppSettings, get_web_settings

__all__ = ["render_trading_config", "TradingConfig"]


class TradingConfig:
    """Trading configuration data class."""

    def __init__(
        self,
        interval: Interval,
        fee_rate: float,
        slippage_rate: float,
        initial_capital: float,
        max_slots: int,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        trailing_stop_pct: float | None = None,
    ) -> None:
        """Initialize trading configuration."""
        self.interval = interval
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.initial_capital = initial_capital
        self.max_slots = max_slots
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct


def _render_interval_selector() -> Interval:
    """Render candle interval selectbox.

    Returns:
        Selected Interval value.
    """
    st.subheader("⏱️ Candle Interval")
    interval_options = {
        "1 min": "minute1",
        "3 min": "minute3",
        "5 min": "minute5",
        "10 min": "minute10",
        "15 min": "minute15",
        "30 min": "minute30",
        "1 hour": "minute60",
        "4 hour": "minute240",
        "Daily": "day",
        "Weekly": "week",
        "Monthly": "month",
    }

    interval_label = st.selectbox(
        "Select Interval",
        options=list(interval_options.keys()),
        index=8,  # Default: Daily
        help="Candle interval to use for backtest",
    )
    interval: Interval = interval_options[interval_label]  # type: ignore
    return interval


def _render_trading_costs(settings: WebAppSettings) -> tuple[float, float]:
    """Render trading costs inputs (fee rate, slippage rate).

    Args:
        settings: Web settings with default_fee_rate and
            default_slippage_rate.

    Returns:
        Tuple of (fee_rate, slippage_rate) as decimals.
    """
    st.subheader("💰 Trading Costs")

    col1, col2 = st.columns(2)

    with col1:
        fee_rate = (
            st.number_input(
                "Fee Rate (%)",
                min_value=0.0,
                max_value=1.0,
                value=settings.default_fee_rate * 100,
                step=0.01,
                format="%.3f",
                help="Fee rate per trade (0.05% = Upbit default)",
            )
            / 100
        )

    with col2:
        slippage_rate = (
            st.number_input(
                "Slippage Rate (%)",
                min_value=0.0,
                max_value=1.0,
                value=settings.default_slippage_rate * 100,
                step=0.01,
                format="%.3f",
                help="Price slippage due to market impact",
            )
            / 100
        )

    return fee_rate, slippage_rate


def _render_portfolio_settings(settings: WebAppSettings) -> tuple[int, int]:
    """Render portfolio settings inputs (initial capital, max slots).

    Args:
        settings: Web settings with default_initial_capital.

    Returns:
        Tuple of (initial_capital, max_slots).
    """
    st.subheader("⚙️ Portfolio Settings")

    col1, col2 = st.columns(2)

    with col1:
        initial_capital = st.number_input(
            "Initial Capital (KRW)",
            min_value=1_000_000,
            max_value=1_000_000_000,
            value=int(settings.default_initial_capital),
            step=1_000_000,
            format="%d",
            help="Starting capital for backtest",
        )

    with col2:
        # Auto-sync with selected ticker count (key changes force widget reset)
        ticker_count = st.session_state.get("selected_ticker_count", 4) or 4
        max_slots = st.number_input(
            "Max Slots",
            min_value=1,
            max_value=20,
            value=ticker_count,
            step=1,
            key=f"max_slots_{ticker_count}",
            help="Maximum number of assets to hold simultaneously (auto-synced with selected assets)",
        )

    return initial_capital, max_slots


def _render_advanced_settings() -> tuple[float | None, float | None, float | None]:
    """Render advanced settings expander (stop loss, take profit, trailing stop).

    Returns:
        Tuple of (stop_loss_pct, take_profit_pct, trailing_stop_pct),
        each None if not enabled, otherwise a decimal fraction.
    """
    with st.expander("🔧 Advanced Settings (Optional)"):
        enable_stop_loss = st.checkbox("Enable Stop Loss", value=False)
        stop_loss_pct = None
        if enable_stop_loss:
            stop_loss_pct = (
                st.slider(
                    "Stop Loss (%)",
                    min_value=1.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.5,
                    help="Loss limit ratio",
                )
                / 100
            )

        enable_take_profit = st.checkbox("Enable Take Profit", value=False)
        take_profit_pct = None
        if enable_take_profit:
            take_profit_pct = (
                st.slider(
                    "Take Profit (%)",
                    min_value=1.0,
                    max_value=50.0,
                    value=10.0,
                    step=1.0,
                    help="Profit taking ratio",
                )
                / 100
            )

        enable_trailing_stop = st.checkbox("Enable Trailing Stop", value=False)
        trailing_stop_pct = None
        if enable_trailing_stop:
            trailing_stop_pct = (
                st.slider(
                    "Trailing Stop (%)",
                    min_value=1.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.5,
                    help="Maximum drawdown from peak",
                )
                / 100
            )

    return stop_loss_pct, take_profit_pct, trailing_stop_pct


def render_trading_config() -> TradingConfig:
    """Render trading configuration UI.

    Returns:
        TradingConfig object
    """
    settings = get_web_settings()

    interval = _render_interval_selector()

    st.markdown("---")

    fee_rate, slippage_rate = _render_trading_costs(settings)

    st.markdown("---")

    initial_capital, max_slots = _render_portfolio_settings(settings)

    st.markdown("---")

    stop_loss_pct, take_profit_pct, trailing_stop_pct = _render_advanced_settings()

    return TradingConfig(
        interval=interval,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        initial_capital=initial_capital,
        max_slots=max_slots,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop_pct=trailing_stop_pct,
    )
