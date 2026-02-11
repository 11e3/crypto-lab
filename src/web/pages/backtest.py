"""Backtest page.

Page for backtest execution and result display.
"""

from __future__ import annotations

from datetime import date as date_type
from typing import TYPE_CHECKING, Any

import streamlit as st

from src.backtester.models import BacktestConfig
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.web.components.sidebar.trading_config import TradingConfig
from src.web.components.results.backtest_results import (
    UnifiedBacktestResult,
    render_backtest_results,
)
from src.web.components.sidebar.asset_selector import render_asset_selector
from src.web.components.sidebar.date_config import render_date_config
from src.web.components.sidebar.strategy_selector import render_strategy_selector
from src.web.components.sidebar.trading_config import render_trading_config
from src.web.services.backtest_runner import run_backtest_service
from src.web.services.data_loader import get_data_files, validate_data_availability

logger = get_logger(__name__)

__all__ = ["render_backtest_page"]


def render_backtest_page() -> None:
    """Render backtest page (settings and results on same page)."""
    st.header("📈 Backtest")

    # Render settings section
    _render_settings_section()

    # Display results below settings if available
    if "vbo_backtest_result" in st.session_state:
        st.divider()
        strategy_name = st.session_state.get("vbo_strategy_name", "VBO")
        unified = UnifiedBacktestResult.from_vbo(st.session_state.vbo_backtest_result, strategy_name)
        render_backtest_results(unified)
    elif "backtest_result" in st.session_state:
        st.divider()
        unified = UnifiedBacktestResult.from_event_driven(st.session_state.backtest_result)
        render_backtest_results(unified)


def _render_settings_section() -> None:
    """Render settings section."""
    st.subheader("⚙️ Backtest Settings")

    # Split settings into 3 columns
    col1, col2, col3 = st.columns([1, 1, 1])

    # Render asset selector first so ticker count is available for max_slots default
    with col3:
        st.markdown("### 🪙 Asset Selection")
        selected_tickers = render_asset_selector()

    # ===== Column 1: Date & Trading Settings =====
    with col1:
        st.markdown("### 📅 Period Settings")
        start_date, end_date = render_date_config()

        st.markdown("### 💰 Trading Settings")
        trading_config = render_trading_config()

    # ===== Column 2: Strategy Settings =====
    with col2:
        st.markdown("### 📈 Strategy Settings")
        strategy_name, strategy_params = render_strategy_selector()

    st.divider()

    # Settings Summary
    with st.expander("📋 Settings Summary", expanded=False):
        _show_config_summary(strategy_name, selected_tickers, trading_config, start_date, end_date)

    # Run Button and Clear Cache
    col_left, col_center, col_right = st.columns([1, 1, 1])
    with col_center:
        run_button = st.button(
            "🚀 Run Backtest",
            type="primary",
            width="stretch",
            disabled=not strategy_name or not selected_tickers,
        )
    with col_right:
        if st.button("🗑️ Clear Cache", width="stretch"):
            st.cache_data.clear()
            if "backtest_result" in st.session_state:
                del st.session_state.backtest_result
            if "vbo_backtest_result" in st.session_state:
                del st.session_state.vbo_backtest_result
            st.success("Cache cleared!")
            st.rerun()

    # Validation
    if not strategy_name:
        st.warning("⚠️ Please select a strategy.")
        return

    if not selected_tickers:
        st.warning("⚠️ Please select at least one asset.")
        return

    # Check data availability
    available_tickers, missing_tickers = validate_data_availability(
        selected_tickers, trading_config.interval
    )

    if missing_tickers:
        st.warning(
            f"⚠️ Missing data for the following assets: {', '.join(missing_tickers)}\n\n"
            f"Available assets: {', '.join(available_tickers) if available_tickers else 'None'}"
        )

        if not available_tickers:
            st.error("❌ No available data. Please collect data first.")
            st.code("uv run python scripts/collect_data.py")
            return

    # Run backtest
    if run_button:
        # Check if bt library strategy
        from src.web.services.strategy_registry import is_vbo_strategy

        if is_vbo_strategy(strategy_name):
            _run_vbo_backtest(
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                available_tickers=available_tickers,
                trading_config=trading_config,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            _run_event_driven_backtest(
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                available_tickers=available_tickers,
                trading_config=trading_config,
                start_date=start_date,
                end_date=end_date,
            )


def _run_event_driven_backtest(
    strategy_name: str,
    strategy_params: dict[str, Any],
    available_tickers: list[str],
    trading_config: TradingConfig,
    start_date: date_type | None,
    end_date: date_type | None,
) -> None:
    """Run backtest using event-driven engine."""
    with st.spinner("Running backtest..."):
        # Create BacktestConfig
        config = BacktestConfig(
            initial_capital=trading_config.initial_capital,
            fee_rate=trading_config.fee_rate,
            slippage_rate=trading_config.slippage_rate,
            max_slots=trading_config.max_slots,
            use_cache=False,
        )

        # Get data file paths
        data_files = get_data_files(available_tickers, trading_config.interval)

        if not data_files:
            st.error("Data files not found.")
            return

        # Run backtest (convert to serializable types for caching)
        data_files_dict = {ticker: str(path) for ticker, path in data_files.items()}
        config_dict = {
            "initial_capital": config.initial_capital,
            "fee_rate": config.fee_rate,
            "slippage_rate": config.slippage_rate,
            "max_slots": config.max_slots,
            "use_cache": config.use_cache,
        }
        start_date_str = start_date.isoformat() if start_date else None
        end_date_str = end_date.isoformat() if end_date else None

        result = run_backtest_service(
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            data_files_dict=data_files_dict,
            config_dict=config_dict,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
        )

        if result:
            # Clear bt result if exists
            if "vbo_backtest_result" in st.session_state:
                del st.session_state.vbo_backtest_result
            st.session_state.backtest_result = result
            st.success("Backtest completed!")
            st.rerun()
        else:
            st.error("Backtest execution failed")


def _run_vbo_backtest(
    strategy_name: str,
    strategy_params: dict[str, Any],
    available_tickers: list[str],
    trading_config: TradingConfig,
    start_date: date_type | None,
    end_date: date_type | None,
) -> None:
    """Run backtest using VBO vectorized engine."""
    from src.web.services.vbo_backtest_runner import (
        run_vbo_backtest_generic_service,
        run_vbo_backtest_service,
    )

    # Convert tickers: KRW-BTC -> BTC
    symbols = [t.replace("KRW-", "") for t in available_tickers]

    # Strategy name to bt strategy type mapping (centralized in strategy_registry)
    from src.web.services.strategy_registry import get_vbo_strategy_type

    vbo_strategy_type, strategy_display = get_vbo_strategy_type(strategy_name)

    # Run backtest based on strategy type
    if strategy_name == "VBO":
        with st.spinner(f"Running {strategy_display} backtest..."):
            result = run_vbo_backtest_service(
                symbols=tuple(symbols),
                interval="day",
                initial_cash=int(trading_config.initial_capital),
                fee=trading_config.fee_rate,
                slippage=trading_config.slippage_rate,
                multiplier=strategy_params.get("multiplier", 2),
                lookback=strategy_params.get("lookback", 5),
                start_date=start_date,
                end_date=end_date,
            )
    else:
        # Use generic service for other strategies
        with st.spinner(f"Running {strategy_display} backtest..."):
            result = run_vbo_backtest_generic_service(
                strategy_type=vbo_strategy_type,
                symbols=tuple(symbols),
                interval="day",
                initial_cash=int(trading_config.initial_capital),
                fee=trading_config.fee_rate,
                slippage=trading_config.slippage_rate,
                start_date=start_date,
                end_date=end_date,
                **strategy_params,
            )

    if result:
        # Clear event-driven result if exists
        if "backtest_result" in st.session_state:
            del st.session_state.backtest_result
        st.session_state.vbo_backtest_result = result
        st.session_state.vbo_strategy_name = strategy_name
        st.success(f"{strategy_display} Backtest completed!")
        st.rerun()
    else:
        st.error("VBO Backtest execution failed")


def _show_config_summary(
    strategy_name: str,
    selected_tickers: list[str],
    trading_config: TradingConfig,
    start_date: date_type | None,
    end_date: date_type | None,
) -> None:
    """Display settings summary."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            **📈 Strategy**
            - Strategy: {strategy_name}
            - Interval: {trading_config.interval}
            """
        )

    with col2:
        st.markdown(
            f"""
            **📅 Period**
            - Start: {start_date if start_date else "All"}
            - End: {end_date if end_date else "All"}
            """
        )

    with col3:
        st.markdown(
            f"""
            **⚙️ Portfolio**
            - Initial Capital: {trading_config.initial_capital:,.0f} KRW
            - Max Slots: {trading_config.max_slots}
            - Assets: {len(selected_tickers)}
            """
        )
