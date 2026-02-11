"""Data collection page.

Page for data collection execution and status display.
"""

import streamlit as st

from src.data.collector_factory import DataCollectorFactory, ExchangeName
from src.utils.logger import get_logger
from src.web.config.constants import (
    BINANCE_DATA_COLLECT_INTERVALS,
    BINANCE_DATA_COLLECT_TICKERS,
    DATA_COLLECT_INTERVALS,
    DATA_COLLECT_TICKERS,
)

logger = get_logger(__name__)

__all__ = ["render_data_collect_page"]


def _get_exchange_config(
    exchange: str,
) -> tuple[list[str], list[tuple[str, str]], list[str], str]:
    """Get ticker list, interval list, default intervals, and placeholder for an exchange.

    Args:
        exchange: Exchange name

    Returns:
        Tuple of (tickers, intervals, default_intervals, custom_placeholder)
    """
    if exchange == "binance":
        return (
            BINANCE_DATA_COLLECT_TICKERS,
            BINANCE_DATA_COLLECT_INTERVALS,
            ["4h", "1d", "1w"],
            "e.g., MATICUSDT",
        )
    # Default: Upbit
    return (
        DATA_COLLECT_TICKERS,
        DATA_COLLECT_INTERVALS,
        ["minute240", "day", "week"],
        "e.g., KRW-MATIC",
    )


def _render_ticker_selection(
    tickers_list: list[str],
    exchange_lower: ExchangeName,
    custom_placeholder: str,
    exchange: str,
) -> list[str]:
    """Render ticker selection UI within the current Streamlit column context.

    Includes multiselect, select all / deselect all buttons, and custom ticker input.

    Args:
        tickers_list: Available tickers for the exchange
        exchange_lower: Lowercase exchange name
        custom_placeholder: Placeholder text for custom ticker input
        exchange: Display name of the exchange

    Returns:
        List of selected tickers (including custom additions)
    """
    st.markdown("### 📈 Ticker Selection")

    # Quick selection buttons
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Select All", key="select_all_tickers"):
            st.session_state.collect_selected_tickers = list(tickers_list)
            st.rerun()
    with btn_col2:
        if st.button("Deselect All", key="deselect_all_tickers"):
            st.session_state.collect_selected_tickers = []
            st.rerun()

    # Ticker multiselect
    if "collect_selected_tickers" not in st.session_state:
        st.session_state.collect_selected_tickers = tickers_list[:6]

    # Ensure defaults are valid for current exchange
    valid_defaults = [t for t in st.session_state.collect_selected_tickers if t in tickers_list]

    selected_tickers = st.multiselect(
        "Select Tickers",
        options=tickers_list,
        default=valid_defaults,
        key=f"collect_ticker_multiselect_{exchange_lower}",
    )

    # Custom ticker input
    if "custom_collect_tickers" not in st.session_state:
        st.session_state.custom_collect_tickers = []

    custom_ticker = st.text_input(
        "Add Custom Ticker",
        placeholder=custom_placeholder,
        help=f"Enter ticker supported by {exchange}",
    )
    if (
        custom_ticker
        and custom_ticker.upper() not in selected_tickers
        and custom_ticker.upper() not in st.session_state.custom_collect_tickers
        and st.button(f"➕ Add {custom_ticker}")
    ):
        st.session_state.custom_collect_tickers.append(custom_ticker.upper())
        st.rerun()

    # Add custom tickers to selected list
    for ticker in st.session_state.custom_collect_tickers:
        selected_tickers.append(ticker)

    return selected_tickers


def _render_interval_selection(
    intervals_list: list[tuple[str, str]],
    default_intervals: list[str],
    exchange_lower: ExchangeName,
) -> list[str]:
    """Render interval selection UI within the current Streamlit column context.

    Args:
        intervals_list: Available intervals as (code, name) tuples
        default_intervals: Default interval codes to pre-select
        exchange_lower: Lowercase exchange name

    Returns:
        List of selected interval codes
    """
    st.markdown("### ⏱️ Interval Selection")

    interval_options = [code for code, _ in intervals_list]
    interval_labels = {code: f"{name} ({code})" for code, name in intervals_list}

    return st.multiselect(
        "Select Intervals",
        options=interval_options,
        default=[i for i in default_intervals if i in interval_options],
        format_func=lambda x: interval_labels.get(x, x),
        key=f"collect_interval_multiselect_{exchange_lower}",
    )


def _render_collection_summary(
    selected_tickers: list[str],
    selected_intervals: list[str],
    intervals_list: list[tuple[str, str]],
    exchange: str,
    full_refresh: bool,
) -> None:
    """Render the collection settings summary expander.

    Args:
        selected_tickers: Currently selected tickers
        selected_intervals: Currently selected interval codes
        intervals_list: Available intervals as (code, name) tuples
        exchange: Display name of the exchange
        full_refresh: Whether full refresh is enabled
    """
    with st.expander("📋 Collection Settings Summary", expanded=True):
        sum_col1, sum_col2, sum_col3 = st.columns(3)

        with sum_col1:
            st.markdown("**📈 Selected Tickers**")
            st.write(", ".join(selected_tickers))
            st.metric("Ticker Count", len(selected_tickers))

        with sum_col2:
            st.markdown("**⏱️ Selected Intervals**")
            interval_names = [name for code, name in intervals_list if code in selected_intervals]
            st.write(", ".join(interval_names))
            st.metric("Interval Count", len(selected_intervals))

        with sum_col3:
            st.markdown("**⚙️ Options**")
            st.write(f"Exchange: {exchange}")
            st.write(f"Full Refresh: {'Yes' if full_refresh else 'No'}")
            total_tasks = len(selected_tickers) * len(selected_intervals)
            st.metric("Total Tasks", total_tasks)


def render_data_collect_page() -> None:
    """Render data collection page."""
    st.header("📥 Data Collection")

    # ===== Exchange Selection =====
    exchange = st.selectbox(
        "Exchange",
        options=["Upbit", "Binance"],
        key="collect_exchange",
        help="Select exchange to collect data from",
    )
    exchange_lower: ExchangeName = "binance" if exchange == "Binance" else "upbit"
    tickers_list, intervals_list, default_intervals, custom_placeholder = _get_exchange_config(
        exchange_lower
    )

    # ===== Settings Section =====
    st.subheader("⚙️ Collection Settings")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        selected_tickers = _render_ticker_selection(
            tickers_list, exchange_lower, custom_placeholder, exchange
        )

    with col2:
        selected_intervals = _render_interval_selection(
            intervals_list, default_intervals, exchange_lower
        )

    # Column 3: Options
    with col3:
        st.markdown("### ⚙️ Options")
        full_refresh = st.checkbox(
            "Full Refresh",
            value=False,
            help="Ignore existing data and collect all data from scratch",
        )

        st.divider()

        # Run button
        run_button = st.button(
            "🚀 Start Collection",
            type="primary",
            width="stretch",
            disabled=not selected_tickers or not selected_intervals,
        )

    st.divider()

    # ===== Validation =====
    if not selected_tickers:
        st.warning("⚠️ Please select at least one ticker.")
        return

    if not selected_intervals:
        st.warning("⚠️ Please select at least one interval.")
        return

    # Current settings summary
    _render_collection_summary(
        selected_tickers, selected_intervals, intervals_list, exchange, full_refresh
    )

    # Execute data collection
    if run_button:
        _run_collection(selected_tickers, selected_intervals, full_refresh, exchange_lower)

    # Display previous collection results
    if "collection_results" in st.session_state:
        _display_collection_results()


def _run_collection(
    tickers: list[str],
    intervals: list[str],
    full_refresh: bool,
    exchange: ExchangeName = "upbit",
) -> None:
    """Execute data collection.

    Args:
        tickers: List of tickers to collect
        intervals: List of intervals to collect
        full_refresh: Whether to perform full refresh
        exchange: Exchange name
    """
    st.subheader("📊 Collection Progress")

    # Progress bar
    total_tasks = len(tickers) * len(intervals)
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Store results
    results: dict[str, int] = {}
    completed = 0

    try:
        collector = DataCollectorFactory.create(exchange_name=exchange)

        for ticker in tickers:
            for interval in intervals:
                status_text.text(f"Collecting: {ticker} ({interval})...")

                try:
                    # Use collect method (supports full_refresh)
                    count = collector.collect(
                        ticker=ticker,
                        interval=interval,  # type: ignore[arg-type]
                        full_refresh=full_refresh,
                    )

                    key = f"{ticker}_{interval}"
                    results[key] = count

                except Exception as e:
                    logger.error(f"Error collecting {ticker} {interval}: {e}")
                    results[f"{ticker}_{interval}"] = -1

                completed += 1
                progress_bar.progress(completed / total_tasks)

        status_text.text("Collection completed!")

        # Store results
        st.session_state.collection_results = results

        # Count success/failure
        success_count = sum(1 for v in results.values() if v >= 0)
        fail_count = sum(1 for v in results.values() if v < 0)
        total_candles = sum(v for v in results.values() if v >= 0)

        if fail_count == 0:
            st.success(f"✅ All collections completed! Total {total_candles:,} candles collected")
        else:
            st.warning(f"⚠️ Collection finished: {success_count} succeeded, {fail_count} failed")

    except Exception as e:
        logger.error(f"Collection error: {e}", exc_info=True)
        st.error(f"❌ Error during collection: {e}")


def _display_collection_results() -> None:
    """Display collection results."""
    results = st.session_state.collection_results

    st.subheader("📊 Recent Collection Results")

    # Display results as table
    import pandas as pd

    data = []
    for key, count in results.items():
        parts = key.rsplit("_", 1)
        ticker = parts[0]
        interval = parts[1] if len(parts) > 1 else "unknown"

        status = "✅ Success" if count >= 0 else "❌ Failed"
        candles = f"{count:,}" if count >= 0 else "-"

        data.append(
            {
                "Ticker": ticker,
                "Interval": interval,
                "Status": status,
                "Candles": candles,
            }
        )

    df = pd.DataFrame(data)
    st.dataframe(df, width="stretch", height=400)

    # Summary
    col1, col2, col3 = st.columns(3)
    success_count = sum(1 for v in results.values() if v >= 0)
    fail_count = sum(1 for v in results.values() if v < 0)
    total_candles = sum(v for v in results.values() if v >= 0)

    with col1:
        st.metric("Success", f"{success_count}")
    with col2:
        st.metric("Failed", f"{fail_count}")
    with col3:
        st.metric("Total Candles", f"{total_candles:,}")
