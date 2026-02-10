"""Data collection page.

Page for data collection execution and status display.
"""

import streamlit as st

from src.data.collector_factory import DataCollectorFactory
from src.utils.logger import get_logger
from src.web.config.constants import DATA_COLLECT_INTERVALS, DATA_COLLECT_TICKERS

logger = get_logger(__name__)

__all__ = ["render_data_collect_page"]

# Local aliases
DEFAULT_TICKERS = DATA_COLLECT_TICKERS
INTERVALS = DATA_COLLECT_INTERVALS


def render_data_collect_page() -> None:
    """Render data collection page."""
    st.header("📥 Data Collection")

    # ===== Settings Section =====
    st.subheader("⚙️ Collection Settings")

    col1, col2, col3 = st.columns([2, 2, 1])

    # Column 1: Ticker Selection
    with col1:
        st.markdown("### 📈 Ticker Selection")

        # Quick selection buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Select All", key="select_all_tickers"):
                st.session_state.collect_selected_tickers = list(DEFAULT_TICKERS)
                st.rerun()
        with btn_col2:
            if st.button("Deselect All", key="deselect_all_tickers"):
                st.session_state.collect_selected_tickers = []
                st.rerun()

        # Ticker multiselect
        if "collect_selected_tickers" not in st.session_state:
            st.session_state.collect_selected_tickers = DEFAULT_TICKERS[:6]

        selected_tickers = st.multiselect(
            "Select Tickers",
            options=DEFAULT_TICKERS,
            default=st.session_state.collect_selected_tickers,
            key="collect_ticker_multiselect",
        )

        # Custom ticker input
        if "custom_collect_tickers" not in st.session_state:
            st.session_state.custom_collect_tickers = []

        custom_ticker = st.text_input(
            "Add Custom Ticker",
            placeholder="e.g., KRW-MATIC",
            help="Enter KRW market ticker supported by Upbit",
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

    # Column 2: Interval Selection
    with col2:
        st.markdown("### ⏱️ Interval Selection")

        # Initialize interval checkbox states if not exists
        if "intervals_initialized" not in st.session_state:
            st.session_state.intervals_initialized = True
            default_intervals = ["minute240", "day", "week"]
            for interval_code, _ in INTERVALS:
                st.session_state[f"interval_{interval_code}"] = interval_code in default_intervals

        selected_intervals = []
        for interval_code, interval_name in INTERVALS:
            if st.checkbox(f"{interval_name} ({interval_code})", key=f"interval_{interval_code}"):
                selected_intervals.append(interval_code)

    # Column 3: Options
    with col3:
        st.markdown("### ⚙️ Options")
        full_refresh = st.checkbox(
            "Full Refresh",
            value=False,
            help="Ignore existing data and collect all data from scratch",
        )

        st.markdown("---")

        # Run button
        run_button = st.button(
            "🚀 Start Collection",
            type="primary",
            width="stretch",
            disabled=not selected_tickers or not selected_intervals,
        )

    st.markdown("---")

    # ===== Validation =====
    if not selected_tickers:
        st.warning("⚠️ Please select at least one ticker.")
        return

    if not selected_intervals:
        st.warning("⚠️ Please select at least one interval.")
        return

    # Current settings summary
    with st.expander("📋 Collection Settings Summary", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📈 Selected Tickers**")
            st.write(", ".join(selected_tickers))
            st.metric("Ticker Count", len(selected_tickers))

        with col2:
            st.markdown("**⏱️ Selected Intervals**")
            interval_names = [name for code, name in INTERVALS if code in selected_intervals]
            st.write(", ".join(interval_names))
            st.metric("Interval Count", len(selected_intervals))

        with col3:
            st.markdown("**⚙️ Options**")
            st.write(f"Full Refresh: {'Yes' if full_refresh else 'No'}")
            total_tasks = len(selected_tickers) * len(selected_intervals)
            st.metric("Total Tasks", total_tasks)

    # Execute data collection
    if run_button:
        _run_collection(selected_tickers, selected_intervals, full_refresh)

    # Display previous collection results
    if "collection_results" in st.session_state:
        _display_collection_results()


def _run_collection(
    tickers: list[str],
    intervals: list[str],
    full_refresh: bool,
) -> None:
    """Execute data collection.

    Args:
        tickers: List of tickers to collect
        intervals: List of intervals to collect
        full_refresh: Whether to perform full refresh
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
        collector = DataCollectorFactory.create()

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
