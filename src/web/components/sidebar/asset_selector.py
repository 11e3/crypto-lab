"""Asset selector component.

Multi-asset selection UI component.
"""

import streamlit as st

__all__ = ["render_asset_selector", "get_available_tickers"]


# Major cryptocurrency list
POPULAR_TICKERS = [
    "KRW-BTC",
    "KRW-ETH",
    "KRW-XRP",
    "KRW-ADA",
    "KRW-SOL",
    "KRW-DOGE",
    "KRW-AVAX",
    "KRW-DOT",
    "KRW-MATIC",
    "KRW-LINK",
    "KRW-TRX",
    "KRW-ATOM",
    "KRW-ETC",
    "KRW-NEO",
    "KRW-WAVES",
]


def get_available_tickers() -> list[str]:
    """Return list of available tickers.

    Returns:
        List of ticker strings
    """
    # TODO: In practice, scan data directory or call Upbit API
    return POPULAR_TICKERS


def render_asset_selector() -> list[str]:
    """Render asset selection UI.

    Returns:
        List of selected tickers
    """
    st.subheader("🪙 Asset Selection")

    available_tickers = get_available_tickers()

    # Selection mode
    selection_mode = st.radio(
        "Selection Mode",
        ["Quick Select", "Individual Select"],
        horizontal=True,
        help="Quick Select: Use presets | Individual Select: Check manually",
    )

    selected_tickers: list[str] = []

    if selection_mode == "Quick Select":
        # Preset selection
        presets = {
            "Top 3 (BTC, ETH, XRP)": ["KRW-BTC", "KRW-ETH", "KRW-XRP"],
            "Top 5": ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-ADA", "KRW-SOL"],
            "Top 10": available_tickers[:10],
            "All": available_tickers,
            "Custom": [],
        }

        preset_name = st.selectbox(
            "Select Preset",
            options=list(presets.keys()),
            help="Predefined asset groups",
        )

        selected_tickers = presets[preset_name]

        if preset_name == "Custom":
            # Custom selection
            selected_tickers = st.multiselect(
                "Select Assets",
                options=available_tickers,
                default=["KRW-BTC", "KRW-ETH"],
                help="Select assets to include in backtest",
            )

    else:
        # Individual selection (checkboxes)
        st.caption("Select with checkboxes:")

        # Grid layout (3 columns)
        cols_per_row = 3
        for i in range(0, len(available_tickers), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(available_tickers):
                    ticker = available_tickers[idx]
                    with col:
                        if st.checkbox(ticker, key=f"ticker_{ticker}"):
                            selected_tickers.append(ticker)

    # Store selected count in session_state for other components
    st.session_state["selected_ticker_count"] = len(selected_tickers)

    # Display selection result
    if selected_tickers:
        st.success(f"✅ Selected assets: **{len(selected_tickers)}**")
        st.caption(", ".join(selected_tickers))
    else:
        st.warning("⚠️ Please select at least one asset.")

    return selected_tickers
