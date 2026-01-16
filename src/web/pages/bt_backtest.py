"""bt Library Backtest page.

Page for running VBO strategy backtest using the external bt library.
Part of the Crypto Quant Ecosystem integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.utils.logger import get_logger
from src.web.services.bt_backtest_runner import (
    BtBacktestResult,
    get_available_bt_symbols,
    is_bt_available,
    run_bt_backtest_service,
)

logger = get_logger(__name__)

__all__ = ["render_bt_backtest_page"]


def render_bt_backtest_page() -> None:
    """Render bt library backtest page."""
    st.header("bt Library VBO Backtest")

    # Check bt library availability
    if not is_bt_available():
        st.error(
            "bt library is not installed. Please install it with:\n\n"
            "```bash\n"
            "pip install -e /path/to/bt\n"
            "```"
        )
        return

    # Create tabs
    if "bt_backtest_result" in st.session_state:
        tab1, tab2 = st.tabs(["Settings", "Results"])
    else:
        tab1 = st.tabs(["Settings"])[0]
        tab2 = None

    # Settings Tab
    with tab1:
        _render_settings_tab()

    # Results Tab
    if tab2 is not None:
        with tab2:
            if "bt_backtest_result" in st.session_state:
                _render_results_tab(st.session_state.bt_backtest_result)


def _render_settings_tab() -> None:
    """Render settings tab."""
    st.subheader("VBO Strategy Settings")

    # Get available symbols
    available_symbols = get_available_bt_symbols("day")

    if not available_symbols:
        st.warning(
            "No data available. Please collect data first:\n\n"
            "```bash\n"
            "python scripts/fetch_data.py --symbols BTC,ETH,XRP,TRX --interval day\n"
            "```"
        )
        return

    # Layout: 3 columns
    col1, col2, col3 = st.columns([1, 1, 1])

    # Column 1: Symbol Selection
    with col1:
        st.markdown("### Asset Selection")

        # Quick selection buttons
        quick_col1, quick_col2 = st.columns(2)
        with quick_col1:
            if st.button("Select All", use_container_width=True):
                st.session_state.bt_selected_symbols = available_symbols
        with quick_col2:
            if st.button("Clear All", use_container_width=True):
                st.session_state.bt_selected_symbols = []

        # Initialize session state
        if "bt_selected_symbols" not in st.session_state:
            st.session_state.bt_selected_symbols = available_symbols[:4]  # Default: first 4

        selected_symbols = st.multiselect(
            "Select Symbols",
            options=available_symbols,
            default=st.session_state.bt_selected_symbols,
            key="bt_symbol_select",
        )
        st.session_state.bt_selected_symbols = selected_symbols

    # Column 2: Trading Config
    with col2:
        st.markdown("### Trading Settings")

        initial_cash = st.number_input(
            "Initial Capital (KRW)",
            min_value=1_000_000,
            max_value=1_000_000_000,
            value=10_000_000,
            step=1_000_000,
            format="%d",
        )

        fee = st.number_input(
            "Fee Rate (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            format="%.2f",
        )

        slippage = st.number_input(
            "Slippage Rate (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            format="%.2f",
        )

    # Column 3: Strategy Parameters
    with col3:
        st.markdown("### VBO Parameters")

        lookback = st.slider(
            "Lookback Period (Short MA)",
            min_value=2,
            max_value=20,
            value=5,
            help="Short-term moving average period",
        )

        multiplier = st.slider(
            "Multiplier (Long MA = Lookback x Multiplier)",
            min_value=1,
            max_value=5,
            value=2,
            help="Multiplier for long-term moving average",
        )

        st.info(f"Long MA Period: {lookback * multiplier}")

    st.markdown("---")

    # Settings Summary
    with st.expander("Settings Summary", expanded=False):
        sum_col1, sum_col2, sum_col3 = st.columns(3)

        with sum_col1:
            st.markdown(
                f"""
                **Assets**
                - Selected: {len(selected_symbols)}
                - Symbols: {", ".join(selected_symbols[:5])}{"..." if len(selected_symbols) > 5 else ""}
                """
            )

        with sum_col2:
            st.markdown(
                f"""
                **Trading**
                - Initial: {initial_cash:,.0f} KRW
                - Fee: {fee:.2f}%
                - Slippage: {slippage:.2f}%
                """
            )

        with sum_col3:
            st.markdown(
                f"""
                **VBO Strategy**
                - Short MA: {lookback}
                - Long MA: {lookback * multiplier}
                """
            )

    # Run Button
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col2:
        run_button = st.button(
            "Run bt VBO Backtest",
            type="primary",
            use_container_width=True,
            disabled=len(selected_symbols) == 0,
        )

    if not selected_symbols:
        st.warning("Please select at least one asset.")
        return

    # Run backtest
    if run_button:
        with st.spinner("Running bt library VBO backtest..."):
            result = run_bt_backtest_service(
                symbols=tuple(selected_symbols),  # tuple for caching
                interval="day",
                initial_cash=initial_cash,
                fee=fee / 100,  # Convert to decimal
                slippage=slippage / 100,
                multiplier=multiplier,
                lookback=lookback,
            )

            if result:
                st.session_state.bt_backtest_result = result
                st.success("Backtest completed! Check the 'Results' tab.")
                st.rerun()
            else:
                st.error("Backtest failed. Check logs for details.")


def _render_results_tab(result: BtBacktestResult) -> None:
    """Render results tab.

    Args:
        result: BtBacktestResult object
    """
    st.subheader("Backtest Results")

    # Metrics Cards
    _render_metrics_cards(result)

    st.markdown("---")

    # Charts Tabs
    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs(
        ["Equity Curve", "Yearly Returns", "Trade History", "Statistics"]
    )

    with chart_tab1:
        _render_equity_chart(result)

    with chart_tab2:
        _render_yearly_chart(result)

    with chart_tab3:
        _render_trade_history(result)

    with chart_tab4:
        _render_statistics(result)


def _render_metrics_cards(result: BtBacktestResult) -> None:
    """Render metrics cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Return",
            f"{result.total_return:,.2f}%",
            delta=f"{result.cagr:.2f}% CAGR",
        )

    with col2:
        st.metric(
            "Max Drawdown",
            f"{result.mdd:.2f}%",
            delta=None,
        )

    with col3:
        st.metric(
            "Sortino Ratio",
            f"{result.sortino_ratio:.2f}",
            delta=f"{result.win_rate:.1f}% Win Rate",
        )

    with col4:
        st.metric(
            "Trades",
            f"{result.num_trades:,}",
            delta=f"PF: {result.profit_factor:.2f}",
        )

    # Second row
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            "Final Equity",
            f"{result.final_equity:,.0f} KRW",
        )

    with col6:
        st.metric(
            "Avg Win",
            f"{result.avg_win:,.0f} KRW",
        )

    with col7:
        st.metric(
            "Avg Loss",
            f"{result.avg_loss:,.0f} KRW",
        )

    with col8:
        st.metric(
            "Profit Factor",
            f"{result.profit_factor:.2f}",
        )


def _render_equity_chart(result: BtBacktestResult) -> None:
    """Render equity curve chart."""
    equity = np.array(result.equity_curve)
    dates = result.dates

    # Calculate drawdown
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax * 100

    # Create subplot
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve (Log Scale)", "Drawdown (%)"),
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity,
            mode="lines",
            name="Equity",
            line={"color": "#2E86AB", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(46, 134, 171, 0.2)",
        ),
        row=1,
        col=1,
    )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line={"color": "#E94F37", "width": 1},
            fill="tozeroy",
            fillcolor="rgba(233, 79, 55, 0.3)",
        ),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        hovermode="x unified",
    )

    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _render_yearly_chart(result: BtBacktestResult) -> None:
    """Render yearly returns bar chart."""
    years = sorted(result.yearly_returns.keys())
    returns = [result.yearly_returns[y] for y in years]

    colors = ["#2ECC71" if r >= 0 else "#E74C3C" for r in returns]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=years,
            y=returns,
            marker_color=colors,
            text=[f"{r:.1f}%" for r in returns],
            textposition="outside",
        )
    )

    # Add average line
    avg_return = np.mean(returns)
    fig.add_hline(
        y=avg_return,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Avg: {avg_return:.1f}%",
        annotation_position="right",
    )

    fig.update_layout(
        title="Yearly Returns",
        xaxis_title="Year",
        yaxis_title="Return (%)",
        height=400,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Yearly returns table
    st.markdown("### Yearly Returns Table")
    yearly_df = pd.DataFrame(
        {
            "Year": years,
            "Return (%)": [f"{r:.2f}%" for r in returns],
        }
    )
    st.dataframe(yearly_df, use_container_width=True, hide_index=True)


def _render_trade_history(result: BtBacktestResult) -> None:
    """Render trade history."""
    if not result.trades:
        st.info("No trades executed.")
        return

    st.markdown(f"### Trade History ({len(result.trades):,} trades)")

    # Convert to DataFrame
    trades_df = pd.DataFrame(result.trades)

    # Format columns
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.strftime("%Y-%m-%d")
    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"]).dt.strftime("%Y-%m-%d")
    trades_df["entry_price"] = trades_df["entry_price"].apply(lambda x: f"{x:,.0f}")
    trades_df["exit_price"] = trades_df["exit_price"].apply(lambda x: f"{x:,.0f}")
    trades_df["pnl"] = trades_df["pnl"].apply(lambda x: f"{x:,.0f}")
    trades_df["return_pct"] = trades_df["return_pct"].apply(lambda x: f"{x:.2f}%")

    # Rename columns for display
    trades_df = trades_df.rename(
        columns={
            "symbol": "Symbol",
            "entry_date": "Entry Date",
            "exit_date": "Exit Date",
            "entry_price": "Entry Price",
            "exit_price": "Exit Price",
            "pnl": "P&L",
            "return_pct": "Return",
        }
    )

    # Display options
    show_count = st.selectbox(
        "Show trades",
        options=[10, 25, 50, 100, "All"],
        index=1,
    )

    display_df = trades_df if show_count == "All" else trades_df.tail(int(str(show_count)))

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_statistics(result: BtBacktestResult) -> None:
    """Render detailed statistics."""
    st.markdown("### Performance Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Return Metrics")
        stats_df = pd.DataFrame(
            {
                "Metric": [
                    "Total Return",
                    "CAGR",
                    "Max Drawdown",
                    "Sortino Ratio",
                ],
                "Value": [
                    f"{result.total_return:,.2f}%",
                    f"{result.cagr:.2f}%",
                    f"{result.mdd:.2f}%",
                    f"{result.sortino_ratio:.2f}",
                ],
            }
        )
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Trade Statistics")
        trade_stats_df = pd.DataFrame(
            {
                "Metric": [
                    "Total Trades",
                    "Win Rate",
                    "Profit Factor",
                    "Avg Win",
                    "Avg Loss",
                ],
                "Value": [
                    f"{result.num_trades:,}",
                    f"{result.win_rate:.2f}%",
                    f"{result.profit_factor:.2f}",
                    f"{result.avg_win:,.0f} KRW",
                    f"{result.avg_loss:,.0f} KRW",
                ],
            }
        )
        st.dataframe(trade_stats_df, use_container_width=True, hide_index=True)

    # Additional metrics
    st.markdown("#### Final Results")
    final_col1, final_col2 = st.columns(2)

    with final_col1:
        st.metric("Initial Capital", "10,000,000 KRW")
    with final_col2:
        st.metric("Final Equity", f"{result.final_equity:,.0f} KRW")


if __name__ == "__main__":
    st.set_page_config(page_title="bt VBO Backtest", layout="wide")
    render_bt_backtest_page()
