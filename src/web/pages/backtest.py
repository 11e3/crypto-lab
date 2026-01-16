"""Backtest page.

Page for backtest execution and result display.
"""

from __future__ import annotations

from datetime import date as date_type
from typing import TYPE_CHECKING

import numpy as np
import streamlit as st

from src.backtester.models import BacktestConfig, BacktestResult
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.web.components.sidebar.trading_config import TradingConfig
    from src.web.services.bt_backtest_runner import BtBacktestResult
from src.web.components.charts.equity_curve import render_equity_curve
from src.web.components.charts.monthly_heatmap import render_monthly_heatmap
from src.web.components.charts.underwater import render_underwater_curve
from src.web.components.charts.yearly_bar import render_yearly_bar_chart
from src.web.components.metrics.metrics_display import (
    render_metrics_cards,
    render_statistical_significance,
)
from src.web.components.sidebar.asset_selector import render_asset_selector
from src.web.components.sidebar.date_config import render_date_config
from src.web.components.sidebar.strategy_selector import render_strategy_selector
from src.web.components.sidebar.trading_config import render_trading_config
from src.web.services.backtest_runner import run_backtest_service
from src.web.services.data_loader import get_data_files, validate_data_availability
from src.web.services.metrics_calculator import calculate_extended_metrics

logger = get_logger(__name__)

__all__ = ["render_backtest_page"]


def render_backtest_page() -> None:
    """Render backtest page (tab-based UI)."""
    st.header("📈 Backtest")

    # Create tabs: Settings and Results
    if "backtest_result" in st.session_state or "bt_backtest_result" in st.session_state:
        # Show both settings and results tabs if results exist
        tab1, tab2 = st.tabs(["⚙️ Settings", "📊 Results"])
    else:
        # Show only settings tab if no results
        tab1 = st.tabs(["⚙️ Settings"])[0]
        tab2 = None

    # ===== Settings Tab =====
    with tab1:
        _render_settings_tab()

    # ===== Results Tab =====
    if tab2 is not None:
        with tab2:
            # Check which result to display
            if "bt_backtest_result" in st.session_state:
                _display_bt_results(st.session_state.bt_backtest_result)
            elif "backtest_result" in st.session_state:
                _display_results(st.session_state.backtest_result)
            else:
                st.info("Run backtest to see results here.")


def _render_settings_tab() -> None:
    """Render settings tab."""
    st.subheader("⚙️ Backtest Settings")

    # Split settings into 3 columns
    col1, col2, col3 = st.columns([1, 1, 1])

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

    # ===== Column 3: Asset Selection =====
    with col3:
        st.markdown("### 🪙 Asset Selection")
        selected_tickers = render_asset_selector()

    st.markdown("---")

    # Settings Summary
    with st.expander("📋 Settings Summary", expanded=False):
        _show_config_summary(strategy_name, selected_tickers, trading_config, start_date, end_date)

    # Run Button and Clear Cache
    col_left, col_center, col_right = st.columns([1, 1, 1])
    with col_center:
        run_button = st.button(
            "🚀 Run Backtest",
            type="primary",
            use_container_width=True,
            disabled=not strategy_name or not selected_tickers,
        )
    with col_right:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            if "backtest_result" in st.session_state:
                del st.session_state.backtest_result
            if "bt_backtest_result" in st.session_state:
                del st.session_state.bt_backtest_result
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
        from src.web.services.strategy_registry import is_bt_strategy

        if is_bt_strategy(strategy_name):
            _run_bt_backtest(
                strategy_params=strategy_params,
                available_tickers=available_tickers,
                trading_config=trading_config,
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
    strategy_params: dict,
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
            if "bt_backtest_result" in st.session_state:
                del st.session_state.bt_backtest_result
            st.session_state.backtest_result = result
            st.success("Backtest completed! Check the 'Results' tab.")
            st.rerun()
        else:
            st.error("Backtest execution failed")


def _run_bt_backtest(
    strategy_params: dict,
    available_tickers: list[str],
    trading_config: TradingConfig,
) -> None:
    """Run backtest using bt library."""
    from src.web.services.bt_backtest_runner import run_bt_backtest_service

    with st.spinner("Running bt library VBO backtest..."):
        # Convert tickers: KRW-BTC -> BTC
        symbols = [t.replace("KRW-", "") for t in available_tickers]

        result = run_bt_backtest_service(
            symbols=tuple(symbols),
            interval="day",
            initial_cash=int(trading_config.initial_capital),
            fee=trading_config.fee_rate,
            slippage=trading_config.slippage_rate,
            multiplier=strategy_params.get("multiplier", 2),
            lookback=strategy_params.get("lookback", 5),
        )

        if result:
            # Clear event-driven result if exists
            if "backtest_result" in st.session_state:
                del st.session_state.backtest_result
            st.session_state.bt_backtest_result = result
            st.success("bt VBO Backtest completed! Check the 'Results' tab.")
            st.rerun()
        else:
            st.error("bt Backtest execution failed")


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


def _display_results(result: BacktestResult) -> None:
    """Display backtest results.

    Args:
        result: BacktestResult object
    """
    st.subheader("📊 Backtest Results")

    # Extract trade returns
    trade_returns = [t.pnl_pct / 100 for t in result.trades if t.pnl_pct is not None]

    # Calculate extended metrics (cached in session state)
    equity = np.array(result.equity_curve)
    dates = np.array(result.dates) if hasattr(result, "dates") else np.arange(len(equity))

    # Generate cache key (cache metrics by equity hash)
    cache_key = f"metrics_{hash(equity.tobytes())}"

    if cache_key not in st.session_state:
        # Calculate metrics (only once)
        st.session_state[cache_key] = calculate_extended_metrics(
            equity=equity,
            trade_returns=trade_returns,
        )

    extended_metrics = st.session_state[cache_key]

    # Tab configuration
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📈 Overview",
            "📊 Equity Curve",
            "📉 Drawdown",
            "📅 Monthly Analysis",
            "📆 Yearly Analysis",
            "🔬 Statistics",
        ]
    )

    with tab1:
        # Metrics cards
        render_metrics_cards(extended_metrics)

        # Trade history
        if result.trades:
            st.markdown("### 📋 Trade History")

            import pandas as pd

            trades_df = pd.DataFrame(
                [
                    {
                        "Ticker": t.ticker,
                        "Entry Date": str(t.entry_date),
                        "Entry Price": f"{t.entry_price:,.0f}",
                        "Exit Date": str(t.exit_date) if t.exit_date else "-",
                        "Exit Price": f"{t.exit_price:,.0f}" if t.exit_price else "-",
                        "P&L": f"{t.pnl:,.0f}",
                        "P&L %": f"{t.pnl_pct:.2f}%",
                    }
                    for t in result.trades[-100:]  # Last 100 trades only
                ]
            )

            st.dataframe(trades_df, width="stretch", height=400)

    with tab2:
        render_equity_curve(dates, equity)

    with tab3:
        render_underwater_curve(dates, equity)

    with tab4:
        render_monthly_heatmap(dates, equity)

    with tab5:
        render_yearly_bar_chart(dates, equity)

    with tab6:
        render_statistical_significance(extended_metrics)


def _display_bt_results(result: BtBacktestResult) -> None:
    """Display bt library backtest results.

    Args:
        result: BtBacktestResult object
    """

    st.subheader("📊 bt VBO Backtest Results")

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Return",
            f"{result.total_return:,.2f}%",
            delta=f"{result.cagr:.2f}% CAGR",
        )

    with col2:
        st.metric("Max Drawdown", f"{result.mdd:.2f}%")

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
        st.metric("Final Equity", f"{result.final_equity:,.0f} KRW")

    with col6:
        st.metric("Avg Win", f"{result.avg_win:,.0f} KRW")

    with col7:
        st.metric("Avg Loss", f"{result.avg_loss:,.0f} KRW")

    with col8:
        st.metric("Profit Factor", f"{result.profit_factor:.2f}")

    st.markdown("---")

    # Tab configuration
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Equity Curve", "📆 Yearly Returns", "📋 Trade History", "🔬 Statistics"]
    )

    with tab1:
        _render_bt_equity_chart(result)

    with tab2:
        _render_bt_yearly_chart(result)

    with tab3:
        _render_bt_trade_history(result)

    with tab4:
        _render_bt_statistics(result)


def _render_bt_equity_chart(result: BtBacktestResult) -> None:
    """Render bt equity curve chart."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    equity = np.array(result.equity_curve)
    dates = result.dates

    # Calculate drawdown
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax * 100

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve (Log Scale)", "Drawdown (%)"),
    )

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

    fig.update_layout(height=600, showlegend=False, hovermode="x unified")
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _render_bt_yearly_chart(result: BtBacktestResult) -> None:
    """Render bt yearly returns chart."""
    import pandas as pd
    import plotly.graph_objects as go

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

    # Table
    yearly_df = pd.DataFrame({"Year": years, "Return (%)": [f"{r:.2f}%" for r in returns]})
    st.dataframe(yearly_df, use_container_width=True, hide_index=True)


def _render_bt_trade_history(result: BtBacktestResult) -> None:
    """Render bt trade history."""
    import pandas as pd

    if not result.trades:
        st.info("No trades executed.")
        return

    st.markdown(f"### Trade History ({len(result.trades):,} trades)")

    trades_df = pd.DataFrame(result.trades)
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.strftime("%Y-%m-%d")
    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"]).dt.strftime("%Y-%m-%d")
    trades_df["entry_price"] = trades_df["entry_price"].apply(lambda x: f"{x:,.0f}")
    trades_df["exit_price"] = trades_df["exit_price"].apply(lambda x: f"{x:,.0f}")
    trades_df["pnl"] = trades_df["pnl"].apply(lambda x: f"{x:,.0f}")
    trades_df["return_pct"] = trades_df["return_pct"].apply(lambda x: f"{x:.2f}%")

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

    show_count = st.selectbox(
        "Show trades", options=[10, 25, 50, 100, "All"], index=1, key="bt_trade_count"
    )
    display_df = trades_df if show_count == "All" else trades_df.tail(int(str(show_count)))
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_bt_statistics(result: BtBacktestResult) -> None:
    """Render bt statistics."""
    import pandas as pd

    st.markdown("### Performance Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Return Metrics")
        stats_df = pd.DataFrame(
            {
                "Metric": ["Total Return", "CAGR", "Max Drawdown", "Sortino Ratio"],
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
                "Metric": ["Total Trades", "Win Rate", "Profit Factor", "Avg Win", "Avg Loss"],
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
