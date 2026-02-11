"""Advanced analysis page.

Advanced analysis (Monte Carlo, Walk-Forward) page.
"""

from typing import cast

import streamlit as st

from src.data.collector_fetch import Interval
from src.utils.logger import get_logger
from src.web.components.sidebar.strategy_selector import get_cached_registry
from src.web.config.constants import (
    DEFAULT_TICKERS,
    INTERVAL_DISPLAY_MAP,
    OPTIMIZATION_METRICS,
)
from src.web.services.analysis_service import execute_monte_carlo, execute_walk_forward
from src.web.services.data_loader import validate_data_availability
from src.web.services.strategy_registry import map_strategy_to_internal_type

logger = get_logger(__name__)

__all__ = ["render_analysis_page"]


def render_analysis_page() -> None:
    """Render advanced analysis page."""
    st.header("📊 Advanced Analysis")

    # Analysis type selection
    analysis_type = st.radio(
        "Select Analysis Type",
        options=["monte_carlo", "walk_forward"],
        format_func=lambda x: "🎲 Monte Carlo Simulation"
        if x == "monte_carlo"
        else "📈 Walk-Forward Analysis",
        horizontal=True,
    )

    st.markdown("---")

    if analysis_type == "monte_carlo":
        _render_monte_carlo()
    else:
        _render_walk_forward()


def _render_monte_carlo() -> None:
    """Monte Carlo simulation page."""
    st.subheader("🎲 Monte Carlo Simulation")

    # Get available strategies from registry
    registry = get_cached_registry()
    all_strategies = registry.list_strategies()

    # Filter for non-bt strategies (bt strategies not supported for Monte Carlo yet)
    available_strategies = [s for s in all_strategies if not s.name.startswith("bt_")]
    strategy_names = [s.name for s in available_strategies]

    # ===== Configuration Section =====
    with st.expander("⚙️ Simulation Configuration", expanded=True):
        # First row: Strategy and Simulation Settings
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 📈 Strategy")

            selected_strategy = st.selectbox(
                "Strategy",
                options=strategy_names if strategy_names else ["VanillaVBO"],
                key="mc_strategy_name",
                help="Select a registered strategy",
            )
            strategy_type = map_strategy_to_internal_type(selected_strategy)

            st.markdown("##### 💰 Trading Settings")
            initial_capital = st.number_input(
                "Initial Capital",
                min_value=0.1,
                max_value=100.0,
                value=1.0,
                step=0.1,
                key="mc_capital",
            )
            fee_rate = st.number_input(
                "Fee Rate",
                min_value=0.0,
                max_value=0.01,
                value=0.0005,
                step=0.0001,
                format="%.4f",
                key="mc_fee",
            )
            max_slots = st.slider("Max Slots", min_value=1, max_value=10, value=4, key="mc_slots")

        with col2:
            st.markdown("##### 🎯 Simulation")
            n_simulations = st.slider(
                "Number of Simulations",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                key="mc_n_sim",
            )

            method = st.radio(
                "Simulation Method",
                options=["bootstrap", "parametric"],
                format_func=lambda x: "Bootstrap (Resampling)"
                if x == "bootstrap"
                else "Parametric (Normal Distribution)",
                horizontal=True,
                key="mc_method",
            )

            seed = st.number_input(
                "Random Seed (Optional)",
                min_value=0,
                max_value=99999,
                value=0,
                help="0 for random, otherwise use specific seed for reproducibility",
                key="mc_seed",
            )

        # Second row: Data Settings
        st.markdown("---")
        st.markdown("##### 📊 Data Settings")
        col3, col4 = st.columns(2)

        with col3:
            interval = st.selectbox(
                "Interval",
                options=["minute240", "day", "week"],
                format_func=lambda x: INTERVAL_DISPLAY_MAP[x],
                index=1,
                key="mc_interval",
            )

        with col4:
            available, _ = validate_data_availability(DEFAULT_TICKERS, cast(Interval, interval))
            selected_tickers = st.multiselect(
                "Tickers",
                options=available if available else DEFAULT_TICKERS,
                default=available[:2] if available else [],
                key="mc_tickers",
            )

    # Run button
    run_button = st.button(
        "🚀 Run Simulation",
        type="primary",
        width="stretch",
        disabled=not selected_tickers,
        key="mc_run",
    )

    # ===== Main Content =====
    if not selected_tickers:
        st.warning("⚠️ Please select at least one ticker.")
        _show_monte_carlo_help()
        return

    # Configuration summary
    with st.expander("📋 Simulation Settings Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Strategy**")
            st.write(f"Type: {strategy_type}")
        with col2:
            st.markdown("**Simulation**")
            st.write(f"Count: {n_simulations:,}")
            st.write(f"Method: {method}")
        with col3:
            st.markdown("**Data**")
            st.write(f"Tickers: {', '.join(selected_tickers)}")
            st.write(f"Interval: {interval}")

    # Run simulation
    if run_button:
        _run_monte_carlo(
            strategy_type=strategy_type,
            tickers=selected_tickers,
            interval=interval,
            n_simulations=n_simulations,
            method=method,
            seed=seed if seed > 0 else None,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            max_slots=max_slots,
        )

    # Display results
    if "monte_carlo_result" in st.session_state:
        _display_monte_carlo_results()


def _show_monte_carlo_help() -> None:
    """Monte Carlo help information."""
    st.info(
        """
        ### 🎲 Monte Carlo Simulation Guide

        **What is Monte Carlo Simulation?**
        - Randomly rearranges historical trade results to generate various scenarios
        - Identifies the distribution of strategy risk and expected returns

        **Simulation Methods**
        - **Bootstrap**: Resamples actual trade results (recommended)
        - **Parametric**: Simulates based on normal distribution assumption

        **Interpreting Results**
        - Confidence Interval: 95% probability that returns will fall within this range
        - VaR: Expected loss in worst-case scenario
        """
    )


def _run_monte_carlo(
    strategy_type: str,
    tickers: list[str],
    interval: str,
    n_simulations: int,
    method: str,
    seed: int | None,
    initial_capital: float,
    fee_rate: float,
    max_slots: int,
) -> None:
    """Run Monte Carlo simulation."""
    progress = st.empty()
    progress.info("Running simulation...")

    try:
        mc_result, result = execute_monte_carlo(
            strategy_type=strategy_type,
            tickers=tickers,
            interval=interval,
            n_simulations=n_simulations,
            method=method,
            seed=seed,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            max_slots=max_slots,
        )

        st.session_state.monte_carlo_result = mc_result
        st.session_state.backtest_result_for_mc = result
        progress.success("✅ Simulation completed!")

    except Exception as e:
        logger.error(f"Monte Carlo error: {e}", exc_info=True)
        progress.error(f"❌ Simulation failed: {e}")


def _display_monte_carlo_results() -> None:
    """Display Monte Carlo results."""
    mc_result = st.session_state.monte_carlo_result
    backtest_result = st.session_state.backtest_result_for_mc

    st.subheader("📊 Monte Carlo Results")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Original Return",
            f"{backtest_result.total_return:.2%}",
        )
    with col2:
        st.metric(
            "Avg Simulated CAGR",
            f"{mc_result.mean_cagr:.2f}%",
        )
    with col3:
        # Calculate VaR from percentiles
        var_95 = mc_result.cagr_percentiles.get(5, 0.0)
        st.metric(
            "95% VaR",
            f"{var_95:.2f}%",
            help="Maximum expected loss at 95% confidence level",
        )
    with col4:
        st.metric(
            "Avg Simulated MDD",
            f"{mc_result.mean_mdd:.2f}%",
            help="Average maximum drawdown across simulations",
        )

    # Confidence intervals
    st.markdown("### 📈 CAGR Confidence Intervals")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("5th Percentile", f"{mc_result.cagr_percentiles.get(5, 0.0):.2f}%")
    with col2:
        st.metric("50th Percentile (Median)", f"{mc_result.cagr_percentiles.get(50, 0.0):.2f}%")
    with col3:
        st.metric("95th Percentile", f"{mc_result.cagr_percentiles.get(95, 0.0):.2f}%")

    # Distribution chart
    st.markdown("### 📊 CAGR Distribution")
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=mc_result.simulated_cagrs,
            nbinsx=50,
            name="Simulated CAGRs",
            marker_color="lightblue",
        )
    )

    # Original CAGR line
    fig.add_vline(
        x=backtest_result.cagr,
        line_dash="dash",
        line_color="red",
        annotation_text="Original CAGR",
    )

    fig.update_layout(
        title="Monte Carlo CAGR Distribution",
        xaxis_title="CAGR (%)",
        yaxis_title="Frequency",
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_walk_forward() -> None:
    """Walk-Forward analysis page."""
    st.subheader("📈 Walk-Forward Analysis")

    # Get available strategies from registry
    registry = get_cached_registry()
    all_strategies = registry.list_strategies()

    # Filter for non-bt strategies (Walk-Forward uses internal backtester)
    available_strategies = [s for s in all_strategies if not s.name.startswith("bt_")]
    strategy_names = [s.name for s in available_strategies]

    # ===== Configuration Section =====
    with st.expander("⚙️ Analysis Configuration", expanded=True):
        # First row: Strategy and Period Settings
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 📈 Strategy")

            selected_strategy = st.selectbox(
                "Strategy",
                options=strategy_names if strategy_names else ["VanillaVBO"],
                key="wf_strategy_name",
                help="Select a registered strategy",
            )
            strategy_type = map_strategy_to_internal_type(selected_strategy)

            st.markdown("##### 📅 Period Settings")
            optimization_days = st.slider(
                "Optimization Period (days)",
                min_value=90,
                max_value=730,
                value=365,
                step=30,
                key="wf_opt_days",
            )
            test_days = st.slider(
                "Test Period (days)",
                min_value=30,
                max_value=180,
                value=90,
                step=30,
                key="wf_test_days",
            )
            step_days = st.slider(
                "Step Size (days)",
                min_value=30,
                max_value=180,
                value=90,
                step=30,
                key="wf_step_days",
            )

        with col2:
            st.markdown("##### 📊 Optimization Metric")
            metric = st.selectbox(
                "Optimization Target",
                options=[m[0] for m in OPTIMIZATION_METRICS],
                format_func=lambda x: next(
                    name for code, name in OPTIMIZATION_METRICS if code == x
                ),
                key="wf_metric",
            )

            st.markdown("##### 📐 Parameter Range")
            sma_range = st.text_input(
                "SMA Period",
                value="4,5,6",
                key="wf_sma",
            )
            trend_range = st.text_input(
                "Trend SMA Period",
                value="8,10,12",
                key="wf_trend",
            )

            st.markdown("##### ⚙️ Parallel Processing")
            workers = st.slider(
                "Number of Workers", min_value=1, max_value=8, value=4, key="wf_workers"
            )

        # Second row: Trading and Data Settings
        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("##### 💰 Trading Settings")
            initial_capital = st.number_input(
                "Initial Capital",
                min_value=0.1,
                max_value=100.0,
                value=1.0,
                step=0.1,
                key="wf_capital",
            )
            fee_rate = st.number_input(
                "Fee Rate",
                min_value=0.0,
                max_value=0.01,
                value=0.0005,
                format="%.4f",
                key="wf_fee",
            )
            max_slots = st.slider("Max Slots", min_value=1, max_value=10, value=4, key="wf_slots")

        with col4:
            st.markdown("##### 📊 Data Settings")
            interval = st.selectbox(
                "Interval",
                options=["minute240", "day", "week"],
                format_func=lambda x: INTERVAL_DISPLAY_MAP[x],
                index=1,
                key="wf_interval",
            )

            available, _ = validate_data_availability(DEFAULT_TICKERS, cast(Interval, interval))
            selected_tickers = st.multiselect(
                "Tickers",
                options=available if available else DEFAULT_TICKERS,
                default=available[:2] if available else [],
                key="wf_tickers",
            )

    # Run button
    run_button = st.button(
        "🚀 Run Analysis",
        type="primary",
        width="stretch",
        disabled=not selected_tickers,
        key="wf_run",
    )

    # ===== Main Content =====
    if not selected_tickers:
        st.warning("⚠️ Please select at least one ticker.")
        _show_walk_forward_help()
        return

    # Parse parameters
    try:
        param_grid = {
            "sma_period": [int(x.strip()) for x in sma_range.split(",")],
            "trend_sma_period": [int(x.strip()) for x in trend_range.split(",")],
        }
        param_grid["short_noise_period"] = param_grid["sma_period"]
        param_grid["long_noise_period"] = param_grid["trend_sma_period"]
    except ValueError as e:
        st.error(f"❌ Parameter error: {e}")
        return

    # Configuration summary
    with st.expander("📋 Walk-Forward Settings Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Period Settings**")
            st.write(f"Optimization: {optimization_days} days")
            st.write(f"Test: {test_days} days")
            st.write(f"Step: {step_days} days")
        with col2:
            st.markdown("**Parameters**")
            st.write(f"SMA: {param_grid['sma_period']}")
            st.write(f"Trend: {param_grid['trend_sma_period']}")
        with col3:
            st.markdown("**Data**")
            st.write(f"Tickers: {', '.join(selected_tickers)}")
            st.write(f"Interval: {interval}")

    # Run analysis
    if run_button:
        _run_walk_forward(
            strategy_type=strategy_type,
            param_grid=param_grid,
            tickers=selected_tickers,
            interval=interval,
            optimization_days=optimization_days,
            test_days=test_days,
            step_days=step_days,
            metric=metric,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            max_slots=max_slots,
            workers=workers,
        )

    # Display results
    if "walk_forward_result" in st.session_state:
        _display_walk_forward_results()


def _show_walk_forward_help() -> None:
    """Walk-Forward help information."""
    st.info(
        """
        ### 📈 Walk-Forward Analysis Guide

        **What is Walk-Forward Analysis?**
        - Divides data into multiple periods and tests sequentially
        - Analysis method to prevent overfitting

        **Period Settings**
        - **Optimization Period**: Training period to find parameters
        - **Test Period**: Period to validate found parameters
        - **Step Size**: Window movement interval

        **Interpreting Results**
        - Compare In-Sample vs Out-of-Sample performance
        - WFE (Walk-Forward Efficiency): Closer to 1 is better
        """
    )


def _run_walk_forward(
    strategy_type: str,
    param_grid: dict[str, list[int]],
    tickers: list[str],
    interval: str,
    optimization_days: int,
    test_days: int,
    step_days: int,
    metric: str,
    initial_capital: float,
    fee_rate: float,
    max_slots: int,
    workers: int,
) -> None:
    """Run Walk-Forward analysis."""
    progress = st.empty()
    progress.info("Running Walk-Forward analysis... (this may take a while)")

    try:
        result = execute_walk_forward(
            strategy_type=strategy_type,
            param_grid=param_grid,
            tickers=tickers,
            interval=interval,
            optimization_days=optimization_days,
            test_days=test_days,
            step_days=step_days,
            metric=metric,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            max_slots=max_slots,
            workers=workers,
        )

        st.session_state.walk_forward_result = result
        progress.success("✅ Walk-Forward analysis completed!")

    except Exception as e:
        logger.error(f"Walk-Forward error: {e}", exc_info=True)
        progress.error(f"❌ Analysis failed: {e}")


def _display_walk_forward_results() -> None:
    """Display Walk-Forward results."""
    result = st.session_state.walk_forward_result

    st.subheader("📊 Walk-Forward Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Avg Optimization CAGR",
            f"{result.avg_optimization_cagr:.2f}%",
        )
    with col2:
        st.metric(
            "Avg Test CAGR",
            f"{result.avg_test_cagr:.2f}%",
        )
    with col3:
        # Calculate walk-forward efficiency (test / optimization)
        wfe = (
            result.avg_test_cagr / result.avg_optimization_cagr
            if result.avg_optimization_cagr != 0
            else 0.0
        )
        st.metric(
            "Walk-Forward Efficiency",
            f"{wfe:.2f}",
            help="Closer to 1 indicates less overfitting",
        )
    with col4:
        st.metric(
            "Consistency Rate",
            f"{result.consistency_rate:.1f}%",
            help=f"{result.positive_periods}/{result.total_periods} periods profitable",
        )

    # Additional metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Test Sharpe", f"{result.avg_test_sharpe:.2f}")
    with col2:
        st.metric("Avg Test MDD", f"{result.avg_test_mdd:.2f}%")
    with col3:
        st.metric("Total Periods", f"{result.total_periods}")
    with col4:
        st.metric("Profitable Periods", f"{result.positive_periods}")

    # Period-by-period results table
    st.markdown("### 📋 Period-by-Period Results")

    import pandas as pd

    data = []
    for i, period in enumerate(result.periods):
        # Get optimization result info
        opt_cagr = 0.0
        best_params = {}
        if period.optimization_result and period.optimization_result.best_result:
            opt_cagr = period.optimization_result.best_result.cagr
            best_params = period.optimization_result.best_params

        # Get test result info
        test_cagr = period.test_result.cagr if period.test_result else 0.0

        data.append(
            {
                "Period": i + 1,
                "Opt Start": period.optimization_start.strftime("%Y-%m-%d"),
                "Opt End": period.optimization_end.strftime("%Y-%m-%d"),
                "Test Start": period.test_start.strftime("%Y-%m-%d"),
                "Test End": period.test_end.strftime("%Y-%m-%d"),
                "Best Params": str(best_params),
                "Opt CAGR": f"{opt_cagr:.2f}%",
                "Test CAGR": f"{test_cagr:.2f}%",
            }
        )

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Chart
    st.markdown("### 📈 Period-by-Period CAGR Comparison")

    import plotly.graph_objects as go

    period_nums = list(range(1, len(result.periods) + 1))
    opt_cagrs = []
    test_cagrs = []

    for period in result.periods:
        if period.optimization_result and period.optimization_result.best_result:
            opt_cagrs.append(period.optimization_result.best_result.cagr)
        else:
            opt_cagrs.append(0.0)
        test_cagrs.append(period.test_result.cagr if period.test_result else 0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Optimization (In-Sample)",
            x=period_nums,
            y=opt_cagrs,
            marker_color="lightblue",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Test (Out-of-Sample)",
            x=period_nums,
            y=test_cagrs,
            marker_color="orange",
        )
    )

    fig.update_layout(
        title="Walk-Forward Period CAGR Comparison",
        xaxis_title="Period",
        yaxis_title="CAGR (%)",
        barmode="group",
    )

    st.plotly_chart(fig, use_container_width=True)
