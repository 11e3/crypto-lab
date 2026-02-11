"""Optimization page.

Strategy parameter optimization page.
"""

from typing import Any, cast

import streamlit as st

from src.data.collector_fetch import Interval
from src.utils.logger import get_logger
from src.web.components.sidebar.strategy_selector import get_cached_registry
from src.web.config.constants import DEFAULT_TICKERS, INTERVAL_DISPLAY_MAP, OPTIMIZATION_METRICS
from src.web.services.bt_backtest_runner import get_available_bt_symbols
from src.web.services.data_loader import validate_data_availability
from src.web.services.optimization_service import (
    execute_bt_optimization,
    execute_native_optimization,
    get_default_param_range,
    parse_dynamic_param_grid,
)
from src.web.services.strategy_registry import is_bt_strategy

logger = get_logger(__name__)

__all__ = ["render_optimization_page"]


def render_optimization_page() -> None:
    """Render optimization page."""
    st.header("🔧 Parameter Optimization")

    # Get strategy registry (same as backtest page)
    registry = get_cached_registry()
    all_strategies = registry.list_strategies()

    # Separate bt and non-bt strategies
    native_strategies = [s for s in all_strategies if not is_bt_strategy(s.name)]
    bt_strategies = [s for s in all_strategies if is_bt_strategy(s.name)]

    # Combine all strategies (native first, then bt)
    strategies = native_strategies + bt_strategies

    if not strategies:
        st.error("⚠️ No strategies available for optimization.")
        return

    # ===== Configuration Section =====
    with st.expander("⚙️ Optimization Settings", expanded=True):
        # Row 1: Strategy and Method
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Strategy")
            strategy_names = [s.name for s in strategies]

            # Format strategy names to show engine type
            def format_strategy_name(name: str) -> str:
                if is_bt_strategy(name):
                    return f"{name} [bt]"
                return name

            selected_strategy_name = st.selectbox(
                "Strategy",
                options=strategy_names,
                format_func=format_strategy_name,
                help="Select strategy to optimize. [bt] strategies use bt library backtest engine.",
            )

            # Get selected strategy info
            selected_strategy = registry.get_strategy(selected_strategy_name)
            is_bt = is_bt_strategy(selected_strategy_name)

            if selected_strategy and selected_strategy.description:
                st.caption(f"ℹ️ {selected_strategy.description}")

        with col2:
            st.subheader("⚙️ Optimization Method")
            method = st.radio(
                "Search Method",
                options=["grid", "random"],
                format_func=lambda x: "Grid Search (Full exploration)"
                if x == "grid"
                else "Random Search (Random sampling)",
                horizontal=True,
            )

            if method == "random":
                n_iter = st.slider(
                    "Number of Iterations", min_value=10, max_value=500, value=100, step=10
                )
            else:
                n_iter = 100  # Not used in grid search

        st.markdown("---")

        # Row 2: Metric and Trading Settings
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Optimization Metric")
            metric = st.selectbox(
                "Optimization Target",
                options=[m[0] for m in OPTIMIZATION_METRICS],
                format_func=lambda x: next(
                    name for code, name in OPTIMIZATION_METRICS if code == x
                ),
                index=0,
            )

        with col2:
            st.subheader("💰 Trading Settings")
            initial_capital = st.number_input(
                "Initial Capital",
                min_value=0.1,
                max_value=100.0,
                value=1.0,
                step=0.1,
                format="%.1f",
            )
            fee_rate = st.number_input(
                "Fee Rate",
                min_value=0.0,
                max_value=0.01,
                value=0.0005,
                step=0.0001,
                format="%.4f",
            )
            max_slots = st.slider("Maximum Slots", min_value=1, max_value=10, value=4)

        st.markdown("---")

        # Row 3: Parameter Ranges (dynamically generated from strategy)
        st.subheader("📐 Parameter Ranges")

        param_ranges: dict[str, str] = {}
        if selected_strategy and selected_strategy.parameters:
            # Create dynamic input fields for each parameter
            params_list = list(selected_strategy.parameters.items())
            n_params = len(params_list)

            if n_params > 0:
                # Create two columns for parameters
                col1, col2 = st.columns(2)
                for i, (param_name, spec) in enumerate(params_list):
                    target_col = col1 if i % 2 == 0 else col2
                    with target_col:
                        label = param_name.replace("_", " ").title()
                        default_values = get_default_param_range(spec)
                        param_ranges[param_name] = st.text_input(
                            label,
                            value=default_values,
                            help=f"{spec.description or param_name} - Enter comma-separated values",
                            key=f"opt_param_{param_name}",
                        )
            else:
                st.info("📌 This strategy has no configurable parameters.")
        else:
            st.warning("⚠️ No strategy selected or no parameters available.")

        st.markdown("---")

        # Row 4: Data Settings
        col1, col2, col3 = st.columns(3)

        with col1:
            interval = st.selectbox(
                "Data Interval",
                options=["minute240", "day", "week"],
                format_func=lambda x: INTERVAL_DISPLAY_MAP[x],
                index=1,
            )

        with col2:
            st.subheader("📈 Ticker/Symbol Selection")

            if is_bt:
                # bt strategies use symbol names without KRW- prefix
                bt_interval = "day" if interval == "day" else "day"  # bt only supports day
                available_bt_symbols = get_available_bt_symbols(bt_interval)

                if not available_bt_symbols:
                    st.warning("⚠️ No data available for bt backtest.")

                selected_symbols = st.multiselect(
                    "Symbols",
                    options=available_bt_symbols,
                    default=available_bt_symbols[:4] if available_bt_symbols else [],
                    help="Select symbols for bt backtest (without KRW- prefix)",
                )
                # Convert to tickers format for consistency
                selected_tickers = [f"KRW-{s}" for s in selected_symbols]
            else:
                # Native strategies use full ticker names
                available, missing = validate_data_availability(
                    DEFAULT_TICKERS, cast(Interval, interval)
                )

                selected_tickers = st.multiselect(
                    "Tickers",
                    options=available if available else DEFAULT_TICKERS,
                    default=available[:2] if available else [],
                )

        with col3:
            workers = st.slider(
                "Parallel Workers",
                min_value=1,
                max_value=8,
                value=4,
                help="Adjust according to your CPU cores",
            )

        st.markdown("---")

        # Run Button
        run_button = st.button(
            "🚀 Run Optimization",
            type="primary",
            width="stretch",
            disabled=not selected_tickers,
        )

    # ===== Main Area =====

    # Validation
    if not selected_tickers:
        st.warning("⚠️ Please select at least one ticker.")
        _show_help()
        return

    if not selected_strategy:
        st.warning("⚠️ Please select a strategy.")
        return

    # Parse parameter ranges (dynamic based on strategy parameters)
    try:
        param_grid = parse_dynamic_param_grid(param_ranges, selected_strategy.parameters)
    except ValueError as e:
        st.error(f"❌ Parameter range error: {e}")
        return

    # Configuration summary
    _show_config_summary(
        selected_strategy_name, method, metric, param_grid, selected_tickers, interval, n_iter
    )

    # Run optimization
    if run_button:
        if is_bt:
            # bt strategy optimization
            _run_bt_optimization(
                strategy_name=selected_strategy_name,
                param_grid=param_grid,
                symbols=[t.replace("KRW-", "") for t in selected_tickers],
                metric=metric,
                method=method,
                n_iter=n_iter,
                initial_capital=int(initial_capital * 10_000_000),  # Convert to KRW
                fee_rate=fee_rate,
            )
        else:
            # Native strategy optimization
            _run_optimization(
                strategy_name=selected_strategy_name,
                strategy_class=selected_strategy.strategy_class,
                param_grid=param_grid,
                tickers=selected_tickers,
                interval=interval,
                metric=metric,
                method=method,
                n_iter=n_iter,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                max_slots=max_slots,
                workers=workers,
            )

    # Display previous results
    if "optimization_result" in st.session_state:
        _display_optimization_results()


def _show_config_summary(
    strategy_type: str,
    method: str,
    metric: str,
    param_grid: dict[str, list[int]],
    tickers: list[str],
    interval: str,
    n_iter: int,
) -> None:
    """Display configuration summary."""
    with st.expander("📋 Optimization Configuration Summary", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📈 Strategy & Method**")
            st.write(f"- Strategy: {strategy_type}")
            st.write(f"- Method: {method}")
            st.write(f"- Metric: {metric}")

        with col2:
            st.markdown("**📐 Parameter Ranges**")
            for key, values in param_grid.items():
                st.write(f"- {key}: {values}")

        with col3:
            st.markdown("**📊 Data**")
            st.write(f"- Tickers: {', '.join(tickers)}")
            st.write(f"- Interval: {interval}")

            # Calculate total combinations
            if method == "grid":
                total_combinations = 1
                for values in param_grid.values():
                    total_combinations *= len(values)
                st.metric("Total Combinations", f"{total_combinations:,}")
            else:
                st.metric("Search Iterations", f"{n_iter}")


def _show_help() -> None:
    """Display help information."""
    st.info(
        """
        ### 🔧 Parameter Optimization Guide

        **1. Strategy Selection**
        - Select any strategy from the dropdown (same as backtest page)
        - bt library strategies are not supported for optimization

        **2. Search Method**
        - Grid Search: Tests all combinations (accurate but slow)
        - Random Search: Random sampling (fast but may miss optimal solution)

        **3. Parameter Ranges**
        - Enter comma-separated values for each parameter
        - Example: "3,4,5,6,7" for integers
        - Example: "0.1,0.2,0.3" for floats

        **4. Optimization Metrics**
        - Sharpe Ratio: Risk-adjusted return (recommended)
        - CAGR: Compound Annual Growth Rate
        - Calmar Ratio: Return relative to maximum drawdown
        """
    )


def _run_optimization(
    strategy_name: str,
    strategy_class: type | None,
    param_grid: dict[str, list[Any]],
    tickers: list[str],
    interval: str,
    metric: str,
    method: str,
    n_iter: int,
    initial_capital: float,
    fee_rate: float,
    max_slots: int,
    workers: int,
) -> None:
    """Run native strategy optimization."""
    st.subheader("🔄 Optimization in Progress...")
    progress_placeholder = st.empty()
    progress_placeholder.info("Running backtests... (this may take a while)")

    if strategy_class is None:
        progress_placeholder.error("❌ Strategy class not found")
        return

    try:
        result = execute_native_optimization(
            strategy_class=strategy_class,
            param_grid=param_grid,
            tickers=tickers,
            interval=interval,
            metric=metric,
            method=method,
            n_iter=n_iter,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            max_slots=max_slots,
            workers=workers,
        )

        st.session_state.optimization_result = result
        st.session_state.optimization_metric = metric
        progress_placeholder.success("✅ Optimization completed!")

    except Exception as e:
        logger.error(f"Optimization error: {e}", exc_info=True)
        progress_placeholder.error(f"❌ Optimization failed: {e}")


def _run_bt_optimization(
    strategy_name: str,
    param_grid: dict[str, list[Any]],
    symbols: list[str],
    metric: str,
    method: str,
    n_iter: int,
    initial_capital: int,
    fee_rate: float,
) -> None:
    """Run bt strategy optimization."""
    st.subheader("🔄 bt Optimization in Progress...")
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    progress_placeholder.info("Starting bt optimization...")

    def on_progress(current: int, total: int) -> None:
        progress_bar.progress(current / total)
        progress_placeholder.info(f"Running backtests... ({current}/{total})")

    try:
        result_obj = execute_bt_optimization(
            strategy_name=strategy_name,
            param_grid=param_grid,
            symbols=symbols,
            metric=metric,
            method=method,
            n_iter=n_iter,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            on_progress=on_progress,
        )

        st.session_state.optimization_result = result_obj
        st.session_state.optimization_metric = metric
        progress_bar.progress(1.0)
        progress_placeholder.success(
            f"✅ bt Optimization completed! Best {metric}: {result_obj.best_score:.4f}"
        )

    except RuntimeError as e:
        progress_placeholder.error(f"❌ {e}")
    except Exception as e:
        logger.error(f"bt Optimization error: {e}", exc_info=True)
        progress_placeholder.error(f"❌ Optimization failed: {e}")


def _display_optimization_results() -> None:
    """Display optimization results."""
    result = st.session_state.optimization_result
    metric = st.session_state.optimization_metric

    st.subheader("📊 Optimization Results")

    # Best parameters
    st.markdown("### 🏆 Best Parameters")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Parameters**")
        for key, value in result.best_params.items():
            st.write(f"- {key}: **{value}**")

    with col2:
        st.markdown("**Performance**")
        st.metric(f"Best {metric}", f"{result.best_score:.4f}")

    # Full results table
    st.markdown("### 📋 All Results")

    import pandas as pd

    # Convert results to DataFrame
    data = []
    for params, score in zip(result.all_params, result.all_scores, strict=False):
        row = params.copy()
        row[metric] = score
        data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values(metric, ascending=False)

    st.dataframe(df, width="stretch", height=400)

    # Top 10 results
    st.markdown("### 🔝 Top 10 Results")
    top_10 = df.head(10)
    st.dataframe(top_10, width="stretch")
