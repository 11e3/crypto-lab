"""Optimization page.

Strategy parameter optimization page.
"""

from typing import Any, cast

import streamlit as st

from src.backtester import BacktestConfig, optimize_strategy_parameters
from src.data.collector_fetch import Interval
from src.strategies.volatility_breakout import create_vbo_strategy
from src.utils.logger import get_logger
from src.web.services.data_loader import validate_data_availability

logger = get_logger(__name__)

__all__ = ["render_optimization_page"]

# Optimization metric options
METRICS = [
    ("sharpe_ratio", "Sharpe Ratio"),
    ("cagr", "CAGR"),
    ("total_return", "Total Return"),
    ("calmar_ratio", "Calmar Ratio"),
    ("win_rate", "Win Rate"),
    ("profit_factor", "Profit Factor"),
]

# Default tickers
DEFAULT_TICKERS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL"]


def render_optimization_page() -> None:
    """Render optimization page."""
    st.header("🔧 Parameter Optimization")

    # ===== Configuration Section =====
    with st.expander("⚙️ Optimization Settings", expanded=True):
        # Row 1: Strategy and Method
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Strategy")
            strategy_type = st.selectbox(
                "Strategy Type",
                options=["vanilla", "legacy"],
                format_func=lambda x: "Vanilla VBO" if x == "vanilla" else "Legacy VBO",
            )

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
                options=[m[0] for m in METRICS],
                format_func=lambda x: next(name for code, name in METRICS if code == x),
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

        # Row 3: Parameter Ranges
        st.subheader("📐 Parameter Ranges")

        col1, col2 = st.columns(2)

        with col1:
            sma_range = st.text_input(
                "SMA Period",
                value="3,4,5,6,7",
                help="Enter comma-separated values (e.g., 3,4,5,6,7)",
            )
            trend_range = st.text_input(
                "Trend SMA Period",
                value="8,10,12,14",
                help="Enter comma-separated values",
            )

        with col2:
            short_noise = st.text_input(
                "Short Noise Period (Optional)",
                value="",
                help="Leave empty to use SMA Period values",
            )
            long_noise = st.text_input(
                "Long Noise Period (Optional)",
                value="",
                help="Leave empty to use Trend SMA Period values",
            )

        st.markdown("---")

        # Row 4: Data Settings
        col1, col2, col3 = st.columns(3)

        with col1:
            interval = st.selectbox(
                "Data Interval",
                options=["minute240", "day", "week"],
                format_func=lambda x: {"minute240": "4 Hours", "day": "Daily", "week": "Weekly"}[x],
                index=1,
            )

        with col2:
            st.subheader("📈 Ticker Selection")
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
            use_container_width=True,
            disabled=not selected_tickers,
        )

    # ===== Main Area =====

    # Validation
    if not selected_tickers:
        st.warning("⚠️ Please select at least one ticker.")
        _show_help()
        return

    # Parse parameter ranges
    try:
        param_grid = _parse_param_grid(sma_range, trend_range, short_noise, long_noise)
    except ValueError as e:
        st.error(f"❌ Parameter range error: {e}")
        return

    # Configuration summary
    _show_config_summary(
        strategy_type, method, metric, param_grid, selected_tickers, interval, n_iter
    )

    # Run optimization
    if run_button:
        _run_optimization(
            strategy_type=strategy_type,
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


def _parse_param_grid(
    sma_range: str,
    trend_range: str,
    short_noise: str,
    long_noise: str,
) -> dict[str, list[int]]:
    """Parse parameter ranges.

    Args:
        sma_range: SMA period range
        trend_range: Trend SMA period range
        short_noise: Short noise period range
        long_noise: Long noise period range

    Returns:
        Parameter grid dictionary

    Raises:
        ValueError: If parsing error occurs
    """

    def parse_range(s: str) -> list[int]:
        if not s.strip():
            return []
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    sma_values = parse_range(sma_range)
    trend_values = parse_range(trend_range)

    if not sma_values:
        raise ValueError("Please enter SMA Period values")
    if not trend_values:
        raise ValueError("Please enter Trend SMA Period values")

    param_grid = {
        "sma_period": sma_values,
        "trend_sma_period": trend_values,
        "short_noise_period": parse_range(short_noise) or sma_values,
        "long_noise_period": parse_range(long_noise) or trend_values,
    }

    return param_grid


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
        - Vanilla VBO: Basic volatility breakout strategy
        - Legacy VBO: Version with noise filter

        **2. Search Method**
        - Grid Search: Tests all combinations (accurate but slow)
        - Random Search: Random sampling (fast but may miss optimal solution)

        **3. Parameter Ranges**
        - Enter comma-separated integer values
        - Example: "3,4,5,6,7"

        **4. Optimization Metrics**
        - Sharpe Ratio: Risk-adjusted return (recommended)
        - CAGR: Compound Annual Growth Rate
        - Calmar Ratio: Return relative to maximum drawdown
        """
    )


def _run_optimization(
    strategy_type: str,
    param_grid: dict[str, list[int]],
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
    """Run optimization."""
    st.subheader("🔄 Optimization in Progress...")

    # Progress indicator
    progress_placeholder = st.empty()
    progress_placeholder.info("Starting optimization...")

    try:
        # Create strategy factory
        def create_strategy(**kwargs: Any) -> Any:
            if strategy_type == "vanilla":
                return create_vbo_strategy(
                    name="VanillaVBO",
                    use_trend_filter=False,
                    use_noise_filter=False,
                    **kwargs,
                )
            else:
                return create_vbo_strategy(
                    name="LegacyVBO",
                    use_trend_filter=True,
                    use_noise_filter=True,
                    **kwargs,
                )

        # Create configuration
        config = BacktestConfig(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=fee_rate,
            max_slots=max_slots,
            use_cache=True,
        )

        progress_placeholder.info("Running backtests... (this may take a while)")

        # Run optimization
        result = optimize_strategy_parameters(
            strategy_factory=create_strategy,
            param_grid=param_grid,
            tickers=tickers,
            interval=interval,
            config=config,
            metric=metric,
            maximize=True,
            method=method,
            n_iter=n_iter,
            n_workers=workers,
        )

        # Save results
        st.session_state.optimization_result = result
        st.session_state.optimization_metric = metric

        progress_placeholder.success("✅ Optimization completed!")

    except Exception as e:
        logger.error(f"Optimization error: {e}", exc_info=True)
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
