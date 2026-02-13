"""Streamlit Backtest UI - Main Entry Point.

Main application for cryptocurrency backtesting web interface.
"""

import sys
from pathlib import Path

# Add project root to sys.path so `src.*` imports work with streamlit
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # Load .env file before anything else

import streamlit as st  # noqa: E402

from src.utils.logger import get_logger, setup_logging  # noqa: E402

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Crypto Lab Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": "https://github.com/11e3/crypto-lab",
        "Report a bug": "https://github.com/11e3/crypto-lab/issues",
        "About": "# Crypto Lab Backtest UI\nEvent-driven backtesting engine based web interface",
    },
)


def main() -> None:
    """Main application entry point."""
    st.title("📊 Crypto Lab Backtest System")

    # Tab-based navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🏠 Home",
            "📥 Data Collection",
            "📈 Backtest",
            "🔧 Optimization",
            "📊 Analysis",
            "🤖 Bot Monitor",
        ]
    )

    with tab1:
        show_home()

    with tab2:
        show_data_collect()

    with tab3:
        show_backtest()

    with tab4:
        show_optimization()

    with tab5:
        show_analysis()

    with tab6:
        show_monitor()


def show_home() -> None:
    """Home page."""
    st.header("🏠 Welcome to Crypto Lab Backtest")

    st.markdown(
        """
    ## 🎯 Key Features

    ### 📥 Data Collection
    - **Upbit API** integration for real-time OHLCV data collection
    - **Various intervals**: Support from 1-minute to monthly candles
    - **Incremental updates**: Add only new data to existing datasets

    ### 📈 Backtest
    - **Event-driven engine** for accurate simulation
    - **Dynamic parameter configuration**: Auto-generated UI based on strategy selection
    - **Multi-asset support**: Backtest multiple cryptocurrencies simultaneously
    - **Real-time metrics**: CAGR, Sharpe, MDD, and 30+ metrics
    - **Interactive charts**: Plotly-based zoomable and pannable charts

    ### 🔧 Parameter Optimization
    - **Grid Search**: Test all parameter combinations
    - **Random Search**: Fast exploration
    - **Parallel processing**: Leverage multi-core CPUs
    - **Metric selection**: Choose from Sharpe, CAGR, Calmar, etc.

    ### 📊 Advanced Analysis
    - **Walk-Forward Analysis**: Prevent overfitting
    - **Monte Carlo**: Risk simulation
    - **VaR/CVaR**: Portfolio risk analysis

    ### 🤖 Bot Monitor
    - **Live positions**: Real-time position tracking
    - **Trade history**: Historical trades from GCS logs
    - **PnL summary**: Daily/weekly/monthly returns
    - **Alerts**: Error notifications and warnings

    ---

    ## 🚀 Getting Started

    1. **📥 Data Collection**: Select ticker and interval to collect data
    2. **📈 Backtest**: Configure strategy and parameters, then run backtest
    3. **🔧 Optimization**: Set parameter grid to find optimal combinations
    4. **📊 Advanced Analysis**: Validate strategy with Monte Carlo and Walk-Forward

    ---

    ## 📚 Supported Strategies

    - **VBO (Volatility Breakout)**: Volatility breakout strategy
    - **Momentum**: Momentum trend following
    - **Mean Reversion**: Mean reversion strategy
    - **Pair Trading**: Pairs trading
    - **ORB (Opening Range Breakout)**: Opening range breakout

    """
    )

    # System status
    st.markdown("### 🔍 System Status")
    try:
        from src.web.services.strategy_registry import StrategyRegistry

        registry = StrategyRegistry()
        strategy_count = len(registry.list_strategies())
    except (ImportError, AttributeError) as e:
        logger.debug(f"Strategy registry unavailable: {e}")
        strategy_count = 0

    try:
        from src.web.config.constants import DATA_COLLECT_TICKERS

        asset_count = len(DATA_COLLECT_TICKERS)
    except (ImportError, AttributeError) as e:
        logger.debug(f"Data collect constants unavailable: {e}")
        asset_count = 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Registered Strategies", str(strategy_count) if strategy_count else "–")
    with col2:
        st.metric("Available Indicators", "20+")
    with col3:
        st.metric("Supported Assets", str(asset_count) if asset_count else "–")


def show_data_collect() -> None:
    """Data collection page."""
    from src.web.pages.data_collect import render_data_collect_page

    render_data_collect_page()


def show_backtest() -> None:
    """Backtest page."""
    from src.web.pages.backtest import render_backtest_page

    render_backtest_page()


def show_optimization() -> None:
    """Parameter optimization page."""
    from src.web.pages.optimization import render_optimization_page

    render_optimization_page()


def show_analysis() -> None:
    """Advanced analysis page."""
    from src.web.pages.analysis import render_analysis_page

    render_analysis_page()


def show_monitor() -> None:
    """Bot monitor page."""
    from src.web.pages.monitor import render_monitor_page

    render_monitor_page()


if __name__ == "__main__":
    main()
