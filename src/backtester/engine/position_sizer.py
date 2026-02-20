"""
Position sizing calculations for backtesting.

Handles portfolio optimization and position size calculations.
"""

from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.backtester.models import BacktestConfig
from src.risk.portfolio_optimization import optimize_portfolio
from src.risk.position_sizing import calculate_multi_asset_position_sizes
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.backtester.engine.trade_simulator import SimulationState

logger = get_logger(__name__)


def calculate_position_sizes_for_entries(
    state: "SimulationState",
    config: BacktestConfig,
    candidate_idx: np.ndarray,
    tickers: list[str],
    entry_prices: np.ndarray,
    d_idx: int,
    current_date: date,
    ticker_historical_data: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """
    Calculate position sizes for candidate entries.

    Args:
        state: Current simulation state
        config: Backtest configuration
        candidate_idx: Indices of candidate entries
        tickers: List of ticker symbols
        entry_prices: Entry prices array
        d_idx: Current date index
        current_date: Current date
        ticker_historical_data: Historical data per ticker

    Returns:
        Dictionary of ticker -> position size
    """
    position_sizes: dict[str, float] = {}

    if config.position_sizing == "equal" or len(candidate_idx) <= 1:
        return position_sizes

    optimization_method = config.portfolio_optimization_method or config.position_sizing

    if optimization_method in ["mpt", "risk_parity"]:
        position_sizes = _calculate_mpt_sizes(
            state, config, candidate_idx, tickers, optimization_method
        )
    elif optimization_method == "kelly":
        position_sizes = _calculate_kelly_sizes(state, config)

    if not position_sizes:
        position_sizes = _calculate_fallback_sizes(
            state, config, candidate_idx, tickers, entry_prices, d_idx, ticker_historical_data
        )

    return position_sizes


def _calculate_mpt_sizes(
    state: "SimulationState",
    config: BacktestConfig,
    candidate_idx: np.ndarray,
    tickers: list[str],
    optimization_method: str,
) -> dict[str, float]:
    """Calculate MPT/risk parity position sizes."""
    position_sizes: dict[str, float] = {}
    candidate_tickers = [tickers[idx] for idx in candidate_idx]

    returns_data: dict[str, list[float]] = {}
    for idx in candidate_idx:
        ticker = tickers[idx]
        if (
            ticker in state.asset_returns
            and len(state.asset_returns[ticker]) >= config.position_sizing_lookback
        ):
            returns_data[ticker] = state.asset_returns[ticker][-config.position_sizing_lookback :]

    if len(returns_data) < 2:
        return position_sizes

    max_len = max(len(r) for r in returns_data.values())
    returns_df = pd.DataFrame(
        {ticker: r + [np.nan] * (max_len - len(r)) for ticker, r in returns_data.items()}
    ).dropna()

    if returns_df.empty or len(returns_df) < 10:
        return position_sizes

    try:
        weights = optimize_portfolio(
            returns_df,
            method=optimization_method,
            risk_free_rate=config.risk_free_rate,
        )
        for ticker, weight in weights.weights.items():
            if ticker in candidate_tickers:
                position_sizes[ticker] = state.cash * weight
    except Exception as e:
        logger.warning(f"Portfolio optimization failed: {e}")

    return position_sizes


def _calculate_kelly_sizes(
    state: "SimulationState",
    config: BacktestConfig,
) -> dict[str, float]:
    """Calculate Kelly criterion position sizes."""
    if len(state.trades_list) < 10:
        return {}

    try:
        trades_df = pd.DataFrame(state.trades_list)
        if "pnl_pct" not in trades_df.columns:
            return {}

        from src.risk.portfolio_optimization import PortfolioOptimizer

        optimizer = PortfolioOptimizer()
        return optimizer.optimize_kelly_portfolio(
            trades_df,
            available_cash=state.cash,
            max_kelly=config.max_kelly,
        )
    except Exception as e:
        logger.warning(f"Kelly optimization failed: {e}")
        return {}


def _calculate_fallback_sizes(
    state: "SimulationState",
    config: BacktestConfig,
    candidate_idx: np.ndarray,
    tickers: list[str],
    entry_prices: np.ndarray,
    d_idx: int,
    ticker_historical_data: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Calculate fallback position sizes using multi-asset method."""
    candidate_tickers = [tickers[idx] for idx in candidate_idx]
    candidate_prices = {
        tickers[idx]: entry_prices[idx, d_idx]
        for idx in candidate_idx
        if not np.isnan(entry_prices[idx, d_idx])
    }

    candidate_historical: dict[str, pd.DataFrame] = {}
    for idx in candidate_idx:
        ticker = tickers[idx]
        if ticker in ticker_historical_data:
            hist_df = ticker_historical_data[ticker]
            candidate_historical[ticker] = hist_df.iloc[:d_idx] if d_idx > 0 else hist_df.iloc[:1]

    return calculate_multi_asset_position_sizes(
        method=config.position_sizing,
        available_cash=state.cash,
        tickers=candidate_tickers,
        current_prices=candidate_prices,
        historical_data=candidate_historical,
        target_risk_pct=config.position_sizing_risk_pct,
        lookback_period=config.position_sizing_lookback,
    )
