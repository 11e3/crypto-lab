"""
Metrics calculation for vectorized backtesting.

Handles performance metrics, trade statistics, and risk metrics.

Metrics layer: Layer 3 — Vector Engine Aggregator.
Wraps src/utils/metrics_core.py (Layer 1) directly (numpy arrays, no Trade models).
Produces BacktestResult for the VectorizedBacktestEngine.
"""

import numpy as np
import pandas as pd

from src.backtester.models import BacktestConfig, BacktestResult
from src.config import ANNUALIZATION_FACTOR
from src.risk.metrics import PortfolioRiskMetrics, calculate_portfolio_risk_metrics
from src.utils.logger import get_logger
from src.utils.metrics_core import (
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_daily_returns,
    calculate_mdd,
    calculate_sharpe_ratio,
)

logger = get_logger(__name__)


def calculate_metrics_vectorized(
    equity_curve: np.ndarray,
    dates: np.ndarray,
    trades_df: pd.DataFrame,
    config: BacktestConfig,
    asset_returns: dict[str, list[float]] | None = None,
) -> BacktestResult:
    """
    Calculate performance metrics using vectorized operations.

    Args:
        equity_curve: Daily portfolio values
        dates: Date array
        trades_df: All trades DataFrame
        config: Backtest configuration
        asset_returns: Per-asset returns for correlation analysis

    Returns:
        BacktestResult with all metrics
    """
    result = BacktestResult(
        equity_curve=equity_curve,
        dates=dates,
        config=config,
    )

    if len(equity_curve) < 2:
        return result

    initial = config.initial_capital
    final = equity_curve[-1]

    # Total return
    result.total_return = (final / initial - 1) * 100

    # CAGR (annualized return)
    total_days = (dates[-1] - dates[0]).days
    result.cagr = calculate_cagr(initial, final, total_days)

    # MDD (maximum drawdown)
    result.mdd = calculate_mdd(equity_curve)

    # Calmar Ratio
    result.calmar_ratio = calculate_calmar_ratio(result.cagr, result.mdd)

    # Sharpe Ratio & daily returns
    daily_returns = calculate_daily_returns(equity_curve, prepend_zero=True)
    returns = daily_returns[1:]  # without prepended zero for Sharpe
    result.sharpe_ratio = calculate_sharpe_ratio(returns, ANNUALIZATION_FACTOR)

    # Risk metrics
    result.risk_metrics = _calculate_risk_metrics(
        equity_curve, daily_returns, trades_df, asset_returns
    )

    # Trade statistics
    _calculate_trade_statistics(result, trades_df)

    return result


def _calculate_risk_metrics(
    equity_curve: np.ndarray,
    daily_returns: np.ndarray,
    trades_df: pd.DataFrame,
    asset_returns: dict[str, list[float]] | None,
) -> PortfolioRiskMetrics | None:
    """Calculate portfolio risk metrics."""
    asset_returns_dict: dict[str, np.ndarray] | None = None
    if asset_returns:
        valid_returns = {k: v for k, v in asset_returns.items() if len(v) > 0}
        if valid_returns:
            min_length = min(len(returns) for returns in valid_returns.values())
            if min_length > 0:
                asset_returns_dict = {
                    ticker: np.array(returns[-min_length:])
                    for ticker, returns in valid_returns.items()
                }

    position_values_dict: dict[str, float] | None = None
    if len(trades_df) > 0 and len(equity_curve) > 0:
        open_trades = trades_df[trades_df["exit_date"].isna()]
        if len(open_trades) > 0:
            position_values_dict = {}
            total_position_value = 0.0
            for _, trade in open_trades.iterrows():
                ticker = trade["ticker"]
                if ticker not in position_values_dict:
                    position_values_dict[ticker] = 0.0
                position_value = trade["amount"] * trade.get("entry_price", 0.0)
                position_values_dict[ticker] += position_value
                total_position_value += position_value

            final_equity = equity_curve[-1] if len(equity_curve) > 0 else 0.0
            if (
                total_position_value > 0
                and final_equity > 0
                and total_position_value > final_equity
            ):
                scale_factor = final_equity / total_position_value
                position_values_dict = {
                    ticker: value * scale_factor for ticker, value in position_values_dict.items()
                }

    try:
        return calculate_portfolio_risk_metrics(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            asset_returns=asset_returns_dict,
            position_values=position_values_dict,
            total_portfolio_value=equity_curve[-1] if len(equity_curve) > 0 else 0.0,
            benchmark_returns=None,
            annualization_factor=ANNUALIZATION_FACTOR,
        )
    except (ValueError, ZeroDivisionError, FloatingPointError) as e:
        logger.warning(f"Failed to calculate risk metrics: {e}")
        return None


def _calculate_trade_statistics(result: BacktestResult, trades_df: pd.DataFrame) -> None:
    """Calculate trade statistics."""
    if len(trades_df) == 0:
        return

    closed = trades_df[trades_df["exit_date"].notna()]
    result.total_trades = len(closed)

    if len(closed) == 0:
        return

    result.winning_trades = int((closed["pnl"] > 0).sum())
    result.losing_trades = int((closed["pnl"] <= 0).sum())
    result.win_rate = (result.winning_trades / len(closed)) * 100
    result.avg_trade_return = float(closed["pnl_pct"].mean())

    total_profit = closed.loc[closed["pnl"] > 0, "pnl"].sum()
    total_loss = abs(closed.loc[closed["pnl"] <= 0, "pnl"].sum())
    result.profit_factor = total_profit / total_loss if total_loss > 0 else 999.99
