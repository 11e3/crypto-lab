"""
Backtesting metrics calculation.

Performance metrics for strategy evaluation.

Metrics layer: Layer 3 — Event Engine Aggregator.
Wraps src/backtester/metrics_helpers.py (Layer 2).
Produces BacktestResult for the EventDrivenBacktestEngine.
"""

import numpy as np
import pandas as pd

from src.backtester.metrics_helpers import (
    calculate_return_metrics,
    calculate_risk_metrics_from_equity,
    calculate_trade_stats,
)
from src.backtester.models import BacktestConfig, BacktestResult, Trade
from src.risk.metrics import calculate_portfolio_risk_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_metrics(
    equity_curve: np.ndarray,
    dates: np.ndarray,
    trades: list[Trade],
    config: BacktestConfig,
    strategy_name: str = "",
    asset_returns: dict[str, list[float]] | None = None,
) -> BacktestResult:
    """
    Calculate performance metrics from backtest results.

    Args:
        equity_curve: Daily equity values
        dates: Corresponding dates
        trades: List of executed trades
        config: Backtest configuration
        strategy_name: Name of strategy
        asset_returns: Asset returns for portfolio metrics (optional)

    Returns:
        BacktestResult with all calculated metrics
    """
    result = BacktestResult(
        equity_curve=equity_curve,
        dates=dates,
        trades=trades,
        config=config,
        strategy_name=strategy_name,
    )

    if len(equity_curve) < 2:
        return result

    initial = config.initial_capital
    final = equity_curve[-1]

    # Calculate return metrics
    result.total_return, result.cagr = calculate_return_metrics(equity_curve, dates, initial)

    # Calculate risk metrics
    result.mdd, result.calmar_ratio, result.sharpe_ratio = calculate_risk_metrics_from_equity(
        equity_curve, result.cagr
    )

    # Calculate trade statistics
    if trades:
        trade_stats = calculate_trade_stats(trades)
        result.total_trades = int(trade_stats["total_trades"])
        result.winning_trades = int(trade_stats["winning_trades"])
        result.losing_trades = int(trade_stats["losing_trades"])
        result.win_rate = trade_stats["win_rate"]
        result.profit_factor = trade_stats["profit_factor"]
        result.avg_trade_return = trade_stats["avg_trade_return"]

    # Portfolio risk metrics
    if asset_returns:
        try:
            asset_returns_dict: dict[str, np.ndarray] = {
                ticker: np.array(returns_list, dtype=np.float32)
                for ticker, returns_list in asset_returns.items()
                if len(returns_list) > 0
            }

            # Estimate position values (equal weight for simplicity)
            if result.total_trades > 0 and asset_returns_dict:
                position_values_dict: dict[str, float] | None = {
                    ticker: float(final / len(asset_returns_dict)) for ticker in asset_returns_dict
                }
            else:
                position_values_dict = None

            if asset_returns_dict:
                daily_returns = np.diff(equity_curve) / equity_curve[:-1]
                result.risk_metrics = calculate_portfolio_risk_metrics(
                    equity_curve=equity_curve,
                    daily_returns=daily_returns.astype(np.float64),
                    asset_returns=asset_returns_dict,
                    position_values=position_values_dict,
                    total_portfolio_value=float(final),
                )
        except Exception as e:
            logger.warning(f"Failed to calculate portfolio risk metrics: {e}")

    return result


def calculate_trade_metrics(trades_df: pd.DataFrame) -> dict[str, float]:
    """
    Calculate trade-level metrics from trades DataFrame.

    Args:
        trades_df: DataFrame with trade records

    Returns:
        Dictionary of trade metrics
    """
    if trades_df.empty:
        return {}

    metrics: dict[str, float] = {}

    # Basic stats
    metrics["total_trades"] = float(len(trades_df))
    metrics["winning_trades"] = float((trades_df["pnl"] > 0).sum())
    metrics["losing_trades"] = float((trades_df["pnl"] < 0).sum())

    # Win rate
    if len(trades_df) > 0:
        metrics["win_rate"] = metrics["winning_trades"] / float(len(trades_df)) * 100

    # Profit factor
    winning = trades_df[trades_df["pnl"] > 0]
    losing = trades_df[trades_df["pnl"] < 0]

    if not winning.empty and not losing.empty:
        total_profit = winning["pnl"].sum()
        total_loss = abs(losing["pnl"].sum())
        metrics["profit_factor"] = total_profit / total_loss if total_loss > 0 else 0.0

    # Average metrics
    metrics["avg_pnl"] = float(trades_df["pnl"].mean())
    metrics["avg_pnl_pct"] = float(trades_df["pnl_pct"].mean())

    if not winning.empty:
        metrics["avg_win"] = float(winning["pnl"].mean())
        metrics["avg_win_pct"] = float(winning["pnl_pct"].mean())

    if not losing.empty:
        metrics["avg_loss"] = float(losing["pnl"].mean())
        metrics["avg_loss_pct"] = float(losing["pnl_pct"].mean())

    # Risk metrics
    metrics["max_win"] = float(trades_df["pnl"].max())
    metrics["max_loss"] = float(trades_df["pnl"].min())
    metrics["max_win_pct"] = float(trades_df["pnl_pct"].max())
    metrics["max_loss_pct"] = float(trades_df["pnl_pct"].min())

    return metrics
