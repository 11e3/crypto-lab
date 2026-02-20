"""
Lightweight vectorized backtest runner for Walk-Forward Analysis.

Used inside WalkForwardAnalyzer to evaluate each fold cheaply without
spinning up the full BacktestEngine.
"""

import numpy as np
import pandas as pd

from src.backtester.models import BacktestResult
from src.strategies.base import Strategy
from src.utils.logger import get_logger
from src.utils.metrics_core import calculate_daily_returns, calculate_mdd, calculate_sharpe_ratio

logger = get_logger(__name__)


def simple_backtest(
    data: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float = 10000000,
) -> BacktestResult:
    """
    Vectorized backtest (lightweight alternative to BacktestEngine).

    Args:
        data: OHLCV DataFrame
        strategy: Trading strategy
        initial_capital: Starting capital

    Returns:
        BacktestResult object
    """
    try:
        df = data.copy()

        df = strategy.calculate_indicators(df)
        df = strategy.generate_signals(df)

        if "signal" not in df.columns:
            if "entry_signal" in df.columns and "exit_signal" in df.columns:
                # Synthesize signal column from entry/exit flags (for strategies like VBOV1)
                # entry_signal=True → 1 (open long), exit_signal=True → -1 (close long)
                df["signal"] = 0
                df.loc[df["entry_signal"].astype(bool), "signal"] = 1
                df.loc[df["exit_signal"].astype(bool), "signal"] = -1
            else:
                return _create_empty_result()

        trades, equity = _simulate_positions(df, initial_capital)

        return _calculate_metrics(trades, equity, initial_capital)

    except Exception as e:
        logger.error(f"Simple backtest error: {e}")
        return _create_empty_result()


def _simulate_positions(
    df: pd.DataFrame,
    initial_capital: float,
) -> tuple[list[float], list[float]]:
    """Simulate positions and record trade returns."""
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    trades: list[float] = []
    equity: list[float] = [initial_capital]

    for _idx, row in df.iterrows():
        signal = row.get("signal", 0)
        close = float(row.get("close", 0))

        if signal != 0 and position == 0:
            # Entry
            entry_price = close
            position = int(signal)
        elif signal * position < 0:
            # Exit on opposing signal
            if position != 0:
                pnl = (close - entry_price) * position / entry_price
                trades.append(pnl)
                equity.append(equity[-1] * (1 + pnl))
                position = int(signal)
                entry_price = close if signal != 0 else 0.0
            else:
                position = 0

        if position == 0 and len(equity) > 1:
            equity.append(equity[-1])

    # Close any open position at the final price
    if position != 0 and len(df) > 0:
        last_close = float(df.iloc[-1].get("close", entry_price))
        pnl = (last_close - entry_price) * position / entry_price
        trades.append(pnl)
        equity.append(equity[-1] * (1 + pnl))

    return trades, equity


def _calculate_metrics(
    trades: list[float],
    equity: list[float],
    initial_capital: float,
) -> BacktestResult:
    """Compute performance metrics from trade returns and equity curve."""
    total_return = (equity[-1] - initial_capital) / initial_capital if equity else 0.0

    equity_arr = np.array(equity)
    returns = calculate_daily_returns(equity_arr)
    sharpe = calculate_sharpe_ratio(returns, 252)
    max_drawdown = calculate_mdd(equity_arr)

    winning_trades, win_rate = _calculate_win_rate(trades)

    result = BacktestResult()
    result.total_return = total_return
    result.sharpe_ratio = sharpe
    result.mdd = max_drawdown
    result.total_trades = len(trades)
    result.winning_trades = winning_trades
    result.win_rate = win_rate
    result.equity_curve = np.array(equity)

    return result


def _calculate_win_rate(trades: list[float]) -> tuple[int, float]:
    """Return (winning_count, win_rate) for a list of trade returns."""
    if not trades:
        return 0, 0.0

    winning = sum(1 for t in trades if t > 0)
    return winning, winning / len(trades)


def _create_empty_result() -> BacktestResult:
    """Return a zeroed-out BacktestResult for error/empty cases."""
    result = BacktestResult()
    result.total_return = 0.0
    result.sharpe_ratio = 0.0
    result.mdd = 0.0
    result.total_trades = 0
    result.winning_trades = 0
    result.win_rate = 0.0
    return result
