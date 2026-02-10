"""
Simple backtest implementation for bootstrap analysis.
"""

import numpy as np
import pandas as pd

from src.backtester.models import BacktestResult
from src.strategies.base import Strategy
from src.utils.logger import get_logger
from src.utils.metrics_core import (
    calculate_daily_returns,
    calculate_mdd,
    calculate_sharpe_ratio,
)

__all__ = ["simple_backtest_vectorized"]

logger = get_logger(__name__)


def simple_backtest_vectorized(
    data: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float,
) -> BacktestResult:
    """
    Simple vectorized backtest for bootstrap sampling.

    Args:
        data: OHLCV DataFrame
        strategy: Strategy to backtest
        initial_capital: Initial capital

    Returns:
        BacktestResult with basic metrics
    """
    try:
        df = data.copy()
        df = strategy.calculate_indicators(df)
        df = strategy.generate_signals(df)

        if "signal" not in df.columns:
            result = BacktestResult()
            result.total_return = 0.0
            result.sharpe_ratio = 0.0
            result.mdd = 0.0
            result.total_trades = 0
            result.winning_trades = 0
            result.win_rate = 0.0
            return result

        position = 0
        entry_price = 0.0
        equity = [initial_capital]

        for _, row in df.iterrows():
            signal = row.get("signal", 0)
            close = row.get("close", 0)

            if signal != 0 and position == 0:
                entry_price = close
                position = signal
            elif signal * position < 0:
                if position != 0:
                    pnl = (close - entry_price) * position / entry_price
                    equity.append(equity[-1] * (1 + pnl))
                    position = signal
                    entry_price = close if signal != 0 else 0.0
                else:
                    position = 0

            if position == 0 and len(equity) > 1:
                equity.append(equity[-1])

        if position != 0 and len(df) > 0:
            last_close = df.iloc[-1].get("close", entry_price)
            pnl = (last_close - entry_price) * position / entry_price
            equity.append(equity[-1] * (1 + pnl))

        result = BacktestResult()
        result.total_return = (equity[-1] - initial_capital) / initial_capital if equity else 0.0

        if len(equity) > 1:
            equity_arr = np.array(equity, dtype=np.float64)
            returns = calculate_daily_returns(equity_arr)
            result.sharpe_ratio = calculate_sharpe_ratio(returns, annualization_factor=252)
            result.mdd = calculate_mdd(equity_arr)
        else:
            result.sharpe_ratio = 0.0
            result.mdd = 0.0

        return result
    except Exception as e:
        logger.error(f"Bootstrap backtest error: {e}")
        result = BacktestResult()
        result.total_return = 0.0
        result.sharpe_ratio = 0.0
        result.mdd = 0.0
        return result
