"""Metrics helpers for research experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.metrics_core import (
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_daily_returns,
    calculate_mdd,
    calculate_sharpe_ratio,
)


@dataclass(frozen=True)
class EqualWeightPortfolioResult:
    """Equal-weight portfolio metrics and timeseries."""

    returns: pd.Series
    equity: pd.Series
    cagr: float
    sharpe: float
    mdd: float


def compute_equity_trade_metrics(
    equity: pd.Series,
    trade_pnls: Sequence[float],
    trade_holding_days: Sequence[int],
    annualization_days: int = 365,
) -> dict[str, float]:
    """Compute core performance metrics from an equity curve and trade history."""
    if len(equity) < 2:
        return {
            "CAGR": 0.0,
            "Sharpe": 0.0,
            "MDD": 0.0,
            "Calmar": 0.0,
            "WinRate": 0.0,
            "AvgPnL": 0.0,
            "NumTrades": 0.0,
            "AvgHold": 0.0,
            "PF": 0.0,
        }

    equity_arr = equity.to_numpy(dtype=float, copy=False)
    initial_value = float(equity_arr[0])
    final_value = float(equity_arr[-1])
    total_days = max(len(equity_arr) - 1, 1)
    returns = calculate_daily_returns(equity_arr)

    cagr_pct = calculate_cagr(initial_value, final_value, total_days)
    mdd_pct = calculate_mdd(equity_arr)
    sharpe = calculate_sharpe_ratio(returns, annualization_factor=annualization_days)
    calmar = calculate_calmar_ratio(cagr_pct, mdd_pct)

    # Keep research script convention:
    # - CAGR as decimal fraction (0.12 = 12%)
    # - MDD as negative fraction (-0.25 = -25%)
    cagr = cagr_pct / 100.0
    mdd = -(mdd_pct / 100.0)

    n_trades = len(trade_pnls)
    if n_trades > 0:
        wins = [pnl for pnl in trade_pnls if pnl > 0.0]
        losses = [pnl for pnl in trade_pnls if pnl <= 0.0]
        win_rate = len(wins) / n_trades
        avg_pnl = float(np.mean(trade_pnls))
        avg_hold = float(np.mean(trade_holding_days))
        gross_win = float(sum(wins)) if wins else 0.0
        gross_loss = abs(float(sum(losses))) if losses else 0.0
        pf = gross_win / max(gross_loss, 1e-8)
    else:
        win_rate = 0.0
        avg_pnl = 0.0
        avg_hold = 0.0
        pf = 0.0

    return {
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "MDD": float(mdd),
        "Calmar": float(calmar),
        "WinRate": float(win_rate),
        "AvgPnL": float(avg_pnl),
        "NumTrades": float(n_trades),
        "AvgHold": float(avg_hold),
        "PF": float(pf),
    }


def build_equal_weight_portfolio(
    equities_by_symbol: dict[str, pd.Series],
    annualization_days: int = 365,
) -> EqualWeightPortfolioResult | None:
    """Build an equal-weight portfolio from symbol equity series."""
    eq_df = pd.DataFrame(equities_by_symbol).dropna(how="any")
    if len(eq_df) <= 1:
        return None

    portfolio_returns = eq_df.pct_change().mean(axis=1).dropna()
    if portfolio_returns.empty:
        return None

    portfolio_equity = (1.0 + portfolio_returns).cumprod()
    portfolio_equity_arr = portfolio_equity.to_numpy(dtype=float, copy=False)
    total_days = max(len(portfolio_equity_arr) - 1, 1)
    returns_arr = portfolio_returns.to_numpy(dtype=float, copy=False)

    cagr_pct = calculate_cagr(
        float(portfolio_equity_arr[0]), float(portfolio_equity_arr[-1]), total_days
    )
    sharpe = calculate_sharpe_ratio(returns_arr, annualization_factor=annualization_days)
    mdd_pct = calculate_mdd(portfolio_equity_arr)

    cagr = cagr_pct / 100.0
    mdd = -(mdd_pct / 100.0)

    return EqualWeightPortfolioResult(
        returns=portfolio_returns,
        equity=portfolio_equity,
        cagr=float(cagr),
        sharpe=float(sharpe),
        mdd=float(mdd),
    )


def compute_yearly_return_and_sharpe(
    returns: pd.Series,
    annualization_days: int = 365,
) -> tuple[pd.Series, pd.Series]:
    """Compute yearly return and yearly Sharpe series."""
    if returns.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    year_index = pd.Index([timestamp.year for timestamp in returns.index], dtype="int64")
    yearly_returns = returns.groupby(year_index).apply(_compound_return)
    yearly_sharpe = returns.groupby(year_index).apply(
        lambda values: _annualized_sharpe(values, annualization_days)
    )
    return yearly_returns.astype(float), yearly_sharpe.astype(float)


def _compound_return(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float, copy=False)
    return float(np.prod(1.0 + array) - 1.0)


def _annualized_sharpe(values: pd.Series, annualization_days: int) -> float:
    array = values.to_numpy(dtype=float, copy=False)
    return calculate_sharpe_ratio(array, annualization_factor=annualization_days)
