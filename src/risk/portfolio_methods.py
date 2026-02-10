"""
Portfolio optimization methods implementation.

Contains MPT, Risk Parity, and Kelly Criterion implementations.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.risk.portfolio_models import PortfolioWeights
from src.utils.logger import get_logger

logger = get_logger(__name__)


def optimize_mpt(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    target_return: float | None = None,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> PortfolioWeights:
    """Optimize portfolio using Mean-Variance Optimization (Modern Portfolio Theory)."""
    if returns.empty or len(returns.columns) == 0:
        raise ValueError("Returns DataFrame is empty or has no columns")

    tickers = list(returns.columns)
    n_assets = len(tickers)

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # Check for singular/ill-conditioned covariance matrix
    cond = np.linalg.cond(cov_matrix.values)
    if cond > 1e12 or np.isnan(cond):
        logger.warning(f"Covariance matrix ill-conditioned (cond={cond:.2e}), using equal weights")
        return PortfolioWeights(weights=dict.fromkeys(tickers, 1.0 / n_assets), method="mpt")

    def objective(weights: np.ndarray) -> float:
        port_ret: float = float(np.dot(weights, mean_returns))
        port_vol: float = float(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))))
        if port_vol == 0:
            return float("inf")
        return -(port_ret - risk_free_rate) / port_vol

    constraints: list[dict[str, Any]] | dict[str, Any] = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
    }
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    initial_weights = np.array([1.0 / n_assets] * n_assets)

    if target_return is not None:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target_return},
        ]

    try:
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )
        weights = result.x if result.success else np.array([1.0 / n_assets] * n_assets)
        weights = weights / np.sum(weights)

        port_ret = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0.0

        weights_dict = {t: float(w) for t, w in zip(tickers, weights, strict=False)}
        return PortfolioWeights(
            weights=weights_dict,
            method="mpt",
            expected_return=float(port_ret),
            portfolio_volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
        )
    except Exception as e:
        logger.error(f"MPT optimization error: {e}")
        return PortfolioWeights(weights=dict.fromkeys(tickers, 1.0 / n_assets), method="mpt")


def optimize_risk_parity(
    returns: pd.DataFrame,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> PortfolioWeights:
    """Optimize portfolio using Risk Parity (equal risk contribution)."""
    if returns.empty or len(returns.columns) == 0:
        raise ValueError("Returns DataFrame is empty or has no columns")

    tickers = list(returns.columns)
    n_assets = len(tickers)
    cov_matrix = returns.cov() * 252

    # Check for singular/ill-conditioned covariance matrix
    cond = np.linalg.cond(cov_matrix.values)
    if cond > 1e12 or np.isnan(cond):
        logger.warning(f"Covariance matrix ill-conditioned (cond={cond:.2e}), using equal weights")
        return PortfolioWeights(
            weights=dict.fromkeys(tickers, 1.0 / n_assets), method="risk_parity"
        )

    def objective(weights: np.ndarray) -> float:
        port_vol = float(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))))
        if port_vol == 0:
            return float("inf")
        marginal = np.dot(cov_matrix, weights) / port_vol
        risk_contrib = weights * marginal
        target = port_vol / n_assets
        return float(np.sum((risk_contrib - target) ** 2))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    vols = np.sqrt(np.diag(cov_matrix))
    inv_vols = 1.0 / (vols + 1e-8)
    initial_weights = inv_vols / np.sum(inv_vols)

    try:
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )
        weights = result.x if result.success else initial_weights
        weights = weights / np.sum(weights)

        mean_returns = returns.mean() * 252
        port_ret = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe = port_ret / port_vol if port_vol > 0 else 0.0

        weights_dict = {t: float(w) for t, w in zip(tickers, weights, strict=False)}
        return PortfolioWeights(
            weights=weights_dict,
            method="risk_parity",
            expected_return=float(port_ret),
            portfolio_volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
        )
    except Exception as e:
        logger.error(f"Risk parity optimization error: {e}")
        vols = np.sqrt(np.diag(cov_matrix))
        inv_vols = 1.0 / (vols + 1e-8)
        weights = inv_vols / np.sum(inv_vols)
        return PortfolioWeights(
            weights={t: float(w) for t, w in zip(tickers, weights, strict=False)},
            method="risk_parity",
        )


def calculate_kelly_criterion(
    win_rate: float, avg_win: float, avg_loss: float, max_kelly: float = 0.25
) -> float:
    """Calculate Kelly Criterion for optimal position sizing."""
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError(f"Win rate must be between 0 and 1, got {win_rate}")
    if avg_loss <= 0:
        raise ValueError(f"Average loss must be positive, got {avg_loss}")
    if avg_win <= 0:
        return 0.0

    payoff_ratio = avg_win / avg_loss
    kelly = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio

    if kelly <= 0:
        return 0.0
    return min(kelly, max_kelly)


def optimize_kelly_portfolio(
    trades: pd.DataFrame, available_cash: float, max_kelly: float = 0.25
) -> dict[str, float]:
    """Calculate portfolio allocation using Kelly Criterion for each asset."""
    if trades.empty:
        raise ValueError("Trades DataFrame is empty")
    if "ticker" not in trades.columns:
        raise ValueError("Trades DataFrame must have 'ticker' column")

    return_col = None
    for col in ["pnl_pct", "return", "return_pct"]:
        if col in trades.columns:
            return_col = col
            break
    if return_col is None:
        raise ValueError("Trades DataFrame must have return column")

    allocations: dict[str, float] = {}

    for ticker in trades["ticker"].unique():
        ticker_trades = trades[trades["ticker"] == ticker]
        if len(ticker_trades) < 2:
            continue

        returns = ticker_trades[return_col].values / 100.0
        wins, losses = returns[returns > 0], returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            continue

        win_rate = len(wins) / len(returns)
        avg_win = float(np.mean(wins))
        avg_loss = abs(float(np.mean(losses)))

        if avg_loss == 0:
            continue

        kelly_pct = calculate_kelly_criterion(win_rate, avg_win, avg_loss, max_kelly)
        allocations[ticker] = available_cash * kelly_pct

    total = sum(allocations.values())
    if total > available_cash:
        scale = available_cash / total
        allocations = {t: a * scale for t, a in allocations.items()}

    return allocations


__all__ = [
    "optimize_mpt",
    "optimize_risk_parity",
    "calculate_kelly_criterion",
    "optimize_kelly_portfolio",
]
