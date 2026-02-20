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

# Minimum volatility to prevent division by zero in inverse-volatility weighting
_MIN_VOLATILITY = 1e-8


def _maximize_sharpe(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> float:
    """Negative Sharpe ratio — used as scipy.minimize objective for MPT."""
    port_ret = float(np.dot(weights, mean_returns))
    port_vol = float(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))))
    if port_vol == 0:
        return float("inf")
    return -(port_ret - risk_free_rate) / port_vol


def _mpt_constraints(
    mean_returns: np.ndarray,
    target_return: float | None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Build scipy constraint list: weights sum to 1, optional target return."""
    eq_sum: dict[str, Any] = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    if target_return is not None:
        return [eq_sum, {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target_return}]
    return eq_sum


def _build_mpt_result(
    opt_weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    tickers: list[str],
) -> PortfolioWeights:
    """Compute portfolio metrics and package as PortfolioWeights."""
    port_ret = float(np.dot(opt_weights, mean_returns))
    port_vol = float(np.sqrt(np.dot(opt_weights, np.dot(cov_matrix, opt_weights))))
    sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0.0
    return PortfolioWeights(
        weights={t: float(w) for t, w in zip(tickers, opt_weights, strict=False)},
        method="mpt",
        expected_return=port_ret,
        portfolio_volatility=port_vol,
        sharpe_ratio=sharpe,
    )


def _risk_parity_objective(weights: np.ndarray, cov_matrix: np.ndarray, n_assets: int) -> float:
    """Sum of squared deviations from equal risk contribution — scipy minimize objective."""
    port_vol = float(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))))
    if port_vol == 0:
        return float("inf")
    marginal = np.dot(cov_matrix, weights) / port_vol
    risk_contrib = weights * marginal
    target = port_vol / n_assets
    return float(np.sum((risk_contrib - target) ** 2))


def _build_risk_parity_result(
    opt_weights: np.ndarray,
    returns: pd.DataFrame,
    cov_matrix: np.ndarray,
    tickers: list[str],
) -> PortfolioWeights:
    """Compute portfolio metrics for risk parity and package as PortfolioWeights."""
    mean_returns = returns.mean() * 252
    port_ret = float(np.dot(opt_weights, mean_returns))
    port_vol = float(np.sqrt(np.dot(opt_weights, np.dot(cov_matrix, opt_weights))))
    sharpe = port_ret / port_vol if port_vol > 0 else 0.0
    return PortfolioWeights(
        weights={t: float(w) for t, w in zip(tickers, opt_weights, strict=False)},
        method="risk_parity",
        expected_return=port_ret,
        portfolio_volatility=port_vol,
        sharpe_ratio=sharpe,
    )


def _validate_covariance(
    cov_matrix: pd.DataFrame,
    tickers: list[str],
    method: str,
) -> PortfolioWeights | None:
    """Check covariance matrix condition; return equal-weight fallback if ill-conditioned."""
    cond = np.linalg.cond(cov_matrix.values)
    if cond > 1e12 or np.isnan(cond):
        logger.warning(f"Covariance matrix ill-conditioned (cond={cond:.2e}), using equal weights")
        return PortfolioWeights(
            weights=dict.fromkeys(tickers, 1.0 / len(tickers)),
            method=method,
        )
    return None


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
    cov_matrix = returns.cov() * 252

    fallback = _validate_covariance(cov_matrix, tickers, "mpt")
    if fallback is not None:
        return fallback

    mean_ret_np: np.ndarray = (returns.mean() * 252).to_numpy()
    cov_mat_np: np.ndarray = cov_matrix.to_numpy()
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    initial_weights = np.array([1.0 / n_assets] * n_assets)
    constraints = _mpt_constraints(mean_ret_np, target_return)

    try:
        result = minimize(
            _maximize_sharpe,
            initial_weights,
            args=(mean_ret_np, cov_mat_np, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )
        weights = result.x if result.success else initial_weights
        weights = weights / np.sum(weights)
        return _build_mpt_result(weights, mean_ret_np, cov_mat_np, risk_free_rate, tickers)
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

    fallback = _validate_covariance(cov_matrix, tickers, "risk_parity")
    if fallback is not None:
        return fallback

    cov_mat_np: np.ndarray = cov_matrix.to_numpy()
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    vols = np.sqrt(np.diag(cov_mat_np))
    inv_vols = 1.0 / (vols + _MIN_VOLATILITY)
    initial_weights = inv_vols / np.sum(inv_vols)

    try:
        result = minimize(
            _risk_parity_objective,
            initial_weights,
            args=(cov_mat_np, n_assets),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )
        weights = result.x if result.success else initial_weights
        weights = weights / np.sum(weights)
        return _build_risk_parity_result(weights, returns, cov_mat_np, tickers)
    except Exception as e:
        logger.error(f"Risk parity optimization error: {e}")
        vols = np.sqrt(np.diag(cov_mat_np))
        inv_vols = 1.0 / (vols + _MIN_VOLATILITY)
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
