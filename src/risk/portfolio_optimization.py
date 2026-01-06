"""
Portfolio optimization methods.

Implements:
- Modern Portfolio Theory (MPT): Mean-variance optimization
- Risk Parity: Equal risk contribution from each asset
- Kelly Criterion: Optimal position sizing based on win rate and payoff ratio
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioWeights:
    """Portfolio allocation weights."""

    weights: dict[str, float]  # ticker -> weight (0.0 to 1.0)
    method: str  # "mpt", "risk_parity", "kelly"
    expected_return: float | None = None  # Expected portfolio return
    portfolio_volatility: float | None = None  # Portfolio volatility
    sharpe_ratio: float | None = None  # Sharpe ratio

    def __repr__(self) -> str:
        return (
            f"PortfolioWeights(method={self.method}, "
            f"assets={len(self.weights)}, "
            f"sharpe={self.sharpe_ratio:.3f if self.sharpe_ratio else None})"
        )


class PortfolioOptimizer:
    """
    Portfolio optimization using various methods.

    Supports Modern Portfolio Theory, Risk Parity, and Kelly Criterion.
    """

    def __init__(self) -> None:
        """Initialize portfolio optimizer."""
        pass

    def optimize_mpt(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
        target_return: float | None = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
    ) -> PortfolioWeights:
        """
        Optimize portfolio using Modern Portfolio Theory (mean-variance optimization).

        Maximizes Sharpe ratio (or minimizes variance for given return).

        Args:
            returns: DataFrame with returns (columns = tickers, index = dates)
            risk_free_rate: Risk-free rate (annualized, default: 0.0)
            target_return: Target expected return (if None, maximizes Sharpe ratio)
            max_weight: Maximum weight per asset (default: 1.0)
            min_weight: Minimum weight per asset (default: 0.0)

        Returns:
            PortfolioWeights with optimized allocation

        Raises:
            ValueError: If optimization fails or returns are invalid
        """
        if returns.empty or len(returns.columns) == 0:
            raise ValueError("Returns DataFrame is empty or has no columns")

        tickers = list(returns.columns)
        n_assets = len(tickers)

        # Calculate expected returns and covariance matrix
        mean_returns = returns.mean() * 252  # Annualize (assuming daily returns)
        cov_matrix = returns.cov() * 252  # Annualize covariance

        # Objective function: negative Sharpe ratio (to minimize)
        def objective(weights: np.ndarray) -> float:
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return np.inf
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe  # Negative because we minimize

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        # Bounds: each weight between min_weight and max_weight
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        # If target return specified, add constraint
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

            if not result.success:
                logger.warning(f"MPT optimization did not converge: {result.message}")
                # Fallback to equal weights
                weights = np.array([1.0 / n_assets] * n_assets)
            else:
                weights = result.x

            # Normalize weights (ensure they sum to 1)
            weights = weights / np.sum(weights)

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0.0

            # Convert to dictionary
            weights_dict = {ticker: float(w) for ticker, w in zip(tickers, weights)}

            return PortfolioWeights(
                weights=weights_dict,
                method="mpt",
                expected_return=float(portfolio_return),
                portfolio_volatility=float(portfolio_vol),
                sharpe_ratio=float(sharpe),
            )

        except Exception as e:
            logger.error(f"Error in MPT optimization: {e}", exc_info=True)
            # Fallback to equal weights
            weights_dict = {ticker: 1.0 / n_assets for ticker in tickers}
            return PortfolioWeights(weights=weights_dict, method="mpt")

    def optimize_risk_parity(
        self,
        returns: pd.DataFrame,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
    ) -> PortfolioWeights:
        """
        Optimize portfolio using Risk Parity (equal risk contribution).

        Each asset contributes equally to portfolio risk.

        Args:
            returns: DataFrame with returns (columns = tickers, index = dates)
            max_weight: Maximum weight per asset (default: 1.0)
            min_weight: Minimum weight per asset (default: 0.0)

        Returns:
            PortfolioWeights with risk parity allocation

        Raises:
            ValueError: If optimization fails or returns are invalid
        """
        if returns.empty or len(returns.columns) == 0:
            raise ValueError("Returns DataFrame is empty or has no columns")

        tickers = list(returns.columns)
        n_assets = len(tickers)

        # Calculate covariance matrix
        cov_matrix = returns.cov() * 252  # Annualize

        # Objective: minimize sum of squared differences in risk contributions
        def objective(weights: np.ndarray) -> float:
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return np.inf

            # Risk contribution of each asset
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib

            # Target: equal risk contribution (1/n for each asset)
            target_contrib = portfolio_vol / n_assets

            # Sum of squared differences from target
            diff = risk_contrib - target_contrib
            return np.sum(diff**2)

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # Initial guess: inverse volatility weights
        vols = np.sqrt(np.diag(cov_matrix))
        inv_vols = 1.0 / (vols + 1e-8)  # Add small epsilon to avoid division by zero
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

            if not result.success:
                logger.warning(f"Risk parity optimization did not converge: {result.message}")
                # Fallback to inverse volatility weights
                weights = initial_weights
            else:
                weights = result.x

            # Normalize weights
            weights = weights / np.sum(weights)

            # Calculate portfolio metrics
            mean_returns = returns.mean() * 252
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0.0

            weights_dict = {ticker: float(w) for ticker, w in zip(tickers, weights)}

            return PortfolioWeights(
                weights=weights_dict,
                method="risk_parity",
                expected_return=float(portfolio_return),
                portfolio_volatility=float(portfolio_vol),
                sharpe_ratio=float(sharpe),
            )

        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}", exc_info=True)
            # Fallback to inverse volatility weights
            vols = np.sqrt(np.diag(cov_matrix))
            inv_vols = 1.0 / (vols + 1e-8)
            weights = inv_vols / np.sum(inv_vols)
            weights_dict = {ticker: float(w) for ticker, w in zip(tickers, weights)}
            return PortfolioWeights(weights=weights_dict, method="risk_parity")

    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_kelly: float = 0.25,
    ) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing.

        Kelly % = (W * R - L) / R
        where:
        - W = win rate (probability of winning)
        - R = average win / average loss (payoff ratio)
        - L = loss rate (1 - W)

        Args:
            win_rate: Win rate (0.0 to 1.0)
            avg_win: Average winning trade return (e.g., 0.10 = 10%)
            avg_loss: Average losing trade return (absolute value, e.g., 0.05 = 5%)
            max_kelly: Maximum Kelly percentage to use (default: 0.25 = 25%)

        Returns:
            Kelly percentage (0.0 to max_kelly)

        Raises:
            ValueError: If parameters are invalid
        """
        if not 0.0 <= win_rate <= 1.0:
            raise ValueError(f"Win rate must be between 0 and 1, got {win_rate}")

        if avg_loss <= 0:
            raise ValueError(f"Average loss must be positive, got {avg_loss}")

        if avg_win <= 0:
            # If no wins, Kelly is 0
            return 0.0

        # Calculate payoff ratio
        payoff_ratio = avg_win / avg_loss

        # Kelly formula
        loss_rate = 1.0 - win_rate
        kelly = (win_rate * payoff_ratio - loss_rate) / payoff_ratio

        # Kelly must be positive
        if kelly <= 0:
            return 0.0

        # Cap at max_kelly (fractional Kelly for safety)
        kelly = min(kelly, max_kelly)

        return kelly

    def optimize_kelly_portfolio(
        self,
        trades: pd.DataFrame,
        available_cash: float,
        max_kelly: float = 0.25,
    ) -> dict[str, float]:
        """
        Calculate portfolio allocation using Kelly Criterion for each asset.

        Args:
            trades: DataFrame with columns: ticker, pnl_pct (or return)
            available_cash: Total cash available
            max_kelly: Maximum Kelly percentage per asset (default: 0.25)

        Returns:
            Dictionary mapping ticker to position size (in cash amount)

        Raises:
            ValueError: If trades DataFrame is invalid
        """
        if trades.empty:
            raise ValueError("Trades DataFrame is empty")

        if "ticker" not in trades.columns:
            raise ValueError("Trades DataFrame must have 'ticker' column")

        # Determine return column
        return_col = None
        for col in ["pnl_pct", "return", "return_pct"]:
            if col in trades.columns:
                return_col = col
                break

        if return_col is None:
            raise ValueError("Trades DataFrame must have return column (pnl_pct, return, or return_pct)")

        allocations: dict[str, float] = {}

        # Calculate Kelly for each ticker
        for ticker in trades["ticker"].unique():
            ticker_trades = trades[trades["ticker"] == ticker]

            if len(ticker_trades) < 2:
                # Not enough data, skip
                continue

            returns = ticker_trades[return_col].values / 100.0  # Convert percentage to decimal

            # Calculate win rate and average win/loss
            wins = returns[returns > 0]
            losses = returns[returns < 0]

            if len(wins) == 0 or len(losses) == 0:
                # No wins or no losses, skip
                continue

            win_rate = len(wins) / len(returns)
            avg_win = np.mean(wins) if len(wins) > 0 else 0.0
            avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0

            if avg_loss == 0:
                continue

            # Calculate Kelly
            kelly_pct = self.calculate_kelly_criterion(
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                max_kelly=max_kelly,
            )

            # Allocate cash based on Kelly
            allocations[ticker] = available_cash * kelly_pct

        # Normalize if total exceeds available cash
        total_allocated = sum(allocations.values())
        if total_allocated > available_cash:
            scale_factor = available_cash / total_allocated
            allocations = {ticker: amount * scale_factor for ticker, amount in allocations.items()}

        return allocations


def optimize_portfolio(
    returns: pd.DataFrame,
    method: str = "mpt",
    **kwargs: Any,
) -> PortfolioWeights:
    """
    Convenience function to optimize portfolio.

    Args:
        returns: DataFrame with returns (columns = tickers, index = dates)
        method: Optimization method ("mpt", "risk_parity", "kelly")
        **kwargs: Additional arguments for optimization

    Returns:
        PortfolioWeights with optimized allocation

    Example:
        Optimize using MPT::
            weights = optimize_portfolio(returns, method="mpt")

        Optimize using Risk Parity::
            weights = optimize_portfolio(returns, method="risk_parity")

        Optimize using Kelly Criterion::
            weights = optimize_portfolio(returns, method="kelly", trades=trades_df)
    """
    optimizer = PortfolioOptimizer()

    if method == "mpt":
        return optimizer.optimize_mpt(returns, **kwargs)
    elif method == "risk_parity":
        return optimizer.optimize_risk_parity(returns, **kwargs)
    elif method == "kelly":
        if "trades" not in kwargs:
            raise ValueError("Kelly method requires 'trades' DataFrame")
        trades = kwargs.pop("trades")
        available_cash = kwargs.pop("available_cash", 1.0)
        max_kelly = kwargs.pop("max_kelly", 0.25)
        allocations = optimizer.optimize_kelly_portfolio(trades, available_cash, max_kelly)
        # Convert to weights
        total = sum(allocations.values())
        weights = {ticker: amount / total if total > 0 else 0.0 for ticker, amount in allocations.items()}
        return PortfolioWeights(weights=weights, method="kelly")
    else:
        raise ValueError(f"Unknown optimization method: {method}")
