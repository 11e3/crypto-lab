"""
Performance attribution analysis.

Provides methods to decompose portfolio returns:
- Brinson attribution: Allocation, Selection, Interaction effects
- Factor attribution: Returns explained by factor exposures
- Risk attribution: Contribution to portfolio risk
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BrinsonResult:
    """Result of Brinson attribution analysis."""

    # Total effects
    total_excess_return: float  # Portfolio return - Benchmark return
    allocation_effect: float  # Effect from weight differences
    selection_effect: float  # Effect from security selection
    interaction_effect: float  # Interaction between allocation and selection

    # Per-asset breakdown
    asset_allocation: dict[str, float] = field(default_factory=dict)
    asset_selection: dict[str, float] = field(default_factory=dict)
    asset_interaction: dict[str, float] = field(default_factory=dict)

    # Metadata
    period_start: datetime | None = None
    period_end: datetime | None = None

    def __post_init__(self):
        # Verify attribution adds up
        total_attributed = (
            self.allocation_effect +
            self.selection_effect +
            self.interaction_effect
        )
        if abs(total_attributed - self.total_excess_return) > 0.0001:
            logger.warning(
                f"Attribution doesn't sum to excess return: "
                f"{total_attributed:.4f} vs {self.total_excess_return:.4f}"
            )

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Brinson Attribution Summary\n"
            f"{'=' * 40}\n"
            f"Total Excess Return: {self.total_excess_return:.2%}\n"
            f"  Allocation Effect: {self.allocation_effect:.2%}\n"
            f"  Selection Effect:  {self.selection_effect:.2%}\n"
            f"  Interaction Effect: {self.interaction_effect:.2%}\n"
        )


@dataclass
class FactorAttributionResult:
    """Result of factor-based attribution."""

    # Factor contributions
    factor_returns: dict[str, float]  # Factor name -> return contribution
    alpha: float  # Unexplained return (skill or luck)
    r_squared: float  # Fraction of variance explained

    # Factor exposures (betas)
    factor_exposures: dict[str, float] = field(default_factory=dict)

    # Statistical significance
    t_statistics: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)

    # Metadata
    period_start: datetime | None = None
    period_end: datetime | None = None

    @property
    def total_factor_return(self) -> float:
        """Total return explained by factors."""
        return sum(self.factor_returns.values())

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Factor Attribution Summary",
            "=" * 40,
            f"R-squared: {self.r_squared:.2%}",
            f"Alpha: {self.alpha:.2%}",
            "",
            "Factor Contributions:",
        ]

        for factor, contribution in sorted(
            self.factor_returns.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        ):
            exposure = self.factor_exposures.get(factor, 0)
            t_stat = self.t_statistics.get(factor, 0)
            sig = "***" if abs(t_stat) > 2.58 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.64 else ""
            lines.append(
                f"  {factor}: {contribution:.2%} (beta={exposure:.2f}) {sig}"
            )

        return "\n".join(lines)


@dataclass
class ReturnDecomposition:
    """Decomposition of portfolio returns by source."""

    # Return components
    total_return: float
    market_return: float  # Beta * market
    factor_return: float  # Sum of factor contributions
    alpha_return: float  # Unexplained return
    timing_return: float  # From timing decisions
    selection_return: float  # From security selection

    # Cost attribution
    trading_cost: float = 0.0
    management_fee: float = 0.0

    # Risk attribution
    systematic_risk_pct: float = 0.0  # % of risk from systematic factors
    idiosyncratic_risk_pct: float = 0.0  # % of risk from stock-specific

    def net_return(self) -> float:
        """Return after costs."""
        return self.total_return - self.trading_cost - self.management_fee


class PerformanceAttribution:
    """
    Performance attribution engine.

    Decomposes portfolio returns to understand sources of alpha and risk.

    Example:
        >>> attr = PerformanceAttribution()
        >>> brinson = attr.brinson_attribution(
        ...     portfolio_weights={"BTC": 0.6, "ETH": 0.4},
        ...     benchmark_weights={"BTC": 0.5, "ETH": 0.5},
        ...     portfolio_returns={"BTC": 0.10, "ETH": 0.05},
        ...     benchmark_returns={"BTC": 0.10, "ETH": 0.05},
        ... )
        >>> print(brinson.summary())
    """

    def __init__(self) -> None:
        """Initialize attribution engine."""
        pass

    def brinson_attribution(
        self,
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
        portfolio_returns: dict[str, float],
        benchmark_returns: dict[str, float],
        period_start: datetime | None = None,
        period_end: datetime | None = None,
    ) -> BrinsonResult:
        """
        Perform Brinson attribution analysis.

        Decomposes excess return into:
        - Allocation: Over/underweighting sectors/assets
        - Selection: Picking better/worse securities within sectors
        - Interaction: Combined effect

        Args:
            portfolio_weights: Portfolio weights by asset
            benchmark_weights: Benchmark weights by asset
            portfolio_returns: Portfolio asset returns
            benchmark_returns: Benchmark asset returns
            period_start: Start of analysis period
            period_end: End of analysis period

        Returns:
            BrinsonResult with attribution breakdown
        """
        # Get all assets
        all_assets = set(portfolio_weights.keys()) | set(benchmark_weights.keys())

        # Calculate portfolio and benchmark returns
        port_return = sum(
            portfolio_weights.get(a, 0) * portfolio_returns.get(a, 0)
            for a in all_assets
        )
        bench_return = sum(
            benchmark_weights.get(a, 0) * benchmark_returns.get(a, 0)
            for a in all_assets
        )
        excess_return = port_return - bench_return

        # Calculate effects for each asset
        allocation_effects = {}
        selection_effects = {}
        interaction_effects = {}

        for asset in all_assets:
            w_p = portfolio_weights.get(asset, 0)
            w_b = benchmark_weights.get(asset, 0)
            r_p = portfolio_returns.get(asset, 0)
            r_b = benchmark_returns.get(asset, 0)

            # Allocation effect: (w_p - w_b) * (r_b - R_b)
            # Being overweight in outperforming sectors
            allocation_effects[asset] = (w_p - w_b) * (r_b - bench_return)

            # Selection effect: w_b * (r_p - r_b)
            # Picking better securities within sectors
            selection_effects[asset] = w_b * (r_p - r_b)

            # Interaction effect: (w_p - w_b) * (r_p - r_b)
            # Combined effect of allocation and selection
            interaction_effects[asset] = (w_p - w_b) * (r_p - r_b)

        # Sum effects
        total_allocation = sum(allocation_effects.values())
        total_selection = sum(selection_effects.values())
        total_interaction = sum(interaction_effects.values())

        return BrinsonResult(
            total_excess_return=excess_return,
            allocation_effect=total_allocation,
            selection_effect=total_selection,
            interaction_effect=total_interaction,
            asset_allocation=allocation_effects,
            asset_selection=selection_effects,
            asset_interaction=interaction_effects,
            period_start=period_start,
            period_end=period_end,
        )

    def brinson_time_series(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate Brinson attribution over time.

        Args:
            portfolio_weights: DataFrame with weights (columns=assets, index=dates)
            benchmark_weights: Benchmark weights DataFrame
            portfolio_returns: Portfolio returns DataFrame
            benchmark_returns: Benchmark returns DataFrame

        Returns:
            DataFrame with attribution effects over time
        """
        results = []

        for date in portfolio_weights.index:
            if date not in benchmark_weights.index:
                continue
            if date not in portfolio_returns.index:
                continue

            result = self.brinson_attribution(
                portfolio_weights=portfolio_weights.loc[date].to_dict(),
                benchmark_weights=benchmark_weights.loc[date].to_dict(),
                portfolio_returns=portfolio_returns.loc[date].to_dict(),
                benchmark_returns=benchmark_returns.loc[date].to_dict(),
            )

            results.append({
                "date": date,
                "excess_return": result.total_excess_return,
                "allocation": result.allocation_effect,
                "selection": result.selection_effect,
                "interaction": result.interaction_effect,
            })

        return pd.DataFrame(results).set_index("date")

    def factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
        period_start: datetime | None = None,
        period_end: datetime | None = None,
    ) -> FactorAttributionResult:
        """
        Perform factor-based attribution.

        Uses regression to decompose returns into factor contributions.

        Args:
            portfolio_returns: Portfolio return series
            factor_returns: DataFrame with factor returns (columns=factors)
            risk_free_rate: Risk-free rate for alpha calculation
            period_start: Start of analysis period
            period_end: End of analysis period

        Returns:
            FactorAttributionResult with factor breakdown
        """
        # Align data
        aligned = pd.concat([portfolio_returns, factor_returns], axis=1).dropna()
        if len(aligned) < 10:
            logger.warning("Insufficient data for factor attribution")
            return FactorAttributionResult(
                factor_returns={},
                alpha=0.0,
                r_squared=0.0,
            )

        y = aligned.iloc[:, 0]  # Portfolio returns
        X = aligned.iloc[:, 1:]  # Factor returns

        # Add constant for alpha
        X_with_const = pd.concat([pd.Series(1, index=X.index, name="const"), X], axis=1)

        # OLS regression
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(
                X_with_const.values, y.values, rcond=None
            )
        except Exception as e:
            logger.error(f"Regression failed: {e}")
            return FactorAttributionResult(
                factor_returns={},
                alpha=0.0,
                r_squared=0.0,
            )

        alpha = coeffs[0]
        betas = dict(zip(X.columns, coeffs[1:]))

        # Calculate R-squared
        y_pred = X_with_const.values @ coeffs
        ss_res = np.sum((y.values - y_pred) ** 2)
        ss_tot = np.sum((y.values - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Factor contributions (beta * mean factor return)
        factor_contributions = {}
        for factor in X.columns:
            factor_contributions[factor] = betas[factor] * X[factor].mean()

        # T-statistics and p-values
        n = len(y)
        k = len(coeffs)
        if n > k:
            mse = ss_res / (n - k)
            var_coef = mse * np.linalg.inv(X_with_const.T @ X_with_const).diagonal()
            se = np.sqrt(var_coef)

            t_stats = coeffs / se
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))

            t_statistics = {"alpha": t_stats[0]}
            p_vals = {"alpha": p_values[0]}
            for i, factor in enumerate(X.columns):
                t_statistics[factor] = t_stats[i + 1]
                p_vals[factor] = p_values[i + 1]
        else:
            t_statistics = {}
            p_vals = {}

        return FactorAttributionResult(
            factor_returns=factor_contributions,
            alpha=alpha * 252,  # Annualize daily alpha
            r_squared=r_squared,
            factor_exposures=betas,
            t_statistics=t_statistics,
            p_values=p_vals,
            period_start=period_start,
            period_end=period_end,
        )

    def rolling_factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        window: int = 60,
    ) -> pd.DataFrame:
        """
        Calculate rolling factor attribution.

        Args:
            portfolio_returns: Portfolio return series
            factor_returns: Factor return DataFrame
            window: Rolling window size

        Returns:
            DataFrame with rolling exposures and alpha
        """
        results = []

        for i in range(window, len(portfolio_returns)):
            window_port = portfolio_returns.iloc[i-window:i]
            window_factors = factor_returns.iloc[i-window:i]

            attr = self.factor_attribution(window_port, window_factors)

            result = {
                "date": portfolio_returns.index[i],
                "alpha": attr.alpha,
                "r_squared": attr.r_squared,
            }
            for factor, exposure in attr.factor_exposures.items():
                result[f"{factor}_beta"] = exposure
            for factor, contrib in attr.factor_returns.items():
                result[f"{factor}_contrib"] = contrib

            results.append(result)

        return pd.DataFrame(results).set_index("date")

    def risk_attribution(
        self,
        portfolio_weights: dict[str, float],
        covariance_matrix: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Calculate risk contribution by asset.

        Each asset's contribution to total portfolio volatility.

        Args:
            portfolio_weights: Portfolio weights
            covariance_matrix: Asset covariance matrix

        Returns:
            Dict of asset -> risk contribution (percentage)
        """
        assets = list(portfolio_weights.keys())
        weights = np.array([portfolio_weights[a] for a in assets])

        # Subset covariance matrix to match assets
        cov = covariance_matrix.loc[assets, assets].values

        # Portfolio variance
        port_var = weights @ cov @ weights
        port_vol = np.sqrt(port_var)

        # Marginal contribution to risk
        mctr = cov @ weights / port_vol

        # Component contribution
        cctr = weights * mctr

        # Percentage contribution
        risk_pct = cctr / port_vol

        return dict(zip(assets, risk_pct))

    def return_decomposition(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: pd.DataFrame | None = None,
        trading_costs: float = 0.0,
        management_fee: float = 0.0,
    ) -> ReturnDecomposition:
        """
        Comprehensive return decomposition.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            factor_returns: Factor returns for factor attribution
            trading_costs: Total trading costs over period
            management_fee: Management fees over period

        Returns:
            ReturnDecomposition with breakdown
        """
        total_return = (1 + portfolio_returns).prod() - 1
        market_return = (1 + benchmark_returns).prod() - 1

        # Factor attribution if available
        if factor_returns is not None:
            factor_attr = self.factor_attribution(portfolio_returns, factor_returns)
            factor_return = factor_attr.total_factor_return
            alpha_return = factor_attr.alpha
            r_squared = factor_attr.r_squared
        else:
            factor_return = 0.0
            alpha_return = total_return - market_return
            r_squared = 0.0

        # Timing vs selection (simplified)
        timing_return = 0.0  # Would need weight changes to calculate
        selection_return = total_return - market_return - timing_return

        return ReturnDecomposition(
            total_return=total_return,
            market_return=market_return,
            factor_return=factor_return,
            alpha_return=alpha_return,
            timing_return=timing_return,
            selection_return=selection_return,
            trading_cost=trading_costs,
            management_fee=management_fee,
            systematic_risk_pct=r_squared,
            idiosyncratic_risk_pct=1 - r_squared,
        )


__all__ = [
    "PerformanceAttribution",
    "BrinsonResult",
    "FactorAttributionResult",
    "ReturnDecomposition",
]
