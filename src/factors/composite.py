"""
Composite factor model implementation.

Combines multiple factors into a single composite score.

Methods:
- Equal weighting
- IC (Information Coefficient) weighting
- Optimized weighting
- Machine learning based
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from src.factors.base import (
    Factor,
    FactorDirection,
    NormalizationMethod,
    calculate_factor_ic,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FactorWeight:
    """Weight configuration for a factor."""

    factor_name: str
    weight: float = 1.0
    active: bool = True

    # Dynamic weighting
    use_ic_weight: bool = False
    ic_lookback: int = 60
    ic_decay: float = 0.94  # Exponential decay for IC averaging


@dataclass
class CompositeScore:
    """Composite factor score for an asset."""

    ticker: str
    composite_score: float
    factor_contributions: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    percentile: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class CompositeFactorModel:
    """
    Composite factor model.

    Combines multiple factors into a unified scoring system.

    Example:
        >>> from src.factors import MomentumFactor, VolatilityFactor, QualityFactor
        >>>
        >>> model = CompositeFactorModel(
        ...     factors=[
        ...         MomentumFactor(),
        ...         VolatilityFactor(),
        ...         QualityFactor(),
        ...     ],
        ...     weights=[
        ...         FactorWeight("momentum", weight=0.4),
        ...         FactorWeight("volatility", weight=0.3),
        ...         FactorWeight("quality", weight=0.3),
        ...     ],
        ... )
        >>> composite_scores = model.calculate(price_data)
        >>> top_picks = model.get_top_n(price_data, n=10)
    """

    def __init__(
        self,
        factors: list[Factor],
        weights: list[FactorWeight] | None = None,
        combination_method: Literal["weighted", "rank", "z_score"] = "z_score",
        normalization: NormalizationMethod = NormalizationMethod.ZSCORE,
    ) -> None:
        """
        Initialize composite factor model.

        Args:
            factors: List of Factor instances
            weights: List of FactorWeight configs (default: equal weights)
            combination_method: How to combine factor scores
            normalization: Final normalization method
        """
        self.factors = {f.name: f for f in factors}
        self.combination_method = combination_method
        self.normalization = normalization

        # Set up weights
        if weights is None:
            n = len(factors)
            self.weights = {
                f.name: FactorWeight(f.name, weight=1.0 / n)
                for f in factors
            }
        else:
            self.weights = {w.factor_name: w for w in weights}

        # Validate weights match factors
        for factor_name in self.factors:
            if factor_name not in self.weights:
                self.weights[factor_name] = FactorWeight(
                    factor_name, weight=1.0 / len(self.factors)
                )

        # IC history for dynamic weighting
        self._ic_history: dict[str, list[float]] = {
            name: [] for name in self.factors
        }

    def calculate(
        self,
        data: pd.DataFrame,
        forward_returns: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Calculate composite factor scores.

        Args:
            data: Price data DataFrame
            forward_returns: Forward returns for IC calculation (optional)

        Returns:
            DataFrame with composite scores and factor contributions
        """
        factor_scores = {}

        # Calculate each factor
        for name, factor in self.factors.items():
            weight_config = self.weights.get(name)
            if weight_config and not weight_config.active:
                continue

            scores = factor.calculate(data)
            if not scores.empty:
                factor_scores[name] = scores["normalized"]

                # Update IC history if forward returns provided
                if forward_returns is not None and weight_config and weight_config.use_ic_weight:
                    ic = calculate_factor_ic(scores["normalized"], forward_returns)
                    self._ic_history[name].append(ic)

        if not factor_scores:
            return pd.DataFrame(columns=["composite", "rank", "percentile"])

        # Get effective weights
        effective_weights = self._get_effective_weights()

        # Combine scores
        scores_df = pd.DataFrame(factor_scores)
        composite = self._combine_scores(scores_df, effective_weights)

        # Rank and percentile
        ranks = composite.rank(ascending=False)
        percentiles = composite.rank(pct=True)

        result = pd.DataFrame({
            "composite": composite,
            "rank": ranks.astype(int),
            "percentile": percentiles,
        })

        # Add factor contributions
        for name in factor_scores:
            weight = effective_weights.get(name, 0)
            result[f"{name}_contribution"] = factor_scores[name] * weight

        return result

    def get_scores(
        self,
        data: pd.DataFrame,
        timestamp: datetime | None = None,
    ) -> list[CompositeScore]:
        """Get CompositeScore objects for all assets."""
        timestamp = timestamp or datetime.now()
        scores_df = self.calculate(data)

        result = []
        for ticker, row in scores_df.iterrows():
            contributions = {
                col.replace("_contribution", ""): row[col]
                for col in scores_df.columns
                if col.endswith("_contribution")
            }

            result.append(CompositeScore(
                ticker=str(ticker),
                composite_score=row["composite"],
                factor_contributions=contributions,
                rank=int(row["rank"]),
                percentile=row["percentile"],
                timestamp=timestamp,
            ))

        return result

    def get_top_n(self, data: pd.DataFrame, n: int = 10) -> list[str]:
        """Get top N tickers by composite score."""
        scores = self.calculate(data)
        return scores.nsmallest(n, "rank").index.tolist()

    def get_bottom_n(self, data: pd.DataFrame, n: int = 10) -> list[str]:
        """Get bottom N tickers by composite score."""
        scores = self.calculate(data)
        return scores.nlargest(n, "rank").index.tolist()

    def get_factor_exposures(
        self,
        data: pd.DataFrame,
        portfolio_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculate portfolio's exposure to each factor.

        Args:
            data: Price data
            portfolio_weights: Current portfolio weights

        Returns:
            Dict of factor_name -> exposure
        """
        exposures = {}

        for name, factor in self.factors.items():
            scores = factor.calculate(data)
            if scores.empty:
                exposures[name] = 0.0
                continue

            # Weighted average of factor scores
            exposure = 0.0
            for ticker, weight in portfolio_weights.items():
                if ticker in scores.index:
                    exposure += weight * scores.loc[ticker, "normalized"]

            exposures[name] = exposure

        return exposures

    def _combine_scores(
        self,
        scores_df: pd.DataFrame,
        weights: dict[str, float],
    ) -> pd.Series:
        """Combine factor scores using configured method."""
        if self.combination_method == "weighted":
            # Simple weighted average
            combined = pd.Series(0.0, index=scores_df.index)
            for name in scores_df.columns:
                weight = weights.get(name, 0)
                combined += scores_df[name].fillna(0) * weight
            return combined

        elif self.combination_method == "rank":
            # Average of ranks
            ranks = scores_df.rank(ascending=False)
            combined = pd.Series(0.0, index=scores_df.index)
            for name in ranks.columns:
                weight = weights.get(name, 0)
                combined += ranks[name].fillna(ranks[name].max()) * weight
            # Invert so lower rank = higher score
            return -combined

        elif self.combination_method == "z_score":
            # Weighted average of z-scores
            combined = pd.Series(0.0, index=scores_df.index)
            for name in scores_df.columns:
                weight = weights.get(name, 0)
                # Already z-scored in factor calculation
                combined += scores_df[name].fillna(0) * weight
            return combined

        return scores_df.mean(axis=1)

    def _get_effective_weights(self) -> dict[str, float]:
        """Get effective weights (possibly IC-adjusted)."""
        weights = {}
        total_weight = 0.0

        for name, config in self.weights.items():
            if not config.active:
                continue

            if config.use_ic_weight and self._ic_history.get(name):
                # Exponentially weighted IC average
                ics = self._ic_history[name][-config.ic_lookback:]
                if ics:
                    decay_weights = [
                        config.ic_decay ** i for i in range(len(ics) - 1, -1, -1)
                    ]
                    ic_weight = sum(
                        ic * dw for ic, dw in zip(ics, decay_weights)
                    ) / sum(decay_weights)
                    weight = max(0, ic_weight) * config.weight
                else:
                    weight = config.weight
            else:
                weight = config.weight

            weights[name] = weight
            total_weight += weight

        # Normalize to sum to 1
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def set_weight(self, factor_name: str, weight: float) -> None:
        """Set weight for a factor."""
        if factor_name in self.weights:
            self.weights[factor_name].weight = weight
        else:
            self.weights[factor_name] = FactorWeight(factor_name, weight=weight)

    def toggle_factor(self, factor_name: str, active: bool) -> None:
        """Enable or disable a factor."""
        if factor_name in self.weights:
            self.weights[factor_name].active = active

    def get_factor_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation between factors."""
        factor_scores = {}

        for name, factor in self.factors.items():
            scores = factor.calculate(data)
            if not scores.empty:
                factor_scores[name] = scores["normalized"]

        if not factor_scores:
            return pd.DataFrame()

        scores_df = pd.DataFrame(factor_scores)
        return scores_df.corr()

    def backtest_factor(
        self,
        data: pd.DataFrame,
        returns: pd.DataFrame,
        holding_period: int = 21,
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        """
        Backtest individual factors.

        Args:
            data: Price data
            returns: Forward returns
            holding_period: Holding period in days
            n_quantiles: Number of quantiles for analysis

        Returns:
            DataFrame with factor performance metrics
        """
        results = []

        for name, factor in self.factors.items():
            scores = factor.calculate(data)
            if scores.empty:
                continue

            # Calculate IC
            forward_rets = returns.iloc[holding_period] if len(returns) > holding_period else returns.iloc[-1]
            ic = calculate_factor_ic(scores["normalized"], forward_rets)

            # Quantile returns
            aligned = pd.DataFrame({
                "score": scores["normalized"],
                "return": forward_rets,
            }).dropna()

            if len(aligned) < n_quantiles:
                continue

            aligned["quantile"] = pd.qcut(aligned["score"], n_quantiles, labels=False)
            q_returns = aligned.groupby("quantile")["return"].mean()

            spread = q_returns.iloc[-1] - q_returns.iloc[0] if len(q_returns) >= 2 else 0

            results.append({
                "factor": name,
                "ic": ic,
                "spread": spread,
                "top_quintile_return": q_returns.iloc[-1] if len(q_returns) > 0 else 0,
                "bottom_quintile_return": q_returns.iloc[0] if len(q_returns) > 0 else 0,
            })

        return pd.DataFrame(results)


def create_standard_model(
    momentum_weight: float = 0.30,
    volatility_weight: float = 0.25,
    value_weight: float = 0.20,
    quality_weight: float = 0.25,
) -> CompositeFactorModel:
    """
    Create a standard multi-factor model.

    Uses common factor definitions with customizable weights.
    """
    from src.factors.momentum import MomentumFactor
    from src.factors.volatility import VolatilityFactor
    from src.factors.value import ValueFactor
    from src.factors.quality import QualityFactor

    factors = [
        MomentumFactor(lookback_periods=(21, 63, 126)),
        VolatilityFactor(lookback=60),
        ValueFactor(value_type="price_to_ma", ma_period=200),
        QualityFactor(quality_type="sharpe", lookback=60),
    ]

    weights = [
        FactorWeight("momentum", weight=momentum_weight),
        FactorWeight("volatility", weight=volatility_weight),
        FactorWeight("value", weight=value_weight),
        FactorWeight("quality", weight=quality_weight),
    ]

    return CompositeFactorModel(
        factors=factors,
        weights=weights,
        combination_method="z_score",
    )


__all__ = [
    "FactorWeight",
    "CompositeScore",
    "CompositeFactorModel",
    "create_standard_model",
]
