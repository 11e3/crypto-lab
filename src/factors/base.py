"""
Base classes for factor-based investing.

Provides abstract Factor class and common data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd


class FactorDirection(str, Enum):
    """Direction of factor signal."""

    LONG = "long"  # Higher score = buy
    SHORT = "short"  # Higher score = sell
    NEUTRAL = "neutral"  # No directional preference


class NormalizationMethod(str, Enum):
    """Method for normalizing factor scores."""

    ZSCORE = "zscore"  # Cross-sectional z-score
    RANK = "rank"  # Percentile rank (0-1)
    MINMAX = "minmax"  # Min-max scaling (0-1)
    ROBUST = "robust"  # Robust z-score (median, MAD)


@dataclass
class FactorScore:
    """Factor score for a single asset at a point in time."""

    ticker: str
    factor_name: str
    raw_score: float
    normalized_score: float
    rank: int  # Rank among universe (1 = best)
    percentile: float  # Percentile (0-1)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_top_quintile(self) -> bool:
        """Check if in top 20%."""
        return self.percentile >= 0.80

    @property
    def is_bottom_quintile(self) -> bool:
        """Check if in bottom 20%."""
        return self.percentile <= 0.20


@dataclass
class FactorExposure:
    """Portfolio's exposure to a factor."""

    factor_name: str
    exposure: float  # Portfolio's factor loading
    benchmark_exposure: float = 0.0
    active_exposure: float = 0.0  # exposure - benchmark_exposure
    t_statistic: float | None = None
    p_value: float | None = None

    def __post_init__(self):
        self.active_exposure = self.exposure - self.benchmark_exposure


class Factor(ABC):
    """
    Abstract base class for investment factors.

    Factors capture systematic sources of return and risk.
    Each factor calculates scores for assets that can be used
    for ranking, portfolio construction, and risk analysis.

    Example:
        >>> momentum = MomentumFactor(lookback_periods=[21, 63, 126])
        >>> scores = momentum.calculate(price_data)
        >>> top_quintile = momentum.get_top_quintile(scores)
    """

    def __init__(
        self,
        name: str,
        direction: FactorDirection = FactorDirection.LONG,
        normalization: NormalizationMethod = NormalizationMethod.ZSCORE,
    ) -> None:
        """
        Initialize factor.

        Args:
            name: Factor name
            direction: Direction of factor signal
            normalization: Normalization method for scores
        """
        self.name = name
        self.direction = direction
        self.normalization = normalization

    @abstractmethod
    def calculate_raw(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate raw factor values.

        Args:
            data: DataFrame with required data (implementation-specific)

        Returns:
            Series with raw factor values, indexed by ticker
        """
        pass

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate normalized factor scores.

        Args:
            data: DataFrame with required data

        Returns:
            DataFrame with columns:
            - raw: Raw factor values
            - normalized: Normalized scores
            - rank: Cross-sectional rank
            - percentile: Percentile (0-1)
        """
        raw_values = self.calculate_raw(data)

        # Remove NaN/Inf
        raw_values = raw_values.replace([np.inf, -np.inf], np.nan).dropna()

        if len(raw_values) == 0:
            return pd.DataFrame(columns=["raw", "normalized", "rank", "percentile"])

        # Normalize
        normalized = self._normalize(raw_values)

        # Rank (1 = best based on direction)
        if self.direction == FactorDirection.LONG:
            ranks = raw_values.rank(ascending=False)
        else:
            ranks = raw_values.rank(ascending=True)

        # Percentile
        percentiles = raw_values.rank(pct=True)
        if self.direction == FactorDirection.SHORT:
            percentiles = 1 - percentiles

        result = pd.DataFrame(
            {
                "raw": raw_values,
                "normalized": normalized,
                "rank": ranks.astype(int),
                "percentile": percentiles,
            }
        )

        return result

    def get_scores(
        self,
        data: pd.DataFrame,
        timestamp: datetime | None = None,
    ) -> list[FactorScore]:
        """
        Get FactorScore objects for all assets.

        Args:
            data: DataFrame with required data
            timestamp: Timestamp for scores

        Returns:
            List of FactorScore objects
        """
        timestamp = timestamp or datetime.now()
        scores_df = self.calculate(data)

        return [
            FactorScore(
                ticker=ticker,
                factor_name=self.name,
                raw_score=row["raw"],
                normalized_score=row["normalized"],
                rank=int(row["rank"]),
                percentile=row["percentile"],
                timestamp=timestamp,
            )
            for ticker, row in scores_df.iterrows()
        ]

    def get_top_n(self, data: pd.DataFrame, n: int = 10) -> list[str]:
        """Get top N tickers by factor score."""
        scores = self.calculate(data)
        return scores.nsmallest(n, "rank").index.tolist()

    def get_bottom_n(self, data: pd.DataFrame, n: int = 10) -> list[str]:
        """Get bottom N tickers by factor score."""
        scores = self.calculate(data)
        return scores.nlargest(n, "rank").index.tolist()

    def get_top_quintile(self, data: pd.DataFrame) -> list[str]:
        """Get tickers in top 20%."""
        scores = self.calculate(data)
        return scores[scores["percentile"] >= 0.80].index.tolist()

    def get_bottom_quintile(self, data: pd.DataFrame) -> list[str]:
        """Get tickers in bottom 20%."""
        scores = self.calculate(data)
        return scores[scores["percentile"] <= 0.20].index.tolist()

    def _normalize(self, values: pd.Series) -> pd.Series:
        """Normalize values using configured method."""
        if len(values) == 0:
            return values

        if self.normalization == NormalizationMethod.ZSCORE:
            mean = values.mean()
            std = values.std()
            if std == 0:
                return pd.Series(0, index=values.index)
            return (values - mean) / std

        elif self.normalization == NormalizationMethod.RANK:
            return values.rank(pct=True)

        elif self.normalization == NormalizationMethod.MINMAX:
            min_val = values.min()
            max_val = values.max()
            if max_val == min_val:
                return pd.Series(0.5, index=values.index)
            return (values - min_val) / (max_val - min_val)

        elif self.normalization == NormalizationMethod.ROBUST:
            median = values.median()
            mad = (values - median).abs().median()
            if mad == 0:
                return pd.Series(0, index=values.index)
            return (values - median) / (1.4826 * mad)

        return values

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', direction='{self.direction.value}')"


def calculate_factor_correlation(
    factor_scores: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Calculate correlation between factors.

    Args:
        factor_scores: Dict of factor_name -> scores Series

    Returns:
        Correlation matrix DataFrame
    """
    scores_df = pd.DataFrame(factor_scores)
    return scores_df.corr()


def calculate_factor_ic(
    factor_scores: pd.Series,
    forward_returns: pd.Series,
    method: Literal["pearson", "spearman"] = "spearman",
) -> float:
    """
    Calculate Information Coefficient (IC).

    IC measures correlation between factor scores and subsequent returns.

    Args:
        factor_scores: Factor scores
        forward_returns: Forward returns (aligned with scores)
        method: Correlation method

    Returns:
        Information coefficient
    """
    aligned = pd.DataFrame({"score": factor_scores, "return": forward_returns}).dropna()

    if len(aligned) < 3:
        return 0.0

    if method == "spearman":
        return float(aligned["score"].corr(aligned["return"], method="spearman"))
    return float(aligned["score"].corr(aligned["return"]))


def calculate_factor_returns(
    factor_scores: pd.Series,
    returns: pd.Series,
    n_quantiles: int = 5,
) -> pd.Series:
    """
    Calculate factor returns by quantile.

    Long top quantile, short bottom quantile.

    Args:
        factor_scores: Factor scores
        returns: Asset returns
        n_quantiles: Number of quantiles

    Returns:
        Factor return (long-short spread)
    """
    aligned = pd.DataFrame({"score": factor_scores, "return": returns}).dropna()

    if len(aligned) < n_quantiles:
        return pd.Series(dtype=float)

    # Create quantiles
    aligned["quantile"] = pd.qcut(aligned["score"], n_quantiles, labels=False)

    # Calculate quantile returns
    quantile_returns = aligned.groupby("quantile")["return"].mean()

    # Long-short return
    long_return = quantile_returns.iloc[-1]  # Top quantile
    short_return = quantile_returns.iloc[0]  # Bottom quantile

    return long_return - short_return


__all__ = [
    "Factor",
    "FactorScore",
    "FactorExposure",
    "FactorDirection",
    "NormalizationMethod",
    "calculate_factor_correlation",
    "calculate_factor_ic",
    "calculate_factor_returns",
]
