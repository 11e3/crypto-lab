"""
Value factor implementation.

Value investing: buying assets that are cheap relative to fundamentals.

For crypto assets without traditional fundamentals:
- Network value to transactions (NVT)
- Price to realized value
- MVRV (Market Value to Realized Value)
- Price relative to moving averages (mean reversion proxy)
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.factors.base import Factor, FactorDirection, NormalizationMethod


@dataclass
class ValueConfig:
    """Configuration for value factor."""

    # Value metric type
    value_type: Literal[
        "price_to_ma", "distance_from_high", "mean_reversion", "nvt", "mvrv"
    ] = "price_to_ma"

    # For price_to_ma
    ma_period: int = 200

    # For distance_from_high
    high_lookback: int = 252

    # For mean_reversion
    z_score_lookback: int = 60


class ValueFactor(Factor):
    """
    Value factor for crypto assets.

    Since crypto lacks traditional fundamentals (P/E, P/B),
    we use price-based proxies for value:
    - Price relative to moving average
    - Distance from all-time high
    - Mean reversion metrics

    Lower price relative to these measures = higher value score.

    Example:
        >>> value = ValueFactor(value_type="price_to_ma", ma_period=200)
        >>> scores = value.calculate(prices)
        >>> undervalued = value.get_top_quintile(prices)
    """

    def __init__(
        self,
        value_type: Literal[
            "price_to_ma", "distance_from_high", "mean_reversion", "nvt", "mvrv"
        ] = "price_to_ma",
        ma_period: int = 200,
        high_lookback: int = 252,
        z_score_lookback: int = 60,
        normalization: NormalizationMethod = NormalizationMethod.ZSCORE,
    ) -> None:
        """
        Initialize value factor.

        Args:
            value_type: Type of value calculation
            ma_period: MA period for price_to_ma
            high_lookback: Lookback for distance_from_high
            z_score_lookback: Lookback for mean_reversion
            normalization: Normalization method
        """
        # Lower valuation = better value = SHORT direction
        super().__init__(
            name="value",
            direction=FactorDirection.SHORT,
            normalization=normalization,
        )

        self.config = ValueConfig(
            value_type=value_type,
            ma_period=ma_period,
            high_lookback=high_lookback,
            z_score_lookback=z_score_lookback,
        )

    def calculate_raw(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate raw value scores.

        Args:
            data: DataFrame with price data

        Returns:
            Series with value metrics (lower = cheaper = better)
        """
        if data.empty:
            return pd.Series(dtype=float)

        data = data.sort_index()

        if self.config.value_type == "price_to_ma":
            return self._price_to_ma(data)
        elif self.config.value_type == "distance_from_high":
            return self._distance_from_high(data)
        elif self.config.value_type == "mean_reversion":
            return self._mean_reversion_score(data)
        elif self.config.value_type == "nvt":
            return self._nvt_ratio(data)
        elif self.config.value_type == "mvrv":
            return self._mvrv_ratio(data)

        return self._price_to_ma(data)

    def _price_to_ma(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate price relative to moving average.

        Lower = trading below MA = potentially undervalued.
        """
        if len(data) < self.config.ma_period:
            return pd.Series(dtype=float)

        ma = data.iloc[-self.config.ma_period:].mean()
        current = data.iloc[-1]

        # Ratio: current price / MA (lower = cheaper)
        ratio = current / ma
        return ratio

    def _distance_from_high(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate distance from recent high.

        Greater distance = potentially undervalued.
        Inverted so lower score = closer to high = expensive.
        """
        if len(data) < self.config.high_lookback:
            lookback_data = data
        else:
            lookback_data = data.iloc[-self.config.high_lookback:]

        high = lookback_data.max()
        current = data.iloc[-1]

        # Ratio: current / high (lower = further from high = cheaper)
        ratio = current / high
        return ratio

    def _mean_reversion_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate mean reversion z-score.

        Negative z-score = below mean = potentially undervalued.
        """
        if len(data) < self.config.z_score_lookback:
            return pd.Series(dtype=float)

        lookback_data = data.iloc[-self.config.z_score_lookback:]
        mean = lookback_data.mean()
        std = lookback_data.std()

        current = data.iloc[-1]

        # Z-score (negative = below mean = cheap)
        z_score = (current - mean) / std.replace(0, np.nan)
        return z_score

    def _nvt_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        Network Value to Transactions ratio proxy.

        Uses volume as proxy for transaction value.
        Higher NVT = potentially overvalued.

        Note: This requires volume data in the DataFrame.
        """
        # This is a placeholder - actual implementation needs
        # on-chain data or volume data
        return pd.Series(dtype=float)

    def _mvrv_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        Market Value to Realized Value proxy.

        Uses price relative to volume-weighted average as proxy.

        Note: This requires additional data for accurate calculation.
        """
        # Placeholder for MVRV calculation
        return pd.Series(dtype=float)

    def calculate_value_metrics(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate comprehensive value metrics.

        Returns DataFrame with multiple value measures.
        """
        if data.empty:
            return pd.DataFrame()

        metrics = {}

        for ticker in data.columns:
            ticker_data = data[ticker].dropna()
            if len(ticker_data) < 20:
                continue

            current = ticker_data.iloc[-1]

            # Price to various MAs
            ma_50 = ticker_data.iloc[-50:].mean() if len(ticker_data) >= 50 else np.nan
            ma_200 = ticker_data.iloc[-200:].mean() if len(ticker_data) >= 200 else np.nan

            # Distance from high
            high_52w = ticker_data.iloc[-252:].max() if len(ticker_data) >= 252 else ticker_data.max()
            low_52w = ticker_data.iloc[-252:].min() if len(ticker_data) >= 252 else ticker_data.min()

            # Z-score
            mean_60 = ticker_data.iloc[-60:].mean() if len(ticker_data) >= 60 else np.nan
            std_60 = ticker_data.iloc[-60:].std() if len(ticker_data) >= 60 else np.nan

            metrics[ticker] = {
                "price_to_ma50": current / ma_50 if ma_50 else np.nan,
                "price_to_ma200": current / ma_200 if ma_200 else np.nan,
                "pct_from_high": (current - high_52w) / high_52w,
                "pct_from_low": (current - low_52w) / low_52w,
                "z_score_60d": (current - mean_60) / std_60 if std_60 else np.nan,
            }

        return pd.DataFrame(metrics).T


class RelativeValueFactor(Factor):
    """
    Relative value factor.

    Compares assets within a group/sector to find relative mispricings.
    """

    def __init__(
        self,
        comparison_period: int = 60,
        normalization: NormalizationMethod = NormalizationMethod.ZSCORE,
    ) -> None:
        super().__init__(
            name="relative_value",
            direction=FactorDirection.SHORT,
            normalization=normalization,
        )
        self.comparison_period = comparison_period

    def calculate_raw(self, data: pd.DataFrame) -> pd.Series:
        """Calculate relative value within universe."""
        if data.empty or len(data) < self.comparison_period:
            return pd.Series(dtype=float)

        # Calculate returns over period
        returns = data.iloc[-1] / data.iloc[-self.comparison_period] - 1

        # Cross-sectional z-score of returns
        # Lower return = relatively cheaper
        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return pd.Series(0, index=returns.index)

        z_scores = (returns - mean_return) / std_return
        return z_scores  # Lower z-score = underperformed = potentially cheap


__all__ = [
    "ValueFactor",
    "ValueConfig",
    "RelativeValueFactor",
]
