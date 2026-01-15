"""
Momentum factor implementation.

Momentum captures the tendency of assets that have performed well
to continue performing well (and vice versa).

Variants:
- Price momentum: Past returns (1M, 3M, 6M, 12M)
- Risk-adjusted momentum: Returns / volatility
- Residual momentum: Returns after removing market beta
- Time-series momentum: Trend following within asset
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.factors.base import Factor, FactorDirection, NormalizationMethod


@dataclass
class MomentumConfig:
    """Configuration for momentum factor."""

    # Lookback periods in trading days
    lookback_periods: tuple[int, ...] = (21, 63, 126, 252)

    # Skip recent days (to avoid short-term reversal)
    skip_recent: int = 5

    # Weighting scheme for multiple periods
    period_weights: tuple[float, ...] | None = None

    # Momentum type
    momentum_type: Literal["price", "risk_adjusted", "residual"] = "price"

    # For risk-adjusted momentum
    volatility_lookback: int = 21


class MomentumFactor(Factor):
    """
    Momentum factor.

    Calculates momentum based on past returns over multiple lookback periods.
    Higher momentum score indicates stronger recent performance.

    Example:
        >>> momentum = MomentumFactor(
        ...     lookback_periods=(21, 63, 126),
        ...     momentum_type="risk_adjusted",
        ... )
        >>> # prices: DataFrame with columns = tickers, rows = dates
        >>> scores = momentum.calculate(prices)
        >>> top_momentum = momentum.get_top_quintile(prices)
    """

    def __init__(
        self,
        lookback_periods: tuple[int, ...] = (21, 63, 126, 252),
        skip_recent: int = 5,
        period_weights: tuple[float, ...] | None = None,
        momentum_type: Literal["price", "risk_adjusted", "residual"] = "price",
        volatility_lookback: int = 21,
        normalization: NormalizationMethod = NormalizationMethod.ZSCORE,
    ) -> None:
        """
        Initialize momentum factor.

        Args:
            lookback_periods: Periods to calculate returns (trading days)
            skip_recent: Skip recent days to avoid reversal
            period_weights: Weights for each period (default: equal)
            momentum_type: Type of momentum calculation
            volatility_lookback: Lookback for volatility (risk-adjusted)
            normalization: Normalization method
        """
        super().__init__(
            name="momentum",
            direction=FactorDirection.LONG,
            normalization=normalization,
        )

        self.config = MomentumConfig(
            lookback_periods=lookback_periods,
            skip_recent=skip_recent,
            period_weights=period_weights,
            momentum_type=momentum_type,
            volatility_lookback=volatility_lookback,
        )

        # Default equal weights
        if self.config.period_weights is None:
            n = len(lookback_periods)
            self.config.period_weights = tuple(1.0 / n for _ in range(n))

    def calculate_raw(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate raw momentum scores.

        Args:
            data: DataFrame with price data
                  - Columns: tickers
                  - Index: dates (sorted ascending)

        Returns:
            Series with momentum scores, indexed by ticker
        """
        if data.empty:
            return pd.Series(dtype=float)

        # Ensure data is sorted
        data = data.sort_index()

        # Skip recent observations
        if self.config.skip_recent > 0:
            data = data.iloc[:-self.config.skip_recent]

        if len(data) < max(self.config.lookback_periods):
            return pd.Series(dtype=float)

        # Calculate momentum for each period
        period_momentums = []
        for period, weight in zip(
            self.config.lookback_periods, self.config.period_weights
        ):
            if len(data) < period:
                continue

            if self.config.momentum_type == "price":
                mom = self._calculate_price_momentum(data, period)
            elif self.config.momentum_type == "risk_adjusted":
                mom = self._calculate_risk_adjusted_momentum(data, period)
            elif self.config.momentum_type == "residual":
                mom = self._calculate_residual_momentum(data, period)
            else:
                mom = self._calculate_price_momentum(data, period)

            period_momentums.append(mom * weight)

        if not period_momentums:
            return pd.Series(dtype=float)

        # Combine weighted momentums
        combined = pd.concat(period_momentums, axis=1).sum(axis=1)

        return combined

    def _calculate_price_momentum(
        self,
        data: pd.DataFrame,
        period: int,
    ) -> pd.Series:
        """Calculate simple price momentum (returns)."""
        returns = data.iloc[-1] / data.iloc[-period] - 1
        return returns

    def _calculate_risk_adjusted_momentum(
        self,
        data: pd.DataFrame,
        period: int,
    ) -> pd.Series:
        """Calculate risk-adjusted momentum (return / volatility)."""
        returns = data.iloc[-1] / data.iloc[-period] - 1

        # Calculate volatility
        vol_data = data.iloc[-self.config.volatility_lookback:]
        daily_returns = vol_data.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)

        # Avoid division by zero
        volatility = volatility.replace(0, np.nan)

        return returns / volatility

    def _calculate_residual_momentum(
        self,
        data: pd.DataFrame,
        period: int,
    ) -> pd.Series:
        """
        Calculate residual momentum (alpha after removing beta).

        Uses equal-weighted market return as benchmark.
        """
        period_data = data.iloc[-period:]
        daily_returns = period_data.pct_change().dropna()

        if len(daily_returns) < 10:
            return pd.Series(dtype=float)

        # Market return (equal-weighted)
        market_return = daily_returns.mean(axis=1)

        residuals = {}
        for ticker in daily_returns.columns:
            asset_returns = daily_returns[ticker]

            # Simple beta calculation
            cov = asset_returns.cov(market_return)
            var = market_return.var()
            beta = cov / var if var > 0 else 0

            # Residual return
            expected_return = beta * market_return
            residual = (asset_returns - expected_return).sum()
            residuals[ticker] = residual

        return pd.Series(residuals)

    def calculate_momentum_strength(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate momentum strength metrics.

        Returns DataFrame with:
        - momentum: Combined momentum score
        - consistency: Fraction of positive periods
        - acceleration: Recent vs long-term momentum
        """
        if data.empty:
            return pd.DataFrame()

        data = data.sort_index()
        if self.config.skip_recent > 0:
            data = data.iloc[:-self.config.skip_recent]

        results = {}

        for ticker in data.columns:
            ticker_data = data[ticker].dropna()
            if len(ticker_data) < max(self.config.lookback_periods):
                continue

            # Momentum per period
            period_moms = []
            for period in self.config.lookback_periods:
                if len(ticker_data) >= period:
                    mom = ticker_data.iloc[-1] / ticker_data.iloc[-period] - 1
                    period_moms.append(mom)

            if not period_moms:
                continue

            # Consistency: fraction of periods with positive momentum
            consistency = sum(1 for m in period_moms if m > 0) / len(period_moms)

            # Acceleration: short-term momentum vs long-term
            if len(period_moms) >= 2:
                acceleration = period_moms[0] - period_moms[-1]
            else:
                acceleration = 0

            results[ticker] = {
                "momentum": np.mean(period_moms),
                "consistency": consistency,
                "acceleration": acceleration,
            }

        return pd.DataFrame(results).T


class TimeSeriesMomentum(Factor):
    """
    Time-series momentum (trend following).

    Unlike cross-sectional momentum, this measures an asset's own trend
    without comparing to other assets.

    Signals:
    - Positive when asset is above its moving average or has positive returns
    - Useful for timing individual assets or markets
    """

    def __init__(
        self,
        lookback: int = 252,
        signal_type: Literal["return", "ma_cross", "breakout"] = "return",
        ma_period: int = 200,
        normalization: NormalizationMethod = NormalizationMethod.ZSCORE,
    ) -> None:
        """
        Initialize time-series momentum.

        Args:
            lookback: Lookback period for return calculation
            signal_type: Type of signal generation
            ma_period: Moving average period (for ma_cross)
            normalization: Normalization method
        """
        super().__init__(
            name="ts_momentum",
            direction=FactorDirection.LONG,
            normalization=normalization,
        )
        self.lookback = lookback
        self.signal_type = signal_type
        self.ma_period = ma_period

    def calculate_raw(self, data: pd.DataFrame) -> pd.Series:
        """Calculate time-series momentum signal."""
        if data.empty:
            return pd.Series(dtype=float)

        data = data.sort_index()

        if self.signal_type == "return":
            return self._return_signal(data)
        elif self.signal_type == "ma_cross":
            return self._ma_cross_signal(data)
        elif self.signal_type == "breakout":
            return self._breakout_signal(data)

        return self._return_signal(data)

    def _return_signal(self, data: pd.DataFrame) -> pd.Series:
        """Signal based on lookback return."""
        if len(data) < self.lookback:
            return pd.Series(dtype=float)

        returns = data.iloc[-1] / data.iloc[-self.lookback] - 1
        return returns

    def _ma_cross_signal(self, data: pd.DataFrame) -> pd.Series:
        """Signal based on price vs moving average."""
        if len(data) < self.ma_period:
            return pd.Series(dtype=float)

        ma = data.iloc[-self.ma_period:].mean()
        current = data.iloc[-1]

        # Percentage above/below MA
        signal = (current - ma) / ma
        return signal

    def _breakout_signal(self, data: pd.DataFrame) -> pd.Series:
        """Signal based on breakout from range."""
        if len(data) < self.lookback:
            return pd.Series(dtype=float)

        lookback_data = data.iloc[-self.lookback:]
        high = lookback_data.max()
        low = lookback_data.min()
        current = data.iloc[-1]

        # Normalize position within range
        range_width = high - low
        range_width = range_width.replace(0, np.nan)

        signal = (current - low) / range_width - 0.5  # -0.5 to 0.5
        return signal * 2  # Scale to -1 to 1


__all__ = [
    "MomentumFactor",
    "MomentumConfig",
    "TimeSeriesMomentum",
]
