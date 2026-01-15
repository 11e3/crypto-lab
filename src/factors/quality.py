"""
Quality factor implementation.

Quality measures the reliability and consistency of returns.

For crypto assets:
- Return consistency (Sharpe-like measures)
- Trend strength (ADX-like measures)
- Volume quality (liquidity consistency)
- Price stability (lower volatility of volatility)
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.factors.base import Factor, FactorDirection, NormalizationMethod


@dataclass
class QualityConfig:
    """Configuration for quality factor."""

    # Quality metric type
    quality_type: Literal[
        "sharpe", "consistency", "trend_strength", "liquidity", "stability"
    ] = "sharpe"

    # Lookback period
    lookback: int = 60

    # For consistency
    min_positive_days_pct: float = 0.55


class QualityFactor(Factor):
    """
    Quality factor for crypto assets.

    Measures quality characteristics like:
    - Risk-adjusted returns (Sharpe)
    - Return consistency
    - Trend strength
    - Liquidity quality
    - Price stability

    Higher quality = higher score.

    Example:
        >>> quality = QualityFactor(quality_type="sharpe", lookback=60)
        >>> scores = quality.calculate(prices)
        >>> high_quality = quality.get_top_quintile(prices)
    """

    def __init__(
        self,
        quality_type: Literal[
            "sharpe", "consistency", "trend_strength", "liquidity", "stability"
        ] = "sharpe",
        lookback: int = 60,
        normalization: NormalizationMethod = NormalizationMethod.ZSCORE,
    ) -> None:
        """
        Initialize quality factor.

        Args:
            quality_type: Type of quality metric
            lookback: Lookback period
            normalization: Normalization method
        """
        super().__init__(
            name="quality",
            direction=FactorDirection.LONG,  # Higher quality = better
            normalization=normalization,
        )

        self.config = QualityConfig(
            quality_type=quality_type,
            lookback=lookback,
        )

    def calculate_raw(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate raw quality scores.

        Args:
            data: DataFrame with price data

        Returns:
            Series with quality scores (higher = better)
        """
        if data.empty:
            return pd.Series(dtype=float)

        data = data.sort_index()

        if len(data) < self.config.lookback:
            return pd.Series(dtype=float)

        data = data.iloc[-self.config.lookback:]

        if self.config.quality_type == "sharpe":
            return self._sharpe_ratio(data)
        elif self.config.quality_type == "consistency":
            return self._consistency_score(data)
        elif self.config.quality_type == "trend_strength":
            return self._trend_strength(data)
        elif self.config.quality_type == "liquidity":
            return self._liquidity_quality(data)
        elif self.config.quality_type == "stability":
            return self._stability_score(data)

        return self._sharpe_ratio(data)

    def _sharpe_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Sharpe ratio.

        Higher Sharpe = better risk-adjusted returns.
        """
        returns = data.pct_change().dropna()

        mean_return = returns.mean()
        std_return = returns.std()

        # Annualized Sharpe
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return.any() else 0
        sharpe = sharpe.replace([np.inf, -np.inf], 0)

        return sharpe

    def _consistency_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate return consistency.

        Measures the fraction of positive return days.
        """
        returns = data.pct_change().dropna()

        # Fraction of positive days
        positive_days = (returns > 0).sum()
        total_days = len(returns)

        consistency = positive_days / total_days if total_days > 0 else 0
        return consistency

    def _trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate trend strength (ADX-like measure).

        Measures how trending (vs ranging) the price action is.
        """
        returns = data.pct_change().dropna()

        # Simple trend strength: |cumulative return| / sum of |daily returns|
        cum_return = (1 + returns).prod() - 1
        sum_abs_returns = returns.abs().sum()

        # Efficiency ratio (1 = perfect trend, 0 = choppy)
        efficiency = cum_return.abs() / sum_abs_returns.replace(0, np.nan)
        efficiency = efficiency.fillna(0)

        return efficiency

    def _liquidity_quality(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate liquidity quality.

        For price-only data, use bid-ask spread proxy
        based on high-low range.
        """
        # Use return volatility as inverse liquidity proxy
        # More volatile = less liquid = lower quality
        returns = data.pct_change().dropna()
        volatility = returns.std()

        # Invert so higher = better liquidity
        liquidity_score = 1 / (1 + volatility)
        return liquidity_score

    def _stability_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate price stability.

        Measures volatility of volatility (vol-of-vol).
        More stable volatility = higher quality.
        """
        returns = data.pct_change().dropna()

        # Rolling volatility
        rolling_vol = returns.rolling(10).std()

        # Volatility of volatility
        vol_of_vol = rolling_vol.std()

        # Invert so higher = more stable
        stability = 1 / (1 + vol_of_vol)
        return stability

    def calculate_quality_metrics(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate comprehensive quality metrics.

        Returns DataFrame with multiple quality measures.
        """
        if data.empty or len(data) < self.config.lookback:
            return pd.DataFrame()

        data = data.iloc[-self.config.lookback:]
        returns = data.pct_change().dropna()

        metrics = {}

        for ticker in data.columns:
            ticker_returns = returns[ticker].dropna()
            if len(ticker_returns) < 20:
                continue

            # Sharpe ratio
            mean_ret = ticker_returns.mean()
            std_ret = ticker_returns.std()
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

            # Sortino ratio (downside risk only)
            neg_returns = ticker_returns[ticker_returns < 0]
            downside_std = neg_returns.std() if len(neg_returns) > 0 else std_ret
            sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0

            # Win rate
            win_rate = (ticker_returns > 0).mean()

            # Gain/loss ratio
            avg_gain = ticker_returns[ticker_returns > 0].mean() if (ticker_returns > 0).any() else 0
            avg_loss = abs(ticker_returns[ticker_returns < 0].mean()) if (ticker_returns < 0).any() else 1
            gain_loss_ratio = avg_gain / avg_loss if avg_loss != 0 else np.inf

            # Trend efficiency
            cum_ret = (1 + ticker_returns).prod() - 1
            sum_abs = ticker_returns.abs().sum()
            efficiency = abs(cum_ret) / sum_abs if sum_abs > 0 else 0

            # Calmar ratio proxy
            cum_returns = (1 + ticker_returns).cumprod()
            rolling_max = cum_returns.cummax()
            max_dd = ((cum_returns - rolling_max) / rolling_max).min()
            calmar = (cum_ret / abs(max_dd)) if max_dd < 0 else np.inf

            metrics[ticker] = {
                "sharpe": sharpe,
                "sortino": sortino,
                "win_rate": win_rate,
                "gain_loss_ratio": gain_loss_ratio,
                "trend_efficiency": efficiency,
                "calmar": calmar if calmar != np.inf else 10,  # Cap
            }

        return pd.DataFrame(metrics).T


class SharpeQualityFactor(Factor):
    """
    Sharpe-based quality factor.

    Simple implementation focusing on risk-adjusted returns.
    """

    def __init__(
        self,
        lookback: int = 252,
        risk_free_rate: float = 0.0,
        normalization: NormalizationMethod = NormalizationMethod.ZSCORE,
    ) -> None:
        super().__init__(
            name="sharpe_quality",
            direction=FactorDirection.LONG,
            normalization=normalization,
        )
        self.lookback = lookback
        self.risk_free_rate = risk_free_rate

    def calculate_raw(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Sharpe ratios."""
        if data.empty or len(data) < self.lookback:
            return pd.Series(dtype=float)

        data = data.iloc[-self.lookback:]
        returns = data.pct_change().dropna()

        daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
        excess_returns = returns - daily_rf

        mean_excess = excess_returns.mean()
        std_returns = returns.std()

        sharpe = (mean_excess / std_returns) * np.sqrt(252)
        sharpe = sharpe.replace([np.inf, -np.inf], 0)

        return sharpe


__all__ = [
    "QualityFactor",
    "QualityConfig",
    "SharpeQualityFactor",
]
