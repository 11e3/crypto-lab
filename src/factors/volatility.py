"""
Volatility factor implementation.

Low volatility anomaly: historically, lower volatility assets
have delivered better risk-adjusted returns.

Variants:
- Historical volatility: Standard deviation of returns
- Beta: Sensitivity to market movements
- Idiosyncratic volatility: Volatility after removing market factor
- Downside volatility: Volatility of negative returns only
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.factors.base import Factor, FactorDirection, NormalizationMethod


@dataclass
class VolatilityConfig:
    """Configuration for volatility factor."""

    # Lookback period in trading days
    lookback: int = 60

    # Volatility type
    volatility_type: Literal[
        "historical", "beta", "idiosyncratic", "downside", "range"
    ] = "historical"

    # Annualization factor
    annualize: bool = True
    trading_days_per_year: int = 252

    # For downside volatility
    target_return: float = 0.0

    # For beta calculation
    market_ticker: str | None = None


class VolatilityFactor(Factor):
    """
    Volatility factor (Low Volatility Anomaly).

    Lower volatility stocks tend to outperform higher volatility stocks
    on a risk-adjusted basis.

    The factor is inverted so higher score = lower volatility = better.

    Example:
        >>> vol_factor = VolatilityFactor(
        ...     lookback=60,
        ...     volatility_type="historical",
        ... )
        >>> scores = vol_factor.calculate(prices)
        >>> low_vol_stocks = vol_factor.get_top_quintile(prices)
    """

    def __init__(
        self,
        lookback: int = 60,
        volatility_type: Literal[
            "historical", "beta", "idiosyncratic", "downside", "range"
        ] = "historical",
        annualize: bool = True,
        target_return: float = 0.0,
        market_ticker: str | None = None,
        normalization: NormalizationMethod = NormalizationMethod.ZSCORE,
    ) -> None:
        """
        Initialize volatility factor.

        Args:
            lookback: Lookback period in trading days
            volatility_type: Type of volatility calculation
            annualize: Whether to annualize volatility
            target_return: Target return for downside volatility
            market_ticker: Market proxy ticker for beta calculation
            normalization: Normalization method
        """
        # LOW volatility is good, so we invert (SHORT direction)
        super().__init__(
            name="volatility",
            direction=FactorDirection.SHORT,  # Lower vol = higher score
            normalization=normalization,
        )

        self.config = VolatilityConfig(
            lookback=lookback,
            volatility_type=volatility_type,
            annualize=annualize,
            target_return=target_return,
            market_ticker=market_ticker,
        )

    def calculate_raw(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate raw volatility values.

        Args:
            data: DataFrame with price data
                  - Columns: tickers
                  - Index: dates

        Returns:
            Series with volatility values, indexed by ticker
            (Note: lower is better, handled by direction=SHORT)
        """
        if data.empty:
            return pd.Series(dtype=float)

        data = data.sort_index()

        if len(data) < self.config.lookback:
            return pd.Series(dtype=float)

        # Use lookback period
        data = data.iloc[-self.config.lookback:]

        if self.config.volatility_type == "historical":
            return self._historical_volatility(data)
        elif self.config.volatility_type == "beta":
            return self._beta(data)
        elif self.config.volatility_type == "idiosyncratic":
            return self._idiosyncratic_volatility(data)
        elif self.config.volatility_type == "downside":
            return self._downside_volatility(data)
        elif self.config.volatility_type == "range":
            return self._range_volatility(data)

        return self._historical_volatility(data)

    def _historical_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate historical volatility (std of returns)."""
        returns = data.pct_change().dropna()
        volatility = returns.std()

        if self.config.annualize:
            volatility *= np.sqrt(self.config.trading_days_per_year)

        return volatility

    def _beta(self, data: pd.DataFrame) -> pd.Series:
        """Calculate beta relative to market."""
        returns = data.pct_change().dropna()

        # Market return
        if self.config.market_ticker and self.config.market_ticker in returns.columns:
            market_returns = returns[self.config.market_ticker]
        else:
            # Use equal-weighted average as market proxy
            market_returns = returns.mean(axis=1)

        betas = {}
        market_var = market_returns.var()

        for ticker in returns.columns:
            if ticker == self.config.market_ticker:
                betas[ticker] = 1.0
                continue

            cov = returns[ticker].cov(market_returns)
            beta = cov / market_var if market_var > 0 else 0
            betas[ticker] = abs(beta)  # Use absolute beta

        return pd.Series(betas)

    def _idiosyncratic_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate idiosyncratic (residual) volatility."""
        returns = data.pct_change().dropna()

        # Market return
        if self.config.market_ticker and self.config.market_ticker in returns.columns:
            market_returns = returns[self.config.market_ticker]
        else:
            market_returns = returns.mean(axis=1)

        market_var = market_returns.var()
        idio_vols = {}

        for ticker in returns.columns:
            asset_returns = returns[ticker]

            # Calculate beta
            cov = asset_returns.cov(market_returns)
            beta = cov / market_var if market_var > 0 else 0

            # Residual returns
            residuals = asset_returns - beta * market_returns
            idio_vol = residuals.std()

            if self.config.annualize:
                idio_vol *= np.sqrt(self.config.trading_days_per_year)

            idio_vols[ticker] = idio_vol

        return pd.Series(idio_vols)

    def _downside_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate downside volatility (semi-deviation)."""
        returns = data.pct_change().dropna()

        # Filter negative returns
        negative_returns = returns.where(returns < self.config.target_return, np.nan)
        downside_vol = negative_returns.std()

        if self.config.annualize:
            downside_vol *= np.sqrt(self.config.trading_days_per_year)

        return downside_vol

    def _range_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate range-based volatility (high-low range)."""
        # This assumes OHLC data is available
        # For simple price data, approximate with rolling range
        rolling_range = data.rolling(5).apply(lambda x: (x.max() - x.min()) / x.mean())
        avg_range = rolling_range.mean()

        if self.config.annualize:
            avg_range *= np.sqrt(self.config.trading_days_per_year / 5)

        return avg_range

    def calculate_volatility_metrics(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate comprehensive volatility metrics.

        Returns DataFrame with multiple volatility measures.
        """
        if data.empty or len(data) < self.config.lookback:
            return pd.DataFrame()

        data = data.iloc[-self.config.lookback:]
        returns = data.pct_change().dropna()

        metrics = {}
        for ticker in data.columns:
            ticker_returns = returns[ticker].dropna()
            if len(ticker_returns) < 10:
                continue

            # Historical volatility
            hist_vol = ticker_returns.std() * np.sqrt(252)

            # Downside volatility
            neg_returns = ticker_returns[ticker_returns < 0]
            down_vol = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else 0

            # Upside volatility
            pos_returns = ticker_returns[ticker_returns > 0]
            up_vol = pos_returns.std() * np.sqrt(252) if len(pos_returns) > 0 else 0

            # Volatility skew
            vol_skew = up_vol / down_vol if down_vol > 0 else np.inf

            # Max drawdown
            cum_returns = (1 + ticker_returns).cumprod()
            rolling_max = cum_returns.cummax()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_dd = drawdowns.min()

            metrics[ticker] = {
                "historical_vol": hist_vol,
                "downside_vol": down_vol,
                "upside_vol": up_vol,
                "vol_skew": vol_skew,
                "max_drawdown": max_dd,
            }

        return pd.DataFrame(metrics).T


class BetaFactor(Factor):
    """
    Beta factor.

    Measures sensitivity to market movements.
    Low beta stocks tend to have better risk-adjusted returns.
    """

    def __init__(
        self,
        lookback: int = 252,
        market_ticker: str | None = None,
        normalization: NormalizationMethod = NormalizationMethod.ZSCORE,
    ) -> None:
        super().__init__(
            name="beta",
            direction=FactorDirection.SHORT,  # Lower beta preferred
            normalization=normalization,
        )
        self.lookback = lookback
        self.market_ticker = market_ticker

    def calculate_raw(self, data: pd.DataFrame) -> pd.Series:
        """Calculate beta for each asset."""
        if data.empty or len(data) < self.lookback:
            return pd.Series(dtype=float)

        data = data.iloc[-self.lookback:]
        returns = data.pct_change().dropna()

        if self.market_ticker and self.market_ticker in returns.columns:
            market_returns = returns[self.market_ticker]
        else:
            market_returns = returns.mean(axis=1)

        market_var = market_returns.var()
        betas = {}

        for ticker in returns.columns:
            cov = returns[ticker].cov(market_returns)
            beta = cov / market_var if market_var > 0 else 0
            betas[ticker] = beta

        return pd.Series(betas)


__all__ = [
    "VolatilityFactor",
    "VolatilityConfig",
    "BetaFactor",
]
