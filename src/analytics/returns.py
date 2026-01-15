"""
Return analysis utilities.

Provides:
- Rolling metrics calculation
- Drawdown analysis
- Return distribution analysis
- Risk-adjusted return metrics
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
class RollingMetrics:
    """Rolling performance metrics."""

    # Time series
    rolling_return: pd.Series
    rolling_volatility: pd.Series
    rolling_sharpe: pd.Series
    rolling_sortino: pd.Series
    rolling_max_drawdown: pd.Series

    # Window size
    window: int

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            "return": self.rolling_return,
            "volatility": self.rolling_volatility,
            "sharpe": self.rolling_sharpe,
            "sortino": self.rolling_sortino,
            "max_drawdown": self.rolling_max_drawdown,
        })


@dataclass
class DrawdownAnalysis:
    """Drawdown analysis results."""

    # Current state
    current_drawdown: float
    current_drawdown_duration: int  # Days in current drawdown

    # Historical
    max_drawdown: float
    max_drawdown_start: datetime
    max_drawdown_end: datetime
    max_drawdown_duration: int
    recovery_time: int | None  # Days to recover (None if not recovered)

    # Statistics
    avg_drawdown: float
    avg_drawdown_duration: float
    num_drawdowns: int  # Number of drawdowns > threshold

    # Time series
    drawdown_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    underwater_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def summary(self) -> str:
        """Generate summary string."""
        recovery_str = f"{self.recovery_time} days" if self.recovery_time else "Not recovered"
        return (
            f"Drawdown Analysis\n"
            f"{'=' * 40}\n"
            f"Current Drawdown: {self.current_drawdown:.2%}\n"
            f"Max Drawdown: {self.max_drawdown:.2%}\n"
            f"Max DD Duration: {self.max_drawdown_duration} days\n"
            f"Recovery Time: {recovery_str}\n"
            f"Avg Drawdown: {self.avg_drawdown:.2%}\n"
            f"Number of Drawdowns: {self.num_drawdowns}\n"
        )


@dataclass
class ReturnDistribution:
    """Return distribution statistics."""

    # Central tendency
    mean: float
    median: float
    mode: float | None

    # Dispersion
    std: float
    variance: float
    range: float

    # Shape
    skewness: float
    kurtosis: float  # Excess kurtosis

    # Quantiles
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float

    # Tail risk
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (Expected Shortfall)

    # Normality
    is_normal: bool  # Jarque-Bera test
    jb_statistic: float
    jb_pvalue: float


class ReturnAnalyzer:
    """
    Comprehensive return analysis.

    Provides rolling metrics, drawdown analysis, distribution analysis,
    and various risk-adjusted return calculations.

    Example:
        >>> analyzer = ReturnAnalyzer()
        >>> rolling = analyzer.calculate_rolling_metrics(returns, window=60)
        >>> drawdown = analyzer.analyze_drawdowns(returns)
        >>> dist = analyzer.analyze_distribution(returns)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        annualization_factor: int = 252,
    ) -> None:
        """
        Initialize return analyzer.

        Args:
            risk_free_rate: Annual risk-free rate
            annualization_factor: Trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

    def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 60,
    ) -> RollingMetrics:
        """
        Calculate rolling performance metrics.

        Args:
            returns: Return series
            window: Rolling window size

        Returns:
            RollingMetrics with time series
        """
        daily_rf = (1 + self.risk_free_rate) ** (1/self.annualization_factor) - 1

        # Rolling return (annualized)
        rolling_return = returns.rolling(window).mean() * self.annualization_factor

        # Rolling volatility (annualized)
        rolling_vol = returns.rolling(window).std() * np.sqrt(self.annualization_factor)

        # Rolling Sharpe
        excess_returns = returns - daily_rf
        rolling_sharpe = (
            excess_returns.rolling(window).mean() /
            returns.rolling(window).std()
        ) * np.sqrt(self.annualization_factor)

        # Rolling Sortino (downside only)
        def sortino_window(x):
            excess = x - daily_rf
            downside = x[x < 0]
            if len(downside) < 2:
                return np.nan
            downside_std = downside.std()
            if downside_std == 0:
                return np.nan
            return excess.mean() / downside_std * np.sqrt(self.annualization_factor)

        rolling_sortino = returns.rolling(window).apply(sortino_window)

        # Rolling max drawdown
        def max_dd_window(x):
            cum = (1 + x).cumprod()
            rolling_max = cum.cummax()
            dd = (cum - rolling_max) / rolling_max
            return dd.min()

        rolling_max_dd = returns.rolling(window).apply(max_dd_window)

        return RollingMetrics(
            rolling_return=rolling_return,
            rolling_volatility=rolling_vol,
            rolling_sharpe=rolling_sharpe,
            rolling_sortino=rolling_sortino,
            rolling_max_drawdown=rolling_max_dd,
            window=window,
        )

    def analyze_drawdowns(
        self,
        returns: pd.Series,
        threshold: float = 0.05,
    ) -> DrawdownAnalysis:
        """
        Analyze drawdowns in return series.

        Args:
            returns: Return series
            threshold: Minimum drawdown to count

        Returns:
            DrawdownAnalysis with detailed breakdown
        """
        # Calculate cumulative returns and drawdown series
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max

        # Current state
        current_dd = drawdown.iloc[-1]

        # Find drawdown periods
        is_drawdown = drawdown < 0
        dd_start = is_drawdown & ~is_drawdown.shift(1).fillna(False)
        dd_end = ~is_drawdown & is_drawdown.shift(1).fillna(False)

        # Calculate duration of current drawdown
        if current_dd < 0:
            last_peak_idx = drawdown[drawdown == 0].index[-1] if (drawdown == 0).any() else drawdown.index[0]
            current_duration = len(drawdown[last_peak_idx:])
        else:
            current_duration = 0

        # Find max drawdown
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()

        # Find start of max drawdown
        max_dd_start_idx = rolling_max[:max_dd_idx].idxmax()

        # Find end of max drawdown (recovery)
        post_max_dd = cum_returns[max_dd_idx:]
        recovery_mask = post_max_dd >= rolling_max[max_dd_start_idx]
        if recovery_mask.any():
            max_dd_end_idx = recovery_mask.idxmax()
            recovery_time = len(cum_returns[max_dd_idx:max_dd_end_idx])
        else:
            max_dd_end_idx = cum_returns.index[-1]
            recovery_time = None

        max_dd_duration = len(cum_returns[max_dd_start_idx:max_dd_idx])

        # Count significant drawdowns
        dd_depths = []
        dd_durations = []
        current_dd_depth = 0
        current_dd_len = 0

        for i, (dd, is_dd) in enumerate(zip(drawdown, is_drawdown)):
            if is_dd:
                current_dd_depth = min(current_dd_depth, dd)
                current_dd_len += 1
            else:
                if current_dd_depth < -threshold:
                    dd_depths.append(current_dd_depth)
                    dd_durations.append(current_dd_len)
                current_dd_depth = 0
                current_dd_len = 0

        avg_dd = np.mean(dd_depths) if dd_depths else 0
        avg_dd_duration = np.mean(dd_durations) if dd_durations else 0

        return DrawdownAnalysis(
            current_drawdown=current_dd,
            current_drawdown_duration=current_duration,
            max_drawdown=max_dd,
            max_drawdown_start=max_dd_start_idx,
            max_drawdown_end=max_dd_end_idx,
            max_drawdown_duration=max_dd_duration,
            recovery_time=recovery_time,
            avg_drawdown=avg_dd,
            avg_drawdown_duration=avg_dd_duration,
            num_drawdowns=len(dd_depths),
            drawdown_series=drawdown,
            underwater_series=drawdown,
        )

    def analyze_distribution(
        self,
        returns: pd.Series,
    ) -> ReturnDistribution:
        """
        Analyze return distribution.

        Args:
            returns: Return series

        Returns:
            ReturnDistribution with statistics
        """
        returns = returns.dropna()

        if len(returns) < 10:
            raise ValueError("Insufficient data for distribution analysis")

        # Central tendency
        mean = returns.mean()
        median = returns.median()
        try:
            mode = stats.mode(returns, keepdims=True).mode[0]
        except Exception:
            mode = None

        # Dispersion
        std = returns.std()
        variance = returns.var()
        range_val = returns.max() - returns.min()

        # Shape
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)  # Excess kurtosis

        # Quantiles
        percentiles = returns.quantile([0.05, 0.25, 0.75, 0.95])

        # VaR and CVaR
        var_95 = returns.quantile(0.05)  # 5th percentile (loss)
        cvar_95 = returns[returns <= var_95].mean()

        # Normality test
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        is_normal = jb_pvalue > 0.05

        return ReturnDistribution(
            mean=mean,
            median=median,
            mode=mode,
            std=std,
            variance=variance,
            range=range_val,
            skewness=skewness,
            kurtosis=kurtosis,
            percentile_5=percentiles[0.05],
            percentile_25=percentiles[0.25],
            percentile_75=percentiles[0.75],
            percentile_95=percentiles[0.95],
            var_95=var_95,
            cvar_95=cvar_95,
            is_normal=is_normal,
            jb_statistic=jb_stat,
            jb_pvalue=jb_pvalue,
        )

    def calculate_metrics(
        self,
        returns: pd.Series,
    ) -> dict[str, float]:
        """
        Calculate comprehensive return metrics.

        Args:
            returns: Return series

        Returns:
            Dict with all metrics
        """
        returns = returns.dropna()
        n = len(returns)

        if n < 2:
            return {}

        daily_rf = (1 + self.risk_free_rate) ** (1/self.annualization_factor) - 1

        # Basic metrics
        total_return = (1 + returns).prod() - 1
        years = n / self.annualization_factor
        cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0

        # Volatility
        volatility = returns.std() * np.sqrt(self.annualization_factor)

        # Risk-adjusted returns
        excess_returns = returns - daily_rf
        sharpe = (
            excess_returns.mean() / returns.std() * np.sqrt(self.annualization_factor)
            if returns.std() > 0 else 0
        )

        # Sortino
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        sortino = (
            excess_returns.mean() / downside_std * np.sqrt(self.annualization_factor)
            if downside_std > 0 else 0
        )

        # Drawdown
        cum = (1 + returns).cumprod()
        rolling_max = cum.cummax()
        max_dd = ((cum - rolling_max) / rolling_max).min()

        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # Win rate
        win_rate = (returns > 0).mean()

        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else np.inf

        # Skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        return {
            "total_return": total_return,
            "cagr": cagr,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "num_observations": n,
        }

    def compare_returns(
        self,
        returns_dict: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Compare multiple return series.

        Args:
            returns_dict: Dict of name -> return series

        Returns:
            DataFrame with metrics for each series
        """
        results = {}
        for name, returns in returns_dict.items():
            results[name] = self.calculate_metrics(returns)

        return pd.DataFrame(results).T

    def calculate_correlation(
        self,
        returns_dict: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Calculate correlation between return series.

        Args:
            returns_dict: Dict of name -> return series

        Returns:
            Correlation matrix
        """
        df = pd.DataFrame(returns_dict)
        return df.corr()

    def calculate_beta(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> dict[str, float]:
        """
        Calculate beta and related metrics.

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns

        Returns:
            Dict with beta, alpha, R-squared, etc.
        """
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 10:
            return {}

        port = aligned.iloc[:, 0]
        bench = aligned.iloc[:, 1]

        # Beta
        cov = port.cov(bench)
        var = bench.var()
        beta = cov / var if var > 0 else 0

        # Alpha (annualized)
        alpha = (port.mean() - beta * bench.mean()) * self.annualization_factor

        # R-squared
        correlation = port.corr(bench)
        r_squared = correlation ** 2

        # Tracking error
        tracking_diff = port - bench
        tracking_error = tracking_diff.std() * np.sqrt(self.annualization_factor)

        # Information ratio
        info_ratio = tracking_diff.mean() / tracking_diff.std() * np.sqrt(self.annualization_factor) if tracking_diff.std() > 0 else 0

        return {
            "beta": beta,
            "alpha": alpha,
            "r_squared": r_squared,
            "correlation": correlation,
            "tracking_error": tracking_error,
            "information_ratio": info_ratio,
        }


__all__ = [
    "ReturnAnalyzer",
    "RollingMetrics",
    "DrawdownAnalysis",
    "ReturnDistribution",
]
