"""
Bootstrap Analysis: Phase 3 Statistical Reliability

Resamples the dataset using block bootstrap to estimate confidence intervals
for strategy performance metrics (return, Sharpe) and risk (MDD).
"""

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtester.analysis.bootstrap_backtest import simple_backtest_vectorized
from src.backtester.engine import run_backtest
from src.backtester.models import BacktestConfig, BacktestResult
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BootstrapResult:
    returns: list[float]
    sharpes: list[float]
    mdds: list[float]
    mean_return: float = 0.0
    ci_return_95: tuple[float, float] = (0.0, 0.0)
    mean_sharpe: float = 0.0
    ci_sharpe_95: tuple[float, float] = (0.0, 0.0)
    mean_mdd: float = 0.0
    ci_mdd_95: tuple[float, float] = (0.0, 0.0)


class BootstrapAnalyzer:
    def __init__(
        self,
        data: pd.DataFrame,
        strategy_factory: Callable[[dict[str, Any]], Strategy] | Callable[[], Strategy],
        backtest_config: BacktestConfig | None = None,
        ticker: str = "KRW-BTC",
        interval: str = "day",
    ):
        self.data = data
        self.strategy_factory = strategy_factory
        self.backtest_config = backtest_config or BacktestConfig()
        self.initial_capital = self.backtest_config.initial_capital
        self.ticker = ticker
        self.interval = interval

    def _simple_backtest(self, data: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        """Use engine-based backtest when possible for realistic metrics."""
        try:
            # Save resampled data to a temporary parquet file
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / f"{self.ticker}_{self.interval}.parquet"
                df_to_save = data.copy()
                if df_to_save.index.name is None:
                    df_to_save.index.name = "datetime"
                df_to_save.to_parquet(tmp_path)

                # Run backtest using official engine
                result = run_backtest(
                    strategy=strategy,
                    tickers=[self.ticker],
                    interval=self.interval,
                    data_dir=Path(tmpdir),
                    config=self.backtest_config,
                )
                return result
        except Exception as e:
            logger.debug(f"Engine backtest failed, fallback to simple: {e}")
            return simple_backtest_vectorized(data, strategy, self.initial_capital)

    def _block_bootstrap(self, returns: np.ndarray, n: int, block_size: int) -> np.ndarray:
        blocks: list[np.ndarray] = []
        i = 0
        while i < n:
            start = np.random.randint(0, max(1, n - block_size))
            block: np.ndarray = returns[start : start + block_size]
            blocks.append(block)
            i += block_size
        concatenated: np.ndarray = np.concatenate(blocks)
        result: np.ndarray = concatenated[:n]
        return result

    def _resample_data(self, data: pd.DataFrame, block_size: int) -> pd.DataFrame:
        """
        Resample data using date-based block bootstrap (preserves temporal structure).

        Instead of resampling returns and reconstructing prices (which breaks trend),
        we resample date blocks directly to maintain OHLCV relationships.
        """
        df = data.copy()
        n = len(df)

        # Randomly select starting indices for blocks
        resampled_indices: list[int] = []
        while len(resampled_indices) < n:
            start_idx = np.random.randint(0, max(1, n - block_size + 1))
            block_indices = list(range(start_idx, min(start_idx + block_size, n)))
            resampled_indices.extend(block_indices)

        # Trim to original length
        resampled_indices = resampled_indices[:n]

        # Select rows by index (preserves OHLCV structure)
        resampled_df = df.iloc[resampled_indices].copy()

        # Reset index to maintain continuous datetime (for indicator calculations)
        # Try to infer frequency from original index
        inferred_freq = None
        if hasattr(df.index, "freq") and df.index.freq is not None:
            inferred_freq = df.index.freq
        else:
            try:
                # pd.infer_freq requires DatetimeIndex/Series
                if isinstance(df.index, pd.DatetimeIndex):
                    inferred_freq = pd.infer_freq(df.index)
            except (ValueError, TypeError):
                pass

        resampled_df.index = pd.date_range(
            start=df.index[0],
            periods=len(resampled_df),
            freq=inferred_freq or "D",
        )

        return resampled_df

    def analyze(self, n_samples: int = 300, block_size: int = 30) -> BootstrapResult:
        rets: list[float] = []
        sharps: list[float] = []
        mdds: list[float] = []

        for i in range(n_samples):
            try:
                d = self._resample_data(self.data, block_size)
                try:
                    strategy = self.strategy_factory()  # type: ignore[call-arg]
                except TypeError:
                    strategy = self.strategy_factory()  # type: ignore[call-arg]
                r = self._simple_backtest(d, strategy)

                # Filter extreme outliers (likely due to resampling edge cases)
                # Keep returns within reasonable range for crypto: -100% to +100000% (1000x)
                total_ret = r.total_return
                sharpe = getattr(r, "sharpe_ratio", 0.0)
                mdd = getattr(r, "mdd", 0.0)

                if -1.0 <= total_ret <= 1000.0 and -10.0 <= sharpe <= 10.0:
                    rets.append(total_ret)
                    sharps.append(sharpe)
                    mdds.append(mdd)
                else:
                    logger.debug(
                        f"Filtered outlier sample {i}: return={total_ret:.2%}, sharpe={sharpe:.2f}"
                    )

            except (ValueError, TypeError, KeyError, IndexError) as e:
                logger.debug(f"Bootstrap sample {i} failed: {e}")
                continue

        if not rets:
            logger.warning("No valid bootstrap samples generated")
            return BootstrapResult(returns=[], sharpes=[], mdds=[])

        # Log filtering statistics
        filtered_pct = (1 - len(rets) / n_samples) * 100
        logger.info(
            f"Bootstrap: {len(rets)}/{n_samples} valid samples ({filtered_pct:.1f}% filtered)"
        )

        arr = np.array(rets)
        arr_s = np.array(sharps)
        arr_m = np.array(mdds)

        def ci95(a: np.ndarray) -> tuple[float, float]:
            return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

        return BootstrapResult(
            returns=rets,
            sharpes=sharps,
            mdds=mdds,
            mean_return=float(np.mean(arr)),
            ci_return_95=ci95(arr),
            mean_sharpe=float(np.mean(arr_s)),
            ci_sharpe_95=ci95(arr_s),
            mean_mdd=float(np.mean(arr_m)),
            ci_mdd_95=ci95(arr_m),
        )
