"""
Permutation Test for statistical overfitting detection.

Method:
1. Backtest the strategy on original data → performance S_original
2. Shuffle data randomly 1000 times and backtest each → S_shuffled
3. Is S_original statistically better than random chance?

Hypothesis test:
- H0 (null): "performance is due to luck" (strategy does not work)
- H1 (alternative): "performance is meaningful" (strategy captures real signal)

Decision rule:
- Z-score > 2.0 (5% significance) → reject H0: significant performance
- Z-score < 1.0 → fail to reject H0: likely due to chance (overfitting suspected)
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.backtester.analysis.permutation_loop import run_permutation_loop
from src.backtester.analysis.permutation_stats import compute_statistics
from src.backtester.models import BacktestConfig
from src.backtester.wfa.walk_forward_backtest import simple_backtest
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PermutationTestResult:
    """Results of a permutation test run."""

    original_return: float
    original_sharpe: float
    original_win_rate: float

    shuffled_returns: list[float]
    shuffled_sharpes: list[float]
    shuffled_win_rates: list[float]

    mean_shuffled_return: float = 0.0
    std_shuffled_return: float = 0.0

    z_score: float = 0.0
    p_value: float = 0.0

    is_statistically_significant: bool = False
    confidence_level: str = ""  # "5%", "1%", or "not significant"

    interpretation: str = ""


class PermutationTester:
    """
    Overfitting detector via permutation test.

    Usage::

        tester = PermutationTester(
            data=ohlcv_df,
            strategy_factory=lambda: VBOV1()
        )

        result = tester.run(
            num_shuffles=1000,
            shuffle_columns=['close', 'volume']
        )

        print(f"Z-score: {result.z_score:.2f}")
        print(f"P-value: {result.p_value:.4f}")
        print(result.interpretation)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_factory: Callable[[], Strategy],
        backtest_config: BacktestConfig | None = None,
    ):
        """
        Initialize Permutation Tester.

        Args:
            data: OHLCV DataFrame
            strategy_factory: Zero-argument callable that creates a Strategy instance
            backtest_config: Backtest configuration
        """
        self.data = data
        self.strategy_factory = strategy_factory
        self.backtest_config = backtest_config or BacktestConfig()
        self.initial_capital = self.backtest_config.initial_capital

    def run(
        self,
        num_shuffles: int = 1000,
        shuffle_columns: list[str] | None = None,
        verbose: bool = True,
    ) -> PermutationTestResult:
        """
        Execute the permutation test.

        Args:
            num_shuffles: Number of shuffle iterations
            shuffle_columns: Columns to shuffle (default: ['close'])
            verbose: Log progress

        Returns:
            PermutationTestResult with statistical significance metrics
        """
        if shuffle_columns is None:
            shuffle_columns = ["close"]

        if verbose:
            logger.info("Step 1: Testing with original data")

        try:
            strategy_orig = self.strategy_factory()
            original_result = simple_backtest(self.data, strategy_orig, self.initial_capital)
        except Exception as e:
            logger.error(f"Failed to run original backtest: {e}")
            raise

        original_return = original_result.total_return
        original_sharpe = original_result.sharpe_ratio
        original_win_rate = (
            original_result.win_rate if hasattr(original_result, "win_rate") else 0.0
        )

        if verbose:
            logger.info(
                f"  Original return: {original_return:.2%}, "
                f"Sharpe: {original_sharpe:.2f}, "
                f"Win rate: {original_win_rate:.1%}"
            )

        if verbose:
            logger.info(f"Step 2: Running {num_shuffles} permutations")

        shuffled_returns, shuffled_sharpes, shuffled_win_rates = run_permutation_loop(
            data=self.data,
            strategy_factory=self.strategy_factory,
            initial_capital=self.initial_capital,
            num_shuffles=num_shuffles,
            shuffle_columns=shuffle_columns,
            verbose=verbose,
        )

        if verbose:
            logger.info("Step 3: Computing statistics")

        result = compute_statistics(
            original_return=original_return,
            original_sharpe=original_sharpe,
            original_win_rate=original_win_rate,
            shuffled_returns=shuffled_returns,
            shuffled_sharpes=shuffled_sharpes,
            shuffled_win_rates=shuffled_win_rates,
            result_class=PermutationTestResult,
        )

        if verbose:
            logger.info(
                f"  Z-score: {result.z_score:.2f}, "
                f"P-value: {result.p_value:.4f}, "
                f"Significance: {result.confidence_level}"
            )
            logger.info(f"  Interpretation: {result.interpretation}")

        return result

    def export_report_html(self, result: PermutationTestResult, output_path: str) -> None:
        """Generate and save an HTML permutation test report."""
        from src.backtester.analysis.permutation_html import generate_permutation_html

        html = generate_permutation_html(result)
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"Permutation test report saved to {output_path}")


if __name__ == "__main__":
    print("PermutationTester module loaded successfully")
