"""
Permutation Test: statistical overfitting detection.

Principle:
1. Backtest strategy on original data → performance S_original
2. Randomly shuffle data 1000 times and backtest each → S_shuffled
3. Is S_original statistically better than random?

Hypothesis test:
- H0 (null hypothesis): "performance is random" (strategy has no edge)
- H1 (alternative hypothesis): "performance is meaningful" (strategy actually works)

Decision:
- Z-score > 2.0 (5% significance level) → reject H0: significant performance
- Z-score < 1.0 → accept H0: likely due to chance (overfitting suspected)
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.backtester.engine import BacktestConfig, BacktestEngine
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PermutationTestResult:
    """Permutation test result."""

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
    confidence_level: str = ""  # "5%", "1%", "not significant"

    interpretation: str = ""


class PermutationTester:
    """
    Overfitting validation via Permutation Test.

    Usage:
    ```python
    tester = PermutationTester(
        data=ohlcv_df,
        strategy_factory=lambda: VBOV1()
    )

    result = tester.run(
        num_shuffles=1000,
        shuffle_columns=['close', 'volume']  # columns to shuffle
    )

    print(f"Z-score: {result.z_score:.2f}")
    print(f"P-value: {result.p_value:.4f}")
    print(result.interpretation)
    ```
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
            strategy_factory: Callable that returns a fresh Strategy instance (no args)
            backtest_config: Backtest configuration
        """
        self.data = data
        self.strategy_factory = strategy_factory
        self.backtest_config = backtest_config or BacktestConfig()
        self.engine = BacktestEngine(self.backtest_config)

    def run(
        self,
        num_shuffles: int = 1000,
        shuffle_columns: list[str] | None = None,
        verbose: bool = True,
    ) -> PermutationTestResult:
        """
        Run Permutation Test.

        Args:
            num_shuffles: Number of shuffle iterations
            shuffle_columns: Columns to shuffle (default: 'close')
            verbose: Log progress

        Returns:
            PermutationTestResult with statistical significance metrics
        """
        if shuffle_columns is None:
            shuffle_columns = ["close"]

        # Step 1: backtest on original data
        if verbose:
            logger.info("Step 1: Testing with original data")

        try:
            strategy_orig = self.strategy_factory()
            original_result = self.engine.run(self.data, strategy_orig)
        except Exception as e:
            logger.error(f"Failed to run original backtest: {e}")
            raise

        original_return = original_result.total_return
        original_sharpe = original_result.sharpe
        original_win_rate = (
            original_result.win_rate if hasattr(original_result, "win_rate") else 0.0
        )

        if verbose:
            logger.info(
                f"  Original return: {original_return:.2%}, "
                f"Sharpe: {original_sharpe:.2f}, "
                f"Win rate: {original_win_rate:.1%}"
            )

        # Step 2: backtest on shuffled data multiple times
        if verbose:
            logger.info(f"Step 2: Running {num_shuffles} permutations")

        shuffled_returns = []
        shuffled_sharpes = []
        shuffled_win_rates = []

        for i in range(num_shuffles):
            try:
                # Shuffle data
                shuffled_data = self._shuffle_data(self.data, shuffle_columns)

                # Backtest
                strategy_shuffled = self.strategy_factory()
                shuffled_result = self.engine.run(shuffled_data, strategy_shuffled)

                shuffled_returns.append(shuffled_result.total_return)
                shuffled_sharpes.append(shuffled_result.sharpe)

                if hasattr(shuffled_result, "win_rate"):
                    shuffled_win_rates.append(shuffled_result.win_rate)

                if verbose and (i + 1) % max(1, num_shuffles // 10) == 0:
                    logger.info(f"  Completed {i + 1}/{num_shuffles} permutations")

            except Exception as e:
                logger.debug(f"Permutation {i} failed: {e}")
                continue

        # Step 3: compute statistics
        if verbose:
            logger.info("Step 3: Computing statistics")

        result = self._compute_statistics(
            original_return=original_return,
            original_sharpe=original_sharpe,
            original_win_rate=original_win_rate,
            shuffled_returns=shuffled_returns,
            shuffled_sharpes=shuffled_sharpes,
            shuffled_win_rates=shuffled_win_rates,
        )

        if verbose:
            logger.info(
                f"  Z-score: {result.z_score:.2f}, "
                f"P-value: {result.p_value:.4f}, "
                f"Significance: {result.confidence_level}"
            )
            logger.info(f"  Interpretation: {result.interpretation}")

        return result

    def _shuffle_data(self, data: pd.DataFrame, columns_to_shuffle: list[str]) -> pd.DataFrame:
        """
        Generate a copy of data with specified columns randomly shuffled.

        Index (dates) is preserved; only the values in the given columns are reordered.
        """
        shuffled = data.copy()

        for col in columns_to_shuffle:
            if col in shuffled.columns:
                # Convert column to numpy array and shuffle in-place
                values = shuffled[col].values.copy()
                np.random.shuffle(values)
                shuffled[col] = values

        return shuffled

    def _compute_statistics(
        self,
        original_return: float,
        original_sharpe: float,
        original_win_rate: float,
        shuffled_returns: list[float],
        shuffled_sharpes: list[float],
        shuffled_win_rates: list[float],
    ) -> PermutationTestResult:
        """Compute Z-score and p-value from permutation results."""
        result = PermutationTestResult(
            original_return=original_return,
            original_sharpe=original_sharpe,
            original_win_rate=original_win_rate,
            shuffled_returns=shuffled_returns,
            shuffled_sharpes=shuffled_sharpes,
            shuffled_win_rates=shuffled_win_rates,
        )

        if not shuffled_returns:
            logger.error("No valid shuffled results")
            return result

        # Compute Z-score based on return distribution
        mean_shuffled = np.mean(shuffled_returns)
        std_shuffled = np.std(shuffled_returns)

        result.mean_shuffled_return = mean_shuffled
        result.std_shuffled_return = std_shuffled

        # Z-score = (X - μ) / σ
        if std_shuffled > 0:
            result.z_score = (original_return - mean_shuffled) / std_shuffled
        else:
            result.z_score = 0.0

        # P-value: probability of achieving the original return by chance (two-tailed)
        result.p_value = 2 * (1 - stats.norm.cdf(abs(result.z_score)))

        # Significance decision
        if result.p_value < 0.01:
            result.confidence_level = "1%"
            result.is_statistically_significant = True
        elif result.p_value < 0.05:
            result.confidence_level = "5%"
            result.is_statistically_significant = True
        else:
            result.confidence_level = "not significant"
            result.is_statistically_significant = False

        result.interpretation = self._interpret_results(result)

        return result

    def _interpret_results(self, result: PermutationTestResult) -> str:
        """Return a human-readable interpretation of the permutation test result."""
        if result.z_score < 0:
            # Original performance is worse than the shuffled mean
            return (
                f"⚠️ Original return ({result.original_return:.2%}) is below the "
                f"shuffled mean ({result.mean_shuffled_return:.2%}). "
                f"The strategy does not appear to capture a real signal."
            )
        elif result.z_score < 1.0:
            # Not significant
            return (
                f"❌ Z-score={result.z_score:.2f} < 1.0: "
                f"not statistically significant (p-value={result.p_value:.4f}). "
                f"Performance is likely due to chance — overfitting suspected."
            )
        elif result.z_score < 2.0:
            # Weakly significant
            return (
                f"⚠️ Z-score={result.z_score:.2f}: "
                f"weakly significant (p-value={result.p_value:.4f}). "
                f"Some signal present, but overfitting risk remains."
            )
        elif result.z_score < 3.0:
            # Significant
            return (
                f"✅ Z-score={result.z_score:.2f} ({result.confidence_level} significance level): "
                f"statistically significant performance. Strategy likely captures a real signal."
            )
        else:
            # Highly significant
            return (
                f"🎯 Z-score={result.z_score:.2f} ({result.confidence_level} significance level): "
                f"very strong statistical significance. Strategy signal quality is excellent."
            )

    def export_report_html(self, result: PermutationTestResult, output_path: str) -> None:
        """Generate and save HTML report."""
        html = self._generate_html(result)

        from pathlib import Path

        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"Permutation test report saved to {output_path}")

    def _generate_html(self, result: PermutationTestResult) -> str:
        """Generate HTML report string."""
        import base64
        from io import BytesIO

        import matplotlib.pyplot as plt

        # Build histogram figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Return distribution
        ax1.hist(result.shuffled_returns, bins=30, alpha=0.7, label="Shuffled")
        ax1.axvline(
            result.original_return,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Original ({result.original_return:.2%})",
        )
        ax1.axvline(
            result.mean_shuffled_return,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Mean ({result.mean_shuffled_return:.2%})",
        )
        ax1.set_xlabel("Total Return")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Return Distribution: Original vs Shuffled")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Sharpe distribution
        ax2.hist(result.shuffled_sharpes, bins=30, alpha=0.7, label="Shuffled")
        ax2.axvline(
            result.original_sharpe,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Original ({result.original_sharpe:.2f})",
        )
        ax2.set_xlabel("Sharpe Ratio")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Sharpe Distribution: Original vs Shuffled")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Encode figure as base64 PNG
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        # Build HTML
        decision = (
            "✅ Strategy shows statistically significant signal."
            if result.is_statistically_significant
            else "❌ Strategy performance is likely due to chance — overfitting suspected."
        )
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Permutation Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                th {{ background-color: #2196F3; color: white; }}
                .significant {{ color: #4CAF50; font-weight: bold; }}
                .warning {{ color: #FF9800; font-weight: bold; }}
                .danger {{ color: #F44336; font-weight: bold; }}
                .interpretation {{
                    background-color: #f0f0f0;
                    padding: 15px;
                    margin: 20px 0;
                    border-left: 4px solid #2196F3;
                    font-size: 14px;
                    line-height: 1.6;
                }}
            </style>
        </head>
        <body>
            <h1>Permutation Test Report</h1>
            <h2>Statistical Significance Test</h2>

            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Original Return</td>
                    <td>{result.original_return:.2%}</td>
                </tr>
                <tr>
                    <td>Original Sharpe</td>
                    <td>{result.original_sharpe:.2f}</td>
                </tr>
                <tr>
                    <td>Original Win Rate</td>
                    <td>{result.original_win_rate:.1%}</td>
                </tr>
                <tr>
                    <td colspan="2"><b>Shuffled Data Statistics (n={len(result.shuffled_returns)})</b></td>
                </tr>
                <tr>
                    <td>Mean Return</td>
                    <td>{result.mean_shuffled_return:.2%}</td>
                </tr>
                <tr>
                    <td>Std Dev Return</td>
                    <td>{result.std_shuffled_return:.2%}</td>
                </tr>
                <tr>
                    <td colspan="2"><b>Hypothesis Test</b></td>
                </tr>
                <tr>
                    <td>Z-score</td>
                    <td class="{"significant" if result.is_statistically_significant else "danger"}">
                        {result.z_score:.2f}
                    </td>
                </tr>
                <tr>
                    <td>P-value</td>
                    <td class="{"significant" if result.is_statistically_significant else "danger"}">
                        {result.p_value:.4f}
                    </td>
                </tr>
                <tr>
                    <td>Significance Level</td>
                    <td class="{"significant" if result.is_statistically_significant else "danger"}">
                        {result.confidence_level}
                    </td>
                </tr>
            </table>

            <h2>Distribution Analysis</h2>
            <img src="data:image/png;base64,{image_base64}" style="max-width: 100%; height: auto;">

            <h2>Interpretation</h2>
            <div class="interpretation">
                {result.interpretation}
            </div>

            <h2>Decision</h2>
            <div class="interpretation">
                {decision}
            </div>
        </body>
        </html>
        """

        return html


if __name__ == "__main__":
    print("PermutationTester module loaded successfully")
