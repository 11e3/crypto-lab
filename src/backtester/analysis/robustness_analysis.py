"""
Parameter robustness analysis engine.

Measures how sensitive a strategy's performance is to parameter changes by
sweeping a range around the optimal values.

Analysis steps:
1. Parameter Sweep: vary each parameter over ±30% of the optimal value
2. Sensitivity Analysis: performance matrix across all parameter combinations
3. Performance Distribution: flat (robust) vs peaked (overfitted) distributions

Overfitting signals:
- Performance collapses with small deviations from the optimal → overfitting
- Stable performance across a wide range → healthy, generalizable strategy
"""

from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtester.analysis.robustness_models import RobustnessReport, RobustnessResult
from src.backtester.analysis.robustness_stats import calculate_sensitivity, find_neighbors
from src.backtester.models import BacktestConfig
from src.backtester.wfa.wfa_backtest import simple_backtest
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Re-export for backward compatibility
__all__ = ["RobustnessAnalyzer", "RobustnessReport", "RobustnessResult"]


class RobustnessAnalyzer:
    """
    Parameter robustness analyzer.

    Usage::

        analyzer = RobustnessAnalyzer(
            data=ohlcv_df,
            strategy_factory=lambda p: VBOV1(**p)
        )

        report = analyzer.analyze(
            optimal_params={'sma_period': 4, 'noise_period': 8},
            parameter_ranges={
                'sma_period': [2, 3, 4, 5, 6],
                'noise_period': [6, 7, 8, 9, 10]
            }
        )
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_factory: Callable[[dict[str, Any]], Strategy],
        backtest_config: BacktestConfig | None = None,
    ):
        """
        Initialize Robustness Analyzer.

        Args:
            data: OHLCV DataFrame
            strategy_factory: Callable that creates a Strategy from a parameter dict
            backtest_config: Backtest configuration
        """
        self.data = data
        self.strategy_factory = strategy_factory
        self.backtest_config = backtest_config or BacktestConfig()
        self.initial_capital = self.backtest_config.initial_capital

    def analyze(
        self,
        optimal_params: dict[str, Any],
        parameter_ranges: dict[str, list[Any]],
        verbose: bool = True,
    ) -> RobustnessReport:
        """
        Run parameter robustness analysis.

        Args:
            optimal_params: Parameter values considered optimal
            parameter_ranges: Each parameter and its candidate values to test
            verbose: Log progress

        Returns:
            RobustnessReport with aggregated statistics
        """
        results = []

        param_keys = list(parameter_ranges.keys())
        param_values = [parameter_ranges[key] for key in param_keys]
        total_combinations = np.prod([len(v) for v in param_values])

        if verbose:
            logger.info(f"Testing {total_combinations} parameter combinations for robustness")

        for idx, param_combo in enumerate(product(*param_values)):
            params = dict(zip(param_keys, param_combo, strict=False))

            try:
                strategy = self.strategy_factory(params)
                result = simple_backtest(self.data, strategy, self.initial_capital)

                robustness_result = RobustnessResult(
                    params=params,
                    total_return=result.total_return,
                    sharpe=result.sharpe_ratio,
                    max_drawdown=result.mdd,
                    win_rate=result.win_rate if hasattr(result, "win_rate") else 0.0,
                    trade_count=result.total_trades if hasattr(result, "total_trades") else 0,
                )
                results.append(robustness_result)

                if verbose and (idx + 1) % max(1, int(total_combinations) // 10) == 0:
                    logger.info(
                        f"  [{idx + 1}/{int(total_combinations)}] "
                        f"Params: {params}, Return: {result.total_return:.2%}"
                    )

            except Exception as e:
                logger.warning(f"Parameter combination {params} failed: {e}")
                continue

        report = self._aggregate_results(optimal_params, results)

        if verbose:
            logger.info(
                f"Robustness analysis complete: "
                f"Mean return {report.mean_return:.2%} ± {report.std_return:.2%}, "
                f"Neighbor success rate: {report.neighbor_success_rate:.1%}"
            )

        return report

    def _aggregate_results(
        self, optimal_params: dict[str, Any], results: list[RobustnessResult]
    ) -> RobustnessReport:
        """Aggregate backtest results into a RobustnessReport."""
        report = RobustnessReport(optimal_params=optimal_params, results=results)

        if not results:
            logger.error("No valid results from robustness analysis")
            return report

        returns = [r.total_return for r in results]
        report.mean_return = float(np.mean(returns))
        report.std_return = float(np.std(returns))
        report.min_return = float(np.min(returns))
        report.max_return = float(np.max(returns))

        # Neighbor stability: how many ±20% neighbors achieve ≥80% of optimal return
        neighbor_results = find_neighbors(optimal_params, results, tolerance=0.20)

        if neighbor_results:
            optimal_returns = [r.total_return for r in results if r.params == optimal_params]
            if optimal_returns:
                optimal_return = max(optimal_returns)
                threshold = optimal_return * 0.80
                successful = sum(1 for r in neighbor_results if r.total_return >= threshold)
                report.neighbor_success_rate = successful / len(neighbor_results)

        report.sensitivity_scores = calculate_sensitivity(results)

        return report

    def export_to_csv(self, report: RobustnessReport, output_path: str | Path) -> None:
        """Export robustness results to CSV."""
        records = [r.to_dict() for r in report.results]
        df = pd.DataFrame(records)

        df.to_csv(output_path, index=False)
        logger.info(f"Robustness results saved to {output_path}")

    def export_report_html(self, report: RobustnessReport, output_path: str | Path) -> None:
        """Generate and save an HTML robustness report."""
        from src.backtester.analysis.robustness_html import generate_robustness_html

        html = generate_robustness_html(report)
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"Robustness report saved to {output_path}")


if __name__ == "__main__":
    print("RobustnessAnalyzer module loaded successfully")
