"""
파라미터 안정성(Robustness) 분석 엔진.

최적 파라미터 주변에서 성과 변화를 분석하여,
전략의 파라미터 변화에 얼마나 민감한지 검증합니다.

분석 내용:
1. Parameter Sweep: 각 파라미터를 ±30% 범위에서 변경
2. Sensitivity Analysis: 파라미터 조합별 성과 매트릭스
3. Performance Distribution: 성과 값의 분포 (완만함 vs 뾰족함)

과적합 신호:
- 최적값에서 조금만 벗어나도 성과가 급격히 하락 → 과적합
- 넓은 범위에서 안정적인 성과 → 건강한 전략
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
    ?�라미터 ?�정??분석�?

    ?�용 ??
    ```python
    analyzer = RobustnessAnalyzer(
        data=ohlcv_df,
        strategy_factory=lambda p: VanillaVBO(**p)
    )

    report = analyzer.analyze(
        optimal_params={'sma_period': 4, 'noise_period': 8},
        parameter_ranges={
            'sma_period': [2, 3, 4, 5, 6],
            'noise_period': [6, 7, 8, 9, 10]
        }
    )
    ```
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
            data: OHLCV ?�이??
            strategy_factory: ?�라미터�?받아 Strategy ?�성
            backtest_config: 백테?�트 ?�정
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
        ?�라미터 ?�정??분석 ?�행.

        Args:
            optimal_params: 최적?�로 간주?�는 ?�라미터
            parameter_ranges: ?�스?�할 �??�라미터??�?리스??
            verbose: 진행 ?�황 로깅

        Returns:
            RobustnessReport: 분석 결과
        """
        results = []

        # ?�라미터 조합 ?�성
        param_keys = list(parameter_ranges.keys())
        param_values = [parameter_ranges[key] for key in param_keys]
        total_combinations = np.prod([len(v) for v in param_values])

        if verbose:
            logger.info(f"Testing {total_combinations} parameter combinations for robustness")

        # 각 조합 테스트
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

        # 결과 집계
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
        """결과 집계 및 통계 계산."""
        report = RobustnessReport(optimal_params=optimal_params, results=results)

        if not results:
            logger.error("No valid results from robustness analysis")
            return report

        # 수익률 통계
        returns = [r.total_return for r in results]
        report.mean_return = float(np.mean(returns))
        report.std_return = float(np.std(returns))
        report.min_return = float(np.min(returns))
        report.max_return = float(np.max(returns))

        # 최적값 주변 안정성 (±20% 범위)
        neighbor_results = find_neighbors(optimal_params, results, tolerance=0.20)

        if neighbor_results:
            optimal_returns = [r.total_return for r in results if r.params == optimal_params]
            if optimal_returns:
                optimal_return = max(optimal_returns)
                threshold = optimal_return * 0.80
                successful = sum(1 for r in neighbor_results if r.total_return >= threshold)
                report.neighbor_success_rate = successful / len(neighbor_results)

        # 파라미터별 민감도
        report.sensitivity_scores = calculate_sensitivity(results)

        return report

    def export_to_csv(self, report: RobustnessReport, output_path: str | Path) -> None:
        """결과를 CSV로 저장."""
        records = [r.to_dict() for r in report.results]
        df = pd.DataFrame(records)

        df.to_csv(output_path, index=False)
        logger.info(f"Robustness results saved to {output_path}")

    def export_report_html(self, report: RobustnessReport, output_path: str | Path) -> None:
        """HTML 리포트 생성 및 저장."""
        from src.backtester.analysis.robustness_html import generate_robustness_html

        html = generate_robustness_html(report)
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"Robustness report saved to {output_path}")


if __name__ == "__main__":
    print("RobustnessAnalyzer module loaded successfully")
