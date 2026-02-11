"""
Permutation Test: 과적합 통계적 검증.

원리:
1. 원본 데이터로 전략을 백테스트 → 성과 S_original
2. 데이터를 무작위로 섞어서 1000번 백테스트 → 성과들 S_shuffled
3. S_original이 우연에 비해 통계적으로 유의한가?

가설 검정:
- H0 (귀무가설): "성과는 우연" (전략이 작동하지 않음)
- H1 (대립가설): "성과는 의미 있음" (전략이 실제로 작동함)

판단:
- Z-score > 2.0 (5% 유의수준) → H1 채택: 유의한 성과
- Z-score < 1.0 → H0 채택: 우연일 가능성 높음 (과적합 의심)
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.backtester.analysis.permutation_loop import run_permutation_loop
from src.backtester.analysis.permutation_stats import compute_statistics
from src.backtester.models import BacktestConfig
from src.backtester.wfa.wfa_backtest import simple_backtest
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PermutationTestResult:
    """Permutation Test 결과."""

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
    Permutation Test를 통한 과적합 검증.

    사용 예:
    ```python
    tester = PermutationTester(
        data=ohlcv_df,
        strategy_factory=lambda: VBOV1()
    )

    result = tester.run(
        num_shuffles=1000,
        shuffle_columns=['close', 'volume']  # 섞을 컬럼
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
            data: OHLCV 데이터
            strategy_factory: Strategy 객체 생성 함수 (파라미터 없음)
            backtest_config: 백테스트 설정
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
        Permutation Test 실행.

        Args:
            num_shuffles: 셔플 횟수
            shuffle_columns: 섞을 컬럼 (기본: 'close')
            verbose: 진행 상황 로깅

        Returns:
            PermutationTestResult: 검증 결과
        """
        if shuffle_columns is None:
            shuffle_columns = ["close"]

        # 1. 원본 데이터로 백테스트
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

        # 2. 셔플 데이터로 여러 번 백테스트
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

        # 3. 통계 계산
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
        """HTML 리포트 생성 및 저장."""
        from src.backtester.analysis.permutation_html import generate_permutation_html

        html = generate_permutation_html(result)
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"Permutation test report saved to {output_path}")


if __name__ == "__main__":
    print("PermutationTester module loaded successfully")
