"""
Parameter optimization for trading strategies.

Provides optimization methods:
- Grid search
- Random search
- Bayesian optimization (future)
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.backtester.engine import BacktestConfig, BacktestResult
from src.backtester.parallel import ParallelBacktestRunner, ParallelBacktestTask
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""

    best_params: dict[str, Any]
    best_result: BacktestResult
    best_score: float
    all_results: list[tuple[dict[str, Any], BacktestResult, float]]
    optimization_metric: str

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(metric={self.optimization_metric}, "
            f"best_score={self.best_score:.2f})"
        )


class ParameterOptimizer:
    """
    Optimize strategy parameters using various search methods.

    파라미터 최적화의 목표:
    - 전략의 핵심 파라미터(SMA 기간, 노이즈 필터, ATR 배수 등)를 체계적으로 조정
    - 최고 수익률(Sharpe Ratio, CAGR 등)을 달성하는 최적 파라미터 조합 탐색
    - 과적합(Overfitting) 회피 = 과거 데이터에만 맞춘 파라미터 제거

    수익 메커니즘:
    1. 파라미터 = 전략의 기술적 설정 값
       예: SMA(20) → SMA(30) = 신호 빈도 감소, 신뢰도 증가
    2. 각 파라미터 조합마다 백테스트 실행
    3. 지정된 메트릭(Sharpe Ratio 등) 계산
    4. 최고 점수 파라미터 조합 선택

    최적화 방식:
    - Grid Search: 모든 조합 시도 (철저하지만 느림)
    - Random Search: 일부 조합만 무작위 선택 (빠르지만 최적 놓칠 수 있음)

    주의사항:
    - 모수가 많으면 조합 수가 폭증 (5개 모수 × 10값 각 = 100,000 조합)
    - 과적합: 특정 기간 최적값이 다른 기간에는 실패할 수 있음
    """

    def __init__(
        self,
        strategy_factory: Callable[[dict[str, Any]], Strategy],
        tickers: list[str],
        interval: str,
        config: BacktestConfig,
        n_workers: int | None = None,
    ) -> None:
        """
        Initialize parameter optimizer.

        Args:
            strategy_factory: Function that creates a strategy from parameters
                            예: lambda p: VolatilityBreakout(**p)
                            입력: {"sma_period": 20, "lookback": 10}
                            출력: 설정된 전략 객체
            tickers: List of tickers to backtest
                    예: ["KRW-BTC", "KRW-ETH"]
            interval: Data interval
                     예: "day", "minute240"
            config: Backtest configuration
                   초기자본, 수수료, position_sizing 등
            n_workers: Number of parallel workers
                      병렬 처리로 백테스트 속도 향상 (10배까지 가능)
        """
        self.strategy_factory = strategy_factory
        self.tickers = tickers
        self.interval = interval
        self.config = config
        self.n_workers = n_workers

    def optimize(
        self,
        param_grid: dict[str, list[Any]],
        metric: str = "sharpe_ratio",
        maximize: bool = True,
        method: str = "grid",
        n_iter: int = 100,
    ) -> OptimizationResult:
        """
        Optimize parameters using specified method.

        파라미터 최적화 메인 함수:
        - param_grid의 모든 조합(또는 일부)에 대해 백테스트 실행
        - 지정된 메트릭 최적화
        - 최고 성능 파라미터 반환

        최적화 대상 메트릭:
        - sharpe_ratio: 변동성 대비 수익 (위험조정수익 최대화)
        - cagr: 연율수익률 (절대 수익 최대화)
        - total_return: 전체 수익률 (기간 전체 수익)
        - calmar_ratio: CAGR/MDD (수익 대비 리스크)
        - profit_factor: 총수익/총손실 (거래 품질)
        - win_rate: 승률 (신호 정확도)

        예시:
        param_grid = {
            'sma_period': [10, 20, 30, 40],  # SMA 기간 4가지
            'lookback': [5, 10, 15],          # Lookback 기간 3가지
        }
        → 4 × 3 = 12 조합 백테스트

        최고 점수 조합의 백테스트 결과를 반환

        Args:
            param_grid: Dictionary mapping parameter names to lists of values
                       파라미터이름 → 시도할 값들의 리스트
                       예: {'sma_period': [10, 20, 30], 'lookback': [5, 10]}
            metric: Metric to optimize (e.g., 'sharpe_ratio', 'cagr', 'total_return')
                   최적화 목표 지표
                   - sharpe_ratio: 변동성 대비 수익 (권장, 위험 고려)
                   - cagr: 순수 수익률 (고수익 추구)
                   - calmar_ratio: 수익/낙폭 (안정성 중시)
            maximize: If True, maximize metric; if False, minimize
                     True: 높을수록 좋음 (대부분의 경우)
                     False: 낮을수록 좋음 (MDD 등)
            method: Optimization method ('grid' or 'random')
                   - grid: 모든 조합 (철저, 느림)
                   - random: 일부 무작위 (빠름, 놓칠 수 있음)
            n_iter: Number of iterations for random search
                   random 방식에서만 사용 (기본값 100)

        Returns:
            OptimizationResult with best parameters and results
        """
        if method == "grid":
            # Grid Search: 모든 파라미터 조합 시도 (철저하지만 느림)
            return self._grid_search(param_grid, metric, maximize)
        elif method == "random":
            # Random Search: 일부 조합만 무작위 시도 (빠르지만 최적 놓칠 수 있음)
            return self._random_search(param_grid, metric, maximize, n_iter)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _grid_search(
        self,
        param_grid: dict[str, list[Any]],
        metric: str,
        maximize: bool,
    ) -> OptimizationResult:
        """Perform grid search over parameter space.

        모든 파라미터 조합을 체계적으로 탐색:

        동작:
        1. param_grid의 모든 값 조합 생성 (Cartesian product)
        2. 각 조합마다 전략 객체 생성 및 백테스트 실행
        3. 병렬 처리로 속도 향상 (n_workers 병렬 작업)
        4. 지정된 메트릭으로 점수 계산
        5. 최고 점수 조합 반환

        예시:
        param_grid = {
            'sma_period': [10, 20],
            'lookback': [5, 10],
        }
        조합: (10, 5), (10, 10), (20, 5), (20, 10) = 4개

        각 조합별 백테스트 → 결과 비교 → 최적값 선택

        장점:
        - 모든 조합을 시도하므로 최적값 발견 확실
        - 공간을 전체적으로 이해 가능

        단점:
        - 조합 수가 지수로 증가 (5 파라미터 × 10값 = 100,000 조합)
        - 시간 오래 소요 (병렬 처리로 완화)

        병렬 처리 효과:
        - 1개 작업 = 5분
        - 100개 조합 = 500분 (순차) vs 50분 (10 workers)
        """
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        # Cartesian product: 모든 조합 생성
        # 예: [10, 20] × [5, 10] → [(10,5), (10,10), (20,5), (20,10)]
        combinations = list(product(*param_values))

        logger.info(f"Grid search: {len(combinations)} parameter combinations")

        tasks = []
        for combo in combinations:
            params = dict(zip(param_names, combo, strict=False))
            strategy = self.strategy_factory(params)
            task_name = f"{strategy.name}_{'_'.join(str(v) for v in combo)}"
            tasks.append(
                ParallelBacktestTask(
                    name=task_name,
                    strategy=strategy,
                    tickers=self.tickers,
                    interval=self.interval,
                    config=self.config,
                )
            )

        # 병렬 백테스트 실행
        # n_workers개의 워커가 동시에 작업 처리
        # 결과: task_name → BacktestResult 맵핑
        runner = ParallelBacktestRunner(n_workers=self.n_workers)
        results = runner.run(tasks)

        # 결과 수집 및 점수 추출
        all_results = []
        for task in tasks:
            task_name = task.name
            result = results.get(task_name)
            if result:
                # 지정된 메트릭 값 추출 (예: result.sharpe_ratio)
                score = self._extract_metric(result, metric)
                # 파라미터 복원 (task name에서 파싱)
                params = self._parse_params_from_name(task_name, param_names)
                all_results.append((params, result, score))

        # 점수로 정렬 (maximize=True면 내림차순, False면 오름차순)
        all_results.sort(key=lambda x: x[2], reverse=maximize)

        # 최고 점수 결과 추출
        best_params, best_result, best_score = (
            all_results[0]
            if all_results
            else (
                {},
                BacktestResult(),
                0.0,
            )
        )

        return OptimizationResult(
            best_params=best_params,
            best_result=best_result,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=metric,
        )

    def _random_search(
        self,
        param_grid: dict[str, list[Any]],
        metric: str,
        maximize: bool,
        n_iter: int,
    ) -> OptimizationResult:
        """Perform random search over parameter space.

        파라미터 공간에서 일부 조합을 무작위로 탐색:

        동작:
        1. n_iter번 반복
        2. 각 반복마다 param_grid의 각 파라미터에서 무작위로 값 선택
        3. 그 조합으로 전략 생성 및 백테스트 실행
        4. 결과 수집 및 정렬
        5. 최고 점수 조합 반환

        무작위 선택 예시:
        param_grid = {'sma_period': [10, 20, 30], 'lookback': [5, 10, 15]}

        반복1: sma_period=20, lookback=5 → 백테스트
        반복2: sma_period=10, lookback=15 → 백테스트
        반복3: sma_period=30, lookback=10 → 백테스트
        ... (n_iter번 반복)

        특징:
        - Grid Search보다 빠름 (조합 수 제한)
        - 최적값을 놓칠 수 있지만, 다양한 조합 탐색
        - 탐색 공간이 넓을 때(100개 조합 이상) 추천

        사용 시나리오:
        - 파라미터가 많거나 값의 범위가 넓을 때
        - 대략적인 최적값만 필요할 때
        - 계산 자원 제약이 있을 때

        예시:
        n_iter=100, n_workers=10 → 약 10분 (기존 1000분)
        """
        import random

        list(param_grid.keys())
        tasks = []

        logger.info(f"Random search: {n_iter} iterations")

        for i in range(n_iter):
            # 각 파라미터에서 무작위로 값 선택
            # 예: random.choice([10, 20, 30]) = 20
            params = {name: random.choice(values) for name, values in param_grid.items()}
            strategy = self.strategy_factory(params)
            task_name = f"{strategy.name}_iter{i}"
            tasks.append(
                ParallelBacktestTask(
                    name=task_name,
                    strategy=strategy,
                    tickers=self.tickers,
                    interval=self.interval,
                    config=self.config,
                    params=params,  # Store params for later retrieval
                )
            )

        # 병렬 백테스트 실행
        runner = ParallelBacktestRunner(n_workers=self.n_workers)
        results = runner.run(tasks)

        # 결과 수집 및 정렬
        all_results = []
        for task in tasks:
            result = results.get(task.name)
            if result and task.params:
                score = self._extract_metric(result, metric)
                all_results.append((task.params, result, score))

        # Sort by score
        all_results.sort(key=lambda x: x[2], reverse=maximize)

        best_params, best_result, best_score = (
            all_results[0]
            if all_results
            else (
                {},
                BacktestResult(),
                0.0,
            )
        )

        return OptimizationResult(
            best_params=best_params,
            best_result=best_result,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=metric,
        )

    def _extract_metric(self, result: BacktestResult, metric: str) -> float:
        """
        Extract metric value from backtest result.

        Args:
            result: BacktestResult
            metric: Metric name

        Returns:
            Metric value
        """
        if metric == "sharpe_ratio":
            return result.sharpe_ratio if hasattr(result, "sharpe_ratio") else 0.0
        elif metric == "cagr":
            return result.cagr if hasattr(result, "cagr") else 0.0
        elif metric == "total_return":
            return result.total_return if hasattr(result, "total_return") else 0.0
        elif metric == "calmar_ratio":
            return result.calmar_ratio if hasattr(result, "calmar_ratio") else 0.0
        elif metric == "win_rate":
            return result.win_rate if hasattr(result, "win_rate") else 0.0
        elif metric == "profit_factor":
            return result.profit_factor if hasattr(result, "profit_factor") else 0.0
        else:
            logger.warning(f"Unknown metric: {metric}, using sharpe_ratio")
            return result.sharpe_ratio if hasattr(result, "sharpe_ratio") else 0.0

    def _parse_params_from_name(self, name: str, param_names: list[str]) -> dict[str, Any]:
        """Parse parameters from task name (simplified)."""
        # This is a simplified parser - in practice, store params separately
        parts = name.split("_")
        params = {}
        # Try to extract numeric values
        for part in parts:
            try:
                value = float(part)
                if param_names:
                    # Assign to first unassigned param (simplified)
                    for pname in param_names:
                        if pname not in params:
                            params[pname] = int(value) if value.is_integer() else value
                            break
            except ValueError:
                pass
        return params


def optimize_strategy_parameters(
    strategy_factory: Callable[[dict[str, Any]], Strategy],
    param_grid: dict[str, list[Any]],
    tickers: list[str],
    interval: str,
    config: BacktestConfig,
    metric: str = "sharpe_ratio",
    maximize: bool = True,
    method: str = "grid",
    n_iter: int = 100,
    n_workers: int | None = None,
) -> OptimizationResult:
    """
    Optimize strategy parameters.

    Args:
        strategy_factory: Function that creates a strategy from parameters
        param_grid: Dictionary mapping parameter names to lists of values
        tickers: List of tickers to backtest
        interval: Data interval
        config: Backtest configuration
        metric: Metric to optimize
        maximize: If True, maximize metric
        method: Optimization method ('grid' or 'random')
        n_iter: Number of iterations for random search
        n_workers: Number of parallel workers

    Returns:
        OptimizationResult

    Example:
        Optimize VBO parameters::

            from src.strategies.volatility_breakout import create_vbo_strategy
            from src.backtester import BacktestConfig

            def create_strategy(params):
                return create_vbo_strategy(
                    name=f"VBO_{params['sma']}_{params['trend']}",
                    sma_period=params['sma'],
                    trend_sma_period=params['trend'],
                )

            param_grid = {
                'sma': [4, 5, 6],
                'trend': [8, 10, 12],
            }

            result = optimize_strategy_parameters(
                strategy_factory=create_strategy,
                param_grid=param_grid,
                tickers=["KRW-BTC"],
                interval="day",
                config=BacktestConfig(),
                metric="sharpe_ratio",
            )

            print(f"Best parameters: {result.best_params}")
            print(f"Best Sharpe ratio: {result.best_score}")
    """
    optimizer = ParameterOptimizer(
        strategy_factory=strategy_factory,
        tickers=tickers,
        interval=interval,
        config=config,
        n_workers=n_workers,
    )

    return optimizer.optimize(
        param_grid=param_grid,
        metric=metric,
        maximize=maximize,
        method=method,
        n_iter=n_iter,
    )
