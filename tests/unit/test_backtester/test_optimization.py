"""
Unit tests for parameter optimization module.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.backtester.engine import BacktestConfig, BacktestResult
from src.backtester.optimization import (
    OptimizationResult,
    ParameterOptimizer,
    optimize_strategy_parameters,
)
from src.strategies.base import Strategy


@pytest.fixture
def mock_strategy_factory() -> MagicMock:
    """Mock strategy factory function."""
    mock_strategy = MagicMock(spec=Strategy)
    mock_strategy.name = "MockStrategy"
    mock_strategy.param1 = 10  # Example param for name generation
    mock_strategy.param2 = "A"  # Example param for name generation
    return MagicMock(return_value=mock_strategy)


@pytest.fixture
def mock_backtest_config() -> BacktestConfig:
    """Mock BacktestConfig instance."""
    return BacktestConfig(initial_capital=1000000)


@pytest.fixture
def mock_backtest_result() -> BacktestResult:
    """Mock BacktestResult instance."""
    result = MagicMock(spec=BacktestResult)
    result.cagr = 10.0
    result.sharpe_ratio = 1.5
    result.mdd = 0.05
    result.total_return = 0.1
    result.calmar_ratio = 2.0
    result.win_rate = 0.6
    result.profit_factor = 1.2
    return result


@pytest.fixture
def param_grid_simple() -> dict[str, list[Any]]:
    """A simple parameter grid for testing."""
    return {"p1": [1, 2], "p2": ["A", "B"]}


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_initialization(self, mock_backtest_result: BacktestResult) -> None:
        result = OptimizationResult(
            best_params={"p1": 1},
            best_result=mock_backtest_result,
            best_score=1.5,
            all_results=[({"p1": 1}, mock_backtest_result, 1.5)],
            optimization_metric="sharpe_ratio",
        )
        assert result.best_params == {"p1": 1}
        assert result.best_score == 1.5

    def test_repr(self) -> None:
        result = OptimizationResult(
            best_params={"p1": 1},
            best_result=MagicMock(spec=BacktestResult),
            best_score=1.5,
            all_results=[],
            optimization_metric="sharpe_ratio",
        )
        assert repr(result) == "OptimizationResult(metric=sharpe_ratio, best_score=1.50)"


class TestParameterOptimizer:
    """Tests for ParameterOptimizer class."""

    @pytest.fixture
    def optimizer(
        self,
        mock_strategy_factory: MagicMock,
        mock_backtest_config: BacktestConfig,
    ) -> ParameterOptimizer:
        return ParameterOptimizer(
            strategy_factory=mock_strategy_factory,
            tickers=["KRW-BTC"],
            interval="day",
            config=mock_backtest_config,
        )

    def test_init(
        self,
        mock_strategy_factory: MagicMock,
        mock_backtest_config: BacktestConfig,
    ) -> None:
        optimizer = ParameterOptimizer(
            strategy_factory=mock_strategy_factory,
            tickers=["KRW-BTC"],
            interval="day",
            config=mock_backtest_config,
            n_workers=4,
        )
        assert optimizer.strategy_factory == mock_strategy_factory
        assert optimizer.n_workers == 4

    def test_extract_metric(
        self, optimizer: ParameterOptimizer, mock_backtest_result: BacktestResult
    ) -> None:
        assert optimizer._extract_metric(mock_backtest_result, "sharpe_ratio") == 1.5
        assert optimizer._extract_metric(mock_backtest_result, "cagr") == 10.0
        assert (
            optimizer._extract_metric(mock_backtest_result, "invalid_metric") == 1.5
        )  # Falls back to sharpe_ratio

    def test_parse_params_from_name(self, optimizer: ParameterOptimizer) -> None:
        param_names = ["p1", "p2"]
        name = "MockStrategy_1_A"  # Format generated in _grid_search
        parsed_params = optimizer._parse_params_from_name(name, param_names)
        # Simplified parsing expects numbers, so it should extract 1 and skip A
        # This implementation detail makes testing a bit tricky, might be better to store params
        assert parsed_params == {"p1": 1}  # Only p1=1 is parsed, 'A' is skipped

        name_complex = "MockStrategy_5_10_True"
        param_names_complex = ["sma", "trend"]
        parsed_params_complex = optimizer._parse_params_from_name(name_complex, param_names_complex)
        assert parsed_params_complex == {"sma": 5, "trend": 10}

    def test_parse_params_from_name_no_params(self, optimizer: ParameterOptimizer) -> None:
        name = "MockStrategy_justname"
        param_names = ["p1"]
        parsed_params = optimizer._parse_params_from_name(name, param_names)
        assert parsed_params == {}

    @patch("src.backtester.optimization.ParallelBacktestRunner")
    def test_grid_search(
        self,
        mock_parallel_backtest_runner: MagicMock,
        optimizer: ParameterOptimizer,
        mock_backtest_result: BacktestResult,
        param_grid_simple: dict[str, list[Any]],
    ) -> None:
        mock_runner_instance = mock_parallel_backtest_runner.return_value
        # Simulate runner returning results for each task
        mock_runner_instance.run.return_value = {
            "MockStrategy_1_A": mock_backtest_result,
            "MockStrategy_1_B": mock_backtest_result,
            "MockStrategy_2_A": mock_backtest_result,
            "MockStrategy_2_B": mock_backtest_result,
        }

        # Configure the mock strategy factory to always return a strategy with "MockStrategy" name
        optimizer.strategy_factory.return_value.name = "MockStrategy"

        result = optimizer._grid_search(param_grid_simple, "sharpe_ratio", maximize=True)

        assert isinstance(result, OptimizationResult)
        assert result.best_score == 1.5  # All mock results have sharpe_ratio 1.5
        assert len(result.all_results) == 4
        mock_parallel_backtest_runner.assert_called_once()
        mock_runner_instance.run.assert_called_once()

    @patch("src.backtester.optimization.ParallelBacktestRunner")
    @patch("random.choice")
    def test_random_search(
        self,
        mock_random_choice: MagicMock,
        mock_parallel_backtest_runner: MagicMock,
        optimizer: ParameterOptimizer,
        mock_backtest_result: BacktestResult,
        param_grid_simple: dict[str, list[Any]],
    ) -> None:
        # Simulate random choice returning specific values for determinism
        mock_random_choice.side_effect = [1, "A"] * 5  # Simulate 5 iterations

        mock_runner_instance = mock_parallel_backtest_runner.return_value
        # Simulate runner returning results for each task
        mock_runner_instance.run.return_value = {
            f"MockStrategy_iter{i}": mock_backtest_result for i in range(5)
        }

        # Mock strategy factory to ensure strategy.name is as expected by the search
        optimizer.strategy_factory.return_value.name = "MockStrategy"

        result = optimizer._random_search(
            param_grid_simple, "sharpe_ratio", maximize=True, n_iter=5
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_score == 1.5
        assert len(result.all_results) == 5
        assert mock_random_choice.call_count == 5 * len(param_grid_simple)  # 5 iter * 2 params
        mock_parallel_backtest_runner.assert_called_once()

    def test_optimize_method_dispatch(
        self,
        optimizer: ParameterOptimizer,
        param_grid_simple: dict[str, list[Any]],
    ) -> None:
        with patch.object(optimizer, "_grid_search") as mock_grid_search:
            optimizer.optimize(param_grid_simple, method="grid")
            mock_grid_search.assert_called_once()

        with patch.object(optimizer, "_random_search") as mock_random_search:
            optimizer.optimize(param_grid_simple, method="random")
            mock_random_search.assert_called_once()

        with pytest.raises(ValueError, match="Unknown optimization method: invalid"):
            optimizer.optimize(param_grid_simple, method="invalid")


class TestOptimizeStrategyParameters:
    """Tests for optimize_strategy_parameters convenience function."""

    @patch("src.backtester.optimization.ParameterOptimizer")
    def test_optimize_strategy_parameters(
        self,
        mock_parameter_optimizer: MagicMock,
        mock_strategy_factory: MagicMock,
        mock_backtest_config: BacktestConfig,
        param_grid_simple: dict[str, list[Any]],
    ) -> None:
        # Mock optimizer instance and its optimize method
        mock_optimizer_instance = mock_parameter_optimizer.return_value
        mock_optimization_result = MagicMock(spec=OptimizationResult)
        mock_optimizer_instance.optimize.return_value = mock_optimization_result

        tickers = ["KRW-BTC"]
        interval = "day"

        result = optimize_strategy_parameters(
            strategy_factory=mock_strategy_factory,
            param_grid=param_grid_simple,
            tickers=tickers,
            interval=interval,
            config=mock_backtest_config,
            metric="cagr",
            maximize=False,
            method="random",
            n_iter=50,
            n_workers=2,
        )

        mock_parameter_optimizer.assert_called_once_with(
            strategy_factory=mock_strategy_factory,
            tickers=tickers,
            interval=interval,
            config=mock_backtest_config,
            n_workers=2,
        )
        mock_optimizer_instance.optimize.assert_called_once_with(
            param_grid=param_grid_simple,
            metric="cagr",
            maximize=False,
            method="random",
            n_iter=50,
        )
        assert result == mock_optimization_result
