"""
Tests for parallel backtesting functionality.
"""

import multiprocessing as mp
from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.backtester.engine import BacktestConfig, BacktestResult
from src.backtester.parallel import (
    ParallelBacktestRunner,
    ParallelBacktestTask,
    _run_single_backtest,
    compare_strategies,
    optimize_parameters,
)
from src.strategies.base import Strategy

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def mock_config() -> BacktestConfig:
    return BacktestConfig(
        initial_capital=10000.0,
        fee_rate=0.001,
        slippage_rate=0.001,
        max_slots=2,
        position_sizing="equal",
        use_cache=False,
    )


@pytest.fixture
def mock_strategy() -> MagicMock:
    """Create a mock strategy."""
    strategy = MagicMock(spec=Strategy)
    strategy.name = "TestStrategy"
    strategy.entry_conditions = MagicMock()
    strategy.entry_conditions.conditions = []
    strategy.exit_conditions = MagicMock()
    strategy.exit_conditions.conditions = []
    strategy.required_indicators.return_value = []
    return strategy


@pytest.fixture
def mock_result() -> BacktestResult:
    """Create a mock BacktestResult."""
    result = BacktestResult()
    result.strategy_name = "TestStrategy"
    result.total_trades = 10
    result.win_rate = 0.5
    result.total_return = 0.1
    return result


@pytest.fixture
def sample_task(mock_strategy: MagicMock, mock_config: BacktestConfig) -> ParallelBacktestTask:
    """Create a sample ParallelBacktestTask."""
    return ParallelBacktestTask(
        name="TestTask",
        strategy=mock_strategy,
        tickers=["KRW-BTC", "KRW-ETH"],
        interval="day",
        config=mock_config,
        params={"param1": 1, "param2": 2},
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
    )


# -------------------------------------------------------------------------
# ParallelBacktestTask Tests
# -------------------------------------------------------------------------


class TestParallelBacktestTask:
    """Test ParallelBacktestTask dataclass."""

    def test_task_creation(self, sample_task: ParallelBacktestTask) -> None:
        """Test task creation with all parameters."""
        assert sample_task.name == "TestTask"
        assert sample_task.tickers == ["KRW-BTC", "KRW-ETH"]
        assert sample_task.interval == "day"
        assert sample_task.params == {"param1": 1, "param2": 2}
        assert sample_task.start_date == date(2023, 1, 1)
        assert sample_task.end_date == date(2023, 12, 31)

    def test_task_repr(self, sample_task: ParallelBacktestTask) -> None:
        """Test task string representation."""
        repr_str = repr(sample_task)
        assert "TestTask" in repr_str
        assert "KRW-BTC" in repr_str

    def test_task_without_optional_params(
        self, mock_strategy: MagicMock, mock_config: BacktestConfig
    ) -> None:
        """Test task creation without optional parameters."""
        task = ParallelBacktestTask(
            name="SimpleTask",
            strategy=mock_strategy,
            tickers=["KRW-BTC"],
            interval="day",
            config=mock_config,
        )
        assert task.params is None
        assert task.start_date is None
        assert task.end_date is None


# -------------------------------------------------------------------------
# _run_single_backtest Tests
# -------------------------------------------------------------------------


class TestRunSingleBacktest:
    """Test single backtest worker function."""

    @patch("src.backtester.parallel.run_backtest")
    def test_run_single_backtest_success(
        self,
        mock_run_backtest: MagicMock,
        sample_task: ParallelBacktestTask,
        mock_result: BacktestResult,
    ) -> None:
        """Test successful single backtest execution."""
        mock_run_backtest.return_value = mock_result

        task_name, result = _run_single_backtest(sample_task)

        assert task_name == "TestTask"
        assert result == mock_result
        mock_run_backtest.assert_called_once()
        call_kwargs = mock_run_backtest.call_args[1]
        assert call_kwargs["strategy"] == sample_task.strategy
        assert call_kwargs["tickers"] == sample_task.tickers
        assert call_kwargs["interval"] == sample_task.interval
        assert call_kwargs["start_date"] == sample_task.start_date
        assert call_kwargs["end_date"] == sample_task.end_date

    @patch("src.backtester.parallel.run_backtest")
    def test_run_single_backtest_exception(
        self, mock_run_backtest: MagicMock, sample_task: ParallelBacktestTask
    ) -> None:
        """Test single backtest with exception handling."""
        mock_run_backtest.side_effect = ValueError("Test error")

        task_name, result = _run_single_backtest(sample_task)

        assert task_name == "TestTask"
        assert isinstance(result, BacktestResult)
        # Result should be empty on error
        assert result.strategy_name == "TestTask"


# -------------------------------------------------------------------------
# ParallelBacktestRunner Tests
# -------------------------------------------------------------------------


class TestParallelBacktestRunner:
    """Test ParallelBacktestRunner class."""

    def test_runner_initialization_default(self) -> None:
        """Test runner initialization with default worker count."""
        runner = ParallelBacktestRunner()
        assert runner.n_workers == max(1, mp.cpu_count() - 1)

    def test_runner_initialization_custom_workers(self) -> None:
        """Test runner initialization with custom worker count."""
        runner = ParallelBacktestRunner(n_workers=2)
        assert runner.n_workers == 2

    def test_runner_initialization_single_worker(self) -> None:
        """Test runner initialization with single worker."""
        runner = ParallelBacktestRunner(n_workers=1)
        assert runner.n_workers == 1

    @patch("src.backtester.parallel._run_single_backtest")
    @patch("multiprocessing.Pool")
    def test_run_empty_tasks(
        self,
        mock_pool_class: MagicMock,
        mock_run: MagicMock,
        mock_config: BacktestConfig,
        mock_strategy: MagicMock,
    ) -> None:
        """Test run with empty task list."""
        runner = ParallelBacktestRunner(n_workers=2)
        result = runner.run([])

        assert result == {}
        mock_pool_class.assert_not_called()

    @patch("src.backtester.parallel._run_single_backtest")
    def test_run_sequential_empty_tasks(
        self, mock_run: MagicMock, mock_config: BacktestConfig, mock_strategy: MagicMock
    ) -> None:
        """Test sequential run with empty task list."""
        runner = ParallelBacktestRunner(n_workers=2)
        result = runner.run_sequential([])

        assert result == {}
        mock_run.assert_not_called()

    @patch("src.backtester.parallel._run_single_backtest")
    def test_run_sequential_single_task(
        self, mock_run: MagicMock, sample_task: ParallelBacktestTask, mock_result: BacktestResult
    ) -> None:
        """Test sequential run with single task."""
        mock_run.return_value = ("TestTask", mock_result)
        runner = ParallelBacktestRunner(n_workers=2)

        result = runner.run_sequential([sample_task])

        assert "TestTask" in result
        assert result["TestTask"] == mock_result
        mock_run.assert_called_once_with(sample_task)

    @patch("src.backtester.parallel._run_single_backtest")
    def test_run_sequential_multiple_tasks(
        self,
        mock_run: MagicMock,
        mock_config: BacktestConfig,
        mock_strategy: MagicMock,
        mock_result: BacktestResult,
    ) -> None:
        """Test sequential run with multiple tasks."""
        task1 = ParallelBacktestTask(
            name="Task1",
            strategy=mock_strategy,
            tickers=["KRW-BTC"],
            interval="day",
            config=mock_config,
        )
        task2 = ParallelBacktestTask(
            name="Task2",
            strategy=mock_strategy,
            tickers=["KRW-ETH"],
            interval="day",
            config=mock_config,
        )

        mock_result1 = mock_result
        mock_result1.strategy_name = "Task1"
        mock_result2 = BacktestResult()
        mock_result2.strategy_name = "Task2"

        mock_run.side_effect = [("Task1", mock_result1), ("Task2", mock_result2)]
        runner = ParallelBacktestRunner(n_workers=2)

        result = runner.run_sequential([task1, task2])

        assert len(result) == 2
        assert "Task1" in result
        assert "Task2" in result
        assert mock_run.call_count == 2

    @patch("src.backtester.parallel._run_single_backtest")
    def test_run_sequential_with_progress_callback(
        self, mock_run: MagicMock, sample_task: ParallelBacktestTask, mock_result: BacktestResult
    ) -> None:
        """Test sequential run with progress callback."""
        mock_run.return_value = ("TestTask", mock_result)
        runner = ParallelBacktestRunner(n_workers=1)

        callback = MagicMock()
        result = runner.run_sequential([sample_task], progress_callback=callback)

        assert "TestTask" in result
        callback.assert_called_once_with("TestTask", mock_result)

    @patch("src.backtester.parallel._run_single_backtest")
    def test_run_with_progress_callback(
        self, mock_run: MagicMock, sample_task: ParallelBacktestTask, mock_result: BacktestResult
    ) -> None:
        """Test parallel run with progress callback."""
        mock_run.return_value = ("TestTask", mock_result)
        runner = ParallelBacktestRunner(n_workers=1)

        callback = MagicMock()

        # We can't easily mock multiprocessing.Pool.map directly,
        # so we test the sequential fallback behavior is correct
        with patch("multiprocessing.Pool") as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value.__enter__.return_value = mock_pool
            mock_pool.map.return_value = [("TestTask", mock_result)]

            result = runner.run([sample_task], progress_callback=callback)

            assert "TestTask" in result
            callback.assert_called_once_with("TestTask", mock_result)


# -------------------------------------------------------------------------
# compare_strategies Tests
# -------------------------------------------------------------------------


class TestCompareStrategies:
    """Test strategy comparison functionality."""

    @patch("src.backtester.parallel.ParallelBacktestRunner.run")
    def test_compare_strategies(
        self, mock_runner_run: MagicMock, mock_config: BacktestConfig
    ) -> None:
        """Test comparing multiple strategies."""
        strategy1 = MagicMock(spec=Strategy)
        strategy1.name = "Strategy1"
        strategy2 = MagicMock(spec=Strategy)
        strategy2.name = "Strategy2"

        result1 = BacktestResult()
        result1.strategy_name = "Strategy1"
        result2 = BacktestResult()
        result2.strategy_name = "Strategy2"

        mock_runner_run.return_value = {
            "Strategy1": result1,
            "Strategy2": result2,
        }

        results = compare_strategies(
            strategies=[strategy1, strategy2],
            tickers=["KRW-BTC"],
            interval="day",
            config=mock_config,
            n_workers=2,
        )

        assert len(results) == 2
        assert "Strategy1" in results
        assert "Strategy2" in results

    @patch("src.backtester.parallel.ParallelBacktestRunner.run")
    def test_compare_single_strategy(
        self, mock_runner_run: MagicMock, mock_config: BacktestConfig, mock_strategy: MagicMock
    ) -> None:
        """Test comparing single strategy."""
        result = BacktestResult()
        result.strategy_name = "TestStrategy"
        mock_runner_run.return_value = {"TestStrategy": result}

        results = compare_strategies(
            strategies=[mock_strategy],
            tickers=["KRW-BTC"],
            interval="day",
            config=mock_config,
        )

        assert len(results) == 1
        assert "TestStrategy" in results


# -------------------------------------------------------------------------
# optimize_parameters Tests
# -------------------------------------------------------------------------


class TestOptimizeParameters:
    """Test parameter optimization functionality."""

    @patch("src.backtester.parallel.ParallelBacktestRunner.run")
    def test_optimize_parameters_simple_grid(
        self, mock_runner_run: MagicMock, mock_config: BacktestConfig, mock_strategy: MagicMock
    ) -> None:
        """Test parameter optimization with simple grid."""

        def strategy_factory(params: dict[str, Any]) -> MagicMock:
            strategy = MagicMock(spec=Strategy)
            strategy.name = f"Strategy_{params['param1']}"
            return strategy

        param_grid = {
            "param1": [1, 2],
            "param2": [10, 20],
        }

        # Should generate 4 combinations: (1,10), (1,20), (2,10), (2,20)
        results_mock = {f"Strategy_{i}_{j}": BacktestResult() for i in [1, 2] for j in [10, 20]}
        mock_runner_run.return_value = results_mock

        optimize_parameters(
            strategy_factory=strategy_factory,
            param_grid=param_grid,
            tickers=["KRW-BTC"],
            interval="day",
            config=mock_config,
            n_workers=2,
        )

        # Verify run was called with correct number of tasks
        call_args = mock_runner_run.call_args
        tasks = call_args[0][0]
        assert len(tasks) == 4  # 2 x 2 combinations

    @patch("src.backtester.parallel.ParallelBacktestRunner.run")
    def test_optimize_parameters_single_param(
        self, mock_runner_run: MagicMock, mock_config: BacktestConfig, mock_strategy: MagicMock
    ) -> None:
        """Test parameter optimization with single parameter."""

        def strategy_factory(params: dict[str, Any]) -> MagicMock:
            strategy = MagicMock(spec=Strategy)
            strategy.name = f"Strategy_{params['sma']}"
            return strategy

        param_grid = {
            "sma": [5, 10, 15],
        }

        results_mock = {f"Strategy_{i}": BacktestResult() for i in [5, 10, 15]}
        mock_runner_run.return_value = results_mock

        optimize_parameters(
            strategy_factory=strategy_factory,
            param_grid=param_grid,
            tickers=["KRW-BTC"],
            interval="day",
            config=mock_config,
        )

        call_args = mock_runner_run.call_args
        tasks = call_args[0][0]
        assert len(tasks) == 3  # 3 values for single parameter

    @patch("src.backtester.parallel.ParallelBacktestRunner.run")
    def test_optimize_parameters_three_dimensions(
        self, mock_runner_run: MagicMock, mock_config: BacktestConfig
    ) -> None:
        """Test parameter optimization with three-dimensional grid."""

        def strategy_factory(params: dict[str, Any]) -> MagicMock:
            strategy = MagicMock(spec=Strategy)
            strategy.name = f"S_{params['a']}_{params['b']}_{params['c']}"
            return strategy

        param_grid = {
            "a": [1, 2],
            "b": [10, 20],
            "c": [100],
        }

        # Should generate 4 combinations: 2 x 2 x 1
        mock_runner_run.return_value = {}

        optimize_parameters(
            strategy_factory=strategy_factory,
            param_grid=param_grid,
            tickers=["KRW-BTC"],
            interval="day",
            config=mock_config,
        )

        call_args = mock_runner_run.call_args
        tasks = call_args[0][0]
        assert len(tasks) == 4  # 2 x 2 x 1 combinations


# -------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------


class TestParallelIntegration:
    """Integration tests for parallel backtesting."""

    @patch("src.backtester.parallel.run_backtest")
    def test_sequential_backtest_workflow(
        self, mock_run_backtest: MagicMock, mock_config: BacktestConfig, mock_strategy: MagicMock
    ) -> None:
        """Test complete sequential backtest workflow."""
        result = BacktestResult()
        result.strategy_name = "TestStrategy"
        result.total_trades = 5
        mock_run_backtest.return_value = result

        task = ParallelBacktestTask(
            name="Test",
            strategy=mock_strategy,
            tickers=["KRW-BTC"],
            interval="day",
            config=mock_config,
        )

        runner = ParallelBacktestRunner(n_workers=1)
        results = runner.run_sequential([task])

        assert "Test" in results
        assert results["Test"].total_trades == 5

    @patch("src.backtester.parallel.run_backtest")
    def test_error_handling_in_sequential(
        self, mock_run_backtest: MagicMock, mock_config: BacktestConfig, mock_strategy: MagicMock
    ) -> None:
        """Test error handling in sequential execution."""
        mock_run_backtest.side_effect = [
            BacktestResult(),  # Success
            ValueError("Error"),  # Failure
            BacktestResult(),  # Success
        ]

        tasks = [
            ParallelBacktestTask(
                name=f"Task{i}",
                strategy=mock_strategy,
                tickers=["KRW-BTC"],
                interval="day",
                config=mock_config,
            )
            for i in range(3)
        ]

        ParallelBacktestRunner(n_workers=1)
        # Manually test error handling in worker
        result_task = tasks[1]
        task_name, result = _run_single_backtest(result_task)

        assert isinstance(result, BacktestResult)
