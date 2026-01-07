"""
Unit tests for walk-forward analysis module.
"""

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtester.engine import BacktestConfig, BacktestResult
from src.backtester.optimization import OptimizationResult
from src.backtester.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardPeriod,
    WalkForwardResult,
    run_walk_forward_analysis,
)
from src.strategies.base import Strategy


@pytest.fixture
def mock_strategy_factory() -> MagicMock:
    """Mock strategy factory function."""
    mock_strategy = MagicMock(spec=Strategy)
    mock_strategy.name = "MockStrategy"
    mock_strategy.param_attr1 = 10
    mock_strategy.param_attr2 = "value"
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
    result.equity_curve = np.array([100, 110, 120])
    result.dates = np.array(
        [datetime.date(2023, 1, 1), datetime.date(2023, 1, 2), datetime.date(2023, 1, 3)]
    )
    return result


@pytest.fixture
def mock_optimization_result(mock_backtest_result: BacktestResult) -> OptimizationResult:
    """Mock OptimizationResult instance."""
    result = MagicMock(spec=OptimizationResult)
    result.best_params = {"param1": 1, "param2": "A"}
    result.best_score = 1.8
    result.best_result = mock_backtest_result
    return result


class TestWalkForwardPeriod:
    """Tests for WalkForwardPeriod dataclass."""

    def test_initialization(self) -> None:
        period = WalkForwardPeriod(
            period_num=1,
            optimization_start=datetime.date(2023, 1, 1),
            optimization_end=datetime.date(2023, 12, 31),
            test_start=datetime.date(2024, 1, 1),
            test_end=datetime.date(2024, 3, 31),
        )
        assert period.period_num == 1
        assert period.optimization_start == datetime.date(2023, 1, 1)

    def test_repr(self) -> None:
        period = WalkForwardPeriod(
            period_num=1,
            optimization_start=datetime.date(2023, 1, 1),
            optimization_end=datetime.date(2023, 12, 31),
            test_start=datetime.date(2024, 1, 1),
            test_end=datetime.date(2024, 3, 31),
        )
        assert (
            repr(period)
            == "WalkForwardPeriod(1, opt=2023-01-01 to 2023-12-31, test=2024-01-01 to 2024-03-31)"
        )


class TestWalkForwardResult:
    """Tests for WalkForwardResult dataclass."""

    def test_initialization(self) -> None:
        result = WalkForwardResult(periods=[])
        assert result.periods == []
        assert result.avg_test_cagr == 0.0

    def test_repr(self) -> None:
        period = WalkForwardPeriod(
            period_num=1,
            optimization_start=datetime.date(2023, 1, 1),
            optimization_end=datetime.date(2023, 12, 31),
            test_start=datetime.date(2024, 1, 1),
            test_end=datetime.date(2024, 3, 31),
        )
        result = WalkForwardResult(periods=[period], avg_test_cagr=15.0, total_periods=1)
        assert repr(result) == "WalkForwardResult(1 periods, avg_test_cagr=15.00%)"


class TestWalkForwardAnalyzer:
    """Tests for WalkForwardAnalyzer class."""

    @pytest.fixture
    def analyzer(
        self, mock_strategy_factory: MagicMock, mock_backtest_config: BacktestConfig
    ) -> WalkForwardAnalyzer:
        return WalkForwardAnalyzer(
            strategy_factory=mock_strategy_factory,
            tickers=["KRW-BTC"],
            interval="day",
            config=mock_backtest_config,
        )

    def test_init(
        self, mock_strategy_factory: MagicMock, mock_backtest_config: BacktestConfig
    ) -> None:
        analyzer = WalkForwardAnalyzer(
            strategy_factory=mock_strategy_factory,
            tickers=["KRW-BTC"],
            interval="day",
            config=mock_backtest_config,
        )
        assert analyzer.strategy_factory == mock_strategy_factory
        assert analyzer.tickers == ["KRW-BTC"]

    def test_generate_periods(self, analyzer: WalkForwardAnalyzer) -> None:
        periods = analyzer._generate_periods(
            start_date=datetime.date(2023, 1, 1),
            end_date=datetime.date(2025, 12, 31),
            optimization_days=365,
            test_days=90,
            step_days=90,
        )
        assert len(periods) > 0
        assert periods[0].period_num == 1
        assert periods[0].optimization_start == datetime.date(2023, 1, 1)
        assert periods[0].test_end == datetime.date(2024, 3, 31)

    def test_generate_periods_insufficient_data(self, analyzer: WalkForwardAnalyzer) -> None:
        periods = analyzer._generate_periods(
            start_date=datetime.date(2023, 1, 1),
            end_date=datetime.date(2023, 3, 31),  # Less than test_days
            optimization_days=365,
            test_days=90,
            step_days=90,
        )
        assert len(periods) == 0

    def test_extract_metric(
        self, analyzer: WalkForwardAnalyzer, mock_backtest_result: BacktestResult
    ) -> None:
        assert analyzer._extract_metric(mock_backtest_result, "sharpe_ratio") == 1.5
        assert analyzer._extract_metric(mock_backtest_result, "cagr") == 10.0
        assert (
            analyzer._extract_metric(mock_backtest_result, "invalid_metric") == 1.5
        )  # Falls back to sharpe_ratio

    @patch("src.backtester.parallel.ParallelBacktestRunner")
    @patch("src.backtester.walk_forward.OptimizationResult")
    def test_optimize_period(
        self,
        mock_optimization_result: MagicMock,
        mock_parallel_backtest_runner: MagicMock,
        analyzer: WalkForwardAnalyzer,
        mock_backtest_result: BacktestResult,
    ) -> None:
        # Mock ParallelBacktestRunner.run to return a result
        mock_runner_instance = mock_parallel_backtest_runner.return_value
        mock_runner_instance.run.return_value = {
            "MockStrategy_1": mock_backtest_result,
            "MockStrategy_2": mock_backtest_result,
        }

        # Mock OptimizationResult to be returned by _optimize_period
        mock_opt_result_instance = mock_optimization_result.return_value
        mock_opt_result_instance.best_params = {"param": 1}
        mock_opt_result_instance.best_score = 1.5
        mock_opt_result_instance.best_result = mock_backtest_result

        period = WalkForwardPeriod(
            period_num=1,
            optimization_start=datetime.date(2023, 1, 1),
            optimization_end=datetime.date(2023, 12, 31),
            test_start=datetime.date(2024, 1, 1),
            test_end=datetime.date(2024, 3, 31),
        )
        param_grid = {"param": [1, 2]}
        metric = "sharpe_ratio"

        opt_result = analyzer._optimize_period(period, param_grid, metric, n_workers=1)
        assert opt_result is not None
        assert opt_result.best_params == {
            "param": 1
        }  # Depends on sorting, this may need to be dynamic
        mock_parallel_backtest_runner.assert_called_once()
        mock_runner_instance.run.assert_called_once()

    @patch("src.backtester.walk_forward.run_backtest")
    def test_test_period(
        self,
        mock_run_backtest: MagicMock,
        analyzer: WalkForwardAnalyzer,
        mock_backtest_result: BacktestResult,
    ) -> None:
        mock_run_backtest.return_value = mock_backtest_result
        period = WalkForwardPeriod(
            period_num=1,
            optimization_start=datetime.date(2023, 1, 1),
            optimization_end=datetime.date(2023, 12, 31),
            test_start=datetime.date(2024, 1, 1),
            test_end=datetime.date(2024, 3, 31),
        )
        best_params = {"param": 1}

        test_result = analyzer._test_period(period, best_params)
        assert test_result == mock_backtest_result
        mock_run_backtest.assert_called_once_with(
            strategy=analyzer.strategy_factory.return_value,
            tickers=analyzer.tickers,
            interval=analyzer.interval,
            config=analyzer.config,
            start_date=period.test_start,
            end_date=period.test_end,
        )

    def test_calculate_statistics(
        self,
        analyzer: WalkForwardAnalyzer,
        mock_backtest_result: BacktestResult,
        mock_optimization_result: OptimizationResult,
    ) -> None:
        period1 = WalkForwardPeriod(
            period_num=1,
            optimization_start=datetime.date(2023, 1, 1),
            optimization_end=datetime.date(2023, 12, 31),
            test_start=datetime.date(2024, 1, 1),
            test_end=datetime.date(2024, 3, 31),
            optimization_result=mock_optimization_result,
            test_result=mock_backtest_result,
        )
        period2 = WalkForwardPeriod(
            period_num=2,
            optimization_start=datetime.date(2023, 4, 1),
            optimization_end=datetime.date(2024, 3, 31),
            test_start=datetime.date(2024, 4, 1),
            test_end=datetime.date(2024, 6, 30),
            optimization_result=mock_optimization_result,
            test_result=mock_backtest_result,  # Another positive result
        )
        period3 = WalkForwardPeriod(
            period_num=3,
            optimization_start=datetime.date(2023, 7, 1),
            optimization_end=datetime.date(2024, 6, 30),
            test_start=datetime.date(2024, 7, 1),
            test_end=datetime.date(2024, 9, 30),
            optimization_result=mock_optimization_result,
            # No test result for this period to test robustness
        )

        # Make one of the test results negative for CAGR
        negative_result = MagicMock(spec=BacktestResult)
        negative_result.cagr = -5.0
        negative_result.sharpe_ratio = -0.5
        negative_result.mdd = 0.1

        period4 = WalkForwardPeriod(
            period_num=4,
            optimization_start=datetime.date(2023, 10, 1),
            optimization_end=datetime.date(2024, 9, 30),
            test_start=datetime.date(2024, 10, 1),
            test_end=datetime.date(2024, 12, 31),
            optimization_result=mock_optimization_result,
            test_result=negative_result,
        )

        periods = [period1, period2, period3, period4]
        stats = analyzer._calculate_statistics(periods)

        assert stats.total_periods == 3  # period3 has no test_result
        assert stats.positive_periods == 2  # period1, period2
        assert stats.consistency_rate == pytest.approx((2 / 3) * 100)
        assert stats.avg_test_cagr == pytest.approx((10.0 + 10.0 + (-5.0)) / 3)
        assert stats.avg_test_sharpe == pytest.approx((1.5 + 1.5 + (-0.5)) / 3)
        assert stats.avg_test_mdd == pytest.approx((0.05 + 0.05 + 0.1) / 3)
        assert stats.avg_optimization_cagr == pytest.approx(
            mock_backtest_result.cagr
        )  # opt_result.best_result.cagr

    @patch("src.backtester.walk_forward.WalkForwardAnalyzer._calculate_statistics")
    @patch("src.backtester.walk_forward.WalkForwardAnalyzer._test_period")
    @patch("src.backtester.walk_forward.WalkForwardAnalyzer._optimize_period")
    @patch("src.backtester.walk_forward.WalkForwardAnalyzer._generate_periods")
    @patch("src.data.upbit_source.UpbitDataSource")
    def test_analyze(
        self,
        mock_upbit_data_source: MagicMock,
        mock_generate_periods: MagicMock,
        mock_optimize_period: MagicMock,
        mock_test_period: MagicMock,
        mock_calculate_statistics: MagicMock,
        analyzer: WalkForwardAnalyzer,
        mock_backtest_result: BacktestResult,
        mock_optimization_result: OptimizationResult,
    ) -> None:
        # Mock data source to return some data
        mock_data_source_instance = mock_upbit_data_source.return_value
        mock_data_source_instance.load_ohlcv.return_value = pd.DataFrame(
            index=pd.to_datetime(pd.date_range("2023-01-01", periods=1000, freq="D"))
        )

        # Mock generated periods
        period = WalkForwardPeriod(
            period_num=1,
            optimization_start=datetime.date(2023, 1, 1),
            optimization_end=datetime.date(2023, 12, 31),
            test_start=datetime.date(2024, 1, 1),
            test_end=datetime.date(2024, 3, 31),
        )
        mock_generate_periods.return_value = [period]

        # Mock optimization and test results
        mock_optimize_period.return_value = mock_optimization_result
        mock_test_period.return_value = mock_backtest_result

        # Mock calculate_statistics
        mock_overall_result = WalkForwardResult(periods=[period], avg_test_cagr=15.0)
        mock_calculate_statistics.return_value = mock_overall_result

        param_grid = {"sma": [10]}
        result = analyzer.analyze(param_grid)

        mock_upbit_data_source.assert_called_once()
        mock_data_source_instance.load_ohlcv.assert_called_once()
        mock_generate_periods.assert_called_once()
        mock_optimize_period.assert_called_once_with(
            period=period, param_grid=param_grid, metric="sharpe_ratio", n_workers=None
        )
        mock_test_period.assert_called_once_with(
            period=period, best_params=mock_optimization_result.best_params
        )
        mock_calculate_statistics.assert_called_once_with([period])
        assert result == mock_overall_result
        assert period.optimization_result == mock_optimization_result
        assert period.test_result == mock_backtest_result

    @patch("src.data.upbit_source.UpbitDataSource")
    def test_analyze_no_data(
        self, mock_upbit_data_source: MagicMock, analyzer: WalkForwardAnalyzer
    ) -> None:
        mock_data_source_instance = mock_upbit_data_source.return_value
        mock_data_source_instance.load_ohlcv.return_value = pd.DataFrame()  # Empty DataFrame

        param_grid = {"sma": [10]}
        with pytest.raises(ValueError, match="No data available for walk-forward analysis"):
            analyzer.analyze(param_grid)

    @patch("src.backtester.walk_forward.WalkForwardAnalyzer")
    def test_run_walk_forward_analysis(
        self,
        mock_walk_forward_analyzer: MagicMock,
        mock_strategy_factory: MagicMock,
        mock_backtest_config: BacktestConfig,
    ) -> None:
        """Test run_walk_forward_analysis convenience function."""
        # Mock analyzer instance and its analyze method
        mock_analyzer_instance = mock_walk_forward_analyzer.return_value
        mock_result = WalkForwardResult(periods=[], avg_test_cagr=20.0)
        mock_analyzer_instance.analyze.return_value = mock_result

        param_grid = {"sma": [10]}
        tickers = ["KRW-BTC"]
        interval = "day"

        result = run_walk_forward_analysis(
            strategy_factory=mock_strategy_factory,
            param_grid=param_grid,
            tickers=tickers,
            interval=interval,
            config=mock_backtest_config,
        )

        mock_walk_forward_analyzer.assert_called_once_with(
            strategy_factory=mock_strategy_factory,
            tickers=tickers,
            interval=interval,
            config=mock_backtest_config,
        )
        mock_analyzer_instance.analyze.assert_called_once_with(
            param_grid=param_grid,
            optimization_days=365,
            test_days=90,
            step_days=90,
            metric="sharpe_ratio",
            start_date=None,
            end_date=None,
            n_workers=None,
        )
        assert result == mock_result
