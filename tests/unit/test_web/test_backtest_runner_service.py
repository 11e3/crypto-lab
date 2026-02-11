"""Tests for backtest runner service — BacktestService and execute_backtest.

Tests:
- BacktestService initialization (vectorized vs event-driven)
- BacktestService.run (success, failure)
- execute_backtest pure function (no Streamlit dependency)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.backtester.models import BacktestConfig
from src.web.services.backtest_runner import BacktestService, execute_backtest

# ── BacktestService ──


class TestBacktestServiceInit:
    """Test BacktestService initialization."""

    def test_default_uses_vectorized(self) -> None:
        config = BacktestConfig(initial_capital=1.0)
        service = BacktestService(config)
        assert service.engine is not None
        assert service.config == config

    def test_vectorized_flag_true(self) -> None:
        config = BacktestConfig(initial_capital=1.0)
        service = BacktestService(config, use_vectorized=True)
        from src.backtester.engine import VectorizedBacktestEngine

        assert isinstance(service.engine, VectorizedBacktestEngine)

    def test_vectorized_flag_false(self) -> None:
        config = BacktestConfig(initial_capital=1.0)
        service = BacktestService(config, use_vectorized=False)
        from src.backtester.engine import EventDrivenBacktestEngine

        assert isinstance(service.engine, EventDrivenBacktestEngine)

    def test_custom_engine(self) -> None:
        config = BacktestConfig(initial_capital=1.0)
        custom_engine = MagicMock()
        service = BacktestService(config, engine=custom_engine)
        assert service.engine is custom_engine


class TestBacktestServiceRun:
    """Test BacktestService.run method."""

    def test_successful_run(self) -> None:
        config = BacktestConfig(initial_capital=1.0)
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.total_return = 10.0
        mock_result.total_trades = 50
        mock_engine.run.return_value = mock_result

        service = BacktestService(config, engine=mock_engine)
        strategy = MagicMock()
        strategy.name = "TestStrategy"
        data_files = {"KRW-BTC": Path("/data/btc.parquet")}

        result = service.run(strategy, data_files)

        assert result is not None
        assert result.total_return == 10.0
        mock_engine.run.assert_called_once_with(
            strategy=strategy,
            data_files=data_files,
            start_date=None,
            end_date=None,
        )

    def test_with_date_range(self) -> None:
        config = BacktestConfig(initial_capital=1.0)
        mock_engine = MagicMock()
        mock_engine.run.return_value = MagicMock()

        service = BacktestService(config, engine=mock_engine)
        strategy = MagicMock()
        strategy.name = "TestStrategy"

        start = date(2024, 1, 1)
        end = date(2024, 12, 31)

        service.run(strategy, {}, start_date=start, end_date=end)

        call_kwargs = mock_engine.run.call_args.kwargs
        assert call_kwargs["start_date"] == start
        assert call_kwargs["end_date"] == end

    def test_engine_exception_returns_none(self) -> None:
        config = BacktestConfig(initial_capital=1.0)
        mock_engine = MagicMock()
        mock_engine.run.side_effect = RuntimeError("Engine failed")

        service = BacktestService(config, engine=mock_engine)
        strategy = MagicMock()
        strategy.name = "TestStrategy"

        result = service.run(strategy, {})
        assert result is None


# ── execute_backtest ──


class TestExecuteBacktest:
    """Test execute_backtest pure function."""

    @patch("src.web.services.backtest_runner.BacktestService")
    @patch("src.web.services.backtest_runner.BacktestConfig")
    @patch("src.web.components.sidebar.strategy_selector.create_strategy_instance")
    def test_successful_execution(
        self,
        mock_create: MagicMock,
        mock_config_cls: MagicMock,
        mock_service_cls: MagicMock,
    ) -> None:
        mock_strategy = MagicMock()
        mock_create.return_value = mock_strategy

        mock_config = MagicMock()
        mock_config_cls.return_value = mock_config

        mock_result = MagicMock()
        mock_service = MagicMock()
        mock_service.run.return_value = mock_result
        mock_service_cls.return_value = mock_service

        result = execute_backtest(
            strategy_name="VanillaVBO",
            strategy_params={"lookback": 5},
            data_files_dict={"KRW-BTC": "/data/btc.parquet"},
            config_dict={"initial_capital": 1.0},
            start_date_str="2024-01-01",
            end_date_str="2024-12-31",
        )

        assert result is not None
        mock_create.assert_called_once_with("VanillaVBO", {"lookback": 5})
        mock_service.run.assert_called_once()

    @patch("src.web.components.sidebar.strategy_selector.create_strategy_instance")
    def test_strategy_creation_failure(self, mock_create: MagicMock) -> None:
        mock_create.return_value = None

        result = execute_backtest(
            strategy_name="BadStrategy",
            strategy_params={},
            data_files_dict={},
            config_dict={"initial_capital": 1.0},
            start_date_str=None,
            end_date_str=None,
        )

        assert result is None

    @patch("src.web.components.sidebar.strategy_selector.create_strategy_instance")
    def test_exception_returns_none(self, mock_create: MagicMock) -> None:
        mock_create.side_effect = ImportError("Missing dependency")

        result = execute_backtest(
            strategy_name="TestStrategy",
            strategy_params={},
            data_files_dict={},
            config_dict={"initial_capital": 1.0},
            start_date_str=None,
            end_date_str=None,
        )

        assert result is None

    @patch("src.web.services.backtest_runner.BacktestService")
    @patch("src.web.services.backtest_runner.BacktestConfig")
    @patch("src.web.components.sidebar.strategy_selector.create_strategy_instance")
    def test_date_parsing(
        self,
        mock_create: MagicMock,
        mock_config_cls: MagicMock,
        mock_service_cls: MagicMock,
    ) -> None:
        mock_create.return_value = MagicMock()
        mock_config_cls.return_value = MagicMock()
        mock_service = MagicMock()
        mock_service.run.return_value = MagicMock()
        mock_service_cls.return_value = mock_service

        execute_backtest(
            strategy_name="TestStrategy",
            strategy_params={},
            data_files_dict={},
            config_dict={"initial_capital": 1.0},
            start_date_str="2024-06-15",
            end_date_str="2024-12-31",
        )

        # execute_backtest calls service.run(strategy, data_files, start_date, end_date) positionally
        call_args = mock_service.run.call_args[0]
        assert call_args[2] == date(2024, 6, 15)
        assert call_args[3] == date(2024, 12, 31)

    @patch("src.web.services.backtest_runner.BacktestService")
    @patch("src.web.services.backtest_runner.BacktestConfig")
    @patch("src.web.components.sidebar.strategy_selector.create_strategy_instance")
    def test_none_dates(
        self,
        mock_create: MagicMock,
        mock_config_cls: MagicMock,
        mock_service_cls: MagicMock,
    ) -> None:
        mock_create.return_value = MagicMock()
        mock_config_cls.return_value = MagicMock()
        mock_service = MagicMock()
        mock_service.run.return_value = MagicMock()
        mock_service_cls.return_value = mock_service

        execute_backtest(
            strategy_name="TestStrategy",
            strategy_params={},
            data_files_dict={},
            config_dict={"initial_capital": 1.0},
            start_date_str=None,
            end_date_str=None,
        )

        # Should pass None for dates
        assert mock_service.run.called
        run_args = mock_service.run.call_args[0]
        assert run_args[2] is None  # start_date
        assert run_args[3] is None  # end_date

    @patch("src.web.services.backtest_runner.BacktestService")
    @patch("src.web.services.backtest_runner.BacktestConfig")
    @patch("src.web.components.sidebar.strategy_selector.create_strategy_instance")
    def test_data_files_converted_to_paths(
        self,
        mock_create: MagicMock,
        mock_config_cls: MagicMock,
        mock_service_cls: MagicMock,
    ) -> None:
        mock_create.return_value = MagicMock()
        mock_config_cls.return_value = MagicMock()
        mock_service = MagicMock()
        mock_service.run.return_value = MagicMock()
        mock_service_cls.return_value = mock_service

        execute_backtest(
            strategy_name="TestStrategy",
            strategy_params={},
            data_files_dict={"KRW-BTC": "/data/btc.parquet"},
            config_dict={"initial_capital": 1.0},
            start_date_str=None,
            end_date_str=None,
        )

        call_args = mock_service.run.call_args
        data_files = (
            call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("data_files")
        )
        if data_files:
            assert isinstance(data_files["KRW-BTC"], Path)
