"""Tests for analysis service — Monte Carlo and Walk-Forward computation.

Tests use mocks to avoid actual backtest computation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.web.services.analysis_service import execute_monte_carlo, execute_walk_forward


class TestExecuteMonteCarlo:
    """Test Monte Carlo execution service."""

    @patch("src.web.services.analysis_service.run_monte_carlo")
    @patch("src.web.services.analysis_service.run_backtest")
    @patch("src.web.services.analysis_service.create_analysis_strategy")
    def test_returns_tuple(
        self,
        mock_strategy: MagicMock,
        mock_backtest: MagicMock,
        mock_mc: MagicMock,
    ) -> None:
        mock_strategy.return_value = MagicMock(name="TestStrategy")
        mock_backtest.return_value = MagicMock(name="BacktestResult")
        mock_mc.return_value = MagicMock(name="MonteCarloResult")

        mc_result, bt_result = execute_monte_carlo(
            strategy_type="vanilla",
            tickers=["KRW-BTC"],
            interval="day",
            n_simulations=100,
            method="bootstrap",
            seed=42,
            initial_capital=1.0,
            fee_rate=0.0005,
            max_slots=4,
        )

        assert mc_result is not None
        assert bt_result is not None

    @patch("src.web.services.analysis_service.run_monte_carlo")
    @patch("src.web.services.analysis_service.run_backtest")
    @patch("src.web.services.analysis_service.create_analysis_strategy")
    def test_passes_correct_config(
        self,
        mock_strategy: MagicMock,
        mock_backtest: MagicMock,
        mock_mc: MagicMock,
    ) -> None:
        mock_strategy.return_value = MagicMock()
        mock_backtest.return_value = MagicMock()
        mock_mc.return_value = MagicMock()

        execute_monte_carlo(
            strategy_type="vanilla",
            tickers=["KRW-BTC", "KRW-ETH"],
            interval="minute240",
            n_simulations=500,
            method="parametric",
            seed=None,
            initial_capital=10.0,
            fee_rate=0.001,
            max_slots=2,
        )

        # Verify backtest was called with correct params
        call_kwargs = mock_backtest.call_args
        assert call_kwargs.kwargs["tickers"] == ["KRW-BTC", "KRW-ETH"]
        assert call_kwargs.kwargs["interval"] == "minute240"

        config = call_kwargs.kwargs["config"]
        assert config.initial_capital == 10.0
        assert config.fee_rate == 0.001
        assert config.max_slots == 2

        # Verify monte carlo was called correctly
        mc_kwargs = mock_mc.call_args.kwargs
        assert mc_kwargs["n_simulations"] == 500
        assert mc_kwargs["method"] == "parametric"
        assert mc_kwargs["random_seed"] is None

    @patch("src.web.services.analysis_service.run_monte_carlo")
    @patch("src.web.services.analysis_service.run_backtest")
    @patch("src.web.services.analysis_service.create_analysis_strategy")
    def test_strategy_type_forwarded(
        self,
        mock_strategy: MagicMock,
        mock_backtest: MagicMock,
        mock_mc: MagicMock,
    ) -> None:
        mock_strategy.return_value = MagicMock()
        mock_backtest.return_value = MagicMock()
        mock_mc.return_value = MagicMock()

        execute_monte_carlo(
            strategy_type="momentum",
            tickers=["KRW-BTC"],
            interval="day",
            n_simulations=100,
            method="bootstrap",
            seed=42,
            initial_capital=1.0,
            fee_rate=0.0005,
            max_slots=4,
        )

        mock_strategy.assert_called_once_with("momentum")

    @patch("src.web.services.analysis_service.run_backtest")
    @patch("src.web.services.analysis_service.create_analysis_strategy")
    def test_backtest_failure_propagates(
        self,
        mock_strategy: MagicMock,
        mock_backtest: MagicMock,
    ) -> None:
        mock_strategy.return_value = MagicMock()
        mock_backtest.side_effect = RuntimeError("No data")

        with pytest.raises(RuntimeError, match="No data"):
            execute_monte_carlo(
                strategy_type="vanilla",
                tickers=["KRW-BTC"],
                interval="day",
                n_simulations=100,
                method="bootstrap",
                seed=42,
                initial_capital=1.0,
                fee_rate=0.0005,
                max_slots=4,
            )


class TestExecuteWalkForward:
    """Test Walk-Forward analysis service."""

    @patch("src.web.services.analysis_service.run_walk_forward_analysis")
    def test_returns_walk_forward_result(self, mock_wfa: MagicMock) -> None:
        mock_wfa.return_value = MagicMock(name="WalkForwardResult")

        result = execute_walk_forward(
            strategy_type="vanilla",
            param_grid={"lookback": [3, 5, 7]},
            tickers=["KRW-BTC"],
            interval="day",
            optimization_days=180,
            test_days=30,
            step_days=30,
            metric="sharpe_ratio",
            initial_capital=1.0,
            fee_rate=0.0005,
            max_slots=4,
            workers=2,
        )

        assert result is not None
        mock_wfa.assert_called_once()

    @patch("src.web.services.analysis_service.run_walk_forward_analysis")
    def test_passes_correct_wfa_params(self, mock_wfa: MagicMock) -> None:
        mock_wfa.return_value = MagicMock()

        execute_walk_forward(
            strategy_type="legacy",
            param_grid={"lookback": [5, 10]},
            tickers=["KRW-BTC", "KRW-ETH"],
            interval="minute240",
            optimization_days=365,
            test_days=60,
            step_days=30,
            metric="cagr",
            initial_capital=5.0,
            fee_rate=0.001,
            max_slots=3,
            workers=4,
        )

        call_kwargs = mock_wfa.call_args.kwargs
        assert call_kwargs["param_grid"] == {"lookback": [5, 10]}
        assert call_kwargs["tickers"] == ["KRW-BTC", "KRW-ETH"]
        assert call_kwargs["interval"] == "minute240"
        assert call_kwargs["optimization_days"] == 365
        assert call_kwargs["test_days"] == 60
        assert call_kwargs["step_days"] == 30
        assert call_kwargs["metric"] == "cagr"
        assert call_kwargs["n_workers"] == 4

        config = call_kwargs["config"]
        assert config.initial_capital == 5.0
        assert config.max_slots == 3

    @patch("src.web.services.analysis_service.run_walk_forward_analysis")
    def test_vanilla_strategy_factory(self, mock_wfa: MagicMock) -> None:
        """Test that vanilla strategy type creates correct strategy."""
        mock_wfa.return_value = MagicMock()

        execute_walk_forward(
            strategy_type="vanilla",
            param_grid={"sma_period": [3, 5]},
            tickers=["KRW-BTC"],
            interval="day",
            optimization_days=180,
            test_days=30,
            step_days=30,
            metric="sharpe_ratio",
            initial_capital=1.0,
            fee_rate=0.0005,
            max_slots=4,
            workers=2,
        )

        # Get the strategy_factory and test it with valid VBO param
        call_kwargs = mock_wfa.call_args.kwargs
        factory = call_kwargs["strategy_factory"]
        strategy = factory({"sma_period": 5})
        assert strategy is not None
        assert strategy.name == "VanillaVBO"

    @patch("src.web.services.analysis_service.run_walk_forward_analysis")
    def test_legacy_strategy_factory(self, mock_wfa: MagicMock) -> None:
        """Test that legacy strategy type creates correct strategy."""
        mock_wfa.return_value = MagicMock()

        execute_walk_forward(
            strategy_type="legacy",
            param_grid={"sma_period": [3, 5]},
            tickers=["KRW-BTC"],
            interval="day",
            optimization_days=180,
            test_days=30,
            step_days=30,
            metric="sharpe_ratio",
            initial_capital=1.0,
            fee_rate=0.0005,
            max_slots=4,
            workers=2,
        )

        call_kwargs = mock_wfa.call_args.kwargs
        factory = call_kwargs["strategy_factory"]
        strategy = factory({"sma_period": 5})
        assert strategy is not None
        assert strategy.name == "LegacyVBO"

    @patch("src.web.services.analysis_service.run_walk_forward_analysis")
    def test_wfa_failure_propagates(self, mock_wfa: MagicMock) -> None:
        mock_wfa.side_effect = ValueError("Not enough data")

        with pytest.raises(ValueError, match="Not enough data"):
            execute_walk_forward(
                strategy_type="vanilla",
                param_grid={"lookback": [5]},
                tickers=["KRW-BTC"],
                interval="day",
                optimization_days=180,
                test_days=30,
                step_days=30,
                metric="sharpe_ratio",
                initial_capital=1.0,
                fee_rate=0.0005,
                max_slots=4,
                workers=2,
            )
