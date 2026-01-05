"""
Unit tests for CLI backtest command.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.backtester.engine import BacktestResult
from src.cli.commands.backtest import backtest


class TestBacktestCommand:
    """Test cases for backtest command."""

    def test_backtest_command_help(self) -> None:
        """Test backtest command help."""
        runner = CliRunner()
        result = runner.invoke(backtest, ["--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output or "help" in result.output.lower()

    @patch("src.cli.commands.backtest.run_backtest", create=True)
    @patch("src.cli.commands.backtest.create_vbo_strategy", create=True)
    def test_backtest_command_vanilla_strategy(
        self, mock_create_strategy: MagicMock, mock_run_backtest: MagicMock
    ) -> None:
        """Test backtest command with vanilla strategy."""
        mock_strategy = MagicMock()
        mock_create_strategy.return_value = mock_strategy

        mock_result = BacktestResult(
            total_return=0.1,
            cagr=0.15,
            mdd=0.05,
            sharpe_ratio=1.5,
            calmar_ratio=3.0,
            total_trades=10,
            win_rate=0.6,
            profit_factor=1.5,
            dates=[],
        )
        mock_run_backtest.return_value = mock_result

        runner = CliRunner()
        result = runner.invoke(backtest, ["--strategy", "vanilla", "--tickers", "KRW-BTC"])

        assert result.exit_code == 0
        mock_create_strategy.assert_called_once_with(
            name="VanillaVBO",
            use_trend_filter=True,
            use_noise_filter=True,
        )
        mock_run_backtest.assert_called_once()

    @patch("src.cli.commands.backtest.run_backtest", create=True)
    @patch("src.cli.commands.backtest.create_vbo_strategy", create=True)
    def test_backtest_command_minimal_strategy(
        self, mock_create_strategy: MagicMock, mock_run_backtest: MagicMock
    ) -> None:
        """Test backtest command with minimal strategy."""
        mock_strategy = MagicMock()
        mock_create_strategy.return_value = mock_strategy

        mock_result = BacktestResult(
            total_return=0.1,
            cagr=0.15,
            mdd=0.05,
            sharpe_ratio=1.5,
            calmar_ratio=3.0,
            total_trades=10,
            win_rate=0.6,
            profit_factor=1.5,
            dates=[],
        )
        mock_run_backtest.return_value = mock_result

        runner = CliRunner()
        result = runner.invoke(backtest, ["--strategy", "minimal"])

        assert result.exit_code == 0
        mock_create_strategy.assert_called_once_with(
            name="MinimalVBO",
            use_trend_filter=False,
            use_noise_filter=False,
        )

    @patch("src.cli.commands.backtest.run_backtest", create=True)
    @patch("src.cli.commands.backtest.create_vbo_strategy", create=True)
    def test_backtest_command_legacy_strategy(
        self, mock_create_strategy: MagicMock, mock_run_backtest: MagicMock
    ) -> None:
        """Test backtest command with legacy strategy."""
        mock_strategy = MagicMock()
        mock_create_strategy.return_value = mock_strategy

        mock_result = BacktestResult(
            total_return=0.1,
            cagr=0.15,
            mdd=0.05,
            sharpe_ratio=1.5,
            calmar_ratio=3.0,
            total_trades=10,
            win_rate=0.6,
            profit_factor=1.5,
            dates=[],
        )
        mock_run_backtest.return_value = mock_result

        runner = CliRunner()
        result = runner.invoke(backtest, ["--strategy", "legacy"])

        assert result.exit_code == 0
        mock_create_strategy.assert_called_once_with(
            name="LegacyBT",
            sma_period=5,
            trend_sma_period=10,
            short_noise_period=5,
            long_noise_period=10,
            use_trend_filter=True,
            use_noise_filter=True,
            exclude_current=True,
        )

    @patch("src.cli.commands.backtest.run_backtest", create=True)
    @patch("src.cli.commands.backtest.create_vbo_strategy", create=True)
    def test_backtest_command_custom_options(
        self, mock_create_strategy: MagicMock, mock_run_backtest: MagicMock
    ) -> None:
        """Test backtest command with custom options."""
        mock_strategy = MagicMock()
        mock_create_strategy.return_value = mock_strategy

        mock_result = BacktestResult(
            total_return=0.1,
            cagr=0.15,
            mdd=0.05,
            sharpe_ratio=1.5,
            calmar_ratio=3.0,
            total_trades=10,
            win_rate=0.6,
            profit_factor=1.5,
            dates=[],
        )
        mock_run_backtest.return_value = mock_result

        runner = CliRunner()
        # Use simpler options to avoid Click parsing issues
        result = runner.invoke(
            backtest,
            [
                "--tickers",
                "KRW-BTC",
                "--no-cache",
            ],
        )

        assert result.exit_code == 0
        mock_run_backtest.assert_called_once()
        call_args = mock_run_backtest.call_args
        assert call_args is not None
        config = call_args.kwargs.get("config")
        assert config is not None
        assert config.use_cache is False

    @patch("src.backtester.report.generate_report", create=True)
    @patch("src.cli.commands.backtest.run_backtest", create=True)
    @patch("src.cli.commands.backtest.create_vbo_strategy", create=True)
    def test_backtest_command_with_output(
        self,
        mock_create_strategy: MagicMock,
        mock_run_backtest: MagicMock,
        mock_generate_report: MagicMock,
    ) -> None:
        """Test backtest command with output directory."""
        mock_strategy = MagicMock()
        mock_create_strategy.return_value = mock_strategy

        mock_result = BacktestResult(
            total_return=0.1,
            cagr=0.15,
            mdd=0.05,
            sharpe_ratio=1.5,
            calmar_ratio=3.0,
            total_trades=10,
            win_rate=0.6,
            profit_factor=1.5,
            dates=[],
        )
        mock_run_backtest.return_value = mock_result

        runner = CliRunner()
        with runner.isolated_filesystem():
            output_dir = Path("test_output")
            output_dir.mkdir()

            result = runner.invoke(backtest, ["--output", str(output_dir)])

            assert result.exit_code == 0
            mock_generate_report.assert_called_once()
            call_args = mock_generate_report.call_args
            assert call_args is not None
            assert call_args.kwargs.get("save_path") == output_dir
            assert call_args.kwargs.get("show") is False

    @patch("src.cli.commands.backtest.run_backtest", create=True)
    @patch("src.cli.commands.backtest.create_vbo_strategy", create=True)
    def test_backtest_command_invalid_strategy(
        self, mock_create_strategy: MagicMock, mock_run_backtest: MagicMock
    ) -> None:
        """Test backtest command with invalid strategy (Click Choice validation)."""
        runner = CliRunner()
        result = runner.invoke(backtest, ["--strategy", "invalid"])

        assert result.exit_code != 0
        assert "invalid choice" in result.output.lower() or "invalid" in result.output.lower()

    @patch("src.cli.commands.backtest.run_backtest", create=True)
    @patch("src.cli.commands.backtest.create_vbo_strategy", create=True)
    def test_backtest_command_unknown_strategy_value_error(
        self, mock_create_strategy: MagicMock, mock_run_backtest: MagicMock
    ) -> None:
        """Test backtest command with unknown strategy raises ValueError (covers line 123)."""
        # Directly call the function to bypass Click's Choice validation
        # This tests the else clause that raises ValueError on line 123
        from src.cli.commands.backtest import backtest

        # Access the underlying callback function from the Click command
        # Click stores the original function in the 'callback' attribute
        callback_func = backtest.callback if hasattr(backtest, "callback") else backtest

        # Call the callback directly with an invalid strategy to test line 123
        with pytest.raises(ValueError, match="Unknown strategy: invalid_strategy"):
            callback_func(
                tickers=("KRW-BTC",),
                interval="day",
                initial_capital=1.0,
                fee_rate=0.0005,
                max_slots=4,
                strategy="invalid_strategy",  # Invalid strategy that triggers else clause (line 123)
                output=None,
                no_cache=False,
            )

    @patch("src.cli.commands.backtest.run_backtest", create=True)
    @patch("src.cli.commands.backtest.create_vbo_strategy", create=True)
    def test_backtest_command_empty_dates(
        self, mock_create_strategy: MagicMock, mock_run_backtest: MagicMock
    ) -> None:
        """Test backtest command with empty dates (covers line 144 edge case)."""
        mock_strategy = MagicMock()
        mock_create_strategy.return_value = mock_strategy

        # Result with empty dates array
        mock_result = BacktestResult(
            total_return=0.0,
            cagr=0.0,
            mdd=0.0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            dates=[],  # Empty dates array
        )
        mock_run_backtest.return_value = mock_result

        runner = CliRunner()
        result = runner.invoke(backtest, ["--tickers", "KRW-BTC"])

        # Should handle empty dates gracefully (period_days = 0)
        assert result.exit_code == 0
        mock_run_backtest.assert_called_once()

    @patch("src.cli.commands.backtest.run_backtest", create=True)
    @patch("src.cli.commands.backtest.create_vbo_strategy", create=True)
    def test_backtest_command_all_options(
        self, mock_create_strategy: MagicMock, mock_run_backtest: MagicMock
    ) -> None:
        """Test backtest command with all options specified."""
        mock_strategy = MagicMock()
        mock_create_strategy.return_value = mock_strategy

        from datetime import date

        mock_result = BacktestResult(
            total_return=100.0,
            cagr=50.0,
            mdd=10.0,
            sharpe_ratio=2.0,
            calmar_ratio=5.0,
            total_trades=100,
            win_rate=60.0,
            profit_factor=2.0,
            dates=[date(2020, 1, 1), date(2023, 1, 1)],
        )
        mock_run_backtest.return_value = mock_result

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                backtest,
                [
                    "--tickers",
                    "KRW-BTC",
                    "--tickers",
                    "KRW-ETH",
                    "--interval",
                    "day",
                    "--initial-capital",
                    "1000000",
                    "--fee-rate",
                    "0.001",
                    "--max-slots",
                    "2",
                    "--strategy",
                    "vanilla",
                    "--no-cache",
                ],
            )

            assert result.exit_code == 0
            call_args = mock_run_backtest.call_args
            assert call_args is not None
            config = call_args.kwargs.get("config")
            assert config is not None
            assert config.initial_capital == 1000000.0
            assert config.fee_rate == 0.001
            assert config.max_slots == 2
            assert config.use_cache is False

    @patch("src.backtester.report.generate_report", create=True)
    @patch("src.cli.commands.backtest.run_backtest", create=True)
    @patch("src.cli.commands.backtest.create_vbo_strategy", create=True)
    def test_backtest_command_output_with_string_path(
        self,
        mock_create_strategy: MagicMock,
        mock_run_backtest: MagicMock,
        mock_generate_report: MagicMock,
    ) -> None:
        """Test backtest command with string output path."""
        mock_strategy = MagicMock()
        mock_create_strategy.return_value = mock_strategy

        mock_result = BacktestResult(
            total_return=0.1,
            cagr=0.15,
            mdd=0.05,
            sharpe_ratio=1.5,
            calmar_ratio=3.0,
            total_trades=10,
            win_rate=0.6,
            profit_factor=1.5,
            dates=[],
        )
        mock_run_backtest.return_value = mock_result

        runner = CliRunner()
        with runner.isolated_filesystem():
            output_path = "test_reports"
            result = runner.invoke(backtest, ["--output", output_path])

            assert result.exit_code == 0
            mock_generate_report.assert_called_once()
            call_args = mock_generate_report.call_args
            assert call_args is not None
            # Should convert string to Path
            save_path = call_args.kwargs.get("save_path")
            assert save_path is not None
            assert str(save_path) == output_path

    @patch("src.cli.commands.backtest.run_backtest", create=True)
    @patch("src.cli.commands.backtest.create_vbo_strategy", create=True)
    def test_backtest_command_run_backtest_error(
        self, mock_create_strategy: MagicMock, mock_run_backtest: MagicMock
    ) -> None:
        """Test backtest command handles run_backtest errors."""
        mock_strategy = MagicMock()
        mock_create_strategy.return_value = mock_strategy
        mock_run_backtest.side_effect = ValueError("Test error")

        runner = CliRunner()
        result = runner.invoke(backtest, ["--tickers", "KRW-BTC"])

        # Should propagate error (Click will handle it)
        assert result.exit_code != 0
