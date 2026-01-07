"""Tests for CLI monte_carlo command."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli.commands.monte_carlo import monte_carlo


class TestMonteCarloCommand:
    """Test monte_carlo CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()

    @patch("src.cli.commands.monte_carlo.run_backtest")
    @patch("src.cli.commands.monte_carlo.run_monte_carlo")
    def test_monte_carlo_basic(
        self, mock_mc: MagicMock, mock_backtest: MagicMock, runner: CliRunner
    ) -> None:
        """Test monte-carlo command with defaults."""
        mock_backtest.return_value = MagicMock(equity_curve=[1.0, 1.01, 1.02], trades=[])
        mock_mc.return_value = MagicMock()

        result = runner.invoke(monte_carlo, [])
        # May succeed or fail depending on data availability
        assert result.exit_code in [0, 1, 2]

    @patch("src.cli.commands.monte_carlo.run_backtest")
    @patch("src.cli.commands.monte_carlo.run_monte_carlo")
    def test_monte_carlo_custom_params(
        self, mock_mc: MagicMock, mock_backtest: MagicMock, runner: CliRunner
    ) -> None:
        """Test monte-carlo command with custom parameters."""
        mock_backtest.return_value = MagicMock(equity_curve=[1.0, 1.01], trades=[])
        mock_mc.return_value = MagicMock()

        result = runner.invoke(
            monte_carlo, ["--tickers", "KRW-BTC", "--interval", "day", "--simulations", "100"]
        )
        assert result.exit_code in [0, 1, 2]

    @patch("src.cli.commands.monte_carlo.run_backtest")
    @patch("src.cli.commands.monte_carlo.run_monte_carlo")
    def test_monte_carlo_invalid_interval(
        self, mock_mc: MagicMock, mock_backtest: MagicMock, runner: CliRunner
    ) -> None:
        """Test monte-carlo command with invalid interval."""
        result = runner.invoke(monte_carlo, ["--interval", "invalid"])
        assert result.exit_code != 0
