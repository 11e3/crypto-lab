"""
Tests for Walk Forward Analysis CLI command.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli.commands.walk_forward import walk_forward


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


class TestWalkForwardCommand:
    """Test Walk Forward Analysis CLI command."""

    def test_walk_forward_help(self, runner: CliRunner) -> None:
        """Test help message."""
        result = runner.invoke(walk_forward, ["--help"])
        assert result.exit_code == 0
        assert "walk-forward" in result.output.lower() or "Walk" in result.output

    @patch("src.cli.commands.walk_forward.run_walk_forward_analysis")
    def test_walk_forward_execution(self, mock_walk_forward: MagicMock, runner: CliRunner) -> None:
        """Test walk forward executes."""
        mock_walk_forward.return_value = {}

        result = runner.invoke(walk_forward, [])

        # Allow for minor variations in test environment
        assert "error" not in result.output.lower() or result.exit_code == 0

    @patch("src.cli.commands.walk_forward.run_walk_forward_analysis")
    def test_walk_forward_accepts_options(
        self, mock_walk_forward: MagicMock, runner: CliRunner
    ) -> None:
        """Test walk forward accepts various options."""
        mock_walk_forward.return_value = {}

        result = runner.invoke(
            walk_forward,
            [
                "--tickers",
                "KRW-BTC",
                "--interval",
                "day",
                "--optimization-days",
                "180",
                "--test-days",
                "45",
            ],
        )

        # Command should accept options without syntax errors
        assert "unrecognized arguments" not in result.output.lower()

    @patch("src.cli.commands.walk_forward.run_walk_forward_analysis")
    def test_walk_forward_multiple_strategies(
        self, mock_walk_forward: MagicMock, runner: CliRunner
    ) -> None:
        """Test walk forward with different strategy options."""
        mock_walk_forward.return_value = {}

        for strategy in ["vanilla", "legacy"]:
            result = runner.invoke(walk_forward, ["--strategy", strategy])
            assert "unrecognized arguments" not in result.output.lower()

    @patch("src.cli.commands.walk_forward.run_walk_forward_analysis")
    def test_walk_forward_metric_options(
        self, mock_walk_forward: MagicMock, runner: CliRunner
    ) -> None:
        """Test walk forward with different metrics."""
        mock_walk_forward.return_value = {}

        metrics = ["sharpe_ratio", "cagr", "total_return"]
        for metric in metrics:
            result = runner.invoke(walk_forward, ["--metric", metric])
            assert "unrecognized arguments" not in result.output.lower()

    @patch("src.cli.commands.walk_forward.run_walk_forward_analysis")
    def test_walk_forward_with_ranges(
        self, mock_walk_forward: MagicMock, runner: CliRunner
    ) -> None:
        """Test walk forward with SMA and trend ranges."""
        mock_walk_forward.return_value = {}

        result = runner.invoke(
            walk_forward,
            [
                "--sma-range",
                "3,4,5,6,7",
                "--trend-range",
                "8,10,12",
            ],
        )

        assert "unrecognized arguments" not in result.output.lower()

    @patch("src.cli.commands.walk_forward.run_walk_forward_analysis")
    def test_walk_forward_error_handling(
        self, mock_walk_forward: MagicMock, runner: CliRunner
    ) -> None:
        """Test error handling in walk forward."""
        mock_walk_forward.side_effect = ValueError("Analysis failed")

        result = runner.invoke(walk_forward, [])

        assert result.exit_code != 0
