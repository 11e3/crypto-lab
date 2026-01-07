"""
Tests for Compare and Optimize CLI commands.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli.commands.compare import compare
from src.cli.commands.optimize import optimize


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


class TestCompareCommand:
    """Test Strategy Comparison CLI command."""

    def test_compare_help(self, runner: CliRunner) -> None:
        """Test help message."""
        result = runner.invoke(compare, ["--help"])
        assert result.exit_code == 0
        assert "compare" in result.output.lower()

    @patch("src.cli.commands.compare.compare_strategies")
    def test_compare_with_defaults(self, mock_compare: MagicMock, runner: CliRunner) -> None:
        """Test compare with default parameters."""
        mock_compare.return_value = {}

        result = runner.invoke(compare, [])

        assert result.exit_code == 0
        mock_compare.assert_called_once()

    @patch("src.cli.commands.compare.compare_strategies")
    def test_compare_with_custom_tickers(self, mock_compare: MagicMock, runner: CliRunner) -> None:
        """Test compare with custom tickers."""
        mock_compare.return_value = {}

        result = runner.invoke(
            compare,
            [
                "--tickers",
                "KRW-BTC",
                "--tickers",
                "KRW-ETH",
            ],
        )

        assert result.exit_code == 0

    @patch("src.cli.commands.compare.compare_strategies")
    def test_compare_with_custom_params(self, mock_compare: MagicMock, runner: CliRunner) -> None:
        """Test compare with custom parameters."""
        mock_compare.return_value = {}

        result = runner.invoke(
            compare,
            [
                "--initial-capital",
                "50000",
                "--fee-rate",
                "0.001",
                "--max-slots",
                "8",
                "--interval",
                "day",
            ],
        )

        assert result.exit_code == 0

    @patch("src.cli.commands.compare.compare_strategies")
    def test_compare_with_strategies(self, mock_compare: MagicMock, runner: CliRunner) -> None:
        """Test compare with multiple strategies."""
        mock_compare.return_value = {}

        result = runner.invoke(
            compare,
            [
                "--strategies",
                "vanilla",
                "--strategies",
                "minimal",
                "--strategies",
                "legacy",
            ],
        )

        assert result.exit_code == 0

    @patch("src.cli.commands.compare.compare_strategies")
    def test_compare_with_output(
        self, mock_compare: MagicMock, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test compare with output file."""
        mock_compare.return_value = {}

        output_file = tmp_path / "compare_report.json"

        result = runner.invoke(
            compare,
            [
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0

    @patch("src.cli.commands.compare.compare_strategies")
    def test_compare_with_workers(self, mock_compare: MagicMock, runner: CliRunner) -> None:
        """Test compare with custom worker count."""
        mock_compare.return_value = {}

        result = runner.invoke(
            compare,
            [
                "--workers",
                "4",
            ],
        )

        assert result.exit_code == 0

    @patch("src.cli.commands.compare.compare_strategies")
    def test_compare_error_handling(self, mock_compare: MagicMock, runner: CliRunner) -> None:
        """Test error handling in compare."""
        mock_compare.side_effect = ValueError("Comparison failed")

        result = runner.invoke(compare, [])

        assert result.exit_code != 0

    @patch("src.cli.commands.compare.compare_strategies")
    def test_compare_with_all_options(
        self, mock_compare: MagicMock, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test compare with all options combined."""
        mock_compare.return_value = {}

        output_file = tmp_path / "compare_report.json"

        result = runner.invoke(
            compare,
            [
                "--tickers",
                "KRW-BTC",
                "--tickers",
                "KRW-ETH",
                "--tickers",
                "KRW-XRP",
                "--interval",
                "day",
                "--initial-capital",
                "100000",
                "--fee-rate",
                "0.001",
                "--max-slots",
                "5",
                "--strategies",
                "vanilla",
                "--strategies",
                "minimal",
                "--workers",
                "4",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0


class TestOptimizeCommand:
    """Test Parameter Optimization CLI command."""

    def test_optimize_help(self, runner: CliRunner) -> None:
        """Test help message."""
        result = runner.invoke(optimize, ["--help"])
        assert result.exit_code == 0
        assert "optimize" in result.output.lower()
