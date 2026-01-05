"""
Unit tests for CLI run_bot command.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from src.cli.commands.run_bot import run_bot


class TestRunBotCommand:
    """Test cases for run_bot command."""

    def test_run_bot_command_help(self) -> None:
        """Test run_bot command help."""
        runner = CliRunner()
        result = runner.invoke(run_bot, ["--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output or "help" in result.output.lower()

    @patch("src.cli.commands.run_bot.create_bot", create=True)
    def test_run_bot_command_default_config(self, mock_create_bot: MagicMock) -> None:
        """Test run_bot command with default config."""
        mock_bot = MagicMock()
        mock_create_bot.return_value = mock_bot

        runner = CliRunner()

        # Use isolated filesystem to avoid actual file operations
        with runner.isolated_filesystem():
            # Mock the bot.run() to raise KeyboardInterrupt to exit cleanly
            mock_bot.run.side_effect = KeyboardInterrupt()

            runner.invoke(run_bot, [])

            # Should create bot with None config_path
            mock_create_bot.assert_called_once_with(config_path=None)
            mock_bot.run.assert_called_once()

    @patch("src.cli.commands.run_bot.create_bot", create=True)
    def test_run_bot_command_with_config(self, mock_create_bot: MagicMock) -> None:
        """Test run_bot command with custom config path."""
        mock_bot = MagicMock()
        mock_create_bot.return_value = mock_bot

        runner = CliRunner()

        with runner.isolated_filesystem():
            config_path = Path("custom_config.yaml")
            config_path.write_text("test: config")

            mock_bot.run.side_effect = KeyboardInterrupt()

            runner.invoke(run_bot, ["--config", str(config_path)])

            mock_create_bot.assert_called_once()
            call_args = mock_create_bot.call_args
            assert call_args is not None
            assert call_args.kwargs.get("config_path") == config_path
            mock_bot.run.assert_called_once()

    @patch("src.cli.commands.run_bot.create_bot", create=True)
    def test_run_bot_command_dry_run(self, mock_create_bot: MagicMock) -> None:
        """Test run_bot command with dry-run flag."""
        mock_bot = MagicMock()
        mock_create_bot.return_value = mock_bot

        runner = CliRunner()

        with runner.isolated_filesystem():
            mock_bot.run.side_effect = KeyboardInterrupt()

            result = runner.invoke(run_bot, ["--dry-run"])

            # Dry-run flag is logged but doesn't affect bot creation
            # Logger output may not appear in result.output, but command should succeed
            assert result.exit_code == 0
            mock_create_bot.assert_called_once()
            mock_bot.run.assert_called_once()

    @patch("src.cli.commands.run_bot.create_bot", create=True)
    def test_run_bot_command_keyboard_interrupt(self, mock_create_bot: MagicMock) -> None:
        """Test run_bot command handles KeyboardInterrupt gracefully."""
        mock_bot = MagicMock()
        mock_create_bot.return_value = mock_bot
        mock_bot.run.side_effect = KeyboardInterrupt()

        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(run_bot, [])

            # Should exit gracefully (KeyboardInterrupt is caught and logged)
            # Logger output may not appear in result.output, but exit_code should be 0
            assert result.exit_code == 0
            mock_create_bot.assert_called_once()
            mock_bot.run.assert_called_once()

    @patch("src.cli.commands.run_bot.create_bot", create=True)
    def test_run_bot_command_error_handling(self, mock_create_bot: MagicMock) -> None:
        """Test run_bot command handles errors."""
        mock_bot = MagicMock()
        mock_create_bot.return_value = mock_bot
        mock_bot.run.side_effect = ValueError("Test error")

        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(run_bot, [])

            # Should handle error and exit with non-zero code
            assert result.exit_code != 0
            assert "error" in result.output.lower() or "failed" in result.output.lower()

    @patch("src.cli.commands.run_bot.create_bot", create=True)
    def test_run_bot_command_config_path_conversion(self, mock_create_bot: MagicMock) -> None:
        """Test run_bot command converts string config path to Path object."""
        mock_bot = MagicMock()
        mock_create_bot.return_value = mock_bot
        mock_bot.run.side_effect = KeyboardInterrupt()

        runner = CliRunner()

        with runner.isolated_filesystem():
            config_path = Path("test_config.yaml")
            config_path.write_text("test: config")

            runner.invoke(run_bot, ["--config", str(config_path)])

            # Should convert string to Path
            mock_create_bot.assert_called_once()
            call_args = mock_create_bot.call_args
            assert call_args is not None
            passed_path = call_args.kwargs.get("config_path")
            assert passed_path is not None
            assert isinstance(passed_path, Path)
            assert passed_path == config_path

    @patch("src.cli.commands.run_bot.create_bot", create=True)
    def test_run_bot_command_none_config_path(self, mock_create_bot: MagicMock) -> None:
        """Test run_bot command with None config path (default)."""
        mock_bot = MagicMock()
        mock_create_bot.return_value = mock_bot
        mock_bot.run.side_effect = KeyboardInterrupt()

        runner = CliRunner()

        with runner.isolated_filesystem():
            runner.invoke(run_bot, [])

            # Should pass None for config_path
            mock_create_bot.assert_called_once_with(config_path=None)
