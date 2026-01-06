"""
Run bot command.

Starts the live trading bot.
"""

import click

from src.execution.bot_facade import create_bot
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@click.command(name="run-bot")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=str),
    default=None,
    help="Path to config file (default: config/settings.yaml)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Dry run mode (don't execute real trades)",
)
def run_bot(config: str | None, dry_run: bool) -> None:
    """
    Start the live trading bot.

    The bot will connect to Upbit API and execute trades based on the VBO strategy.

    Example:
        crypto-quant run-bot --config custom_config.yaml
    """
    if dry_run:
        logger.warning("DRY RUN MODE: No real trades will be executed")
        # TODO: Implement dry-run mode with mock exchange

    logger.info("Starting trading bot...")

    try:
        from pathlib import Path

        config_path = Path(config) if config else None
        bot = create_bot(config_path=config_path)
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}", exc_info=True)
        raise click.ClickException(f"Failed to start bot: {e}") from e
