"""
Main CLI entry point.

Provides command-line interface for the trading system.
"""

import sys
from pathlib import Path

import click

from src.cli.commands.backtest import backtest
from src.cli.commands.collect import collect
from src.cli.commands.compare import compare
from src.cli.commands.monte_carlo import monte_carlo
from src.cli.commands.optimize import optimize
from src.cli.commands.run_bot import run_bot
from src.cli.commands.walk_forward import walk_forward

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@click.group()
@click.version_option(version="0.1.0", prog_name="crypto-quant")
def cli() -> None:
    """
    Crypto Quant System - Automated trading system using volatility breakout strategy.

    Supports multiple cryptocurrency exchanges (Upbit, etc.).
    Provides commands for data collection, backtesting, and live trading.
    """
    pass


# Register commands
cli.add_command(collect)
cli.add_command(backtest)
cli.add_command(compare)
cli.add_command(optimize)
cli.add_command(monte_carlo)
cli.add_command(walk_forward)
cli.add_command(run_bot)


def main() -> None:
    """Main entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
