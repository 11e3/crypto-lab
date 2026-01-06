"""
Live Trading Example

⚠️  WARNING: This example uses REAL MONEY!
Test thoroughly in backtesting before using live trading.

This example demonstrates how to set up and run a live trading bot.
It shows configuration, risk management, and monitoring setup.
"""

import os
from pathlib import Path

from src.execution.bot_facade import TradingBotFacade
from src.utils.logger import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


def check_environment() -> bool:
    """Check if required environment variables are set."""
    required_vars = ["UPBIT_ACCESS_KEY", "UPBIT_SECRET_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   - {var}")
        print("\nPlease set these variables before running live trading.")
        print("\nExample (Linux/Mac):")
        print("  export UPBIT_ACCESS_KEY='your-access-key'")
        print("  export UPBIT_SECRET_KEY='your-secret-key'")
        print("\nExample (Windows PowerShell):")
        print("  $env:UPBIT_ACCESS_KEY='your-access-key'")
        print("  $env:UPBIT_SECRET_KEY='your-secret-key'")
        return False

    return True


def main() -> None:
    """Run live trading example."""
    print("=" * 60)
    print("Live Trading Example")
    print("=" * 60)
    print()
    print("⚠️  WARNING: This will use REAL MONEY!")
    print("⚠️  Make sure you have:")
    print("   1. Tested your strategy thoroughly in backtesting")
    print("   2. Set appropriate position sizes")
    print("   3. Configured risk management settings")
    print("   4. Set up monitoring and alerts")
    print()

    # Safety check
    response = input("Do you want to continue? (type 'YES' to confirm): ")
    if response != "YES":
        print("Aborted. Safety first!")
        return

    # Check environment
    print("\nStep 1: Checking environment...")
    if not check_environment():
        return
    print("✓ Environment variables set")
    print()

    # Check configuration file
    print("Step 2: Loading configuration...")
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("   Please create it from config/settings.yaml.example")
        return
    print(f"✓ Configuration file found: {config_path}")
    print()

    # Create bot
    print("Step 3: Creating trading bot...")
    try:
        bot = TradingBotFacade(config_path=config_path)
        print("✓ Bot created successfully")
        print()

        # Display configuration summary
        print("Step 4: Configuration Summary")
        print("-" * 60)
        config = bot._config  # Access internal config for display
        print(f"Tickers:          {', '.join(config.trading.tickers)}")
        print(f"Max Positions:    {config.trading.max_slots}")
        print(f"Initial Capital:  {config.trading.initial_capital:,.0f} KRW")
        print(f"Strategy:         {config.strategy.name}")
        print()

        # Final confirmation
        print("=" * 60)
        print("Ready to start live trading")
        print("=" * 60)
        print("\nThe bot will:")
        print("  - Monitor market data in real-time")
        print("  - Execute trades based on strategy signals")
        print("  - Manage positions and risk")
        print("  - Send notifications (if configured)")
        print("\nPress Ctrl+C to stop the bot")
        print()

        # Start bot
        print("Starting bot...")
        bot.run()

    except KeyboardInterrupt:
        print("\n\nBot stopped by user")
    except Exception as e:
        logger.exception("Error running bot")
        print(f"\n❌ Error: {e}")
        print("Please check your configuration and try again")


if __name__ == "__main__":
    main()
