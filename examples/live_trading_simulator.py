#!/usr/bin/env python
"""
Live Trading Simulator - Paper Trading Mode

This script demonstrates live trading in PAPER MODE (simulated trading without real capital).
Perfect for testing strategies before going live.

Features:
1. Real-time price simulation (or market data)
2. Order execution simulation
3. Portfolio tracking
4. Performance monitoring
5. Risk management validation

Usage:
    python examples/live_trading_simulator.py

Safety:
    - Uses SIMULATED prices, not real market data
    - No real API connections
    - No risk of losing money
    - Perfect for strategy validation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open trading position."""

    ticker: str
    side: str  # "buy" or "sell"
    quantity: float
    entry_price: float
    entry_time: datetime
    take_profit: float | None = None
    stop_loss: float | None = None

    @property
    def current_value(self, current_price: float) -> float:
        """Calculate current position value."""
        return self.quantity * current_price

    def pnl(self, current_price: float) -> float:
        """Calculate P&L for this position."""
        if self.side == "buy":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def pnl_percent(self, current_price: float) -> float:
        """Calculate P&L percentage."""
        return (self.pnl(current_price) / (self.entry_price * self.quantity)) * 100


@dataclass
class Trade:
    """Represents a completed trade."""

    ticker: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float

    @property
    def duration(self) -> timedelta:
        """Duration of the trade."""
        return self.exit_time - self.entry_time


@dataclass
class Portfolio:
    """Represents trading portfolio."""

    initial_capital: float
    current_cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    balance_history: list[tuple[datetime, float]] = field(default_factory=list)

    def __post_init__(self):
        self.current_cash = self.initial_capital

    def total_equity(self, prices: dict[str, float]) -> float:
        """Calculate total equity (cash + positions)."""
        position_value = sum(
            pos.quantity * prices.get(pos.ticker, 0) for pos in self.positions.values()
        )
        return self.current_cash + position_value

    def open_position(
        self,
        ticker: str,
        quantity: float,
        price: float,
        side: str = "buy",
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> bool:
        """Open a new position."""
        cost = quantity * price

        if cost > self.current_cash:
            logger.warning(
                f"Insufficient cash for {ticker}: need {cost:.0f}, have {self.current_cash:.0f}"
            )
            return False

        self.current_cash -= cost
        self.positions[ticker] = Position(
            ticker=ticker,
            side=side,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        logger.info(f"[OPEN] {side.upper()} {quantity:.4f} {ticker} @ {price:.2f}")
        return True

    def close_position(self, ticker: str, price: float) -> Trade | None:
        """Close an open position."""
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        proceeds = pos.quantity * price
        pnl = pos.pnl(price)
        pnl_percent = pos.pnl_percent(price)

        # Update cash
        self.current_cash += proceeds

        # Record trade
        trade = Trade(
            ticker=ticker,
            side=pos.side,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            exit_price=price,
            entry_time=pos.entry_time,
            exit_time=datetime.now(),
            pnl=pnl,
            pnl_percent=pnl_percent,
        )

        self.trades.append(trade)
        del self.positions[ticker]

        status = "[PROFIT]" if pnl > 0 else "[LOSS]"
        logger.info(
            f"[CLOSE] {ticker} @ {price:.2f} | P&L: {pnl:.0f} ({pnl_percent:+.1f}%) {status}"
        )
        return trade


class PriceSimulator:
    """Simulates price movements for backtesting."""

    def __init__(self, initial_prices: dict[str, float], volatility: float = 0.02):
        """
        Initialize price simulator.

        Args:
            initial_prices: Starting prices per ticker
            volatility: Daily volatility (default 2%)
        """
        self.current_prices = initial_prices.copy()
        self.volatility = volatility
        self.price_history = {ticker: [price] for ticker, price in initial_prices.items()}

    def update_prices(self, steps: int = 1) -> dict[str, float]:
        """Simulate price changes and return updated prices."""
        for _ in range(steps):
            for ticker in self.current_prices:
                # Random walk with drift
                drift = 0.0001  # Slight upward bias
                shock = np.random.normal(drift, self.volatility)
                new_price = self.current_prices[ticker] * (1 + shock)
                self.current_prices[ticker] = max(new_price, self.current_prices[ticker] * 0.5)
                self.price_history[ticker].append(self.current_prices[ticker])

        return self.current_prices.copy()

    def get_trend(self, ticker: str, period: int = 5) -> float:
        """Get price trend (slope)."""
        prices = self.price_history[ticker][-period:]
        if len(prices) < 2:
            return 0.0
        return (prices[-1] - prices[0]) / prices[0]


class SimpleTradingStrategy:
    """Simple momentum-based trading strategy for demonstration."""

    def __init__(self, momentum_period: int = 5, trend_threshold: float = 0.01):
        """
        Initialize strategy.

        Args:
            momentum_period: Period for momentum calculation
            trend_threshold: Threshold for trend detection
        """
        self.momentum_period = momentum_period
        self.trend_threshold = trend_threshold

    def generate_signal(self, simulator: PriceSimulator, ticker: str) -> str | None:
        """
        Generate trading signal.

        Returns:
            "buy", "sell", or None
        """
        trend = simulator.get_trend(ticker, self.momentum_period)

        if trend > self.trend_threshold:
            return "buy"
        elif trend < -self.trend_threshold:
            return "sell"

        return None


class SimulatedBroker:
    """Simulates a trading broker with order execution."""

    def __init__(
        self, portfolio: Portfolio, price_simulator: PriceSimulator, strategy: SimpleTradingStrategy
    ):
        """Initialize broker."""
        self.portfolio = portfolio
        self.simulator = price_simulator
        self.strategy = strategy
        self.step = 0

    async def run_trading_loop(
        self, tickers: list[str], duration_minutes: int = 60, check_interval: int = 1
    ) -> None:
        """
        Run simulated trading loop.

        Args:
            tickers: Tickers to trade
            duration_minutes: Total simulation duration
            check_interval: Check signals every N iterations
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        logger.info("=" * 80)
        logger.info("SIMULATED TRADING SESSION")
        logger.info("=" * 80)
        logger.info(f"Starting capital: {self.portfolio.initial_capital:,.0f}")
        logger.info(f"Tickers: {', '.join(tickers)}")
        logger.info(f"Duration: {duration_minutes} minutes (simulated)")
        logger.info("=" * 80)
        logger.info("")

        iteration = 0
        while datetime.now() < end_time:
            iteration += 1
            self.step += 1

            # Update prices
            prices = self.simulator.update_prices()

            # Check for trading signals every N iterations
            if iteration % check_interval == 0:
                for ticker in tickers:
                    signal = self.strategy.generate_signal(self.simulator, ticker)
                    price = prices[ticker]

                    if signal == "buy" and ticker not in self.portfolio.positions:
                        # Calculate position size (10% of available capital per trade)
                        position_size = (self.portfolio.current_cash * 0.1) / price
                        self.portfolio.open_position(ticker, position_size, price)

                    elif signal == "sell" and ticker in self.portfolio.positions:
                        self.portfolio.close_position(ticker, price)

                    elif ticker in self.portfolio.positions:
                        # Check stop loss and take profit
                        pos = self.portfolio.positions[ticker]

                        if pos.stop_loss and price <= pos.stop_loss:
                            self.portfolio.close_position(ticker, price)
                            logger.info(f"[STOP] {ticker} hit stop loss @ {price:.2f}")

                        elif pos.take_profit and price >= pos.take_profit:
                            self.portfolio.close_position(ticker, price)
                            logger.info(f"[PROFIT] {ticker} hit take profit @ {price:.2f}")

            # Record portfolio state
            if iteration % 10 == 0:
                equity = self.portfolio.total_equity(prices)
                self.portfolio.balance_history.append((datetime.now(), equity))

            # Simulate time passing (in real app, this would be actual market time)
            await asyncio.sleep(0.01)

        logger.info("")
        logger.info("=" * 80)
        logger.info("TRADING SESSION COMPLETE")
        logger.info("=" * 80)

    def print_summary(self) -> None:
        """Print trading summary."""
        if not self.portfolio.balance_history:
            return

        final_equity = self.portfolio.balance_history[-1][1]
        total_return = (
            final_equity - self.portfolio.initial_capital
        ) / self.portfolio.initial_capital

        print("\n" + "=" * 80)
        print("TRADING SUMMARY")
        print("=" * 80)
        print("\nCapital:")
        print(f"  Initial:     {self.portfolio.initial_capital:>15,.0f}")
        print(f"  Final:       {final_equity:>15,.0f}")
        print(f"  P&L:         {final_equity - self.portfolio.initial_capital:>15,.0f}")
        print(f"  Return:      {total_return:>15.1%}")

        print("\nTrades:")
        print(f"  Total:       {len(self.portfolio.trades):>15}")
        if self.portfolio.trades:
            winning = sum(1 for t in self.portfolio.trades if t.pnl > 0)
            win_rate = winning / len(self.portfolio.trades)
            avg_pnl = np.mean([t.pnl for t in self.portfolio.trades])
            avg_pnl_percent = np.mean([t.pnl_percent for t in self.portfolio.trades])

            print(f"  Won:         {winning:>15}")
            print(f"  Lost:        {len(self.portfolio.trades) - winning:>15}")
            print(f"  Win Rate:    {win_rate:>15.1%}")
            print(f"  Avg P&L:     {avg_pnl:>15,.0f}")
            print(f"  Avg % Return: {avg_pnl_percent:>14.1%}")

        print(f"\nOpen Positions: {len(self.portfolio.positions)}")
        for ticker, pos in self.portfolio.positions.items():
            print(f"  {ticker}: {pos.quantity:.4f} @ {pos.entry_price:.2f}")

        print("\n" + "=" * 80)
        print("SAMPLE TRADES")
        print("=" * 80)
        for i, trade in enumerate(self.portfolio.trades[:10], 1):
            print(f"\n[{i}] {trade.ticker}")
            print(f"    Side:     {trade.side.upper()}")
            print(f"    Entry:    {trade.entry_price:.2f}")
            print(f"    Exit:     {trade.exit_price:.2f}")
            print(f"    Quantity: {trade.quantity:.4f}")
            print(f"    P&L:      {trade.pnl:,.0f} ({trade.pnl_percent:+.1f}%)")
            print(f"    Duration: {trade.duration.total_seconds():.0f}s")


async def main():
    """Run simulated live trading session."""
    # Configuration
    initial_capital = 1_000_000  # 1M KRW
    tickers = ["BTC", "ETH", "XRP"]
    initial_prices = {"BTC": 50000, "ETH": 3000, "XRP": 0.8}
    volatility = 0.02  # 2% daily volatility
    duration_minutes = 5  # Simulated time

    # Initialize components
    portfolio = Portfolio(initial_capital=initial_capital)
    price_simulator = PriceSimulator(initial_prices, volatility=volatility)
    strategy = SimpleTradingStrategy(momentum_period=5, trend_threshold=0.01)
    broker = SimulatedBroker(portfolio, price_simulator, strategy)

    # Run trading session
    await broker.run_trading_loop(tickers, duration_minutes=duration_minutes, check_interval=1)

    # Print summary
    broker.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
