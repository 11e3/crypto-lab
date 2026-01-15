"""
Portfolio-level backtesting engine.

Provides comprehensive portfolio simulation with:
- Multi-asset allocation
- Rebalancing
- Factor-based signal integration
- Transaction cost modeling
- Risk management integration
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Literal

import numpy as np
import pandas as pd

from src.backtester.models import BacktestConfig, BacktestResult, Trade
from src.portfolio.models import PortfolioConstraints, TransactionCostModel
from src.portfolio.rebalancing import RebalancingEngine, RebalancingConfig
from src.risk.drawdown_control import DrawdownController
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioBacktestConfig:
    """Configuration for portfolio backtesting."""

    # Capital
    initial_capital: float = 10_000_000

    # Costs
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005

    # Rebalancing
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    rebalance_threshold: float = 0.05  # 5% drift threshold

    # Constraints
    max_position_weight: float = 0.20
    min_position_weight: float = 0.02
    max_holdings: int = 30
    min_holdings: int = 5

    # Risk management
    max_drawdown_limit: float = 0.20
    enable_drawdown_control: bool = True

    # Benchmark
    benchmark_ticker: str | None = None


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time."""

    timestamp: datetime
    holdings: dict[str, float]  # ticker -> quantity
    weights: dict[str, float]  # ticker -> weight
    cash: float
    total_value: float
    daily_return: float = 0.0
    cumulative_return: float = 0.0


@dataclass
class PortfolioBacktestResult:
    """Result of portfolio backtest."""

    # Performance
    total_return: float = 0.0
    cagr: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0

    # Benchmark comparison
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0

    # Trading statistics
    total_trades: int = 0
    turnover: float = 0.0  # Annual turnover
    total_costs: float = 0.0
    num_rebalances: int = 0

    # Time series
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    weights_history: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    trades: list[Trade] = field(default_factory=list)
    snapshots: list[PortfolioSnapshot] = field(default_factory=list)

    # Configuration
    config: PortfolioBacktestConfig | None = None

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"\n{'=' * 50}\n"
            f"Portfolio Backtest Results\n"
            f"{'=' * 50}\n"
            f"Total Return: {self.total_return:.2%}\n"
            f"CAGR: {self.cagr:.2%}\n"
            f"Volatility: {self.volatility:.2%}\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Sortino Ratio: {self.sortino_ratio:.2f}\n"
            f"Max Drawdown: {self.max_drawdown:.2%}\n"
            f"Calmar Ratio: {self.calmar_ratio:.2f}\n"
            f"{'=' * 50}\n"
            f"Alpha: {self.alpha:.2%}\n"
            f"Beta: {self.beta:.2f}\n"
            f"Information Ratio: {self.information_ratio:.2f}\n"
            f"{'=' * 50}\n"
            f"Total Trades: {self.total_trades}\n"
            f"Annual Turnover: {self.turnover:.2%}\n"
            f"Total Costs: {self.total_costs:,.0f}\n"
            f"Rebalances: {self.num_rebalances}\n"
            f"{'=' * 50}"
        )


class PortfolioBacktester:
    """
    Portfolio-level backtesting engine.

    Simulates portfolio management over time with:
    - Dynamic weight allocation from signals
    - Periodic or threshold-based rebalancing
    - Transaction cost modeling
    - Risk management (drawdown control)

    Example:
        >>> backtester = PortfolioBacktester(
        ...     config=PortfolioBacktestConfig(
        ...         initial_capital=10_000_000,
        ...         rebalance_frequency="monthly",
        ...     )
        ... )
        >>> result = backtester.run(
        ...     prices=price_data,
        ...     signal_generator=my_signal_function,
        ... )
        >>> print(result.summary())
    """

    def __init__(
        self,
        config: PortfolioBacktestConfig | None = None,
    ) -> None:
        """
        Initialize portfolio backtester.

        Args:
            config: Backtest configuration
        """
        self.config = config or PortfolioBacktestConfig()

        # Initialize components
        self.rebalancing_engine = RebalancingEngine(
            config=RebalancingConfig(
                method="hybrid",
                frequency=self.config.rebalance_frequency,
                drift_threshold=self.config.rebalance_threshold,
            ),
            constraints=PortfolioConstraints(
                max_position_weight=self.config.max_position_weight,
                min_position_weight=self.config.min_position_weight,
                max_holdings=self.config.max_holdings,
                min_holdings=self.config.min_holdings,
            ),
            cost_model=TransactionCostModel(
                fee_rate=self.config.fee_rate,
                base_slippage=self.config.slippage_rate,
            ),
        )

        if self.config.enable_drawdown_control:
            self.drawdown_controller = DrawdownController(
                max_drawdown_limit=self.config.max_drawdown_limit,
            )
        else:
            self.drawdown_controller = None

    def run(
        self,
        prices: pd.DataFrame,
        signal_generator: Callable[[pd.DataFrame, datetime], dict[str, float]] | None = None,
        target_weights: pd.DataFrame | None = None,
        benchmark_returns: pd.Series | None = None,
    ) -> PortfolioBacktestResult:
        """
        Run portfolio backtest.

        Args:
            prices: Price data (columns=tickers, index=dates)
            signal_generator: Function(prices, date) -> weights dict
            target_weights: Pre-computed target weights DataFrame
            benchmark_returns: Benchmark return series for comparison

        Returns:
            PortfolioBacktestResult
        """
        # Validate inputs
        if signal_generator is None and target_weights is None:
            raise ValueError("Must provide either signal_generator or target_weights")

        prices = prices.sort_index()
        dates = prices.index.tolist()

        # Initialize state
        cash = self.config.initial_capital
        holdings: dict[str, float] = {}  # ticker -> quantity
        equity_history = []
        returns_history = []
        weights_history = []
        snapshots = []
        trades = []
        total_costs = 0.0
        num_rebalances = 0
        prev_value = self.config.initial_capital

        for i, date in enumerate(dates):
            # Skip first few days for indicator warmup
            if i < 20:
                equity_history.append(self.config.initial_capital)
                returns_history.append(0.0)
                continue

            current_prices = prices.loc[date]

            # Calculate current portfolio value
            positions_value = sum(
                holdings.get(ticker, 0) * current_prices.get(ticker, 0)
                for ticker in holdings
            )
            portfolio_value = cash + positions_value

            # Calculate daily return
            daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0

            # Drawdown control
            exposure_multiplier = 1.0
            if self.drawdown_controller:
                dd_state = self.drawdown_controller.update(portfolio_value, date)
                exposure_multiplier = self.drawdown_controller.get_target_exposure(
                    dd_state.current_drawdown
                )

            # Generate target weights
            if target_weights is not None:
                if date in target_weights.index:
                    raw_weights = target_weights.loc[date].to_dict()
                else:
                    raw_weights = {}
            else:
                # Use historical data up to current date
                historical_prices = prices.loc[:date]
                raw_weights = signal_generator(historical_prices, date)

            # Apply exposure multiplier from drawdown control
            target_wts = {
                ticker: weight * exposure_multiplier
                for ticker, weight in raw_weights.items()
            }

            # Check if rebalancing is needed
            current_weights = self._calculate_weights(holdings, current_prices, portfolio_value)

            from src.portfolio.models import PortfolioState
            portfolio_state = PortfolioState(
                holdings=holdings,
                prices=current_prices.to_dict(),
                cash=cash,
            )

            should_rebal, reason = self.rebalancing_engine.should_rebalance(
                portfolio_state, target_wts, date
            )

            if should_rebal and target_wts:
                # Calculate rebalance trades
                rebal_result = self.rebalancing_engine.calculate_rebalance(
                    portfolio_state, target_wts
                )

                # Execute trades
                for trade in rebal_result.trades:
                    trade_value = trade.quantity * trade.estimated_price
                    trade_cost = trade_value * (self.config.fee_rate + self.config.slippage_rate)

                    if trade.side == "buy":
                        if cash >= trade_value + trade_cost:
                            cash -= trade_value + trade_cost
                            holdings[trade.ticker] = holdings.get(trade.ticker, 0) + trade.quantity
                            total_costs += trade_cost

                            trades.append(Trade(
                                ticker=trade.ticker,
                                entry_date=date.date() if hasattr(date, 'date') else date,
                                entry_price=trade.estimated_price,
                                amount=trade_value,
                                commission_cost=trade_cost / 2,
                                slippage_cost=trade_cost / 2,
                            ))

                    else:  # sell
                        if holdings.get(trade.ticker, 0) >= trade.quantity:
                            cash += trade_value - trade_cost
                            holdings[trade.ticker] = holdings.get(trade.ticker, 0) - trade.quantity
                            total_costs += trade_cost

                            # Remove empty positions
                            if holdings[trade.ticker] <= 0:
                                del holdings[trade.ticker]

                num_rebalances += 1
                self.rebalancing_engine._last_rebalance_date = date

            # Record state
            equity_history.append(portfolio_value)
            returns_history.append(daily_return)
            weights_history.append(current_weights)

            snapshot = PortfolioSnapshot(
                timestamp=date,
                holdings=holdings.copy(),
                weights=current_weights,
                cash=cash,
                total_value=portfolio_value,
                daily_return=daily_return,
                cumulative_return=(portfolio_value / self.config.initial_capital) - 1,
            )
            snapshots.append(snapshot)

            prev_value = portfolio_value

        # Calculate performance metrics
        equity_series = pd.Series(equity_history, index=dates)
        returns_series = pd.Series(returns_history, index=dates)

        result = self._calculate_metrics(
            equity_series, returns_series, benchmark_returns
        )

        # Add additional info
        result.total_trades = len(trades)
        result.total_costs = total_costs
        result.num_rebalances = num_rebalances
        result.trades = trades
        result.snapshots = snapshots
        result.equity_curve = equity_series
        result.returns = returns_series
        result.weights_history = pd.DataFrame(weights_history, index=dates)
        result.config = self.config

        # Calculate turnover
        if len(dates) > 252:
            years = len(dates) / 252
            result.turnover = (total_costs / self.config.initial_capital) / years / (
                self.config.fee_rate + self.config.slippage_rate
            )

        return result

    def _calculate_weights(
        self,
        holdings: dict[str, float],
        prices: pd.Series,
        portfolio_value: float,
    ) -> dict[str, float]:
        """Calculate current portfolio weights."""
        if portfolio_value <= 0:
            return {}

        weights = {}
        for ticker, qty in holdings.items():
            if ticker in prices and qty > 0:
                value = qty * prices[ticker]
                weights[ticker] = value / portfolio_value

        return weights

    def _calculate_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        benchmark_returns: pd.Series | None,
    ) -> PortfolioBacktestResult:
        """Calculate performance metrics."""
        result = PortfolioBacktestResult()

        # Clean returns
        returns = returns.replace([np.inf, -np.inf], 0).fillna(0)

        # Basic metrics
        result.total_return = (equity.iloc[-1] / equity.iloc[0]) - 1 if len(equity) > 0 else 0

        # CAGR
        years = len(equity) / 252
        if years > 0 and result.total_return > -1:
            result.cagr = (1 + result.total_return) ** (1 / years) - 1
        else:
            result.cagr = 0

        # Volatility
        result.volatility = returns.std() * np.sqrt(252)

        # Sharpe
        if result.volatility > 0:
            result.sharpe_ratio = (returns.mean() * 252) / result.volatility
        else:
            result.sharpe_ratio = 0

        # Sortino
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(252)
            if downside_vol > 0:
                result.sortino_ratio = (returns.mean() * 252) / downside_vol

        # Max Drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        result.max_drawdown = drawdowns.min()

        # Calmar
        if result.max_drawdown < 0:
            result.calmar_ratio = result.cagr / abs(result.max_drawdown)

        # Benchmark comparison
        if benchmark_returns is not None:
            aligned = pd.DataFrame({
                "portfolio": returns,
                "benchmark": benchmark_returns,
            }).dropna()

            if len(aligned) > 10:
                # Beta
                cov = aligned["portfolio"].cov(aligned["benchmark"])
                var = aligned["benchmark"].var()
                result.beta = cov / var if var > 0 else 0

                # Alpha
                result.alpha = (
                    aligned["portfolio"].mean() - result.beta * aligned["benchmark"].mean()
                ) * 252

                # Tracking error
                tracking_diff = aligned["portfolio"] - aligned["benchmark"]
                result.tracking_error = tracking_diff.std() * np.sqrt(252)

                # Information ratio
                if result.tracking_error > 0:
                    result.information_ratio = (
                        tracking_diff.mean() * 252 / result.tracking_error
                    )

        return result


__all__ = [
    "PortfolioBacktester",
    "PortfolioBacktestConfig",
    "PortfolioBacktestResult",
    "PortfolioSnapshot",
]
