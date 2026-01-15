"""
Portfolio management data models.

Contains data structures for:
- Portfolio state tracking
- Rebalancing trades
- Portfolio constraints
- Transaction cost modeling
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal


class RebalanceReason(str, Enum):
    """Reason for triggering rebalance."""

    PERIODIC = "periodic"  # Scheduled rebalance
    DRIFT = "drift"  # Weight drift exceeded threshold
    SIGNAL = "signal"  # Alpha signal changed
    RISK = "risk"  # Risk limit breached
    MANUAL = "manual"  # Manual trigger


@dataclass
class PortfolioState:
    """Current state of portfolio."""

    # Current holdings: ticker -> quantity
    holdings: dict[str, float] = field(default_factory=dict)

    # Current prices: ticker -> price
    prices: dict[str, float] = field(default_factory=dict)

    # Available cash
    cash: float = 0.0

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def positions_value(self) -> dict[str, float]:
        """Calculate value of each position."""
        return {
            ticker: qty * self.prices.get(ticker, 0.0)
            for ticker, qty in self.holdings.items()
        }

    @property
    def total_value(self) -> float:
        """Total portfolio value including cash."""
        return self.cash + sum(self.positions_value.values())

    @property
    def current_weights(self) -> dict[str, float]:
        """Current weight of each position."""
        total = self.total_value
        if total <= 0:
            return {}
        return {
            ticker: value / total
            for ticker, value in self.positions_value.items()
        }

    @property
    def cash_weight(self) -> float:
        """Current cash weight."""
        total = self.total_value
        return self.cash / total if total > 0 else 1.0


@dataclass
class RebalanceTrade:
    """Single trade required for rebalancing."""

    ticker: str
    side: Literal["buy", "sell"]
    quantity: float
    target_value: float  # Target position value after trade
    current_value: float  # Current position value
    estimated_price: float
    estimated_cost: float  # Transaction cost (fees + slippage)
    priority: int = 0  # Higher = execute first (sells before buys)

    @property
    def trade_value(self) -> float:
        """Absolute value of the trade."""
        return self.quantity * self.estimated_price

    @property
    def net_value(self) -> float:
        """Net value after costs (negative for buys, positive for sells)."""
        if self.side == "sell":
            return self.trade_value - self.estimated_cost
        return -(self.trade_value + self.estimated_cost)


@dataclass
class RebalanceResult:
    """Result of rebalancing calculation."""

    trades: list[RebalanceTrade] = field(default_factory=list)
    reason: RebalanceReason = RebalanceReason.PERIODIC

    # Pre-rebalance state
    pre_weights: dict[str, float] = field(default_factory=dict)
    target_weights: dict[str, float] = field(default_factory=dict)

    # Metrics
    total_turnover: float = 0.0  # Sum of |trade_value| / portfolio_value
    total_cost: float = 0.0  # Total transaction cost
    max_drift: float = 0.0  # Maximum weight deviation

    # Tracking
    timestamp: datetime = field(default_factory=datetime.now)
    executed: bool = False

    @property
    def num_trades(self) -> int:
        """Number of trades required."""
        return len(self.trades)

    @property
    def buy_trades(self) -> list[RebalanceTrade]:
        """List of buy trades."""
        return [t for t in self.trades if t.side == "buy"]

    @property
    def sell_trades(self) -> list[RebalanceTrade]:
        """List of sell trades."""
        return [t for t in self.trades if t.side == "sell"]


@dataclass
class PortfolioConstraints:
    """Constraints for portfolio construction and rebalancing."""

    # Position limits
    max_position_weight: float = 0.20  # Maximum single position (20%)
    min_position_weight: float = 0.01  # Minimum position to hold (1%)

    # Sector/group limits
    max_sector_weight: float = 0.40  # Maximum sector exposure
    max_correlation_group_weight: float = 0.50  # Maximum correlated group

    # Diversification
    min_holdings: int = 5  # Minimum number of holdings
    max_holdings: int = 30  # Maximum number of holdings

    # Turnover limits
    max_single_trade_pct: float = 0.10  # Max single trade as % of portfolio
    max_daily_turnover: float = 0.25  # Max daily turnover (25%)
    max_annual_turnover: float = 4.0  # Max annual turnover (400%)

    # Cash limits
    min_cash_pct: float = 0.02  # Minimum cash buffer (2%)
    max_cash_pct: float = 0.20  # Maximum cash (20%)

    # Trading constraints
    min_trade_value: float = 10_000  # Minimum trade size (KRW)
    long_only: bool = True  # Long only portfolio

    def validate_weights(self, weights: dict[str, float]) -> list[str]:
        """
        Validate weights against constraints.

        Returns list of violation messages.
        """
        violations = []

        # Check position limits
        for ticker, weight in weights.items():
            if weight > self.max_position_weight:
                violations.append(
                    f"{ticker}: weight {weight:.1%} exceeds max {self.max_position_weight:.1%}"
                )
            if 0 < weight < self.min_position_weight:
                violations.append(
                    f"{ticker}: weight {weight:.1%} below min {self.min_position_weight:.1%}"
                )
            if self.long_only and weight < 0:
                violations.append(f"{ticker}: negative weight not allowed (long only)")

        # Check diversification
        num_holdings = sum(1 for w in weights.values() if w > 0)
        if num_holdings < self.min_holdings:
            violations.append(
                f"Holdings {num_holdings} below minimum {self.min_holdings}"
            )
        if num_holdings > self.max_holdings:
            violations.append(
                f"Holdings {num_holdings} exceeds maximum {self.max_holdings}"
            )

        # Check sum
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            violations.append(f"Weights sum to {total_weight:.2%}, expected 100%")

        return violations


@dataclass
class TransactionCostModel:
    """Model for estimating transaction costs."""

    # Fixed costs
    fee_rate: float = 0.0005  # 0.05% trading fee
    min_fee: float = 0.0  # Minimum fee per trade

    # Market impact (slippage)
    base_slippage: float = 0.0005  # 0.05% base slippage
    impact_coefficient: float = 0.1  # Price impact coefficient

    # Volume-based impact
    daily_volume_impact: bool = True  # Consider ADV for impact
    max_participation_rate: float = 0.10  # Max 10% of ADV per trade

    def estimate_cost(
        self,
        trade_value: float,
        avg_daily_volume: float | None = None,
        volatility: float | None = None,
    ) -> float:
        """
        Estimate total transaction cost for a trade.

        Args:
            trade_value: Absolute value of trade
            avg_daily_volume: Average daily volume in value terms
            volatility: Asset volatility (for impact estimation)

        Returns:
            Estimated total cost (fees + slippage + impact)
        """
        # Fixed fee
        fee = max(trade_value * self.fee_rate, self.min_fee)

        # Base slippage
        slippage = trade_value * self.base_slippage

        # Market impact (Almgren-Chriss style)
        impact = 0.0
        if avg_daily_volume and avg_daily_volume > 0:
            participation = trade_value / avg_daily_volume
            # Impact increases with square root of participation
            impact = trade_value * self.impact_coefficient * (participation ** 0.5)

            # Additional volatility-based impact
            if volatility:
                impact *= (1 + volatility)

        return fee + slippage + impact

    def max_trade_size(
        self,
        avg_daily_volume: float,
    ) -> float:
        """
        Calculate maximum recommended trade size.

        Based on participation rate constraint.
        """
        return avg_daily_volume * self.max_participation_rate


__all__ = [
    "RebalanceReason",
    "PortfolioState",
    "RebalanceTrade",
    "RebalanceResult",
    "PortfolioConstraints",
    "TransactionCostModel",
]
