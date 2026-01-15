"""
Portfolio rebalancing engine.

Provides multiple rebalancing strategies:
- Periodic: Calendar-based rebalancing (daily, weekly, monthly, quarterly)
- Threshold: Drift-based rebalancing when weights deviate
- Hybrid: Combination of periodic and threshold
- Tax-aware: Minimize tax impact (for taxable accounts)

Features:
- Transaction cost optimization
- Trade batching and prioritization
- Constraint enforcement
- Turnover control
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal

import numpy as np

from src.portfolio.models import (
    PortfolioConstraints,
    PortfolioState,
    RebalanceReason,
    RebalanceResult,
    RebalanceTrade,
    TransactionCostModel,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

RebalanceMethod = Literal["periodic", "threshold", "hybrid", "tax_aware"]
RebalanceFrequency = Literal["daily", "weekly", "monthly", "quarterly", "annual"]


class DriftType(str, Enum):
    """Type of drift calculation."""

    ABSOLUTE = "absolute"  # |current - target|
    RELATIVE = "relative"  # |current - target| / target


@dataclass
class RebalancingConfig:
    """Configuration for rebalancing engine."""

    # Rebalancing method
    method: RebalanceMethod = "threshold"

    # Periodic settings
    frequency: RebalanceFrequency = "monthly"

    # Threshold settings
    drift_threshold: float = 0.05  # 5% absolute drift triggers rebalance
    drift_type: DriftType = DriftType.ABSOLUTE
    min_rebalance_interval_days: int = 5  # Minimum days between rebalances

    # Cost optimization
    min_trade_improvement: float = 0.001  # Min improvement to justify trade cost
    batch_small_trades: bool = True  # Combine small trades
    small_trade_threshold: float = 0.02  # Trades < 2% considered small

    # Turnover control
    max_turnover_per_rebalance: float = 0.30  # Max 30% turnover per rebalance
    prioritize_sells: bool = True  # Execute sells before buys

    # Cash management
    target_cash_weight: float = 0.05  # Target cash buffer


class RebalancingEngine:
    """
    Portfolio rebalancing engine.

    Determines when and how to rebalance portfolio to target weights.

    Example:
        >>> engine = RebalancingEngine(
        ...     config=RebalancingConfig(method="threshold", drift_threshold=0.05),
        ...     constraints=PortfolioConstraints(max_position_weight=0.20),
        ...     cost_model=TransactionCostModel(fee_rate=0.0005),
        ... )
        >>> result = engine.calculate_rebalance(
        ...     current_state=portfolio_state,
        ...     target_weights={"BTC": 0.4, "ETH": 0.3, "SOL": 0.3},
        ... )
        >>> if result.trades:
        ...     execute_trades(result.trades)
    """

    def __init__(
        self,
        config: RebalancingConfig | None = None,
        constraints: PortfolioConstraints | None = None,
        cost_model: TransactionCostModel | None = None,
    ) -> None:
        """
        Initialize rebalancing engine.

        Args:
            config: Rebalancing configuration
            constraints: Portfolio constraints
            cost_model: Transaction cost model
        """
        self.config = config or RebalancingConfig()
        self.constraints = constraints or PortfolioConstraints()
        self.cost_model = cost_model or TransactionCostModel()

        # State tracking
        self._last_rebalance_date: datetime | None = None
        self._cumulative_turnover: float = 0.0
        self._rebalance_history: list[RebalanceResult] = []

    def should_rebalance(
        self,
        current_state: PortfolioState,
        target_weights: dict[str, float],
        current_date: datetime | None = None,
    ) -> tuple[bool, RebalanceReason]:
        """
        Determine if portfolio should be rebalanced.

        Args:
            current_state: Current portfolio state
            target_weights: Target allocation weights
            current_date: Current date (for periodic checks)

        Returns:
            Tuple of (should_rebalance, reason)
        """
        current_date = current_date or datetime.now()
        current_weights = current_state.current_weights

        # Check minimum interval
        if self._last_rebalance_date:
            days_since = (current_date - self._last_rebalance_date).days
            if days_since < self.config.min_rebalance_interval_days:
                return False, RebalanceReason.PERIODIC

        # Calculate drift metrics
        max_drift = self._calculate_max_drift(current_weights, target_weights)

        if self.config.method == "periodic":
            should = self._check_periodic_trigger(current_date)
            return should, RebalanceReason.PERIODIC

        elif self.config.method == "threshold":
            should = max_drift >= self.config.drift_threshold
            return should, RebalanceReason.DRIFT

        elif self.config.method == "hybrid":
            # Threshold OR periodic
            if max_drift >= self.config.drift_threshold:
                return True, RebalanceReason.DRIFT
            if self._check_periodic_trigger(current_date):
                return True, RebalanceReason.PERIODIC
            return False, RebalanceReason.PERIODIC

        elif self.config.method == "tax_aware":
            # Only rebalance on significant drift to minimize tax events
            tax_drift_threshold = self.config.drift_threshold * 1.5
            should = max_drift >= tax_drift_threshold
            return should, RebalanceReason.DRIFT

        return False, RebalanceReason.PERIODIC

    def calculate_rebalance(
        self,
        current_state: PortfolioState,
        target_weights: dict[str, float],
        avg_daily_volumes: dict[str, float] | None = None,
        volatilities: dict[str, float] | None = None,
        reason: RebalanceReason = RebalanceReason.PERIODIC,
    ) -> RebalanceResult:
        """
        Calculate trades needed to rebalance portfolio.

        Args:
            current_state: Current portfolio state
            target_weights: Target allocation weights
            avg_daily_volumes: Average daily volume per ticker (for cost estimation)
            volatilities: Volatility per ticker (for cost estimation)
            reason: Reason for rebalancing

        Returns:
            RebalanceResult with list of trades
        """
        avg_daily_volumes = avg_daily_volumes or {}
        volatilities = volatilities or {}

        # Normalize target weights
        target_weights = self._normalize_weights(target_weights)

        # Validate constraints
        violations = self.constraints.validate_weights(target_weights)
        if violations:
            logger.warning(f"Target weight constraint violations: {violations}")
            target_weights = self._adjust_for_constraints(target_weights)

        # Get current state
        current_weights = current_state.current_weights
        portfolio_value = current_state.total_value

        # Calculate required trades
        trades = self._calculate_trades(
            current_state=current_state,
            target_weights=target_weights,
            avg_daily_volumes=avg_daily_volumes,
            volatilities=volatilities,
        )

        # Apply turnover limit
        trades = self._apply_turnover_limit(trades, portfolio_value)

        # Filter trades below minimum
        trades = self._filter_small_trades(trades, portfolio_value)

        # Sort trades (sells first, then by priority)
        trades = self._sort_trades(trades)

        # Calculate metrics
        total_turnover = sum(t.trade_value for t in trades) / portfolio_value if portfolio_value > 0 else 0
        total_cost = sum(t.estimated_cost for t in trades)
        max_drift = self._calculate_max_drift(current_weights, target_weights)

        result = RebalanceResult(
            trades=trades,
            reason=reason,
            pre_weights=current_weights,
            target_weights=target_weights,
            total_turnover=total_turnover,
            total_cost=total_cost,
            max_drift=max_drift,
            timestamp=datetime.now(),
        )

        logger.info(
            f"Rebalance calculated: {len(trades)} trades, "
            f"turnover={total_turnover:.2%}, cost={total_cost:,.0f}"
        )

        return result

    def execute_rebalance(
        self,
        result: RebalanceResult,
        current_state: PortfolioState,
    ) -> PortfolioState:
        """
        Simulate execution of rebalance trades.

        Updates portfolio state as if trades were executed.
        For live trading, use actual execution engine.

        Args:
            result: Rebalance result with trades
            current_state: Current portfolio state

        Returns:
            New portfolio state after rebalance
        """
        new_holdings = dict(current_state.holdings)
        new_cash = current_state.cash

        for trade in result.trades:
            if trade.side == "buy":
                # Deduct cash, add position
                new_cash -= trade.trade_value + trade.estimated_cost
                new_holdings[trade.ticker] = new_holdings.get(trade.ticker, 0) + trade.quantity
            else:
                # Add cash, reduce position
                new_cash += trade.trade_value - trade.estimated_cost
                new_holdings[trade.ticker] = new_holdings.get(trade.ticker, 0) - trade.quantity
                if new_holdings[trade.ticker] <= 0:
                    del new_holdings[trade.ticker]

        # Update tracking
        self._last_rebalance_date = datetime.now()
        self._cumulative_turnover += result.total_turnover
        result.executed = True
        self._rebalance_history.append(result)

        return PortfolioState(
            holdings=new_holdings,
            prices=current_state.prices,
            cash=new_cash,
            timestamp=datetime.now(),
        )

    def get_drift_report(
        self,
        current_state: PortfolioState,
        target_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculate drift for each position.

        Args:
            current_state: Current portfolio state
            target_weights: Target weights

        Returns:
            Dict of ticker -> drift (positive = overweight, negative = underweight)
        """
        current_weights = current_state.current_weights
        drift = {}

        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        for ticker in all_tickers:
            current = current_weights.get(ticker, 0.0)
            target = target_weights.get(ticker, 0.0)
            drift[ticker] = current - target

        return dict(sorted(drift.items(), key=lambda x: abs(x[1]), reverse=True))

    def _calculate_trades(
        self,
        current_state: PortfolioState,
        target_weights: dict[str, float],
        avg_daily_volumes: dict[str, float],
        volatilities: dict[str, float],
    ) -> list[RebalanceTrade]:
        """Calculate required trades to reach target weights."""
        trades = []
        portfolio_value = current_state.total_value
        current_weights = current_state.current_weights

        # Consider cash target
        target_cash = self.config.target_cash_weight
        investable_value = portfolio_value * (1 - target_cash)

        all_tickers = set(current_weights.keys()) | set(target_weights.keys())

        for ticker in all_tickers:
            current_weight = current_weights.get(ticker, 0.0)
            target_weight = target_weights.get(ticker, 0.0)

            # Adjust for cash buffer
            adjusted_target = target_weight * (1 - target_cash)

            current_value = portfolio_value * current_weight
            target_value = portfolio_value * adjusted_target

            diff_value = target_value - current_value

            # Skip if difference is negligible
            if abs(diff_value) < self.constraints.min_trade_value:
                continue

            price = current_state.prices.get(ticker, 0)
            if price <= 0:
                logger.warning(f"No price for {ticker}, skipping")
                continue

            quantity = abs(diff_value) / price
            side: Literal["buy", "sell"] = "buy" if diff_value > 0 else "sell"

            # Estimate transaction cost
            adv = avg_daily_volumes.get(ticker)
            vol = volatilities.get(ticker)
            estimated_cost = self.cost_model.estimate_cost(
                abs(diff_value), adv, vol
            )

            # Check if trade is worth the cost
            weight_improvement = abs(target_weight - current_weight)
            if weight_improvement < self.config.min_trade_improvement:
                if estimated_cost > abs(diff_value) * 0.01:  # Cost > 1% of trade
                    logger.debug(f"Skipping {ticker}: cost not justified")
                    continue

            trade = RebalanceTrade(
                ticker=ticker,
                side=side,
                quantity=quantity,
                target_value=target_value,
                current_value=current_value,
                estimated_price=price,
                estimated_cost=estimated_cost,
                priority=1 if side == "sell" else 0,  # Sells first
            )
            trades.append(trade)

        return trades

    def _apply_turnover_limit(
        self,
        trades: list[RebalanceTrade],
        portfolio_value: float,
    ) -> list[RebalanceTrade]:
        """Apply maximum turnover limit per rebalance."""
        if portfolio_value <= 0:
            return trades

        max_turnover_value = portfolio_value * self.config.max_turnover_per_rebalance

        # Sort by priority (sells first) and then by size (largest first)
        sorted_trades = sorted(
            trades,
            key=lambda t: (-t.priority, -t.trade_value),
        )

        result = []
        cumulative_turnover = 0.0

        for trade in sorted_trades:
            if cumulative_turnover + trade.trade_value <= max_turnover_value:
                result.append(trade)
                cumulative_turnover += trade.trade_value
            else:
                # Partial trade to reach limit
                remaining = max_turnover_value - cumulative_turnover
                if remaining >= self.constraints.min_trade_value:
                    scale = remaining / trade.trade_value
                    partial_trade = RebalanceTrade(
                        ticker=trade.ticker,
                        side=trade.side,
                        quantity=trade.quantity * scale,
                        target_value=trade.current_value + (trade.target_value - trade.current_value) * scale,
                        current_value=trade.current_value,
                        estimated_price=trade.estimated_price,
                        estimated_cost=trade.estimated_cost * scale,
                        priority=trade.priority,
                    )
                    result.append(partial_trade)
                break

        if len(result) < len(trades):
            logger.info(
                f"Turnover limit applied: {len(result)}/{len(trades)} trades included"
            )

        return result

    def _filter_small_trades(
        self,
        trades: list[RebalanceTrade],
        portfolio_value: float,
    ) -> list[RebalanceTrade]:
        """Filter out trades below minimum size."""
        return [
            t for t in trades
            if t.trade_value >= self.constraints.min_trade_value
        ]

    def _sort_trades(self, trades: list[RebalanceTrade]) -> list[RebalanceTrade]:
        """Sort trades for optimal execution (sells first)."""
        if self.config.prioritize_sells:
            return sorted(trades, key=lambda t: (-t.priority, t.ticker))
        return sorted(trades, key=lambda t: t.ticker)

    def _calculate_max_drift(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> float:
        """Calculate maximum weight drift."""
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())

        max_drift = 0.0
        for ticker in all_tickers:
            current = current_weights.get(ticker, 0.0)
            target = target_weights.get(ticker, 0.0)

            if self.config.drift_type == DriftType.ABSOLUTE:
                drift = abs(current - target)
            else:  # RELATIVE
                drift = abs(current - target) / target if target > 0 else abs(current)

            max_drift = max(max_drift, drift)

        return max_drift

    def _check_periodic_trigger(self, current_date: datetime) -> bool:
        """Check if periodic rebalance is due."""
        if self._last_rebalance_date is None:
            return True

        freq = self.config.frequency
        last = self._last_rebalance_date

        if freq == "daily":
            return current_date.date() > last.date()

        elif freq == "weekly":
            # Rebalance on Monday
            current_week = current_date.isocalendar()[1]
            last_week = last.isocalendar()[1]
            return current_week != last_week and current_date.weekday() == 0

        elif freq == "monthly":
            # Rebalance on first trading day of month
            return (
                current_date.year != last.year or
                current_date.month != last.month
            ) and current_date.day <= 5

        elif freq == "quarterly":
            current_quarter = (current_date.month - 1) // 3
            last_quarter = (last.month - 1) // 3
            return (
                current_date.year != last.year or
                current_quarter != last_quarter
            )

        elif freq == "annual":
            return current_date.year != last.year

        return False

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        if total <= 0:
            return weights
        return {k: v / total for k, v in weights.items()}

    def _adjust_for_constraints(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Adjust weights to satisfy constraints."""
        adjusted = dict(weights)

        # Cap maximum positions
        for ticker in adjusted:
            if adjusted[ticker] > self.constraints.max_position_weight:
                adjusted[ticker] = self.constraints.max_position_weight

        # Remove positions below minimum
        adjusted = {
            k: v for k, v in adjusted.items()
            if v >= self.constraints.min_position_weight or v == 0
        }

        # Re-normalize
        return self._normalize_weights(adjusted)

    @property
    def last_rebalance_date(self) -> datetime | None:
        """Get date of last rebalance."""
        return self._last_rebalance_date

    @property
    def cumulative_turnover(self) -> float:
        """Get cumulative turnover since inception."""
        return self._cumulative_turnover

    @property
    def rebalance_count(self) -> int:
        """Get number of rebalances executed."""
        return len(self._rebalance_history)

    def reset(self) -> None:
        """Reset engine state."""
        self._last_rebalance_date = None
        self._cumulative_turnover = 0.0
        self._rebalance_history.clear()


__all__ = [
    "RebalanceMethod",
    "RebalanceFrequency",
    "DriftType",
    "RebalancingConfig",
    "RebalancingEngine",
]
