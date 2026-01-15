"""
Tests for portfolio rebalancing engine.
"""

import pytest
from datetime import datetime, timedelta

from src.portfolio.models import (
    PortfolioState,
    PortfolioConstraints,
    TransactionCostModel,
    RebalanceReason,
)
from src.portfolio.rebalancing import (
    RebalancingEngine,
    RebalancingConfig,
    DriftType,
)


class TestPortfolioState:
    """Tests for PortfolioState model."""

    def test_empty_portfolio(self):
        """Test empty portfolio state."""
        state = PortfolioState(cash=1_000_000)
        assert state.total_value == 1_000_000
        assert state.cash_weight == 1.0
        assert state.current_weights == {}

    def test_portfolio_with_holdings(self):
        """Test portfolio with holdings."""
        state = PortfolioState(
            holdings={"BTC": 1.0, "ETH": 10.0},
            prices={"BTC": 50_000_000, "ETH": 3_000_000},
            cash=10_000_000,
        )

        assert state.total_value == 50_000_000 + 30_000_000 + 10_000_000
        assert state.positions_value["BTC"] == 50_000_000
        assert state.positions_value["ETH"] == 30_000_000

        weights = state.current_weights
        assert abs(weights["BTC"] - 50_000_000 / 90_000_000) < 0.001
        assert abs(weights["ETH"] - 30_000_000 / 90_000_000) < 0.001

    def test_cash_weight(self):
        """Test cash weight calculation."""
        state = PortfolioState(
            holdings={"BTC": 1.0},
            prices={"BTC": 90_000_000},
            cash=10_000_000,
        )
        assert abs(state.cash_weight - 0.1) < 0.001


class TestPortfolioConstraints:
    """Tests for PortfolioConstraints."""

    def test_valid_weights(self):
        """Test validation of valid weights."""
        constraints = PortfolioConstraints(
            max_position_weight=0.30,
            min_position_weight=0.05,
            min_holdings=3,
            max_holdings=10,
        )

        valid_weights = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        violations = constraints.validate_weights(valid_weights)
        assert len(violations) == 0

    def test_max_position_violation(self):
        """Test max position weight violation."""
        constraints = PortfolioConstraints(max_position_weight=0.20)
        weights = {"A": 0.50, "B": 0.25, "C": 0.25}
        violations = constraints.validate_weights(weights)
        assert any("exceeds max" in v for v in violations)

    def test_min_holdings_violation(self):
        """Test minimum holdings violation."""
        constraints = PortfolioConstraints(min_holdings=5)
        weights = {"A": 0.50, "B": 0.50}
        violations = constraints.validate_weights(weights)
        assert any("below minimum" in v for v in violations)

    def test_weight_sum_violation(self):
        """Test weight sum validation."""
        constraints = PortfolioConstraints()
        weights = {"A": 0.30, "B": 0.30}  # Sum = 0.60
        violations = constraints.validate_weights(weights)
        assert any("sum to" in v for v in violations)


class TestTransactionCostModel:
    """Tests for TransactionCostModel."""

    def test_basic_cost_estimation(self):
        """Test basic cost estimation."""
        model = TransactionCostModel(fee_rate=0.001, base_slippage=0.001)
        cost = model.estimate_cost(1_000_000)

        # Fee: 1000, Slippage: 1000
        assert cost == 2000

    def test_cost_with_volume_impact(self):
        """Test cost with market impact."""
        model = TransactionCostModel(
            fee_rate=0.001,
            base_slippage=0.001,
            impact_coefficient=0.1,
        )

        # Large trade relative to volume
        cost = model.estimate_cost(
            trade_value=10_000_000,
            avg_daily_volume=100_000_000,  # 10% of ADV
        )

        assert cost > 20_000  # Base cost
        # Impact adds: 10M * 0.1 * sqrt(0.1) ≈ 316k

    def test_max_trade_size(self):
        """Test maximum trade size calculation."""
        model = TransactionCostModel(max_participation_rate=0.10)
        max_size = model.max_trade_size(1_000_000_000)
        assert max_size == 100_000_000


class TestRebalancingEngine:
    """Tests for RebalancingEngine."""

    @pytest.fixture
    def engine(self):
        """Create default rebalancing engine."""
        return RebalancingEngine(
            config=RebalancingConfig(
                method="threshold",
                drift_threshold=0.05,
            ),
            constraints=PortfolioConstraints(
                max_position_weight=0.40,
                min_trade_value=10_000,
            ),
            cost_model=TransactionCostModel(fee_rate=0.0005),
        )

    @pytest.fixture
    def portfolio_state(self):
        """Create sample portfolio state."""
        return PortfolioState(
            holdings={"BTC": 1.0, "ETH": 10.0},
            prices={"BTC": 50_000_000, "ETH": 3_000_000},
            cash=10_000_000,  # Total: 90M
        )

    def test_no_rebalance_when_aligned(self, engine, portfolio_state):
        """Test no rebalance when weights are aligned."""
        # Current weights approximately: BTC 55.6%, ETH 33.3%, Cash 11.1%
        target = {
            "BTC": 0.556,
            "ETH": 0.333,
        }

        should, reason = engine.should_rebalance(portfolio_state, target)
        assert not should

    def test_rebalance_on_drift(self, engine, portfolio_state):
        """Test rebalance triggered by drift."""
        # Significantly different target
        target = {"BTC": 0.30, "ETH": 0.50, "SOL": 0.20}

        should, reason = engine.should_rebalance(portfolio_state, target)
        assert should
        assert reason == RebalanceReason.DRIFT

    def test_calculate_rebalance_trades(self, engine, portfolio_state):
        """Test trade calculation."""
        target = {"BTC": 0.40, "ETH": 0.40, "SOL": 0.20}

        # Add SOL price
        portfolio_state.prices["SOL"] = 200_000

        result = engine.calculate_rebalance(portfolio_state, target)

        assert len(result.trades) > 0
        assert result.total_turnover > 0
        assert result.total_cost >= 0

        # Should have sells (BTC overweight) and buys (SOL underweight)
        sell_trades = [t for t in result.trades if t.side == "sell"]
        buy_trades = [t for t in result.trades if t.side == "buy"]

        assert len(sell_trades) > 0 or len(buy_trades) > 0

    def test_turnover_limit(self, engine, portfolio_state):
        """Test turnover limit is applied."""
        engine.config.max_turnover_per_rebalance = 0.10  # 10% limit

        # Extreme rebalance
        target = {"SOL": 1.0}
        portfolio_state.prices["SOL"] = 200_000

        result = engine.calculate_rebalance(portfolio_state, target)

        # Turnover should be limited
        assert result.total_turnover <= 0.10 + 0.01  # Small tolerance

    def test_execute_rebalance(self, engine, portfolio_state):
        """Test rebalance execution simulation."""
        target = {"BTC": 0.40, "ETH": 0.40, "SOL": 0.20}
        portfolio_state.prices["SOL"] = 200_000

        result = engine.calculate_rebalance(portfolio_state, target)
        new_state = engine.execute_rebalance(result, portfolio_state)

        assert result.executed
        assert engine.rebalance_count == 1
        assert new_state.total_value > 0

    def test_drift_report(self, engine, portfolio_state):
        """Test drift report generation."""
        target = {"BTC": 0.40, "ETH": 0.50, "SOL": 0.10}

        drift = engine.get_drift_report(portfolio_state, target)

        assert "BTC" in drift
        assert "ETH" in drift
        # BTC is overweight, so drift should be positive
        assert drift["BTC"] > 0

    def test_periodic_rebalancing(self):
        """Test periodic rebalancing."""
        engine = RebalancingEngine(
            config=RebalancingConfig(
                method="periodic",
                frequency="monthly",
            )
        )

        state = PortfolioState(
            holdings={"BTC": 1.0},
            prices={"BTC": 50_000_000},
            cash=10_000_000,
        )
        target = {"BTC": 0.80}

        # First time should trigger
        should, _ = engine.should_rebalance(state, target)
        assert should

        # Mark as executed
        engine._last_rebalance_date = datetime.now()

        # Same month should not trigger
        should, _ = engine.should_rebalance(state, target)
        assert not should

        # Next month should trigger
        next_month = datetime.now() + timedelta(days=35)
        should, _ = engine.should_rebalance(state, target, next_month)
        assert should

    def test_hybrid_rebalancing(self):
        """Test hybrid rebalancing (periodic + threshold)."""
        engine = RebalancingEngine(
            config=RebalancingConfig(
                method="hybrid",
                frequency="monthly",
                drift_threshold=0.10,
            )
        )

        state = PortfolioState(
            holdings={"BTC": 1.0},
            prices={"BTC": 50_000_000},
            cash=50_000_000,  # 50% cash, 50% BTC
        )

        # Large drift should trigger even if not periodic
        target = {"BTC": 0.90}  # 40% drift from current 50%
        engine._last_rebalance_date = datetime.now() - timedelta(days=5)

        should, reason = engine.should_rebalance(state, target)
        assert should
        assert reason == RebalanceReason.DRIFT

    def test_min_trade_filter(self, engine):
        """Test minimum trade size filter."""
        engine.constraints.min_trade_value = 1_000_000

        state = PortfolioState(
            holdings={"BTC": 1.0},
            prices={"BTC": 50_000_000},
            cash=50_000_000,
        )

        # Small change that would create trade below minimum
        target = {"BTC": 0.505}  # 0.5% change

        result = engine.calculate_rebalance(state, target)

        # Small trades should be filtered
        for trade in result.trades:
            assert trade.trade_value >= 1_000_000


class TestRebalancingConfig:
    """Tests for RebalancingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RebalancingConfig()
        assert config.method == "threshold"
        assert config.drift_threshold == 0.05
        assert config.frequency == "monthly"

    def test_custom_config(self):
        """Test custom configuration."""
        config = RebalancingConfig(
            method="hybrid",
            drift_threshold=0.10,
            frequency="weekly",
            max_turnover_per_rebalance=0.20,
        )
        assert config.method == "hybrid"
        assert config.drift_threshold == 0.10
        assert config.max_turnover_per_rebalance == 0.20
