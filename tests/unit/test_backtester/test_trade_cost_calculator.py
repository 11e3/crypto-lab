"""Tests for trade cost calculator module."""

from datetime import datetime

import pandas as pd
import pytest

from src.backtester.trade_cost_calculator import (
    TradeAnalyzer,
    TradeCostCalculator,
    TradeExecution,
    UpbitFeeStructure,
)


class TestUpbitFeeStructure:
    """Tests for Upbit fee structure."""

    def test_default_fees(self) -> None:
        """Test default fee rates."""
        fees = UpbitFeeStructure.get_fees(tier=0)
        
        assert fees["maker"] == 0.05
        assert fees["taker"] == 0.05

    def test_vip_tier_1(self) -> None:
        """Test VIP tier 1 fees."""
        fees = UpbitFeeStructure.get_fees(tier=1)
        
        assert fees["maker"] == 0.04
        assert fees["taker"] == 0.05

    def test_vip_tier_4(self) -> None:
        """Test highest VIP tier fees."""
        fees = UpbitFeeStructure.get_fees(tier=4)
        
        assert fees["maker"] == 0.01
        assert fees["taker"] == 0.02

    def test_invalid_tier_clamping(self) -> None:
        """Test that invalid tiers are clamped to valid range."""
        fees_negative = UpbitFeeStructure.get_fees(tier=-1)
        fees_too_high = UpbitFeeStructure.get_fees(tier=10)
        
        assert fees_negative["maker"] == 0.05  # tier 0
        assert fees_too_high["maker"] == 0.01  # tier 4

    def test_fee_dict_is_copy(self) -> None:
        """Test that returned dict is a copy, not reference."""
        fees1 = UpbitFeeStructure.get_fees(tier=0)
        fees2 = UpbitFeeStructure.get_fees(tier=0)
        
        fees1["maker"] = 999.0
        
        assert fees2["maker"] == 0.05  # Not modified


class TestTradeCostCalculator:
    """Tests for trade cost calculator."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        calc = TradeCostCalculator()
        
        assert calc.fees["maker"] == 0.05
        assert calc.fees["taker"] == 0.05
        assert calc.volatility_regime == "medium"
        assert calc.slippage_base["medium"] == 0.02

    def test_initialization_with_vip(self) -> None:
        """Test initialization with VIP tier."""
        calc = TradeCostCalculator(vip_tier=2)
        
        assert calc.fees["maker"] == 0.03
        assert calc.fees["taker"] == 0.04

    def test_initialization_high_volatility(self) -> None:
        """Test initialization with high volatility regime."""
        calc = TradeCostCalculator(volatility_regime="high")
        
        assert calc.slippage_base["high"] == 0.05

    def test_calculate_trade_cost_default(self) -> None:
        """Test single trade cost calculation with defaults."""
        calc = TradeCostCalculator(vip_tier=0, volatility_regime="medium")
        
        cost = calc.calculate_trade_cost_pct("buy")
        
        # medium volatility slippage (0.02%) + taker fee (0.05%)
        expected = 0.02 + 0.05
        assert cost == expected

    def test_calculate_trade_cost_custom_slippage(self) -> None:
        """Test trade cost with custom slippage."""
        calc = TradeCostCalculator(vip_tier=0)
        
        cost = calc.calculate_trade_cost_pct("buy", slippage_pct=0.10)
        
        # custom slippage (0.10%) + taker fee (0.05%)
        expected = 0.10 + 0.05
        assert cost == expected

    def test_calculate_trade_cost_low_volatility(self) -> None:
        """Test trade cost in low volatility regime."""
        calc = TradeCostCalculator(vip_tier=0, volatility_regime="low")
        
        cost = calc.calculate_trade_cost_pct("sell")
        
        # low volatility slippage (0.01%) + taker fee (0.05%)
        expected = 0.01 + 0.05
        assert cost == expected

    def test_calculate_roundtrip_cost_default(self) -> None:
        """Test roundtrip cost calculation."""
        calc = TradeCostCalculator(vip_tier=0, volatility_regime="medium")
        
        roundtrip_cost = calc.calculate_roundtrip_cost_pct()
        
        # Entry: 0.02% + 0.05% = 0.07%
        # Exit:  0.02% + 0.05% = 0.07%
        # Total: 0.14%
        expected = (0.02 + 0.05) * 2
        assert roundtrip_cost == expected

    def test_calculate_roundtrip_cost_custom(self) -> None:
        """Test roundtrip cost with custom slippage."""
        calc = TradeCostCalculator(vip_tier=1, volatility_regime="medium")
        
        roundtrip_cost = calc.calculate_roundtrip_cost_pct(
            entry_slippage=0.03,
            exit_slippage=0.04
        )
        
        # Entry: 0.03% + 0.05% (taker) = 0.08%
        # Exit:  0.04% + 0.05% (taker) = 0.09%
        # Total: 0.17%
        expected = (0.03 + 0.05) + (0.04 + 0.05)
        assert roundtrip_cost == expected

    def test_calculate_net_pnl_profitable_trade(self) -> None:
        """Test net PnL calculation for profitable trade."""
        calc = TradeCostCalculator(vip_tier=0, volatility_regime="medium")
        
        result = calc.calculate_net_pnl(
            entry_price=100_000,
            exit_price=101_000,
        )
        
        # Gross PnL: (101k - 100k) / 100k = 1.0%
        assert result["gross_pnl_pct"] == 1.0
        
        # Costs: slippage (0.04%) + fees (0.10%) = 0.14%
        assert result["total_slippage_pct"] == 0.04
        assert result["total_fee_pct"] == 0.10
        
        # Net PnL: 1.0% - 0.14% = 0.86%
        expected_net = 1.0 - 0.04 - 0.10
        assert result["net_pnl_pct"] == pytest.approx(expected_net)
        
        # Breakeven
        assert result["breakeven_pnl_pct"] == 0.14

    def test_calculate_net_pnl_losing_trade(self) -> None:
        """Test net PnL calculation for losing trade."""
        calc = TradeCostCalculator(vip_tier=0, volatility_regime="medium")
        
        result = calc.calculate_net_pnl(
            entry_price=100_000,
            exit_price=99_000,
        )
        
        # Gross PnL: -1.0%
        assert result["gross_pnl_pct"] == -1.0
        
        # Net PnL: -1.0% - 0.14% = -1.14%
        expected_net = -1.0 - 0.04 - 0.10
        assert result["net_pnl_pct"] == pytest.approx(expected_net)

    def test_calculate_net_pnl_high_volatility(self) -> None:
        """Test net PnL in high volatility regime."""
        calc = TradeCostCalculator(vip_tier=0, volatility_regime="high")
        
        result = calc.calculate_net_pnl(
            entry_price=100_000,
            exit_price=102_000,
        )
        
        # High volatility: 0.05% + 0.05% = 0.10% slippage
        assert result["total_slippage_pct"] == 0.10
        
        # Gross: 2.0%, Net: 2.0% - 0.10% - 0.10% = 1.8%
        expected_net = 2.0 - 0.10 - 0.10
        assert result["net_pnl_pct"] == pytest.approx(expected_net)

    def test_calculate_net_pnl_vip_benefits(self) -> None:
        """Test that VIP tier reduces fees."""
        calc_default = TradeCostCalculator(vip_tier=0, volatility_regime="medium")
        calc_vip4 = TradeCostCalculator(vip_tier=4, volatility_regime="medium")
        
        result_default = calc_default.calculate_net_pnl(
            entry_price=100_000,
            exit_price=101_000,
        )
        
        result_vip = calc_vip4.calculate_net_pnl(
            entry_price=100_000,
            exit_price=101_000,
        )
        
        # VIP tier 4: 0.01% + 0.02% = 0.03% total fees
        assert result_vip["total_fee_pct"] == 0.04  # 0.02 taker + 0.02 taker
        
        # VIP should have higher net PnL
        assert result_vip["net_pnl_pct"] > result_default["net_pnl_pct"]

    def test_calculate_minimum_profit_target(self) -> None:
        """Test minimum profit target calculation."""
        calc = TradeCostCalculator(vip_tier=0, volatility_regime="medium")
        
        exit_price = calc.calculate_minimum_profit_target(
            entry_price=100_000,
            target_pnl=0.5,  # 0.5% net profit target
        )
        
        # Total cost: 0.04% slippage + 0.10% fees = 0.14%
        # Required gross: 0.5% + 0.14% = 0.64%
        # Exit price: 100,000 * 1.0064 = 100,640
        expected = 100_000 * 1.0064
        assert exit_price == pytest.approx(expected)

    def test_calculate_minimum_profit_target_high_costs(self) -> None:
        """Test minimum profit target with high transaction costs."""
        calc = TradeCostCalculator(vip_tier=0, volatility_regime="high")
        
        exit_price = calc.calculate_minimum_profit_target(
            entry_price=100_000,
            target_pnl=1.0,
            entry_slippage=0.10,
            exit_slippage=0.10,
        )
        
        # Total cost: 0.20% slippage + 0.10% fees = 0.30%
        # Required gross: 1.0% + 0.30% = 1.30%
        expected = 100_000 * 1.013
        assert exit_price == pytest.approx(expected)

    def test_calculate_net_pnl_with_position_size(self) -> None:
        """Test that position size is included in result."""
        calc = TradeCostCalculator()
        
        result = calc.calculate_net_pnl(
            entry_price=100_000,
            exit_price=101_000,
            position_size=5.0,
        )
        
        assert result["position_size"] == 5.0


class TestTradeExecution:
    """Tests for TradeExecution dataclass."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        timestamp = pd.Timestamp("2025-01-01")
        trade = TradeExecution(
            timestamp=timestamp,
            side="buy",
            entry_price=100_000,
            entry_size=1.0,
        )
        
        assert trade.timestamp == timestamp
        assert trade.side == "buy"
        assert trade.entry_price == 100_000
        assert trade.entry_size == 1.0
        assert trade.exit_price == 0.0
        assert trade.gross_pnl == 0.0

    def test_with_exit_data(self) -> None:
        """Test initialization with exit data."""
        entry_time = pd.Timestamp("2025-01-01")
        exit_time = pd.Timestamp("2025-01-02")
        
        trade = TradeExecution(
            timestamp=entry_time,
            side="buy",
            entry_price=100_000,
            entry_size=1.0,
            exit_price=101_000,
            exit_size=1.0,
            exit_time=exit_time,
        )
        
        assert trade.exit_price == 101_000
        assert trade.exit_time == exit_time

    def test_with_cost_data(self) -> None:
        """Test initialization with cost calculations."""
        trade = TradeExecution(
            timestamp=pd.Timestamp("2025-01-01"),
            side="buy",
            entry_price=100_000,
            entry_size=1.0,
            entry_slippage_pct=0.02,
            exit_slippage_pct=0.02,
            maker_fee_pct=0.05,
            taker_fee_pct=0.05,
        )
        
        assert trade.entry_slippage_pct == 0.02
        assert trade.exit_slippage_pct == 0.02
        assert trade.taker_fee_pct == 0.05


class TestTradeAnalyzer:
    """Tests for trade analyzer."""

    def test_initialization(self) -> None:
        """Test analyzer initialization."""
        analyzer = TradeAnalyzer(vip_tier=2, volatility_regime="low")
        
        assert analyzer.calculator.fees["maker"] == 0.03
        assert analyzer.calculator.volatility_regime == "low"

    def test_analyze_single_trade(self) -> None:
        """Test analysis of single trade."""
        analyzer = TradeAnalyzer(vip_tier=0, volatility_regime="medium")
        
        trades = [
            {"entry_price": 100_000, "exit_price": 101_000},
        ]
        
        df, summary = analyzer.analyze_trades(trades)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["gross_pnl_pct"] == 1.0
        
        assert isinstance(summary, dict)
        assert summary["total_trades"] == 1
        assert summary["avg_gross_pnl_pct"] == 1.0

    def test_analyze_multiple_trades(self) -> None:
        """Test analysis of multiple trades."""
        analyzer = TradeAnalyzer(vip_tier=0, volatility_regime="medium")
        
        trades = [
            {"entry_price": 100_000, "exit_price": 101_000},
            {"entry_price": 50_000, "exit_price": 49_500},
            {"entry_price": 200_000, "exit_price": 205_000},
        ]
        
        df, summary = analyzer.analyze_trades(trades)
        
        assert len(df) == 3
        assert df.iloc[0]["gross_pnl_pct"] == 1.0  # +1%
        assert df.iloc[1]["gross_pnl_pct"] == -1.0  # -1%
        assert df.iloc[2]["gross_pnl_pct"] == 2.5  # +2.5%
        
        assert summary["total_trades"] == 3
        assert summary["winning_trades"] == 2  # First and third trades


class TestIntegrationScenarios:
    """Integration tests for realistic trading scenarios."""

    def test_breakeven_scenario(self) -> None:
        """Test that we can calculate exact breakeven price."""
        calc = TradeCostCalculator(vip_tier=0, volatility_regime="medium")
        
        entry_price = 100_000
        
        # Calculate breakeven exit price (0% net profit)
        breakeven_price = calc.calculate_minimum_profit_target(
            entry_price=entry_price,
            target_pnl=0.0,
        )
        
        # Verify it results in ~0% net PnL
        result = calc.calculate_net_pnl(
            entry_price=entry_price,
            exit_price=breakeven_price,
        )
        
        assert result["net_pnl_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_cost_reduction_with_vip(self) -> None:
        """Test that VIP tier significantly reduces costs."""
        calc_regular = TradeCostCalculator(vip_tier=0)
        calc_vip = TradeCostCalculator(vip_tier=4)
        
        entry = 100_000
        exit_price = 101_000
        
        result_regular = calc_regular.calculate_net_pnl(entry, exit_price)
        result_vip = calc_vip.calculate_net_pnl(entry, exit_price)
        
        # VIP should have lower total costs
        cost_regular = result_regular["total_slippage_pct"] + result_regular["total_fee_pct"]
        cost_vip = result_vip["total_slippage_pct"] + result_vip["total_fee_pct"]
        
        assert cost_vip < cost_regular
        
        # VIP should have higher net profit
        assert result_vip["net_pnl_pct"] > result_regular["net_pnl_pct"]

    def test_high_volatility_impact(self) -> None:
        """Test that high volatility significantly increases costs."""
        calc_low = TradeCostCalculator(vip_tier=0, volatility_regime="low")
        calc_high = TradeCostCalculator(vip_tier=0, volatility_regime="high")
        
        entry = 100_000
        exit_price = 101_000
        
        result_low = calc_low.calculate_net_pnl(entry, exit_price)
        result_high = calc_high.calculate_net_pnl(entry, exit_price)
        
        # High volatility should have much higher slippage
        assert result_high["total_slippage_pct"] > result_low["total_slippage_pct"]
        
        # High volatility should have lower net profit
        assert result_high["net_pnl_pct"] < result_low["net_pnl_pct"]

    def test_realistic_profitable_trade(self) -> None:
        """Test a realistic profitable trade scenario."""
        # Scenario: Regular user, medium volatility market
        calc = TradeCostCalculator(vip_tier=0, volatility_regime="medium")
        
        # Buy at 50,000 KRW, sell at 51,000 KRW (2% gain)
        result = calc.calculate_net_pnl(
            entry_price=50_000,
            exit_price=51_000,
        )
        
        # Verify all cost components are accounted for
        assert result["gross_pnl_pct"] == 2.0
        assert result["entry_slippage_pct"] > 0
        assert result["exit_slippage_pct"] > 0
        assert result["entry_fee_pct"] > 0
        assert result["exit_fee_pct"] > 0
        
        # Net should be less than gross
        assert result["net_pnl_pct"] < result["gross_pnl_pct"]
        
        # But still profitable
        assert result["net_pnl_pct"] > 0
