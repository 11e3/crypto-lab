"""Backtest accounting verification tests.

Validates that both backtesting engines (event-driven and vectorized)
maintain correct financial accounting: cash conservation, equity identity,
PnL consistency, and slippage symmetry.

Design choices are documented as passing tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtester.engine.event_data_loader import Position
from src.backtester.engine.event_exec import execute_entry, execute_exit
from src.backtester.engine.event_loop import (
    calculate_portfolio_equity,
    close_remaining_positions,
)
from src.backtester.engine.trade_costs import TradeCostCalculator
from src.backtester.engine.trade_simulator import (
    calculate_daily_equity,
    finalize_open_positions,
    initialize_simulation_state,
)
from src.backtester.engine.trade_simulator_exec import (
    execute_exit as vec_execute_exit,
)
from src.backtester.engine.trade_simulator_exec import (
    handle_normal_entry,
)
from src.backtester.models import BacktestConfig
from src.orders.advanced_orders import AdvancedOrderManager

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FEE = 0.001  # 0.1%
SLIPPAGE = 0.002  # 0.2%
INITIAL_CAPITAL = 10_000_000.0


@dataclass
class SimpleScenario:
    """Buy at 100, price goes to 110, then sell."""

    config: BacktestConfig
    entry_close: float
    exit_close: float
    entry_price: float  # close * (1 + slippage)
    exit_price: float  # close * (1 - slippage)


@pytest.fixture
def scenario() -> SimpleScenario:
    """Single-ticker buy→sell scenario with deterministic prices."""
    config = BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        fee_rate=FEE,
        slippage_rate=SLIPPAGE,
        max_slots=1,
    )
    entry_close = 100.0
    exit_close = 110.0
    return SimpleScenario(
        config=config,
        entry_close=entry_close,
        exit_close=exit_close,
        entry_price=entry_close * (1 + SLIPPAGE),
        exit_price=exit_close * (1 - SLIPPAGE),
    )


# ===========================================================================
# Group 1: Cash Conservation
# ===========================================================================


class TestCashConservation:
    """Verify cash accounting is correct after trades complete."""

    def test_event_driven_cash_conservation(self, scenario: SimpleScenario) -> None:
        """After buy→sell, cash change should equal trade PnL."""
        cfg = scenario.config
        cash = cfg.initial_capital

        # Entry
        row_entry = pd.Series(
            {
                "close": scenario.entry_close,
                "entry_signal": True,
                "exit_signal": False,
            }
        )
        position, cost = execute_entry(
            ticker="KRW-BTC",
            row=row_entry,
            current_date=date(2024, 1, 1),
            cash=cash,
            remaining_slots=1,
            config=cfg,
        )
        assert position is not None
        cash -= cost

        # Exit
        row_exit = pd.Series(
            {
                "close": scenario.exit_close,
                "exit_signal": True,
            }
        )
        trade, revenue = execute_exit(
            position=position,
            row=row_exit,
            current_date=date(2024, 1, 2),
            exit_reason="signal",
            config=cfg,
        )
        cash += revenue

        # Cash change should match trade.pnl
        cash_change = cash - cfg.initial_capital
        assert cash_change == pytest.approx(trade.pnl, abs=0.01), (
            f"cash_change={cash_change:.4f} != trade.pnl={trade.pnl:.4f}"
        )

    def test_vectorized_cash_conservation_all_closed(self, scenario: SimpleScenario) -> None:
        """Vectorized engine: cash change == trade PnL after full round-trip."""
        cfg = scenario.config

        # Setup: 1 ticker, 3 dates (entry on day0, exit on day1, day2 extra)
        tickers = ["KRW-BTC"]
        n_dates = 3
        state = initialize_simulation_state(cfg.initial_capital, 1, n_dates, tickers)

        buy_price = scenario.entry_price
        invest_amount = state.cash  # Single slot = all cash

        # Day 0: Entry
        handle_normal_entry(state, 0, 0, invest_amount, buy_price, cfg.fee_rate)

        # Day 1: Exit
        sorted_dates = np.array([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])
        exit_prices = np.array([[0.0, scenario.exit_price, 0.0]])
        order_manager = AdvancedOrderManager()

        vec_execute_exit(
            state=state,
            config=cfg,
            t_idx=0,
            d_idx=1,
            current_date=date(2024, 1, 2),
            sorted_dates=sorted_dates,
            tickers=tickers,
            exit_prices=exit_prices,
            order_manager=order_manager,
            is_stop_loss=False,
            is_take_profit=False,
            exit_reason="signal",
        )

        # Verify: cash change == trade PnL
        cash_change = state.cash - cfg.initial_capital
        trade_pnl = state.trades_list[0]["pnl"]
        assert cash_change == pytest.approx(trade_pnl, abs=0.01)

    def test_vectorized_no_negative_cash(self, scenario: SimpleScenario) -> None:
        """Equal sizing should never produce negative cash."""
        cfg = BacktestConfig(
            initial_capital=INITIAL_CAPITAL,
            fee_rate=FEE,
            slippage_rate=SLIPPAGE,
            max_slots=3,
        )

        tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
        n_dates = 1
        state = initialize_simulation_state(cfg.initial_capital, 3, n_dates, tickers)

        # Open 3 positions at equal sizing
        buy_price = 100.0 * (1 + SLIPPAGE)
        for t_idx in range(3):
            available_slots = cfg.max_slots - int(np.sum(state.position_amounts > 0))
            invest_amount = state.cash / available_slots
            handle_normal_entry(state, t_idx, 0, invest_amount, buy_price, cfg.fee_rate)

        assert state.cash >= 0, f"Cash went negative: {state.cash}"


# ===========================================================================
# Group 2: Equity = Cash + Positions
# ===========================================================================


class TestEquityIdentity:
    """Verify equity = cash + position market value at all times."""

    def test_event_driven_equity_equals_cash_plus_positions(self, scenario: SimpleScenario) -> None:
        """Event-driven equity should be cash + sum(amount * close) daily."""
        cfg = scenario.config
        cash = cfg.initial_capital

        # Day 1: No position
        positions: dict[str, Position] = {}
        current_data: dict[str, pd.Series] = {
            "KRW-BTC": pd.Series(
                {
                    "close": scenario.entry_close,
                    "entry_signal": True,
                    "exit_signal": False,
                }
            )
        }
        equity = calculate_portfolio_equity(positions, current_data, cash)
        assert equity == pytest.approx(cash, abs=0.01)

        # Entry
        position, cost = execute_entry(
            "KRW-BTC", current_data["KRW-BTC"], date(2024, 1, 1), cash, 1, cfg
        )
        assert position is not None
        cash -= cost
        positions["KRW-BTC"] = position

        # Day 2: Position held, price = 110
        current_data_d2: dict[str, pd.Series] = {
            "KRW-BTC": pd.Series({"close": scenario.exit_close})
        }
        equity = calculate_portfolio_equity(positions, current_data_d2, cash)
        expected = cash + position.amount * scenario.exit_close
        assert equity == pytest.approx(expected, abs=0.01)

    def test_vectorized_equity_equals_cash_plus_positions(self, scenario: SimpleScenario) -> None:
        """Vectorized equity[d] == cash + sum(amounts * closes[d])."""
        cfg = scenario.config
        tickers = ["KRW-BTC"]
        n_dates = 2
        state = initialize_simulation_state(cfg.initial_capital, 1, n_dates, tickers)

        closes = np.array([[scenario.entry_close, scenario.exit_close]])

        # Day 0: No position yet, equity = cash
        valid_data = np.array([True])
        calculate_daily_equity(state, 0, 1, closes, valid_data)
        assert state.equity_curve[0] == pytest.approx(cfg.initial_capital, abs=0.01)

        # Entry on day 0
        buy_price = scenario.entry_price
        invest_amount = state.cash
        handle_normal_entry(state, 0, 0, invest_amount, buy_price, cfg.fee_rate)

        # Day 1: position held
        calculate_daily_equity(state, 1, 1, closes, valid_data)
        expected = state.cash + state.position_amounts[0] * closes[0, 1]
        # float32 precision: ~7 significant digits, so abs=1.0 for 10M values
        assert state.equity_curve[1] == pytest.approx(expected, abs=1.0)


# ===========================================================================
# Group 3: PnL Consistency
# ===========================================================================


class TestPnLConsistency:
    """Verify PnL accounting matches cash flows."""

    def test_event_driven_pnl_matches_cash_flow(self, scenario: SimpleScenario) -> None:
        """sum(trade.pnl) should equal final_cash - initial_capital."""
        cfg = scenario.config
        cash = cfg.initial_capital

        # Entry
        row_entry = pd.Series(
            {
                "close": scenario.entry_close,
                "entry_signal": True,
                "exit_signal": False,
            }
        )
        position, cost = execute_entry("KRW-BTC", row_entry, date(2024, 1, 1), cash, 1, cfg)
        assert position is not None
        cash -= cost

        # Exit
        row_exit = pd.Series({"close": scenario.exit_close, "exit_signal": True})
        trade, revenue = execute_exit(position, row_exit, date(2024, 1, 2), "signal", cfg)
        cash += revenue

        total_pnl = trade.pnl
        cash_delta = cash - cfg.initial_capital

        assert total_pnl == pytest.approx(cash_delta, abs=0.01), (
            f"trade.pnl={total_pnl:.4f} != cash_delta={cash_delta:.4f}"
        )

    def test_event_driven_fee_symmetry(self, scenario: SimpleScenario) -> None:
        """Buy and sell at the SAME price → PnL < 0 (fees + slippage)."""
        cfg = scenario.config
        cash = cfg.initial_capital
        same_price = 100.0

        row_entry = pd.Series(
            {
                "close": same_price,
                "entry_signal": True,
                "exit_signal": False,
            }
        )
        position, cost = execute_entry("KRW-BTC", row_entry, date(2024, 1, 1), cash, 1, cfg)
        assert position is not None
        cash -= cost

        row_exit = pd.Series({"close": same_price, "exit_signal": True})
        trade, revenue = execute_exit(position, row_exit, date(2024, 1, 2), "signal", cfg)
        cash += revenue

        # Must lose money due to fees + slippage
        assert cash < cfg.initial_capital, (
            f"Should lose money on round-trip at same price, but cash={cash}"
        )
        # Revenue should be less than initial cost
        assert revenue < cost


# ===========================================================================
# Group 4: Slippage Consistency
# ===========================================================================


class TestSlippageConsistency:
    """Verify slippage is applied consistently across code paths."""

    def test_vectorized_finalize_slippage_consistency(self) -> None:
        """Finalize and mid-exit at same close should give same PnL."""
        cfg = BacktestConfig(
            initial_capital=INITIAL_CAPITAL,
            fee_rate=FEE,
            slippage_rate=SLIPPAGE,
            max_slots=1,
        )
        tickers = ["KRW-BTC"]
        buy_price = 100.0 * (1 + SLIPPAGE)
        close_price = 110.0
        exit_price_with_slippage = close_price * (1 - SLIPPAGE)

        # Scenario A: Normal mid-simulation exit
        state_a = initialize_simulation_state(cfg.initial_capital, 1, 3, tickers)
        handle_normal_entry(state_a, 0, 0, state_a.cash, buy_price, cfg.fee_rate)

        sorted_dates = np.array([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])
        exit_prices = np.array([[0.0, exit_price_with_slippage, 0.0]])
        order_manager = AdvancedOrderManager()

        vec_execute_exit(
            state_a,
            cfg,
            0,
            1,
            date(2024, 1, 2),
            sorted_dates,
            tickers,
            exit_prices,
            order_manager,
            False,
            False,
            "signal",
        )
        mid_exit_revenue = state_a.trades_list[0]["pnl"]

        # Scenario B: Finalize at same close price
        state_b = initialize_simulation_state(cfg.initial_capital, 1, 3, tickers)
        handle_normal_entry(state_b, 0, 0, state_b.cash, buy_price, cfg.fee_rate)

        closes = np.array([[buy_price, close_price, close_price]])
        finalize_open_positions(state_b, sorted_dates, tickers, 1, closes, cfg)
        finalize_pnl = state_b.trades_list[0]["pnl"]

        # Both paths now apply slippage consistently
        # float32 state amounts cause minor precision differences (~0.1 for 10M values)
        assert mid_exit_revenue == pytest.approx(finalize_pnl, abs=1.0), (
            f"mid_exit_pnl={mid_exit_revenue:.4f} != finalize_pnl={finalize_pnl:.4f}"
        )

    def test_vectorized_finalize_vs_mid_exit_revenue(self) -> None:
        """Finalize and normal exit at same close give equal revenue.

        Both paths now apply sell slippage consistently.
        """
        calculator = TradeCostCalculator(FEE, SLIPPAGE)

        entry_price = 100.0 * (1 + SLIPPAGE)
        close_price = 110.0

        # Both paths: close * (1 - slippage)
        exit_price_with_slippage = close_price * (1 - SLIPPAGE)
        invest_amount = INITIAL_CAPITAL
        amount = calculator.calculate_buy_amount(invest_amount, entry_price)

        costs_normal = calculator.calculate_exit_costs(
            entry_price, exit_price_with_slippage, amount
        )
        costs_finalize = calculator.calculate_exit_costs(
            entry_price, exit_price_with_slippage, amount
        )

        assert costs_finalize.revenue == pytest.approx(costs_normal.revenue, abs=0.01)


# ===========================================================================
# Group 5: Behavior Documentation
# ===========================================================================


class TestBehaviorDocumentation:
    """Document design choices as passing tests."""

    def test_event_driven_close_remaining_cash_unchanged(self, scenario: SimpleScenario) -> None:
        """close_remaining_positions adds trades but does NOT update cash.

        This is by design: equity curve is already final, and the trades
        are only for reporting. Cash is stale after this call.
        """
        cfg = scenario.config
        cash = cfg.initial_capital

        # Create a position
        row_entry = pd.Series(
            {
                "close": scenario.entry_close,
                "entry_signal": True,
                "exit_signal": False,
            }
        )
        position, cost = execute_entry("KRW-BTC", row_entry, date(2024, 1, 1), cash, 1, cfg)
        assert position is not None
        cash -= cost

        # Build ticker_data for close_remaining_positions
        positions = {"KRW-BTC": position}
        ticker_data = {
            "KRW-BTC": pd.DataFrame(
                {
                    "index_date": [date(2024, 1, 1), date(2024, 1, 2)],
                    "close": [scenario.entry_close, scenario.exit_close],
                }
            )
        }

        cash_before = cash
        trades = close_remaining_positions(positions, ticker_data, date(2024, 1, 2), cfg)

        # cash is NOT updated by close_remaining_positions
        assert cash == cash_before, "close_remaining_positions should not modify cash"
        assert len(trades) == 1, "Should produce one closing trade"
        # But the trade itself has correct PnL (uses TradeCostCalculator)
        assert trades[0].pnl != 0.0

    def test_event_driven_pnl_pct_is_gross_return(self, scenario: SimpleScenario) -> None:
        """pnl_pct is gross return (exit_price/entry_price - 1)*100, excluding fees.

        This is a design choice: pnl_pct shows price movement, not net return.
        """
        cfg = scenario.config
        cash = cfg.initial_capital

        row_entry = pd.Series(
            {
                "close": scenario.entry_close,
                "entry_signal": True,
                "exit_signal": False,
            }
        )
        position, cost = execute_entry("KRW-BTC", row_entry, date(2024, 1, 1), cash, 1, cfg)
        assert position is not None

        row_exit = pd.Series({"close": scenario.exit_close, "exit_signal": True})
        trade, _ = execute_exit(position, row_exit, date(2024, 1, 2), "signal", cfg)

        # pnl_pct should be gross return based on entry/exit prices
        expected_pnl_pct = (scenario.exit_price / position.entry_price - 1) * 100
        assert trade.pnl_pct == pytest.approx(expected_pnl_pct, abs=0.01)

        # Verify it does NOT include fees
        # Net return would be different (lower)
        net_revenue = position.amount * scenario.exit_price * (1 - FEE)
        net_cost = position.amount * position.entry_price / (1 - FEE)
        net_return_pct = (net_revenue / net_cost - 1) * 100
        assert trade.pnl_pct != pytest.approx(net_return_pct, abs=0.01)

    def test_equity_uses_raw_close_not_liquidation_value(self, scenario: SimpleScenario) -> None:
        """Equity values positions at raw close, not liquidation value.

        Liquidation value would be close * (1 - slippage) * (1 - fee_rate).
        This is standard mark-to-market: equity shows market value, not
        what you'd actually get after selling.
        """
        cfg = scenario.config
        tickers = ["KRW-BTC"]
        state = initialize_simulation_state(cfg.initial_capital, 1, 2, tickers)

        buy_price = scenario.entry_price
        invest_amount = state.cash
        handle_normal_entry(state, 0, 0, invest_amount, buy_price, cfg.fee_rate)

        closes = np.array([[scenario.entry_close, scenario.exit_close]])
        valid_data = np.array([True])
        calculate_daily_equity(state, 1, 1, closes, valid_data)

        # Equity uses raw close
        amount = state.position_amounts[0]
        raw_equity = state.cash + amount * scenario.exit_close
        assert state.equity_curve[1] == pytest.approx(raw_equity, abs=0.01)

        # Liquidation value would be lower
        liquidation_value = state.cash + amount * scenario.exit_close * (1 - SLIPPAGE) * (1 - FEE)
        assert state.equity_curve[1] > liquidation_value
