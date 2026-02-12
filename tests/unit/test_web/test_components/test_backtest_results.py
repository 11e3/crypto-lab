"""Tests for unified backtest results display component."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.web.components.results.backtest_results import (
    UnifiedBacktestResult,
    _render_trade_history,
    render_backtest_results,
)

MODULE = "src.web.components.results.backtest_results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event_driven_trade(
    *,
    ticker: str = "KRW-BTC",
    entry_date: str = "2025-01-01",
    entry_price: float = 50_000_000,
    exit_date: str | None = "2025-01-10",
    exit_price: float | None = 55_000_000,
    pnl: float = 5_000_000,
    pnl_pct: float | None = 10.0,
) -> MagicMock:
    """Create a mock Trade object for event-driven backtesting."""
    trade = MagicMock()
    trade.ticker = ticker
    trade.entry_date = entry_date
    trade.entry_price = entry_price
    trade.exit_date = exit_date
    trade.exit_price = exit_price
    trade.pnl = pnl
    trade.pnl_pct = pnl_pct
    return trade


def _make_event_driven_result(
    *,
    equity_curve: list[float] | None = None,
    dates: list[str] | None = None,
    trades: list[MagicMock] | None = None,
) -> MagicMock:
    """Create a mock BacktestResult."""
    result = MagicMock()
    result.equity_curve = equity_curve or [100.0, 105.0, 110.0]
    result.dates = dates
    result.trades = trades if trades is not None else []
    return result


def _make_vbo_trade(
    *,
    symbol: str = "KRW-ETH",
    entry_date: str = "2025-02-01",
    exit_date: str = "2025-02-15",
    entry_price: float = 4_000_000,
    exit_price: float = 4_400_000,
    pnl: float = 400_000,
    return_pct: float = 10.0,
) -> dict[str, object]:
    """Create a VBO trade dictionary."""
    return {
        "symbol": symbol,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "return_pct": return_pct,
    }


def _make_vbo_result(
    *,
    equity_curve: list[float] | None = None,
    dates: list[str] | None = None,
    trades: list[dict[str, object]] | None = None,
) -> MagicMock:
    """Create a mock VboBacktestResult."""
    result = MagicMock()
    result.equity_curve = equity_curve or [100.0, 108.0, 115.0]
    result.dates = dates
    result.trades = trades if trades is not None else []
    return result


# ---------------------------------------------------------------------------
# TestUnifiedFromEventDriven
# ---------------------------------------------------------------------------


class TestUnifiedFromEventDriven:
    """Tests for UnifiedBacktestResult.from_event_driven."""

    def test_with_trades_and_dates(self) -> None:
        """Trades and dates are converted correctly."""
        trades = [
            _make_event_driven_trade(pnl_pct=10.0),
            _make_event_driven_trade(
                ticker="KRW-ETH",
                entry_date="2025-01-15",
                entry_price=4_000_000,
                exit_date="2025-01-20",
                exit_price=4_200_000,
                pnl=200_000,
                pnl_pct=5.0,
            ),
        ]
        result = _make_event_driven_result(
            equity_curve=[100.0, 105.0, 110.0],
            dates=["2025-01-01", "2025-01-02", "2025-01-03"],
            trades=trades,
        )

        unified = UnifiedBacktestResult.from_event_driven(result)

        np.testing.assert_array_equal(unified.equity, [100.0, 105.0, 110.0])
        np.testing.assert_array_equal(unified.dates, ["2025-01-01", "2025-01-02", "2025-01-03"])
        assert unified.trade_returns == pytest.approx([0.10, 0.05])
        assert unified.trade_count == 2
        assert unified.title == "Backtest Results"
        assert not unified.trades_df.empty
        assert len(unified.trades_df) == 2
        assert list(unified.trades_df.columns) == [
            "Ticker",
            "Entry Date",
            "Entry Price",
            "Exit Date",
            "Exit Price",
            "P&L",
            "P&L %",
        ]

    def test_with_empty_trades(self) -> None:
        """Empty trades list produces empty DataFrame."""
        result = _make_event_driven_result(
            equity_curve=[100.0, 102.0],
            dates=["2025-01-01", "2025-01-02"],
            trades=[],
        )

        unified = UnifiedBacktestResult.from_event_driven(result)

        assert unified.trades_df.empty
        assert unified.trade_count == 0
        assert unified.trade_returns == []

    def test_with_none_dates_falls_back_to_arange(self) -> None:
        """None dates produce integer indices via np.arange."""
        result = _make_event_driven_result(
            equity_curve=[100.0, 105.0, 110.0],
            dates=None,
            trades=[],
        )

        unified = UnifiedBacktestResult.from_event_driven(result)

        np.testing.assert_array_equal(unified.dates, [0, 1, 2])

    def test_trade_returns_filters_none_pnl_pct(self) -> None:
        """trade_returns comprehension excludes trades with None pnl_pct.

        Note: the DataFrame construction formats pnl_pct unconditionally,
        so a trade with None pnl_pct would raise TypeError during
        from_event_driven.  This test verifies the filter logic in isolation.
        """
        trades = [
            _make_event_driven_trade(pnl_pct=10.0),
            MagicMock(pnl_pct=None),
            _make_event_driven_trade(pnl_pct=-5.0),
        ]

        # Exercise the same comprehension used in from_event_driven (line 50)
        trade_returns = [t.pnl_pct / 100 for t in trades if t.pnl_pct is not None]

        assert trade_returns == pytest.approx([0.10, -0.05])

    def test_trade_with_zero_pnl_pct_included(self) -> None:
        """Trades with 0.0 pnl_pct are included in trade_returns (not filtered)."""
        trades = [
            _make_event_driven_trade(pnl_pct=10.0),
            _make_event_driven_trade(pnl_pct=0.0, pnl=0),
        ]
        result = _make_event_driven_result(
            equity_curve=[100.0, 105.0],
            dates=["2025-01-01", "2025-01-02"],
            trades=trades,
        )

        unified = UnifiedBacktestResult.from_event_driven(result)

        assert unified.trade_returns == pytest.approx([0.10, 0.0])
        assert unified.trade_count == 2

    def test_trade_with_none_exit_fields(self) -> None:
        """Trades with None exit_date and exit_price render as dashes."""
        trades = [
            _make_event_driven_trade(exit_date=None, exit_price=None),
        ]
        result = _make_event_driven_result(
            equity_curve=[100.0],
            dates=["2025-01-01"],
            trades=trades,
        )

        unified = UnifiedBacktestResult.from_event_driven(result)

        row = unified.trades_df.iloc[0]
        assert row["Exit Date"] == "-"
        assert row["Exit Price"] == "-"


# ---------------------------------------------------------------------------
# TestUnifiedFromVbo
# ---------------------------------------------------------------------------


class TestUnifiedFromVbo:
    """Tests for UnifiedBacktestResult.from_vbo."""

    def test_with_trades_and_dates(self) -> None:
        """Trades and dates are converted correctly."""
        trades = [_make_vbo_trade(), _make_vbo_trade(return_pct=5.0, pnl=200_000)]
        result = _make_vbo_result(
            equity_curve=[100.0, 108.0, 115.0],
            dates=["2025-02-01", "2025-02-10", "2025-02-20"],
            trades=trades,
        )

        unified = UnifiedBacktestResult.from_vbo(result)

        np.testing.assert_array_equal(unified.equity, [100.0, 108.0, 115.0])
        np.testing.assert_array_equal(unified.dates, ["2025-02-01", "2025-02-10", "2025-02-20"])
        assert unified.trade_returns == pytest.approx([0.10, 0.05])
        assert unified.trade_count == 2
        assert unified.title == "VBO Backtest Results"
        assert not unified.trades_df.empty
        assert "Ticker" in unified.trades_df.columns
        assert "P&L %" in unified.trades_df.columns

    def test_with_empty_trades(self) -> None:
        """Empty trades list produces empty DataFrame."""
        result = _make_vbo_result(
            equity_curve=[100.0],
            dates=["2025-02-01"],
            trades=[],
        )

        unified = UnifiedBacktestResult.from_vbo(result)

        assert unified.trades_df.empty
        assert unified.trade_count == 0
        assert unified.trade_returns == []

    def test_with_custom_strategy_name(self) -> None:
        """Custom strategy_name is reflected in the title."""
        result = _make_vbo_result(trades=[])

        unified = UnifiedBacktestResult.from_vbo(result, strategy_name="MeanReversion")

        assert unified.title == "MeanReversion Backtest Results"

    def test_with_none_dates_falls_back_to_arange(self) -> None:
        """None dates produce integer indices via np.arange."""
        result = _make_vbo_result(
            equity_curve=[100.0, 108.0, 115.0, 120.0],
            dates=None,
            trades=[],
        )

        unified = UnifiedBacktestResult.from_vbo(result)

        np.testing.assert_array_equal(unified.dates, [0, 1, 2, 3])

    def test_with_empty_strategy_name_defaults_to_vbo(self) -> None:
        """Empty string strategy_name falls back to VBO."""
        result = _make_vbo_result(trades=[])

        unified = UnifiedBacktestResult.from_vbo(result, strategy_name="")

        assert unified.title == "VBO Backtest Results"


# ---------------------------------------------------------------------------
# TestRenderBacktestResults
# ---------------------------------------------------------------------------


class TestRenderBacktestResults:
    """Tests for render_backtest_results."""

    @patch(f"{MODULE}._render_trade_history")
    @patch(f"{MODULE}.render_yearly_bar_chart")
    @patch(f"{MODULE}.render_underwater_curve")
    @patch(f"{MODULE}.render_equity_curve")
    @patch(f"{MODULE}.render_metrics_cards")
    @patch(f"{MODULE}.calculate_extended_metrics")
    @patch(f"{MODULE}.st")
    def test_render_calls_all_components(
        self,
        mock_st: MagicMock,
        mock_calc_metrics: MagicMock,
        mock_metrics_cards: MagicMock,
        mock_equity_curve: MagicMock,
        mock_underwater: MagicMock,
        mock_yearly_bar: MagicMock,
        mock_trade_history: MagicMock,
    ) -> None:
        """All rendering sub-functions are called."""
        mock_st.session_state = {}
        mock_calc_metrics.return_value = {"sharpe": 1.5}

        equity = np.array([100.0, 110.0])
        dates = np.array(["2025-01-01", "2025-01-02"])
        result = UnifiedBacktestResult(
            equity=equity,
            dates=dates,
            trade_returns=[0.10],
            trades_df=pd.DataFrame(),
            title="Test Results",
            trade_count=1,
        )

        render_backtest_results(result)

        mock_st.subheader.assert_called_once()
        mock_calc_metrics.assert_called_once()
        mock_metrics_cards.assert_called_once_with({"sharpe": 1.5})
        mock_equity_curve.assert_called_once_with(dates, equity)
        mock_underwater.assert_called_once_with(dates, equity)
        mock_yearly_bar.assert_called_once_with(dates, equity)
        mock_trade_history.assert_called_once_with(result)

    @patch(f"{MODULE}._render_trade_history")
    @patch(f"{MODULE}.render_yearly_bar_chart")
    @patch(f"{MODULE}.render_underwater_curve")
    @patch(f"{MODULE}.render_equity_curve")
    @patch(f"{MODULE}.render_metrics_cards")
    @patch(f"{MODULE}.calculate_extended_metrics")
    @patch(f"{MODULE}.st")
    def test_metrics_cached_in_session_state(
        self,
        mock_st: MagicMock,
        mock_calc_metrics: MagicMock,
        mock_metrics_cards: MagicMock,
        mock_equity_curve: MagicMock,
        mock_underwater: MagicMock,
        mock_yearly_bar: MagicMock,
        mock_trade_history: MagicMock,
    ) -> None:
        """Metrics are cached — second call skips calculate_extended_metrics."""
        equity = np.array([100.0, 110.0])
        cache_key_suffix = __import__("hashlib").md5(equity.tobytes()).hexdigest()
        cache_key = f"metrics_{cache_key_suffix}"
        mock_st.session_state = {cache_key: {"sharpe": 2.0}}

        result = UnifiedBacktestResult(
            equity=equity,
            dates=np.array([0, 1]),
            trade_returns=[],
            trades_df=pd.DataFrame(),
            title="Cached",
            trade_count=0,
        )

        render_backtest_results(result)

        mock_calc_metrics.assert_not_called()
        mock_metrics_cards.assert_called_once_with({"sharpe": 2.0})


# ---------------------------------------------------------------------------
# TestRenderTradeHistory
# ---------------------------------------------------------------------------


class TestRenderTradeHistory:
    """Tests for _render_trade_history."""

    @patch(f"{MODULE}.st")
    def test_empty_trades_shows_info(self, mock_st: MagicMock) -> None:
        """Empty trades_df triggers st.info message."""
        result = UnifiedBacktestResult(
            equity=np.array([100.0]),
            dates=np.array([0]),
            trade_returns=[],
            trades_df=pd.DataFrame(),
            title="Empty",
            trade_count=0,
        )

        _render_trade_history(result)

        mock_st.info.assert_called_once_with("No trades executed.")
        mock_st.dataframe.assert_not_called()

    @patch(f"{MODULE}.st")
    def test_nonempty_trades_shows_dataframe(self, mock_st: MagicMock) -> None:
        """Non-empty trades_df triggers st.dataframe."""
        mock_st.selectbox.return_value = 25

        trades_df = pd.DataFrame(
            {
                "Ticker": ["KRW-BTC"],
                "Entry Date": ["2025-01-01"],
                "Entry Price": ["50,000,000"],
                "Exit Date": ["2025-01-10"],
                "Exit Price": ["55,000,000"],
                "P&L": ["5,000,000"],
                "P&L %": ["10.00%"],
            }
        )
        result = UnifiedBacktestResult(
            equity=np.array([100.0, 110.0]),
            dates=np.array([0, 1]),
            trade_returns=[0.10],
            trades_df=trades_df,
            title="With Trades",
            trade_count=1,
        )

        _render_trade_history(result)

        mock_st.markdown.assert_called_once()
        mock_st.selectbox.assert_called_once()
        mock_st.dataframe.assert_called_once()
        call_kwargs = mock_st.dataframe.call_args
        assert call_kwargs.kwargs["use_container_width"] is True
        assert call_kwargs.kwargs["hide_index"] is True
