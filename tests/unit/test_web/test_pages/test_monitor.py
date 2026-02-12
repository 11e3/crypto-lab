"""Tests for the bot monitor page.

Covers GCS availability checks, account maps, PnL data preparation,
PnL summary calculation, positions card rendering, trade history rendering,
and the top-level monitor page entry point.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.web.pages.monitor import (
    _ACCOUNT_DISPLAY_MAP,
    _ACCOUNT_REVERSE_MAP,
    _calculate_pnl_summary,
    _check_gcs_availability,
    _prepare_pnl_data,
    _render_positions_card,
    _render_trade_history,
    render_monitor_page,
)

# ---------------------------------------------------------------------------
# _check_gcs_availability
# ---------------------------------------------------------------------------


class TestCheckGcsAvailability:
    """Test GCS availability check with import mocking."""

    @patch("src.web.pages.monitor.is_gcs_available", create=True)
    def test_returns_true_when_available(self, mock_is_available: MagicMock) -> None:
        """When is_gcs_available() returns True, _check_gcs_availability returns True."""
        with patch.dict(
            "sys.modules",
            {"src.data.storage": MagicMock(is_gcs_available=lambda: True)},
        ):
            result = _check_gcs_availability()
        assert result is True

    def test_returns_false_on_import_error(self) -> None:
        """When src.data.storage cannot be imported, returns False."""
        with patch.dict("sys.modules", {"src.data.storage": None}):
            result = _check_gcs_availability()
        assert result is False


# ---------------------------------------------------------------------------
# Account maps
# ---------------------------------------------------------------------------


class TestAccountMaps:
    """Verify _ACCOUNT_DISPLAY_MAP and _ACCOUNT_REVERSE_MAP are consistent."""

    def test_reverse_map_is_inverse(self) -> None:
        """Each display->account mapping has a matching account->display entry."""
        for display_name, account_name in _ACCOUNT_DISPLAY_MAP.items():
            assert _ACCOUNT_REVERSE_MAP[account_name] == display_name

    def test_same_length(self) -> None:
        """Both maps contain the same number of entries (no collisions)."""
        assert len(_ACCOUNT_DISPLAY_MAP) == len(_ACCOUNT_REVERSE_MAP)

    def test_maps_are_not_empty(self) -> None:
        """Maps should contain at least one entry."""
        assert len(_ACCOUNT_DISPLAY_MAP) > 0
        assert len(_ACCOUNT_REVERSE_MAP) > 0

    def test_round_trip(self) -> None:
        """display -> account -> display round trip preserves the key."""
        for display_name, account_name in _ACCOUNT_DISPLAY_MAP.items():
            assert _ACCOUNT_REVERSE_MAP[account_name] == display_name
            assert _ACCOUNT_DISPLAY_MAP[display_name] == account_name


# ---------------------------------------------------------------------------
# _prepare_pnl_data
# ---------------------------------------------------------------------------


class TestPreparePnlData:
    """Test the pure PnL data enrichment function."""

    def test_basic_three_days(self) -> None:
        """Three days of known PnL values produce correct equity and return."""
        df = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "pnl": [100_000, -50_000, 200_000],
                "trades": [3, 1, 2],
            }
        )
        initial_capital = 10_000_000.0

        result_df, equity, return_pct, total_trades, avg_daily = _prepare_pnl_data(
            df, initial_capital
        )

        # Cumulative PnL: 100k, 50k, 250k
        assert equity == pytest.approx(10_250_000.0)
        assert return_pct == pytest.approx(2.5)
        assert total_trades == 6

        # All 3 days have trades > 0
        # Daily returns: 100k/10M*100=1.0%, -50k/10M*100=-0.5%, 200k/10M*100=2.0%
        # Average = (1.0 - 0.5 + 2.0) / 3
        assert avg_daily == pytest.approx((1.0 - 0.5 + 2.0) / 3)

        # Enriched columns exist
        assert "cumulative_pnl" in result_df.columns
        assert "equity" in result_df.columns
        assert "return_pct" in result_df.columns

    def test_no_trading_days(self) -> None:
        """When all days have zero trades, avg_daily_return is 0."""
        df = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-02"],
                "pnl": [0, 0],
                "trades": [0, 0],
            }
        )
        initial_capital = 10_000_000.0

        _, equity, return_pct, total_trades, avg_daily = _prepare_pnl_data(df, initial_capital)

        assert equity == pytest.approx(initial_capital)
        assert return_pct == pytest.approx(0.0)
        assert total_trades == 0
        assert avg_daily == pytest.approx(0.0)

    def test_single_day(self) -> None:
        """Single-day input returns correct values."""
        df = pd.DataFrame(
            {
                "date": ["2025-06-15"],
                "pnl": [500_000],
                "trades": [5],
            }
        )
        initial_capital = 10_000_000.0

        _, equity, return_pct, total_trades, avg_daily = _prepare_pnl_data(df, initial_capital)

        assert equity == pytest.approx(10_500_000.0)
        assert return_pct == pytest.approx(5.0)
        assert total_trades == 5
        assert avg_daily == pytest.approx(5.0)

    def test_unsorted_dates_are_sorted(self) -> None:
        """Input with out-of-order dates is sorted by date."""
        df = pd.DataFrame(
            {
                "date": ["2025-01-03", "2025-01-01", "2025-01-02"],
                "pnl": [300, 100, 200],
                "trades": [1, 1, 1],
            }
        )
        result_df, *_ = _prepare_pnl_data(df, 10_000.0)

        assert list(result_df["date"]) == ["2025-01-01", "2025-01-02", "2025-01-03"]
        # After sorting: cumulative = 100, 300, 600
        assert list(result_df["cumulative_pnl"]) == [100, 300, 600]


# ---------------------------------------------------------------------------
# _calculate_pnl_summary
# ---------------------------------------------------------------------------


class TestCalculatePnlSummary:
    """Test PnL summary calculation with mocked storage."""

    def test_with_profit_krw_column(self) -> None:
        """Storage returns trades with profit_krw column."""
        mock_storage = MagicMock()
        trades_df = pd.DataFrame({"profit_krw": [100, 200, -50], "symbol": ["BTC", "ETH", "XRP"]})
        mock_storage.get_bot_logs.return_value = trades_df

        result = _calculate_pnl_summary(mock_storage, "sh", days=3)

        assert not result.empty
        assert len(result) == 3
        assert "pnl" in result.columns
        assert "trades" in result.columns
        assert "date" in result.columns
        # Each day gets the same trades_df => pnl = 250 per day
        assert all(result["pnl"] == 250)
        assert all(result["trades"] == 3)

    def test_with_pnl_column(self) -> None:
        """Storage returns trades with pnl column (fallback)."""
        mock_storage = MagicMock()
        trades_df = pd.DataFrame({"pnl": [1000, -500], "side": ["sell", "sell"]})
        mock_storage.get_bot_logs.return_value = trades_df

        result = _calculate_pnl_summary(mock_storage, "jh", days=2)

        assert len(result) == 2
        assert all(result["pnl"] == 500)

    def test_empty_trades(self) -> None:
        """Storage returns empty DataFrame for all days."""
        mock_storage = MagicMock()
        mock_storage.get_bot_logs.return_value = pd.DataFrame()

        result = _calculate_pnl_summary(mock_storage, "sh", days=5)

        assert len(result) == 5
        assert all(result["pnl"] == 0)
        assert all(result["trades"] == 0)

    def test_exception_during_get_bot_logs(self) -> None:
        """When get_bot_logs raises, those days are skipped."""
        mock_storage = MagicMock()
        mock_storage.get_bot_logs.side_effect = RuntimeError("GCS timeout")

        result = _calculate_pnl_summary(mock_storage, "sh", days=3)

        # All days failed -> empty DataFrame
        assert result.empty

    def test_no_pnl_column(self) -> None:
        """Trades DataFrame has no profit_krw or pnl column => pnl defaults to 0."""
        mock_storage = MagicMock()
        trades_df = pd.DataFrame({"symbol": ["BTC"], "side": ["buy"]})
        mock_storage.get_bot_logs.return_value = trades_df

        result = _calculate_pnl_summary(mock_storage, "sh", days=1)

        assert len(result) == 1
        assert result.iloc[0]["pnl"] == 0
        assert result.iloc[0]["trades"] == 1


# ---------------------------------------------------------------------------
# _render_positions_card
# ---------------------------------------------------------------------------


class TestRenderPositionsCard:
    """Test positions card rendering with mocked streamlit."""

    @patch("src.web.pages.monitor.st")
    def test_empty_positions(self, mock_st: MagicMock) -> None:
        """Empty dict shows 'No open positions' info."""
        _render_positions_card({})

        mock_st.subheader.assert_called_once_with("Current Positions")
        mock_st.info.assert_called_once_with("No open positions")

    @patch("src.web.pages.monitor.st")
    def test_dict_with_positions_key(self, mock_st: MagicMock) -> None:
        """Dict containing a 'positions' key uses that list."""
        positions = {
            "positions": [
                {"symbol": "KRW-BTC", "amount": 0.001, "unrealized_pnl": 5000},
                {"symbol": "KRW-ETH", "amount": 0.1, "unrealized_pnl": -2000},
            ]
        }
        _render_positions_card(positions)

        mock_st.subheader.assert_called_once()
        mock_st.dataframe.assert_called_once()
        # Total unrealized PnL = 5000 - 2000 = 3000 >= 0 -> green
        mock_st.markdown.assert_called_once()
        call_arg = mock_st.markdown.call_args[0][0]
        assert "3,000" in call_arg
        assert "green" in call_arg

    @patch("src.web.pages.monitor.st")
    def test_dict_without_positions_key(self, mock_st: MagicMock) -> None:
        """Dict without 'positions' key treats keys as symbols."""
        positions = {
            "KRW-BTC": {"amount": 0.5, "avg_price": 50_000_000},
            "KRW-ETH": {"amount": 2.0, "avg_price": 3_000_000},
        }
        _render_positions_card(positions)

        mock_st.subheader.assert_called_once()
        mock_st.dataframe.assert_called_once()

    @patch("src.web.pages.monitor.st")
    def test_positions_with_unrealized_pnl_negative(self, mock_st: MagicMock) -> None:
        """Negative total unrealized PnL renders in red."""
        positions = {
            "positions": [
                {"symbol": "KRW-BTC", "unrealized_pnl": -10000},
            ]
        }
        _render_positions_card(positions)

        call_arg = mock_st.markdown.call_args[0][0]
        assert "red" in call_arg

    @patch("src.web.pages.monitor.st")
    def test_empty_positions_list_in_dict(self, mock_st: MagicMock) -> None:
        """Dict with 'positions' key but empty list shows info message."""
        _render_positions_card({"positions": []})

        mock_st.info.assert_called_with("No open positions")

    @patch("src.web.pages.monitor.st")
    def test_dict_with_scalar_values(self, mock_st: MagicMock) -> None:
        """Dict values that are not dicts are wrapped as {'symbol': k, 'amount': v}."""
        positions = {"KRW-BTC": 0.5, "KRW-ETH": 2.0}
        _render_positions_card(positions)

        mock_st.dataframe.assert_called_once()


# ---------------------------------------------------------------------------
# _render_trade_history
# ---------------------------------------------------------------------------


class TestRenderTradeHistory:
    """Test trade history rendering with mocked streamlit."""

    @patch("src.web.pages.monitor.st")
    def test_empty_dataframe(self, mock_st: MagicMock) -> None:
        """Empty DataFrame shows info message."""
        _render_trade_history(pd.DataFrame())

        mock_st.subheader.assert_called_once_with("Trade History")
        mock_st.info.assert_called_once_with("No trades found for selected date")

    @patch("src.web.pages.monitor.st")
    def test_with_price_amount_pnl(self, mock_st: MagicMock) -> None:
        """DataFrame with price/amount/pnl columns gets formatted."""
        df = pd.DataFrame(
            {
                "symbol": ["KRW-BTC", "KRW-ETH"],
                "price": [50_000_000.0, 3_000_000.0],
                "amount": [0.001234, 0.567890],
                "pnl": [15000.0, float("nan")],
            }
        )
        _render_trade_history(df)

        mock_st.subheader.assert_called_once()
        mock_st.dataframe.assert_called_once()

        # Verify the displayed DataFrame has formatted columns
        displayed_df = mock_st.dataframe.call_args[0][0]
        assert displayed_df["price"].iloc[0] == "50,000,000"
        assert displayed_df["amount"].iloc[0] == "0.001234"
        assert displayed_df["pnl"].iloc[0] == "15,000"
        assert displayed_df["pnl"].iloc[1] == "-"

    @patch("src.web.pages.monitor.st")
    def test_without_optional_columns(self, mock_st: MagicMock) -> None:
        """DataFrame without price/amount/pnl columns is displayed as-is."""
        df = pd.DataFrame({"symbol": ["KRW-BTC"], "side": ["buy"], "timestamp": ["2025-01-01"]})
        _render_trade_history(df)

        mock_st.dataframe.assert_called_once()
        displayed_df = mock_st.dataframe.call_args[0][0]
        assert "side" in displayed_df.columns


# ---------------------------------------------------------------------------
# render_monitor_page
# ---------------------------------------------------------------------------


class TestRenderMonitorPage:
    """Test the top-level monitor page entry point."""

    @patch("src.web.pages.monitor._render_gcs_not_configured")
    @patch("src.web.pages.monitor._check_gcs_availability", return_value=False)
    @patch("src.web.pages.monitor.st")
    def test_gcs_not_available(
        self,
        mock_st: MagicMock,
        mock_check: MagicMock,
        mock_render_not_configured: MagicMock,
    ) -> None:
        """When GCS is not available, renders not-configured message and returns."""
        render_monitor_page()

        mock_st.header.assert_called_once_with("Bot Monitor")
        mock_check.assert_called_once()
        mock_render_not_configured.assert_called_once()

    @patch("src.web.pages.monitor._render_gcs_not_configured")
    @patch("src.web.pages.monitor._get_storage", return_value=None)
    @patch("src.web.pages.monitor._check_gcs_availability", return_value=True)
    @patch("src.web.pages.monitor.st")
    def test_storage_none(
        self,
        mock_st: MagicMock,
        mock_check: MagicMock,
        mock_get_storage: MagicMock,
        mock_render_not_configured: MagicMock,
    ) -> None:
        """When storage is None, renders not-configured message."""
        render_monitor_page()

        mock_render_not_configured.assert_called_once()
