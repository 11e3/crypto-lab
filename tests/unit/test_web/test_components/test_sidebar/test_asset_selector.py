"""Tests for asset selector component."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.web.components.sidebar.asset_selector import (
    POPULAR_TICKERS,
    get_available_tickers,
    render_asset_selector,
)

# =========================================================================
# get_available_tickers
# =========================================================================


class TestGetAvailableTickers:
    """Tests for get_available_tickers."""

    def test_returns_tickers_from_files(self, tmp_path: Path) -> None:
        """Should extract tickers from parquet file names."""
        (tmp_path / "KRW-BTC_day.parquet").touch()
        (tmp_path / "KRW-ETH_day.parquet").touch()
        (tmp_path / "KRW-XRP_minute240.parquet").touch()

        with patch("src.config.RAW_DATA_DIR", tmp_path):
            result = get_available_tickers()

        assert "KRW-BTC" in result
        assert "KRW-ETH" in result
        assert "KRW-XRP" in result

    def test_fallback_to_popular_tickers(self, tmp_path: Path) -> None:
        """Should fall back to POPULAR_TICKERS when no data files exist."""
        with patch("src.config.RAW_DATA_DIR", tmp_path):
            result = get_available_tickers()

        assert result == POPULAR_TICKERS

    def test_fallback_on_os_error(self) -> None:
        """Should fall back to POPULAR_TICKERS on OS error."""
        bad_path = MagicMock(spec=Path)
        bad_path.glob.side_effect = OSError("permission denied")

        with patch("src.config.RAW_DATA_DIR", bad_path):
            result = get_available_tickers()
        assert result == POPULAR_TICKERS

    def test_popular_tickers_not_empty(self) -> None:
        """POPULAR_TICKERS constant should have reasonable entries."""
        assert len(POPULAR_TICKERS) > 5
        assert all(t.startswith("KRW-") for t in POPULAR_TICKERS)


# =========================================================================
# render_asset_selector
# =========================================================================


class TestRenderAssetSelector:
    """Tests for render_asset_selector."""

    @patch("src.web.components.sidebar.asset_selector.st")
    @patch("src.web.components.sidebar.asset_selector.get_available_tickers")
    def test_quick_select_preset(self, mock_get: MagicMock, mock_st: MagicMock) -> None:
        """Quick Select with a preset returns tickers."""
        mock_get.return_value = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE"]
        mock_st.radio.return_value = "Quick Select"
        mock_st.selectbox.return_value = "Top 3 (BTC, ETH, XRP)"
        mock_st.session_state = {}

        result = render_asset_selector()
        assert result == ["KRW-BTC", "KRW-ETH", "KRW-XRP"]

    @patch("src.web.components.sidebar.asset_selector.st")
    @patch("src.web.components.sidebar.asset_selector.get_available_tickers")
    def test_quick_select_custom(self, mock_get: MagicMock, mock_st: MagicMock) -> None:
        """Quick Select Custom uses multiselect."""
        mock_get.return_value = ["KRW-BTC", "KRW-ETH"]
        mock_st.radio.return_value = "Quick Select"
        mock_st.selectbox.return_value = "Custom"
        mock_st.multiselect.return_value = ["KRW-BTC"]
        mock_st.session_state = {}

        result = render_asset_selector()
        assert result == ["KRW-BTC"]

    @patch("src.web.components.sidebar.asset_selector.st")
    @patch("src.web.components.sidebar.asset_selector.get_available_tickers")
    def test_individual_select(self, mock_get: MagicMock, mock_st: MagicMock) -> None:
        """Individual Select with checkboxes."""
        mock_get.return_value = ["KRW-BTC", "KRW-ETH"]
        mock_st.radio.return_value = "Individual Select"
        mock_st.session_state = {}

        col_mock = MagicMock()
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [col_mock, col_mock, col_mock]

        mock_st.checkbox.side_effect = [True, False]

        result = render_asset_selector()
        assert "KRW-BTC" in result

    @patch("src.web.components.sidebar.asset_selector.st")
    @patch("src.web.components.sidebar.asset_selector.get_available_tickers")
    def test_empty_selection_shows_warning(self, mock_get: MagicMock, mock_st: MagicMock) -> None:
        """Empty selection should show warning."""
        mock_get.return_value = ["KRW-BTC"]
        mock_st.radio.return_value = "Quick Select"
        mock_st.selectbox.return_value = "Custom"
        mock_st.multiselect.return_value = []
        mock_st.session_state = {}

        result = render_asset_selector()
        assert result == []
        mock_st.warning.assert_called_once()
