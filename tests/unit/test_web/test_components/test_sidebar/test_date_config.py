"""Tests for date configuration component."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

# =========================================================================
# render_date_config
# =========================================================================


class TestRenderDateConfig:
    """Tests for render_date_config."""

    @patch("src.web.components.sidebar.date_config.st")
    @patch("src.web.components.sidebar.date_config.get_data_date_range")
    def test_uses_data_date_range(self, mock_range: MagicMock, mock_st: MagicMock) -> None:
        """Should use get_data_date_range for defaults."""
        from src.web.components.sidebar.date_config import render_date_config

        mock_range.return_value = (date(2020, 1, 1), date(2024, 12, 31))

        col_mock = MagicMock()
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [col_mock, col_mock]

        mock_st.date_input.side_effect = [date(2020, 1, 1), date(2024, 12, 31)]

        start, end = render_date_config()
        assert start == date(2020, 1, 1)
        assert end == date(2024, 12, 31)

    @patch("src.web.components.sidebar.date_config.st")
    @patch("src.web.components.sidebar.date_config.get_data_date_range")
    def test_fallback_when_no_data(self, mock_range: MagicMock, mock_st: MagicMock) -> None:
        """Should fall back to defaults when no data range available."""
        from src.web.components.sidebar.date_config import render_date_config

        mock_range.return_value = (None, None)

        col_mock = MagicMock()
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [col_mock, col_mock]

        mock_st.date_input.side_effect = [date(2018, 1, 1), date(2025, 1, 1)]

        start, end = render_date_config()
        assert start == date(2018, 1, 1)
        assert end == date(2025, 1, 1)

    @patch("src.web.components.sidebar.date_config.st")
    @patch("src.web.components.sidebar.date_config.get_data_date_range")
    def test_invalid_date_range_shows_error(
        self, mock_range: MagicMock, mock_st: MagicMock
    ) -> None:
        """Should show error when start >= end."""
        from src.web.components.sidebar.date_config import render_date_config

        mock_range.return_value = (date(2020, 1, 1), date(2024, 12, 31))

        col_mock = MagicMock()
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [col_mock, col_mock]

        # start >= end
        mock_st.date_input.side_effect = [date(2024, 6, 1), date(2024, 6, 1)]

        start, end = render_date_config()
        mock_st.error.assert_called_once()
        # Returns defaults
        assert start == date(2020, 1, 1)
        assert end == date(2024, 12, 31)
