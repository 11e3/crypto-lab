"""Tests for metrics display component."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.web.services.metrics_calculator import ExtendedMetrics


def _make_metrics(
    *,
    total_return_pct: float = 45.23,
    cagr_pct: float = 18.50,
    max_drawdown_pct: float = -12.34,
    volatility_pct: float = 22.10,
    upside_volatility_pct: float = 14.50,
    downside_volatility_pct: float = 16.80,
    sharpe_ratio: float = 1.25,
    sortino_ratio: float = 1.85,
    calmar_ratio: float = 1.50,
    var_95_pct: float = -2.15,
    var_99_pct: float = -3.40,
    cvar_95_pct: float = -2.85,
    cvar_99_pct: float = -4.10,
    z_score: float = 2.35,
    p_value: float = 0.02,
    skewness: float = 0.45,
    kurtosis: float = 1.20,
    num_trades: int = 150,
    win_rate_pct: float = 58.5,
    avg_win_pct: float = 3.20,
    avg_loss_pct: float = -1.80,
    profit_factor: float = 1.95,
    expectancy: float = 0.85,
    trading_days: int = 365,
    years: float = 1.5,
) -> ExtendedMetrics:
    """Build an ExtendedMetrics with sensible defaults, overridable per field."""
    return ExtendedMetrics(
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        max_drawdown_pct=max_drawdown_pct,
        volatility_pct=volatility_pct,
        upside_volatility_pct=upside_volatility_pct,
        downside_volatility_pct=downside_volatility_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        var_95_pct=var_95_pct,
        var_99_pct=var_99_pct,
        cvar_95_pct=cvar_95_pct,
        cvar_99_pct=cvar_99_pct,
        z_score=z_score,
        p_value=p_value,
        skewness=skewness,
        kurtosis=kurtosis,
        num_trades=num_trades,
        win_rate_pct=win_rate_pct,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        profit_factor=profit_factor,
        expectancy=expectancy,
        trading_days=trading_days,
        years=years,
    )


def _make_mock_st() -> MagicMock:
    """Create a mock streamlit module with context-manager support.

    st.columns(n) returns a list of n MagicMock context managers.
    st.expander(...) returns a single MagicMock context manager.
    """
    mock_st = MagicMock()

    def _columns(n: int) -> list[MagicMock]:
        cols = []
        for _ in range(n):
            col = MagicMock()
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=False)
            cols.append(col)
        return cols

    mock_st.columns.side_effect = _columns

    expander = MagicMock()
    expander.__enter__ = MagicMock(return_value=expander)
    expander.__exit__ = MagicMock(return_value=False)
    mock_st.expander.return_value = expander

    return mock_st


# ---------------------------------------------------------------------------
# _format_value
# ---------------------------------------------------------------------------


class TestFormatValue:
    """Tests for _format_value helper."""

    def test_normal_float(self) -> None:
        from src.web.components.metrics.metrics_display import _format_value

        assert _format_value(12.345) == "12.35"

    def test_positive_infinity(self) -> None:
        from src.web.components.metrics.metrics_display import _format_value

        assert _format_value(float("inf")) == "\u221e"

    def test_negative_infinity(self) -> None:
        from src.web.components.metrics.metrics_display import _format_value

        assert _format_value(float("-inf")) == "-\u221e"

    def test_custom_suffix(self) -> None:
        from src.web.components.metrics.metrics_display import _format_value

        assert _format_value(55.0, "%") == "55.00%"

    def test_custom_precision(self) -> None:
        from src.web.components.metrics.metrics_display import _format_value

        assert _format_value(3.456, "", 1) == "3.5"

    def test_suffix_and_precision(self) -> None:
        from src.web.components.metrics.metrics_display import _format_value

        assert _format_value(7.891, " years", 1) == "7.9 years"

    def test_zero(self) -> None:
        from src.web.components.metrics.metrics_display import _format_value

        assert _format_value(0.0) == "0.00"

    def test_negative_value(self) -> None:
        from src.web.components.metrics.metrics_display import _format_value

        assert _format_value(-12.5, "%") == "-12.50%"

    def test_infinity_ignores_suffix(self) -> None:
        """Infinity formatting ignores suffix and precision."""
        from src.web.components.metrics.metrics_display import _format_value

        assert _format_value(float("inf"), "%", 4) == "\u221e"


# ---------------------------------------------------------------------------
# render_metrics_cards
# ---------------------------------------------------------------------------


class TestRenderMetricsCards:
    """Tests for render_metrics_cards."""

    @patch("src.web.components.metrics.metrics_display.st")
    def test_renders_without_error(self, mock_st: MagicMock) -> None:
        """All code paths execute without exceptions."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import render_metrics_cards

        render_metrics_cards(_make_metrics())

    @patch("src.web.components.metrics.metrics_display.st")
    def test_calls_subheader(self, mock_st: MagicMock) -> None:
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import render_metrics_cards

        render_metrics_cards(_make_metrics())

        mock_st.subheader.assert_called_once()

    @patch("src.web.components.metrics.metrics_display.st")
    def test_creates_column_layouts(self, mock_st: MagicMock) -> None:
        """Should create multiple 4-column layouts for cards."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import render_metrics_cards

        render_metrics_cards(_make_metrics())

        # 2 rows tier-1 + 4 rows tier-2 inside expander = 6 total calls
        assert mock_st.columns.call_count == 6

    @patch("src.web.components.metrics.metrics_display.st")
    def test_metric_calls(self, mock_st: MagicMock) -> None:
        """st.metric is called for every metric card."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import render_metrics_cards

        render_metrics_cards(_make_metrics())

        # Tier 1: 8 metrics + Tier 2: 16 metrics = 24 total
        assert mock_st.metric.call_count == 24

    @patch("src.web.components.metrics.metrics_display.st")
    def test_p_value_significant_icon(self, mock_st: MagicMock) -> None:
        """p < 0.05 should show checkmark icon."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import render_metrics_cards

        render_metrics_cards(_make_metrics(p_value=0.01))

        p_value_calls = [c for c in mock_st.metric.call_args_list if c[0][0] == "P-Value"]
        assert len(p_value_calls) == 1
        assert "\u2705" in p_value_calls[0][0][1]

    @patch("src.web.components.metrics.metrics_display.st")
    def test_p_value_weak_icon(self, mock_st: MagicMock) -> None:
        """0.05 <= p < 0.1 should show warning icon."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import render_metrics_cards

        render_metrics_cards(_make_metrics(p_value=0.07))

        p_value_calls = [c for c in mock_st.metric.call_args_list if c[0][0] == "P-Value"]
        assert len(p_value_calls) == 1
        assert "\u26a0\ufe0f" in p_value_calls[0][0][1]

    @patch("src.web.components.metrics.metrics_display.st")
    def test_p_value_not_significant_icon(self, mock_st: MagicMock) -> None:
        """p >= 0.1 should show x icon."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import render_metrics_cards

        render_metrics_cards(_make_metrics(p_value=0.50))

        p_value_calls = [c for c in mock_st.metric.call_args_list if c[0][0] == "P-Value"]
        assert len(p_value_calls) == 1
        assert "\u274c" in p_value_calls[0][0][1]

    @patch("src.web.components.metrics.metrics_display.st")
    def test_expander_created(self, mock_st: MagicMock) -> None:
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import render_metrics_cards

        render_metrics_cards(_make_metrics())

        mock_st.expander.assert_called_once()


# ---------------------------------------------------------------------------
# render_metrics_table
# ---------------------------------------------------------------------------


class TestRenderMetricsTable:
    """Tests for render_metrics_table."""

    @patch("src.web.components.metrics.metrics_display.st")
    def test_renders_without_error(self, mock_st: MagicMock) -> None:
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect

        from src.web.components.metrics.metrics_display import render_metrics_table

        render_metrics_table(_make_metrics())

    @patch("src.web.components.metrics.metrics_display.st")
    def test_calls_subheader(self, mock_st: MagicMock) -> None:
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect

        from src.web.components.metrics.metrics_display import render_metrics_table

        render_metrics_table(_make_metrics())

        mock_st.subheader.assert_called_once()

    @patch("src.web.components.metrics.metrics_display.st")
    def test_two_column_layout(self, mock_st: MagicMock) -> None:
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect

        from src.web.components.metrics.metrics_display import render_metrics_table

        render_metrics_table(_make_metrics())

        mock_st.columns.assert_called_once_with(2)

    @patch("src.web.components.metrics.metrics_display.st")
    def test_markdown_calls_for_categories(self, mock_st: MagicMock) -> None:
        """Should produce markdown for all 7 categories and their items."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect

        from src.web.components.metrics.metrics_display import render_metrics_table

        render_metrics_table(_make_metrics())

        # 7 categories with bold headers + divider lines + metric items
        assert mock_st.markdown.call_count > 0

    @patch("src.web.components.metrics.metrics_display.st")
    def test_contains_all_category_headers(self, mock_st: MagicMock) -> None:
        """All 7 category headers should appear in markdown calls."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect

        from src.web.components.metrics.metrics_display import render_metrics_table

        render_metrics_table(_make_metrics())

        all_markdown = " ".join(str(c[0][0]) for c in mock_st.markdown.call_args_list)
        assert "Return Metrics" in all_markdown
        assert "Risk Metrics" in all_markdown
        assert "Risk-Adjusted Returns" in all_markdown
        assert "VaR" in all_markdown
        assert "Statistical Analysis" in all_markdown
        assert "Trading Metrics" in all_markdown
        assert "Period Information" in all_markdown


# ---------------------------------------------------------------------------
# render_statistical_significance
# ---------------------------------------------------------------------------


class TestRenderStatisticalSignificance:
    """Tests for render_statistical_significance."""

    @patch("src.web.components.metrics.metrics_display.st")
    def test_highly_significant(self, mock_st: MagicMock) -> None:
        """p < 0.01 branch."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import (
            render_statistical_significance,
        )

        render_statistical_significance(_make_metrics(p_value=0.005))

        md_text = mock_st.markdown.call_args_list[0][0][0]
        assert "Highly Significant" in md_text
        assert "\u2705" in md_text

    @patch("src.web.components.metrics.metrics_display.st")
    def test_significant(self, mock_st: MagicMock) -> None:
        """0.01 <= p < 0.05 branch."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import (
            render_statistical_significance,
        )

        render_statistical_significance(_make_metrics(p_value=0.03))

        md_text = mock_st.markdown.call_args_list[0][0][0]
        assert "Significant (p < 0.05)" in md_text
        assert "\u2705" in md_text

    @patch("src.web.components.metrics.metrics_display.st")
    def test_weakly_significant(self, mock_st: MagicMock) -> None:
        """0.05 <= p < 0.1 branch."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import (
            render_statistical_significance,
        )

        render_statistical_significance(_make_metrics(p_value=0.08))

        md_text = mock_st.markdown.call_args_list[0][0][0]
        assert "Weakly Significant" in md_text
        assert "\u26a0\ufe0f" in md_text

    @patch("src.web.components.metrics.metrics_display.st")
    def test_not_significant(self, mock_st: MagicMock) -> None:
        """p >= 0.1 branch."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import (
            render_statistical_significance,
        )

        render_statistical_significance(_make_metrics(p_value=0.50))

        md_text = mock_st.markdown.call_args_list[0][0][0]
        assert "Not Significant" in md_text
        assert "\u274c" in md_text

    @patch("src.web.components.metrics.metrics_display.st")
    def test_positive_z_score(self, mock_st: MagicMock) -> None:
        """z > 0 should show 'Positive excess return'."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import (
            render_statistical_significance,
        )

        render_statistical_significance(_make_metrics(z_score=2.5))

        md_text = mock_st.markdown.call_args_list[0][0][0]
        assert "Positive excess return" in md_text

    @patch("src.web.components.metrics.metrics_display.st")
    def test_negative_z_score(self, mock_st: MagicMock) -> None:
        """z < 0 should show 'Negative excess return'."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import (
            render_statistical_significance,
        )

        render_statistical_significance(_make_metrics(z_score=-1.5))

        md_text = mock_st.markdown.call_args_list[0][0][0]
        assert "Negative excess return" in md_text

    @patch("src.web.components.metrics.metrics_display.st")
    def test_positive_skewness(self, mock_st: MagicMock) -> None:
        """skewness > 0 should show 'Right tail (positive)'."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import (
            render_statistical_significance,
        )

        render_statistical_significance(_make_metrics(skewness=0.8))

        md_text = mock_st.markdown.call_args_list[0][0][0]
        assert "Right tail (positive)" in md_text

    @patch("src.web.components.metrics.metrics_display.st")
    def test_negative_skewness(self, mock_st: MagicMock) -> None:
        """skewness < 0 should show 'Left tail (negative)'."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import (
            render_statistical_significance,
        )

        render_statistical_significance(_make_metrics(skewness=-0.6))

        md_text = mock_st.markdown.call_args_list[0][0][0]
        assert "Left tail (negative)" in md_text

    @patch("src.web.components.metrics.metrics_display.st")
    def test_positive_kurtosis(self, mock_st: MagicMock) -> None:
        """kurtosis > 0 should show 'Fat tail (increased risk)'."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import (
            render_statistical_significance,
        )

        render_statistical_significance(_make_metrics(kurtosis=1.5))

        md_text = mock_st.markdown.call_args_list[0][0][0]
        assert "Fat tail (increased risk)" in md_text

    @patch("src.web.components.metrics.metrics_display.st")
    def test_negative_kurtosis(self, mock_st: MagicMock) -> None:
        """kurtosis < 0 should show 'Thin tail'."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import (
            render_statistical_significance,
        )

        render_statistical_significance(_make_metrics(kurtosis=-0.3))

        md_text = mock_st.markdown.call_args_list[0][0][0]
        assert "Thin tail" in md_text

    @patch("src.web.components.metrics.metrics_display.st")
    def test_expander_with_guide(self, mock_st: MagicMock) -> None:
        """Interpretation guide expander should be created."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import (
            render_statistical_significance,
        )

        render_statistical_significance(_make_metrics())

        mock_st.expander.assert_called_once()

    @patch("src.web.components.metrics.metrics_display.st")
    def test_p_value_rejection_possible(self, mock_st: MagicMock) -> None:
        """p < 0.05 should say rejection 'possible'."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import (
            render_statistical_significance,
        )

        render_statistical_significance(_make_metrics(p_value=0.03))

        md_text = mock_st.markdown.call_args_list[0][0][0]
        assert "rejection possible" in md_text

    @patch("src.web.components.metrics.metrics_display.st")
    def test_p_value_rejection_not_possible(self, mock_st: MagicMock) -> None:
        """p >= 0.05 should say rejection 'not possible'."""
        mock_st.columns.side_effect = _make_mock_st().columns.side_effect
        expander = MagicMock()
        expander.__enter__ = MagicMock(return_value=expander)
        expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander

        from src.web.components.metrics.metrics_display import (
            render_statistical_significance,
        )

        render_statistical_significance(_make_metrics(p_value=0.50))

        md_text = mock_st.markdown.call_args_list[0][0][0]
        assert "rejection not possible" in md_text
