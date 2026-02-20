"""
Tests for Permutation Test HTML report generation.
"""

from unittest.mock import MagicMock

import pytest

from src.backtester.analysis.permutation_html import generate_permutation_html


@pytest.fixture
def mock_significant_result() -> MagicMock:
    """Create mock PermutationTestResult with significant results."""
    result = MagicMock()
    result.original_return = 0.25
    result.original_sharpe = 1.5
    result.original_win_rate = 0.60
    result.shuffled_returns = [0.05, -0.02, 0.03, 0.01, -0.01] * 20  # 100 items
    result.mean_shuffled_return = 0.012
    result.std_shuffled_return = 0.025
    result.z_score = 9.52
    result.p_value = 0.0001
    result.confidence_level = "99%"
    result.is_statistically_significant = True
    result.interpretation = "The strategy shows statistically significant alpha."
    return result


@pytest.fixture
def mock_insignificant_result() -> MagicMock:
    """Create mock PermutationTestResult with insignificant results."""
    result = MagicMock()
    result.original_return = 0.02
    result.original_sharpe = 0.3
    result.original_win_rate = 0.52
    result.shuffled_returns = [0.01, 0.02, 0.03] * 10
    result.mean_shuffled_return = 0.02
    result.std_shuffled_return = 0.01
    result.z_score = 0.5
    result.p_value = 0.35
    result.confidence_level = "Not Significant"
    result.is_statistically_significant = False
    result.interpretation = "The strategy performance may be due to random chance."
    return result


class TestGeneratePermutationHtml:
    """Tests for generate_permutation_html function."""

    def test_basic_html_structure(self, mock_significant_result: MagicMock) -> None:
        """Test that HTML has correct basic structure."""
        html = generate_permutation_html(mock_significant_result)

        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html
        assert "<title>Permutation Test Report</title>" in html

    def test_original_metrics_section(self, mock_significant_result: MagicMock) -> None:
        """Test that original metrics are included."""
        html = generate_permutation_html(mock_significant_result)

        assert "Original Return" in html
        assert "Original Sharpe" in html
        assert "Original Win Rate" in html
        assert "25.00%" in html  # original_return
        assert "1.50" in html  # original_sharpe
        assert "60.0%" in html  # original_win_rate

    def test_shuffled_statistics_section(self, mock_significant_result: MagicMock) -> None:
        """Test that shuffled data statistics are included."""
        html = generate_permutation_html(mock_significant_result)

        assert "Shuffled Data Statistics" in html
        assert "Mean Return" in html
        assert "Std Dev Return" in html

    def test_hypothesis_test_section(self, mock_significant_result: MagicMock) -> None:
        """Test that hypothesis test results are included."""
        html = generate_permutation_html(mock_significant_result)

        assert "Hypothesis Test" in html
        assert "Z-score" in html
        assert "P-value" in html
        assert "Significance Level" in html

    def test_significant_result_styling(self, mock_significant_result: MagicMock) -> None:
        """Test styling for significant results."""
        html = generate_permutation_html(mock_significant_result)

        assert 'class="significant"' in html

    def test_insignificant_result_styling(self, mock_insignificant_result: MagicMock) -> None:
        """Test styling for insignificant results."""
        html = generate_permutation_html(mock_insignificant_result)

        assert 'class="danger"' in html

    def test_significant_decision(self, mock_significant_result: MagicMock) -> None:
        """Test decision text for significant results."""
        html = generate_permutation_html(mock_significant_result)

        assert "Strategy shows statistically significant signal." in html

    def test_insignificant_decision(self, mock_insignificant_result: MagicMock) -> None:
        """Test decision text for insignificant results."""
        html = generate_permutation_html(mock_insignificant_result)

        assert "Strategy performance is likely due to chance" in html

    def test_interpretation_section(self, mock_significant_result: MagicMock) -> None:
        """Test that interpretation is included."""
        html = generate_permutation_html(mock_significant_result)

        assert "Interpretation" in html
        assert mock_significant_result.interpretation in html

    def test_sample_count(self, mock_significant_result: MagicMock) -> None:
        """Test that sample count is shown correctly."""
        html = generate_permutation_html(mock_significant_result)

        # n=100 (len of shuffled_returns)
        assert "n=100" in html

    def test_css_styles_included(self, mock_significant_result: MagicMock) -> None:
        """Test that CSS styles are included."""
        html = generate_permutation_html(mock_significant_result)

        assert "<style>" in html
        assert ".significant" in html
        assert ".warning" in html
        assert ".danger" in html
        assert ".interpretation" in html

    def test_z_score_formatting(self, mock_significant_result: MagicMock) -> None:
        """Test Z-score is properly formatted."""
        html = generate_permutation_html(mock_significant_result)

        assert "9.52" in html

    def test_p_value_formatting(self, mock_significant_result: MagicMock) -> None:
        """Test P-value is properly formatted."""
        html = generate_permutation_html(mock_significant_result)

        assert "0.0001" in html
