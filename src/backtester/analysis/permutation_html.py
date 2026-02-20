"""HTML report generation for Permutation Test."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtester.analysis.permutation_test import PermutationTestResult


def generate_permutation_html(result: PermutationTestResult) -> str:
    """
    Render a PermutationTestResult as an HTML string.

    Args:
        result: PermutationTestResult object

    Returns:
        HTML string
    """
    sig_class = "significant" if result.is_statistically_significant else "danger"
    decision = (
        "Strategy shows statistically significant signal."
        if result.is_statistically_significant
        else "Strategy performance is likely due to chance — overfitting suspected."
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Permutation Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #2196F3; color: white; }}
        .significant {{ color: #4CAF50; font-weight: bold; }}
        .warning {{ color: #FF9800; font-weight: bold; }}
        .danger {{ color: #F44336; font-weight: bold; }}
        .interpretation {{
            background-color: #f0f0f0;
            padding: 15px;
            margin: 20px 0;
            border-left: 4px solid #2196F3;
            font-size: 14px;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <h1>Permutation Test Report</h1>
    <h2>Statistical Significance Test</h2>

    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Original Return</td>
            <td>{result.original_return:.2%}</td>
        </tr>
        <tr>
            <td>Original Sharpe</td>
            <td>{result.original_sharpe:.2f}</td>
        </tr>
        <tr>
            <td>Original Win Rate</td>
            <td>{result.original_win_rate:.1%}</td>
        </tr>
        <tr>
            <td colspan="2"><b>Shuffled Data Statistics (n={len(result.shuffled_returns)})</b></td>
        </tr>
        <tr>
            <td>Mean Return</td>
            <td>{result.mean_shuffled_return:.2%}</td>
        </tr>
        <tr>
            <td>Std Dev Return</td>
            <td>{result.std_shuffled_return:.2%}</td>
        </tr>
        <tr>
            <td colspan="2"><b>Hypothesis Test</b></td>
        </tr>
        <tr>
            <td>Z-score</td>
            <td class="{sig_class}">{result.z_score:.2f}</td>
        </tr>
        <tr>
            <td>P-value</td>
            <td class="{sig_class}">{result.p_value:.4f}</td>
        </tr>
        <tr>
            <td>Significance Level</td>
            <td class="{sig_class}">{result.confidence_level}</td>
        </tr>
    </table>

    <h2>Interpretation</h2>
    <div class="interpretation">
        {result.interpretation}
    </div>

    <h2>Decision</h2>
    <div class="interpretation">
        {decision}
    </div>
</body>
</html>"""

    return html
