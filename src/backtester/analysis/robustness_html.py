"""HTML report generation for Robustness Analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtester.analysis.robustness_models import RobustnessReport


def generate_robustness_html(report: RobustnessReport) -> str:
    """
    Render a RobustnessReport as an HTML string.

    Args:
        report: RobustnessReport object

    Returns:
        HTML string
    """
    html_parts: list[str] = []

    html_parts.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Robustness Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #2196F3; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .success { color: #4CAF50; font-weight: bold; }
        .warning { color: #FF9800; font-weight: bold; }
        .danger { color: #F44336; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Robustness Analysis Report</h1>
""")

    profit_class = "success" if report.mean_return > 0 else "danger"
    profit_text = "Profitable" if report.mean_return > 0 else "Loss"
    stable_class = "success" if report.std_return < 0.30 else "warning"
    stable_text = "Stable" if report.std_return < 0.30 else "Volatile"

    if report.neighbor_success_rate > 0.70:
        robust_class = "success"
        robust_text = "Robust"
    elif report.neighbor_success_rate > 0.50:
        robust_class = "warning"
        robust_text = "Moderate"
    else:
        robust_class = "danger"
        robust_text = "Fragile"

    html_parts.append(f"""
    <h2>Summary Statistics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Assessment</th>
        </tr>
        <tr>
            <td>Mean Return</td>
            <td>{report.mean_return:.2%}</td>
            <td class="{profit_class}">{profit_text}</td>
        </tr>
        <tr>
            <td>Std Dev Return</td>
            <td>{report.std_return:.2%}</td>
            <td class="{stable_class}">{stable_text}</td>
        </tr>
        <tr>
            <td>Min Return</td>
            <td>{report.min_return:.2%}</td>
            <td></td>
        </tr>
        <tr>
            <td>Max Return</td>
            <td>{report.max_return:.2%}</td>
            <td></td>
        </tr>
        <tr>
            <td>Neighbor Success Rate</td>
            <td>{report.neighbor_success_rate:.1%}</td>
            <td class="{robust_class}">{robust_text}</td>
        </tr>
    </table>
""")

    html_parts.append("""
    <h2>Parameter Sensitivity (0.0 = Insensitive, 1.0 = Highly Sensitive)</h2>
    <table>
        <tr>
            <th>Parameter</th>
            <th>Sensitivity</th>
            <th>Interpretation</th>
        </tr>
""")

    if report.sensitivity_scores is None:
        html_parts.append("        <tr><td colspan='3'>No sensitivity data available</td></tr>\n")
    else:
        for param_name in sorted(report.sensitivity_scores.keys()):
            score = report.sensitivity_scores[param_name]
            if score > 0.7:
                interpretation = "High (Risky)"
            elif score > 0.4:
                interpretation = "Medium"
            else:
                interpretation = "Low (Stable)"

            html_parts.append(f"""        <tr>
            <td>{param_name}</td>
            <td>{score:.3f}</td>
            <td>{interpretation}</td>
        </tr>
""")

    html_parts.append("""    </table>
</body>
</html>
""")

    return "".join(html_parts)
