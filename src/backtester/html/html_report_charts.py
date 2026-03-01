"""
Chart JavaScript generation for HTML reports.

Contains Plotly chart configurations.
"""

import json
from typing import Any


def get_chart_js(
    dates_str: str,
    equity_values: list[float],
    drawdown_values: list[float],
    mdd_pct: float,
    mdd_date: str,
    mdd_drawdown_value: float,
    monthly_returns: dict[str, Any],
) -> str:
    """Generate JavaScript for Plotly charts."""
    equity_js = _get_equity_chart_js(dates_str, equity_values, drawdown_values)
    drawdown_js = _get_drawdown_chart_js(
        dates_str, drawdown_values, mdd_pct, mdd_date, mdd_drawdown_value
    )
    heatmap_js = _get_heatmap_js(monthly_returns)
    yearly_js = _get_yearly_chart_js(monthly_returns)

    return equity_js + drawdown_js + heatmap_js + yearly_js


def _get_equity_chart_js(
    dates_str: str, equity_values: list[float], drawdown_values: list[float]
) -> str:
    """Generate equity curve chart JavaScript."""
    return f"""
        var equityTrace = {{
            x: {dates_str}, y: {equity_values}, type: 'scatter', mode: 'lines',
            name: 'Equity', line: {{color: '#3498db', width: 2}},
            fill: 'tozeroy', fillcolor: 'rgba(52, 152, 219, 0.2)'
        }};
        var drawdownTrace = {{
            x: {dates_str}, y: {drawdown_values}, type: 'scatter', mode: 'lines',
            name: 'Drawdown', yaxis: 'y2', line: {{color: '#e74c3c', width: 1}},
            fill: 'tozeroy', fillcolor: 'rgba(231, 76, 60, 0.3)'
        }};
        var filteredDd = {drawdown_values}.filter(x => !isNaN(x) && isFinite(x));
        var maxDd = filteredDd.length > 0 ? Math.max(...filteredDd) : 0;
        var equityLayout = {{
            title: 'Equity Curve with Drawdown',
            xaxis: {{title: 'Date'}},
            yaxis: {{title: 'Equity', side: 'left', type: 'log'}},
            yaxis2: {{title: 'Drawdown (%)', overlaying: 'y', side: 'right', range: [maxDd * 1.2, 5]}},
            hovermode: 'x unified', showlegend: true
        }};
        Plotly.newPlot('equity-chart', [equityTrace, drawdownTrace], equityLayout, {{responsive: true}});
    """


def _get_drawdown_chart_js(
    dates_str: str,
    drawdown_values: list[float],
    mdd_pct: float,
    mdd_date: str,
    mdd_drawdown_value: float,
) -> str:
    """Generate drawdown chart JavaScript."""
    return f"""
        var ddOnlyTrace = {{
            x: {dates_str}, y: {drawdown_values}, type: 'scatter', mode: 'lines',
            fill: 'tozeroy', fillcolor: 'rgba(231, 76, 60, 0.5)', line: {{color: '#e74c3c', width: 1}}
        }};
        var mddValue = -{mdd_pct:.2f};
        var ddLayout = {{
            title: 'Drawdown Over Time (MDD: ' + mddValue.toFixed(2) + '%)',
            xaxis: {{title: 'Date'}}, yaxis: {{title: 'Drawdown (%)', range: [-100, 0]}},
            hovermode: 'x unified', showlegend: false,
            annotations: [{{
                x: '{mdd_date}', y: -{mdd_drawdown_value:.2f},
                text: 'MDD: ' + mddValue.toFixed(2) + '%',
                showarrow: true, arrowhead: 2, arrowsize: 1.5, arrowwidth: 2, arrowcolor: '#e74c3c',
                bgcolor: 'rgba(231, 76, 60, 0.8)', bordercolor: '#c0392b', borderwidth: 1,
                font: {{color: 'white', size: 12}}, ax: 0, ay: 40
            }}]
        }};
        var ddDiv = document.getElementById('drawdown-chart');
        if (ddDiv) {{ Plotly.newPlot('drawdown-chart', [ddOnlyTrace], ddLayout, {{responsive: true}}); }}
    """


def _get_heatmap_js(monthly_returns: dict[str, Any]) -> str:
    """Generate heatmap chart JavaScript."""
    return f"""
        var hmZ = {json.dumps(monthly_returns["values"])};
        var hmX = {json.dumps(monthly_returns["months"])};
        var hmY = {json.dumps(monthly_returns["years"])};
        var hmText = {json.dumps(monthly_returns["text"])};
        var allVals = hmZ.flat().filter(x => !isNaN(x));
        var absMax = Math.max(Math.abs(Math.min(...allVals)), Math.abs(Math.max(...allVals)));
        var hmData = {{
            z: hmZ, x: hmX, y: hmY, type: 'heatmap',
            colorscale: [[0.0,'#8b0000'],[0.2,'#c0392b'],[0.4,'#e74c3c'],[0.45,'#ec7063'],
                [0.5,'#fadbd8'],[0.55,'#d5f4e6'],[0.6,'#82e0aa'],[0.8,'#27ae60'],[1.0,'#1e8449']],
            zmin: -absMax, zmax: absMax, zmid: 0, colorbar: {{title: 'Return (%)'}},
            text: hmText, texttemplate: '%{{text}}', textfont: {{size: 11, color: '#000'}},
            hovertemplate: 'Year: %{{y}}<br>Month: %{{x}}<br>Return: %{{z:.2f}}%<extra></extra>',
            showscale: true
        }};
        var hmLayout = {{title: 'Monthly Returns Heatmap (%)', xaxis: {{title: 'Month'}},
            yaxis: {{title: 'Year'}}, height: 400}};
        Plotly.newPlot('heatmap-chart', [hmData], hmLayout, {{responsive: true}});
    """


def _get_yearly_chart_js(monthly_returns: dict[str, Any]) -> str:
    """Generate yearly returns bar chart JavaScript."""
    return f"""
        var yrRets = {json.dumps(monthly_returns.get("yearly_returns", []))};
        var yrLabels = {json.dumps(monthly_returns.get("yearly_labels", []))};
        if (yrRets && yrRets.length > 0 && yrLabels && yrLabels.length > 0) {{
            var years = [], values = [], colors = [];
            for (var i = 0; i < yrLabels.length; i++) {{
                if (yrLabels[i] && !isNaN(yrRets[i])) {{
                    var parts = yrLabels[i].split(': ');
                    if (parts.length >= 1) {{
                        years.push(parts[0]); values.push(yrRets[i]);
                        colors.push(yrRets[i] >= 0 ? '#27ae60' : '#e74c3c');
                    }}
                }}
            }}
            if (years.length > 0) {{
                var yrBarData = {{
                    x: years, y: values, type: 'bar', marker: {{color: colors}},
                    text: values.map(v => v.toFixed(1) + '%'), textposition: 'outside',
                    textfont: {{size: 11, color: '#000'}}
                }};
                var yrBarLayout = {{
                    title: 'Yearly Returns (%)', xaxis: {{title: 'Year'}},
                    yaxis: {{title: 'Return (%)'}}, height: 400, showlegend: false
                }};
                var hmContainer = document.getElementById('heatmap-chart');
                if (hmContainer && hmContainer.parentElement) {{
                    var yrDiv = document.createElement('div');
                    yrDiv.id = 'yearly-returns-chart'; yrDiv.style.marginTop = '30px';
                    hmContainer.parentElement.appendChild(yrDiv);
                    Plotly.newPlot('yearly-returns-chart', [yrBarData], yrBarLayout, {{responsive: true}});
                }}
            }}
        }}
    """


__all__ = ["get_chart_js"]
