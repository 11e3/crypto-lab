"""Charts components package."""

from src.web.components.charts.equity_curve import render_equity_curve
from src.web.components.charts.monthly_heatmap import (
    calculate_monthly_returns,
    render_monthly_heatmap,
)
from src.web.components.charts.underwater import render_underwater_curve
from src.web.components.charts.yearly_bar import (
    calculate_yearly_returns,
    render_yearly_bar_chart,
)

__all__ = [
    "render_equity_curve",
    "render_underwater_curve",
    "render_monthly_heatmap",
    "calculate_monthly_returns",
    "render_yearly_bar_chart",
    "calculate_yearly_returns",
]
