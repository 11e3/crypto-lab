"""Equity curve chart component.

Plotly 기반 인터랙티브 수익률 곡선 차트.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.web.utils.chart_utils import downsample_timeseries

__all__ = ["render_equity_curve"]


def render_equity_curve(
    dates: np.ndarray,
    equity: np.ndarray,
    initial_capital: float = 1.0,
    benchmark: np.ndarray | None = None,
    benchmark_name: str = "Benchmark",
    max_points: int = 2000,
) -> None:
    """인터랙티브 수익률 곡선 렌더링.

    대량 데이터의 경우 자동으로 다운샘플링하여 렌더링 성능 향상.

    Args:
        dates: 날짜 배열
        equity: 포트폴리오 가치 배열
        initial_capital: 초기 자본
        benchmark: 벤치마크 가치 배열 (선택)
        benchmark_name: 벤치마크 이름
        max_points: 최대 차트 포인트 수 (기본: 2000, 성능 최적화)
    """
    if len(dates) == 0 or len(equity) == 0:
        st.warning("📊 표시할 데이터가 없습니다.")
        return

    # 데이터 다운샘플링 (대량 데이터 시 성능 향상)
    if len(dates) > max_points:
        downsampled_dates, downsampled_equity = downsample_timeseries(
            dates, equity, max_points=max_points
        )
        dates = downsampled_dates  # type: ignore[assignment]
        equity = downsampled_equity
        if benchmark is not None:
            _, downsampled_benchmark = downsample_timeseries(
                dates, benchmark, max_points=max_points
            )
            benchmark = downsampled_benchmark

    # 수익률로 변환 (%)
    returns_pct = (equity / initial_capital - 1) * 100

    fig = go.Figure()

    # 포트폴리오 곡선
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity,
            mode="lines",
            name="Portfolio",
            line={"color": "#1f77b4", "width": 2},
            hovertemplate=(
                "<b>Date</b>: %{x|%Y-%m-%d}<br>"
                "<b>Value</b>: %{y:,.0f} KRW<br>"
                "<b>Return</b>: %{customdata:.2f}%<extra></extra>"
            ),
            customdata=returns_pct,
        )
    )

    # 벤치마크 곡선 (있는 경우)
    if benchmark is not None and len(benchmark) == len(dates):
        benchmark_returns = (benchmark / benchmark[0] - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=benchmark,
                mode="lines",
                name=benchmark_name,
                line={"color": "#ff7f0e", "width": 1.5, "dash": "dash"},
                hovertemplate=(
                    f"<b>{benchmark_name}</b><br>"
                    "<b>Date</b>: %{x|%Y-%m-%d}<br>"
                    "<b>Value</b>: %{y:,.0f}<br>"
                    "<b>Return</b>: %{customdata:.2f}%<extra></extra>"
                ),
                customdata=benchmark_returns,
            )
        )

    # 초기 자본 기준선
    fig.add_hline(
        y=initial_capital,
        line_dash="dot",
        line_color="gray",
        annotation_text="Initial Capital",
        annotation_position="bottom right",
    )

    # 레이아웃 설정
    fig.update_layout(
        title={
            "text": "📈 Portfolio Equity Curve",
            "font": {"size": 18},
        },
        xaxis={
            "title": "Date",
            "showgrid": True,
            "gridcolor": "rgba(128, 128, 128, 0.2)",
            "rangeslider": {"visible": True},
            "rangeselector": {
                "buttons": [
                    {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                    {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                    {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                    {"count": 1, "label": "YTD", "step": "year", "stepmode": "todate"},
                    {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                    {"step": "all", "label": "All"},
                ]
            },
        },
        yaxis={
            "title": "Portfolio Value (KRW)",
            "showgrid": True,
            "gridcolor": "rgba(128, 128, 128, 0.2)",
            "tickformat": ",",
        },
        hovermode="x unified",
        template="plotly_white",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 60, "r": 20, "t": 80, "b": 60},
    )

    st.plotly_chart(fig, width="stretch")
