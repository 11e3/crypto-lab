"""Underwater (drawdown) chart component.

드로다운을 시각화하는 언더워터 차트.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

__all__ = ["render_underwater_curve"]


def render_underwater_curve(
    dates: np.ndarray,
    equity: np.ndarray,
) -> None:
    """언더워터(드로다운) 곡선 렌더링.
    
    Args:
        dates: 날짜 배열
        equity: 포트폴리오 가치 배열
    """
    if len(dates) == 0 or len(equity) == 0:
        st.warning("📊 표시할 데이터가 없습니다.")
        return

    # 드로다운 계산
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax * 100  # 퍼센트

    # 최대 드로다운 위치
    mdd_idx = np.argmin(drawdown)
    mdd_value = drawdown[mdd_idx]
    mdd_date = dates[mdd_idx]

    fig = go.Figure()

    # 드로다운 영역 차트
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdown,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.3)",
            line=dict(color="red", width=1),
            hovertemplate=(
                "<b>Date</b>: %{x|%Y-%m-%d}<br>"
                "<b>Drawdown</b>: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    # 최대 드로다운 포인트
    fig.add_trace(
        go.Scatter(
            x=[mdd_date],
            y=[mdd_value],
            mode="markers+text",
            name=f"MDD: {mdd_value:.2f}%",
            marker=dict(color="darkred", size=12, symbol="triangle-down"),
            text=[f"MDD: {mdd_value:.2f}%"],
            textposition="bottom center",
            textfont=dict(color="darkred", size=11),
            hoverinfo="skip",
        )
    )

    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text="📉 Underwater Curve (Drawdown)",
            font=dict(size=18),
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
        ),
        yaxis=dict(
            title="Drawdown (%)",
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            ticksuffix="%",
            range=[min(drawdown) * 1.1, 5],  # 약간의 여백
        ),
        hovermode="x unified",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=60, r=20, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)
