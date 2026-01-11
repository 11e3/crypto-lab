"""Yearly returns bar chart component.

연도별 수익률 막대 그래프.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

__all__ = ["render_yearly_bar_chart", "calculate_yearly_returns"]


def calculate_yearly_returns(
    dates: np.ndarray,
    equity: np.ndarray,
) -> pd.DataFrame:
    """연도별 수익률 계산.
    
    Args:
        dates: 날짜 배열
        equity: 포트폴리오 가치 배열
        
    Returns:
        연도별 수익률 DataFrame (columns: year, return_pct)
    """
    if len(dates) == 0 or len(equity) == 0:
        return pd.DataFrame(columns=["year", "return_pct"])

    # DataFrame 생성
    df = pd.DataFrame({"date": pd.to_datetime(dates), "equity": equity})
    df["year"] = df["date"].dt.year

    # 연도별 첫날/마지막날 가치 계산
    yearly = df.groupby("year").agg(
        first_equity=("equity", "first"),
        last_equity=("equity", "last"),
    ).reset_index()

    # 연도별 수익률 계산
    yearly["return_pct"] = (
        (yearly["last_equity"] / yearly["first_equity"] - 1) * 100
    )

    return yearly[["year", "return_pct"]]


def render_yearly_bar_chart(
    dates: np.ndarray,
    equity: np.ndarray,
) -> None:
    """연도별 수익률 막대 그래프 렌더링.
    
    Args:
        dates: 날짜 배열
        equity: 포트폴리오 가치 배열
    """
    if len(dates) == 0 or len(equity) == 0:
        st.warning("📊 표시할 데이터가 없습니다.")
        return

    # 연도별 수익률 계산
    yearly = calculate_yearly_returns(dates, equity)

    if yearly.empty:
        st.warning("📊 연도별 데이터가 없습니다.")
        return

    # 색상: 양수 = 녹색, 음수 = 빨강
    colors = [
        "rgb(0, 150, 0)" if r >= 0 else "rgb(200, 0, 0)"
        for r in yearly["return_pct"]
    ]

    fig = go.Figure()

    # 막대 차트
    fig.add_trace(
        go.Bar(
            x=[str(y) for y in yearly["year"]],
            y=yearly["return_pct"],
            marker_color=colors,
            text=[f"{r:.1f}%" for r in yearly["return_pct"]],
            textposition="outside",
            textfont=dict(size=12),
            hovertemplate=(
                "<b>%{x}년</b><br>"
                "Return: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    # 0% 기준선
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)

    # 평균선
    avg_return = yearly["return_pct"].mean()
    fig.add_hline(
        y=avg_return,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Avg: {avg_return:.1f}%",
        annotation_position="right",
    )

    # 레이아웃
    fig.update_layout(
        title=dict(
            text="📊 Yearly Returns",
            font=dict(size=18),
        ),
        xaxis=dict(
            title="Year",
            tickmode="linear",
        ),
        yaxis=dict(
            title="Return (%)",
            ticksuffix="%",
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
        ),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=60, r=20, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # 통계 요약
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("평균 수익률", f"{avg_return:.1f}%")
    with col2:
        st.metric("최고 수익률", f"{yearly['return_pct'].max():.1f}%")
    with col3:
        st.metric("최저 수익률", f"{yearly['return_pct'].min():.1f}%")
    with col4:
        positive_years = (yearly["return_pct"] > 0).sum()
        total_years = len(yearly)
        st.metric("수익 연도", f"{positive_years}/{total_years}")
