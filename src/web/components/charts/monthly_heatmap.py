"""Monthly returns heatmap component.

월별 수익률 히트맵 차트.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

__all__ = ["render_monthly_heatmap", "calculate_monthly_returns"]


def calculate_monthly_returns(
    dates: np.ndarray,
    equity: np.ndarray,
) -> pd.DataFrame:
    """월별 수익률 계산.

    Args:
        dates: 날짜 배열
        equity: 포트폴리오 가치 배열

    Returns:
        월별 수익률 DataFrame (columns: year, month, return_pct)
    """
    if len(dates) == 0 or len(equity) == 0:
        return pd.DataFrame(columns=["year", "month", "return_pct"])

    # DataFrame 생성
    df = pd.DataFrame({"date": pd.to_datetime(dates), "equity": equity})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # 월별 첫날/마지막날 가치 계산
    monthly = (
        df.groupby(["year", "month"])
        .agg(
            first_equity=("equity", "first"),
            last_equity=("equity", "last"),
        )
        .reset_index()
    )

    # 월별 수익률 계산
    monthly["return_pct"] = (monthly["last_equity"] / monthly["first_equity"] - 1) * 100

    return monthly[["year", "month", "return_pct"]]


def render_monthly_heatmap(
    dates: np.ndarray,
    equity: np.ndarray,
) -> None:
    """월별 수익률 히트맵 렌더링.

    Args:
        dates: 날짜 배열
        equity: 포트폴리오 가치 배열
    """
    if len(dates) == 0 or len(equity) == 0:
        st.warning("📊 표시할 데이터가 없습니다.")
        return

    # 월별 수익률 계산
    monthly = calculate_monthly_returns(dates, equity)

    if monthly.empty:
        st.warning("📊 월별 데이터가 없습니다.")
        return

    # 피벗 테이블 생성 (행: 연도, 열: 월)
    pivot = monthly.pivot(index="year", columns="month", values="return_pct")

    # 빈 월 채우기
    all_months = list(range(1, 13))
    for month in all_months:
        if month not in pivot.columns:
            pivot[month] = np.nan
    pivot = pivot[all_months]

    # 월 이름
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    # 히트맵 데이터
    z_data = pivot.values
    years = pivot.index.tolist()

    # 주석 텍스트 (수익률 값)
    annotations = []
    for i, year in enumerate(years):
        for j, _month in enumerate(all_months):
            value = z_data[i, j]
            if not np.isnan(value):
                annotations.append(
                    {
                        "x": month_names[j],
                        "y": str(year),
                        "text": f"{value:.1f}%",
                        "showarrow": False,
                        "font": {
                            "color": "white" if abs(value) > 5 else "black",
                            "size": 10,
                        },
                    }
                )

    # 히트맵
    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=month_names,
            y=[str(y) for y in years],
            colorscale=[
                [0.0, "rgb(165, 0, 38)"],  # 진한 빨강 (큰 손실)
                [0.25, "rgb(215, 48, 39)"],  # 빨강
                [0.4, "rgb(244, 109, 67)"],  # 연한 빨강
                [0.5, "rgb(255, 255, 255)"],  # 흰색 (0%)
                [0.6, "rgb(166, 217, 106)"],  # 연한 녹색
                [0.75, "rgb(102, 189, 99)"],  # 녹색
                [1.0, "rgb(0, 104, 55)"],  # 진한 녹색 (큰 수익)
            ],
            zmid=0,
            colorbar={
                "title": "Return (%)",
                "ticksuffix": "%",
            },
            hovertemplate=("<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>"),
        )
    )

    # 주석 추가
    fig.update_layout(annotations=annotations)

    # 레이아웃
    fig.update_layout(
        title={
            "text": "📅 Monthly Returns Heatmap",
            "font": {"size": 18},
        },
        xaxis={
            "title": "Month",
            "side": "top",
        },
        yaxis={
            "title": "Year",
            "autorange": "reversed",  # 최신 연도가 위로
        },
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 80, "b": 40},
    )

    st.plotly_chart(fig, width="stretch")

    # 연도별 합계 표시
    yearly_returns = monthly.groupby("year")["return_pct"].sum()
    if not yearly_returns.empty:
        cols = st.columns(len(yearly_returns))
        for i, (year, ret) in enumerate(yearly_returns.items()):
            with cols[i]:
                st.metric(
                    label=f"{year}년",
                    value=f"{ret:.1f}%",
                    delta=None,
                )
