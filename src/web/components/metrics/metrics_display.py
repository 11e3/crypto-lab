"""Metrics display component.

확장 메트릭 카드 및 테이블 표시.
"""

import streamlit as st

from src.web.services.metrics_calculator import ExtendedMetrics

__all__ = ["render_metrics_cards", "render_metrics_table"]


def _format_value(value: float, suffix: str = "", precision: int = 2) -> str:
    """값 포맷팅."""
    if value == float("inf"):
        return "∞"
    if value == float("-inf"):
        return "-∞"
    return f"{value:.{precision}f}{suffix}"


def render_metrics_cards(metrics: ExtendedMetrics) -> None:
    """메트릭 카드 렌더링.

    주요 메트릭을 카드 형태로 표시.

    Args:
        metrics: 확장 메트릭 데이터
    """
    st.subheader("📈 Performance Summary")

    # Row 1: 기본 수익률 메트릭
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "총 수익률",
            _format_value(metrics.total_return_pct, "%"),
            delta=None,
        )
    with col2:
        st.metric(
            "CAGR",
            _format_value(metrics.cagr_pct, "%"),
        )
    with col3:
        st.metric(
            "MDD",
            _format_value(metrics.max_drawdown_pct, "%"),
        )
    with col4:
        st.metric(
            "변동성 (연간)",
            _format_value(metrics.volatility_pct, "%"),
        )

    # Row 2: 리스크 조정 수익률
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sharpe Ratio", _format_value(metrics.sharpe_ratio))
    with col2:
        st.metric("Sortino Ratio", _format_value(metrics.sortino_ratio))
    with col3:
        st.metric("Calmar Ratio", _format_value(metrics.calmar_ratio))
    with col4:
        st.metric("거래 수", str(metrics.num_trades))

    # Row 3: VaR & CVaR
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("VaR (95%)", _format_value(metrics.var_95_pct, "%"))
    with col2:
        st.metric("VaR (99%)", _format_value(metrics.var_99_pct, "%"))
    with col3:
        st.metric("CVaR (95%)", _format_value(metrics.cvar_95_pct, "%"))
    with col4:
        st.metric("CVaR (99%)", _format_value(metrics.cvar_99_pct, "%"))

    # Row 4: 거래 메트릭
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("승률", _format_value(metrics.win_rate_pct, "%", 1))
    with col2:
        st.metric("평균 수익", _format_value(metrics.avg_win_pct, "%"))
    with col3:
        st.metric("평균 손실", _format_value(metrics.avg_loss_pct, "%"))
    with col4:
        st.metric("Profit Factor", _format_value(metrics.profit_factor))

    # Row 5: 통계적 메트릭
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Z-Score", _format_value(metrics.z_score))
    with col2:
        # P-value 색상 표시
        p_val = metrics.p_value
        significance = "✅" if p_val < 0.05 else "⚠️" if p_val < 0.1 else "❌"
        st.metric("P-Value", f"{significance} {p_val:.4f}")
    with col3:
        st.metric("Skewness", _format_value(metrics.skewness))
    with col4:
        st.metric("Kurtosis", _format_value(metrics.kurtosis))

    # Row 6: 변동성 및 기간
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("상방 변동성", _format_value(metrics.upside_volatility_pct, "%"))
    with col2:
        st.metric("하방 변동성", _format_value(metrics.downside_volatility_pct, "%"))
    with col3:
        st.metric("거래일", str(metrics.trading_days))
    with col4:
        st.metric("기간", _format_value(metrics.years, "년", 1))


def render_metrics_table(metrics: ExtendedMetrics) -> None:
    """메트릭 테이블 렌더링.

    모든 메트릭을 테이블 형태로 표시.

    Args:
        metrics: 확장 메트릭 데이터
    """
    st.subheader("📊 Detailed Metrics")

    # 카테고리별 메트릭 그룹화
    categories = {
        "📈 수익률 메트릭": [
            ("총 수익률", _format_value(metrics.total_return_pct, "%")),
            ("CAGR", _format_value(metrics.cagr_pct, "%")),
            ("기대 수익 (Expectancy)", _format_value(metrics.expectancy, "%")),
        ],
        "📉 리스크 메트릭": [
            ("최대 낙폭 (MDD)", _format_value(metrics.max_drawdown_pct, "%")),
            ("변동성 (연간)", _format_value(metrics.volatility_pct, "%")),
            ("상방 변동성", _format_value(metrics.upside_volatility_pct, "%")),
            ("하방 변동성", _format_value(metrics.downside_volatility_pct, "%")),
        ],
        "⚖️ 리스크 조정 수익률": [
            ("Sharpe Ratio", _format_value(metrics.sharpe_ratio)),
            ("Sortino Ratio", _format_value(metrics.sortino_ratio)),
            ("Calmar Ratio", _format_value(metrics.calmar_ratio)),
        ],
        "🎯 VaR & CVaR": [
            ("VaR (95%)", _format_value(metrics.var_95_pct, "%")),
            ("VaR (99%)", _format_value(metrics.var_99_pct, "%")),
            ("CVaR (95%)", _format_value(metrics.cvar_95_pct, "%")),
            ("CVaR (99%)", _format_value(metrics.cvar_99_pct, "%")),
        ],
        "🔢 통계적 분석": [
            ("Z-Score", _format_value(metrics.z_score)),
            ("P-Value", f"{metrics.p_value:.6f}"),
            ("Skewness", _format_value(metrics.skewness)),
            ("Kurtosis", _format_value(metrics.kurtosis)),
        ],
        "💹 거래 메트릭": [
            ("거래 수", str(metrics.num_trades)),
            ("승률", _format_value(metrics.win_rate_pct, "%", 1)),
            ("평균 수익", _format_value(metrics.avg_win_pct, "%")),
            ("평균 손실", _format_value(metrics.avg_loss_pct, "%")),
            ("Profit Factor", _format_value(metrics.profit_factor)),
        ],
        "📅 기간 정보": [
            ("거래일", str(metrics.trading_days)),
            ("기간 (년)", _format_value(metrics.years, "", 2)),
        ],
    }

    # 2열 레이아웃
    col1, col2 = st.columns(2)

    category_items = list(categories.items())
    for i, (category, items) in enumerate(category_items):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            st.markdown(f"**{category}**")
            for name, value in items:
                st.markdown(f"- {name}: **{value}**")
            st.markdown("---")


def render_statistical_significance(metrics: ExtendedMetrics) -> None:
    """통계적 유의성 해석 렌더링.

    Args:
        metrics: 확장 메트릭 데이터
    """
    st.subheader("🔬 Statistical Significance Analysis")

    p_value = metrics.p_value
    z_score = metrics.z_score

    # 유의 수준 판정
    if p_value < 0.01:
        significance = "매우 유의함 (p < 0.01)"
        icon = "✅"
    elif p_value < 0.05:
        significance = "유의함 (p < 0.05)"
        icon = "✅"
    elif p_value < 0.1:
        significance = "약하게 유의함 (p < 0.10)"
        icon = "⚠️"
    else:
        significance = "유의하지 않음 (p ≥ 0.10)"
        icon = "❌"

    st.markdown(f"""
    ### {icon} 결과: {significance}

    | 지표 | 값 | 해석 |
    |------|-----|------|
    | Z-Score | {z_score:.4f} | {"양의 초과 수익" if z_score > 0 else "음의 초과 수익"} |
    | P-Value | {p_value:.6f} | 귀무가설 기각 {"가능" if p_value < 0.05 else "불가"} |
    | Skewness | {metrics.skewness:.4f} | {"우측 꼬리 (긍정적)" if metrics.skewness > 0 else "좌측 꼬리 (부정적)"} |
    | Kurtosis | {metrics.kurtosis:.4f} | {"Fat tail (위험 증가)" if metrics.kurtosis > 0 else "Thin tail"} |
    """)

    # 해석 가이드
    with st.expander("📖 해석 가이드"):
        st.markdown("""
        **Z-Score**: 평균 수익률이 0과 얼마나 다른지 표준편차 단위로 측정
        - |Z| > 1.96: 95% 신뢰수준에서 유의
        - |Z| > 2.58: 99% 신뢰수준에서 유의

        **P-Value**: 귀무가설(수익률=0) 하에서 관측된 결과가 나올 확률
        - p < 0.05: 통계적으로 유의한 수익률
        - p < 0.01: 매우 강한 증거

        **Skewness**: 수익률 분포의 비대칭성
        - 양수: 큰 수익이 큰 손실보다 많음 (바람직)
        - 음수: 큰 손실이 큰 수익보다 많음 (위험)

        **Kurtosis**: 수익률 분포의 꼬리 두께
        - 양수: Fat tail (극단적 사건 빈번)
        - 음수: Thin tail (극단적 사건 희귀)
        """)
