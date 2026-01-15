"""Date configuration component.

시작일과 종료일을 선택하는 UI 컴포넌트.
"""

from datetime import date, timedelta

import streamlit as st

__all__ = ["render_date_config"]


def render_date_config() -> tuple[date, date]:
    """날짜 범위 선택 UI 렌더링.

    Returns:
        (start_date, end_date) 튜플
    """
    st.subheader("📅 기간 설정")

    # 기본값: 최근 1년
    default_end = date.today()
    default_start = default_end - timedelta(days=365)

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "시작일",
            value=default_start,
            min_value=date(2017, 1, 1),  # Upbit 시작일
            max_value=default_end,
            help="백테스트 시작 날짜 (Upbit: 2017년 이후)",
        )

    with col2:
        end_date = st.date_input(
            "종료일",
            value=default_end,
            min_value=start_date,
            max_value=date.today(),
            help="백테스트 종료 날짜",
        )

    # 검증
    if start_date >= end_date:
        st.error("⚠️ 시작일은 종료일보다 이전이어야 합니다.")
        return default_start, default_end

    # 기간 표시
    days_diff = (end_date - start_date).days
    st.caption(f"📊 백테스트 기간: **{days_diff:,}일** ({start_date} ~ {end_date})")

    return start_date, end_date
