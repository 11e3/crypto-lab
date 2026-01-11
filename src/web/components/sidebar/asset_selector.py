"""Asset selector component.

자산군 멀티 선택 UI 컴포넌트.
"""

import streamlit as st

__all__ = ["render_asset_selector", "get_available_tickers"]


# 주요 암호화폐 목록
POPULAR_TICKERS = [
    "KRW-BTC",
    "KRW-ETH",
    "KRW-XRP",
    "KRW-ADA",
    "KRW-SOL",
    "KRW-DOGE",
    "KRW-AVAX",
    "KRW-DOT",
    "KRW-MATIC",
    "KRW-LINK",
    "KRW-TRX",
    "KRW-ATOM",
    "KRW-ETC",
    "KRW-NEO",
    "KRW-WAVES",
]


def get_available_tickers() -> list[str]:
    """사용 가능한 티커 목록 반환.
    
    Returns:
        티커 문자열 리스트
    """
    # TODO: 실제로는 data 디렉토리를 스캔하거나 Upbit API 호출
    return POPULAR_TICKERS


def render_asset_selector() -> list[str]:
    """자산 선택 UI 렌더링.
    
    Returns:
        선택된 티커 리스트
    """
    st.subheader("🪙 자산 선택")

    available_tickers = get_available_tickers()

    # 선택 모드
    selection_mode = st.radio(
        "선택 방식",
        ["빠른 선택", "개별 선택"],
        horizontal=True,
        help="빠른 선택: 프리셋 사용 | 개별 선택: 직접 체크",
    )

    selected_tickers: list[str] = []

    if selection_mode == "빠른 선택":
        # 프리셋 선택
        presets = {
            "Top 3 (BTC, ETH, XRP)": ["KRW-BTC", "KRW-ETH", "KRW-XRP"],
            "Top 5": ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-ADA", "KRW-SOL"],
            "Top 10": available_tickers[:10],
            "전체": available_tickers,
            "커스텀": [],
        }

        preset_name = st.selectbox(
            "프리셋 선택",
            options=list(presets.keys()),
            help="미리 정의된 자산 그룹",
        )

        selected_tickers = presets[preset_name]

        if preset_name == "커스텀":
            # 커스텀 선택
            selected_tickers = st.multiselect(
                "자산 선택",
                options=available_tickers,
                default=["KRW-BTC", "KRW-ETH"],
                help="백테스트에 포함할 자산 선택",
            )

    else:
        # 개별 선택 (체크박스)
        st.caption("체크박스로 선택:")

        # 그리드 레이아웃 (3열)
        cols_per_row = 3
        for i in range(0, len(available_tickers), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(available_tickers):
                    ticker = available_tickers[idx]
                    with col:
                        if st.checkbox(ticker, key=f"ticker_{ticker}"):
                            selected_tickers.append(ticker)

    # 선택 결과 표시
    if selected_tickers:
        st.success(f"✅ 선택된 자산: **{len(selected_tickers)}개**")
        st.caption(", ".join(selected_tickers))
    else:
        st.warning("⚠️ 최소 1개 이상의 자산을 선택하세요.")

    return selected_tickers
