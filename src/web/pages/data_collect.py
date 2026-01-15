"""Data collection page.

데이터 수집 실행 및 상태 표시 페이지.
"""

import streamlit as st

from src.data.collector_factory import DataCollectorFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["render_data_collect_page"]

# 기본 티커 목록
DEFAULT_TICKERS = [
    "KRW-BTC",
    "KRW-ETH",
    "KRW-XRP",
    "KRW-SOL",
    "KRW-DOGE",
    "KRW-TRX",
    "KRW-ADA",
    "KRW-AVAX",
    "KRW-SHIB",
    "KRW-LINK",
]

# 지원하는 인터벌
INTERVALS = [
    ("minute1", "1분"),
    ("minute3", "3분"),
    ("minute5", "5분"),
    ("minute10", "10분"),
    ("minute15", "15분"),
    ("minute30", "30분"),
    ("minute60", "1시간"),
    ("minute240", "4시간"),
    ("day", "일봉"),
    ("week", "주봉"),
    ("month", "월봉"),
]


def render_data_collect_page() -> None:
    """데이터 수집 페이지 렌더링."""
    st.header("📥 데이터 수집")

    # ===== 사이드바 설정 =====
    with st.sidebar:
        st.title("📥 수집 설정")
        st.markdown("---")

        # 1. 티커 선택
        st.subheader("📈 티커 선택")

        # 빠른 선택 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("전체 선택", use_container_width=True):
                st.session_state.selected_collect_tickers = DEFAULT_TICKERS.copy()
        with col2:
            if st.button("선택 해제", use_container_width=True):
                st.session_state.selected_collect_tickers = []

        # 티커 체크박스
        if "selected_collect_tickers" not in st.session_state:
            st.session_state.selected_collect_tickers = DEFAULT_TICKERS[:6]

        selected_tickers = []
        for ticker in DEFAULT_TICKERS:
            checked = ticker in st.session_state.selected_collect_tickers
            if st.checkbox(ticker, value=checked, key=f"collect_{ticker}"):
                selected_tickers.append(ticker)

        st.session_state.selected_collect_tickers = selected_tickers

        # 커스텀 티커 입력
        custom_ticker = st.text_input(
            "커스텀 티커 추가",
            placeholder="예: KRW-MATIC",
            help="Upbit에서 지원하는 KRW 마켓 티커를 입력하세요",
        )
        if (
            custom_ticker
            and custom_ticker not in selected_tickers
            and st.button(f"➕ {custom_ticker} 추가")
        ):
            selected_tickers.append(custom_ticker.upper())
            st.session_state.selected_collect_tickers = selected_tickers
            st.rerun()

        st.markdown("---")

        # 2. 인터벌 선택
        st.subheader("⏱️ 인터벌 선택")

        if "selected_intervals" not in st.session_state:
            st.session_state.selected_intervals = ["minute240", "day", "week"]

        selected_intervals = []
        for interval_code, interval_name in INTERVALS:
            checked = interval_code in st.session_state.selected_intervals
            if st.checkbox(
                f"{interval_name} ({interval_code})", value=checked, key=f"interval_{interval_code}"
            ):
                selected_intervals.append(interval_code)

        st.session_state.selected_intervals = selected_intervals

        st.markdown("---")

        # 3. 수집 옵션
        st.subheader("⚙️ 옵션")
        full_refresh = st.checkbox(
            "전체 새로고침",
            value=False,
            help="기존 데이터를 무시하고 전체 데이터를 다시 수집합니다",
        )

        st.markdown("---")

        # 실행 버튼
        run_button = st.button(
            "🚀 데이터 수집 시작",
            type="primary",
            use_container_width=True,
            disabled=not selected_tickers or not selected_intervals,
        )

    # ===== 메인 화면 =====

    # 검증
    if not selected_tickers:
        st.warning("⚠️ 최소 1개 이상의 티커를 선택하세요.")
        return

    if not selected_intervals:
        st.warning("⚠️ 최소 1개 이상의 인터벌을 선택하세요.")
        return

    # 현재 설정 요약
    with st.expander("📋 수집 설정 요약", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📈 선택된 티커**")
            st.write(", ".join(selected_tickers))
            st.metric("티커 수", len(selected_tickers))

        with col2:
            st.markdown("**⏱️ 선택된 인터벌**")
            interval_names = [name for code, name in INTERVALS if code in selected_intervals]
            st.write(", ".join(interval_names))
            st.metric("인터벌 수", len(selected_intervals))

        with col3:
            st.markdown("**⚙️ 옵션**")
            st.write(f"전체 새로고침: {'예' if full_refresh else '아니오'}")
            total_tasks = len(selected_tickers) * len(selected_intervals)
            st.metric("총 수집 작업", total_tasks)

    # 데이터 수집 실행
    if run_button:
        _run_collection(selected_tickers, selected_intervals, full_refresh)

    # 이전 수집 결과 표시
    if "collection_results" in st.session_state:
        _display_collection_results()


def _run_collection(
    tickers: list[str],
    intervals: list[str],
    full_refresh: bool,
) -> None:
    """데이터 수집 실행.

    Args:
        tickers: 수집할 티커 목록
        intervals: 수집할 인터벌 목록
        full_refresh: 전체 새로고침 여부
    """
    st.subheader("📊 수집 진행 상황")

    # 진행 바
    total_tasks = len(tickers) * len(intervals)
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 결과 저장
    results: dict[str, int] = {}
    completed = 0

    try:
        collector = DataCollectorFactory.create()

        for ticker in tickers:
            for interval in intervals:
                status_text.text(f"수집 중: {ticker} ({interval})...")

                try:
                    # collect 메서드 사용 (full_refresh 지원)
                    count = collector.collect(
                        ticker=ticker,
                        interval=interval,  # type: ignore[arg-type]
                        full_refresh=full_refresh,
                    )

                    key = f"{ticker}_{interval}"
                    results[key] = count

                except Exception as e:
                    logger.error(f"Error collecting {ticker} {interval}: {e}")
                    results[f"{ticker}_{interval}"] = -1

                completed += 1
                progress_bar.progress(completed / total_tasks)

        status_text.text("수집 완료!")

        # 결과 저장
        st.session_state.collection_results = results

        # 성공/실패 카운트
        success_count = sum(1 for v in results.values() if v >= 0)
        fail_count = sum(1 for v in results.values() if v < 0)
        total_candles = sum(v for v in results.values() if v >= 0)

        if fail_count == 0:
            st.success(f"✅ 모든 수집 완료! 총 {total_candles:,}개 캔들 수집됨")
        else:
            st.warning(f"⚠️ 수집 완료: 성공 {success_count}건, 실패 {fail_count}건")

    except Exception as e:
        logger.error(f"Collection error: {e}", exc_info=True)
        st.error(f"❌ 수집 중 오류 발생: {e}")


def _display_collection_results() -> None:
    """수집 결과 표시."""
    results = st.session_state.collection_results

    st.subheader("📊 최근 수집 결과")

    # 결과를 테이블로 표시
    import pandas as pd

    data = []
    for key, count in results.items():
        parts = key.rsplit("_", 1)
        ticker = parts[0]
        interval = parts[1] if len(parts) > 1 else "unknown"

        status = "✅ 성공" if count >= 0 else "❌ 실패"
        candles = f"{count:,}" if count >= 0 else "-"

        data.append(
            {
                "티커": ticker,
                "인터벌": interval,
                "상태": status,
                "캔들 수": candles,
            }
        )

    df = pd.DataFrame(data)
    st.dataframe(df, width="stretch", height=400)

    # 요약
    col1, col2, col3 = st.columns(3)
    success_count = sum(1 for v in results.values() if v >= 0)
    fail_count = sum(1 for v in results.values() if v < 0)
    total_candles = sum(v for v in results.values() if v >= 0)

    with col1:
        st.metric("성공", f"{success_count}건")
    with col2:
        st.metric("실패", f"{fail_count}건")
    with col3:
        st.metric("총 캔들", f"{total_candles:,}개")
