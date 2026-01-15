"""Optimization page.

전략 파라미터 최적화 페이지.
"""

import streamlit as st

from src.backtester import BacktestConfig, optimize_strategy_parameters
from src.strategies.volatility_breakout import create_vbo_strategy
from src.utils.logger import get_logger
from src.web.services.data_loader import get_data_files, validate_data_availability

logger = get_logger(__name__)

__all__ = ["render_optimization_page"]

# 최적화 메트릭 옵션
METRICS = [
    ("sharpe_ratio", "Sharpe Ratio"),
    ("cagr", "CAGR"),
    ("total_return", "Total Return"),
    ("calmar_ratio", "Calmar Ratio"),
    ("win_rate", "Win Rate"),
    ("profit_factor", "Profit Factor"),
]

# 기본 티커
DEFAULT_TICKERS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL"]


def render_optimization_page() -> None:
    """최적화 페이지 렌더링."""
    st.header("🔧 파라미터 최적화")

    # ===== 사이드바 설정 =====
    with st.sidebar:
        st.title("🔧 최적화 설정")
        st.markdown("---")

        # 1. 전략 선택
        st.subheader("📈 전략")
        strategy_type = st.selectbox(
            "전략 유형",
            options=["vanilla", "legacy"],
            format_func=lambda x: "Vanilla VBO" if x == "vanilla" else "Legacy VBO",
        )

        st.markdown("---")

        # 2. 최적화 방법
        st.subheader("⚙️ 최적화 방법")
        method = st.radio(
            "탐색 방법",
            options=["grid", "random"],
            format_func=lambda x: "Grid Search (전체 탐색)" if x == "grid" else "Random Search (무작위 탐색)",
            horizontal=True,
        )

        if method == "random":
            n_iter = st.slider("탐색 횟수", min_value=10, max_value=500, value=100, step=10)
        else:
            n_iter = 100  # grid에서는 사용 안 함

        st.markdown("---")

        # 3. 최적화 메트릭
        st.subheader("📊 최적화 메트릭")
        metric = st.selectbox(
            "최적화 대상",
            options=[m[0] for m in METRICS],
            format_func=lambda x: next(name for code, name in METRICS if code == x),
            index=0,
        )

        st.markdown("---")

        # 4. 거래 설정
        st.subheader("💰 거래 설정")
        initial_capital = st.number_input(
            "초기 자본",
            min_value=0.1,
            max_value=100.0,
            value=1.0,
            step=0.1,
            format="%.1f",
        )
        fee_rate = st.number_input(
            "수수료율",
            min_value=0.0,
            max_value=0.01,
            value=0.0005,
            step=0.0001,
            format="%.4f",
        )
        max_slots = st.slider("최대 슬롯", min_value=1, max_value=10, value=4)

        st.markdown("---")

        # 5. 파라미터 범위
        st.subheader("📐 파라미터 범위")

        sma_range = st.text_input(
            "SMA Period",
            value="3,4,5,6,7",
            help="쉼표로 구분된 값 입력 (예: 3,4,5,6,7)",
        )
        trend_range = st.text_input(
            "Trend SMA Period",
            value="8,10,12,14",
            help="쉼표로 구분된 값 입력",
        )
        short_noise = st.text_input(
            "Short Noise Period (선택)",
            value="",
            help="비워두면 SMA Period와 동일",
        )
        long_noise = st.text_input(
            "Long Noise Period (선택)",
            value="",
            help="비워두면 Trend SMA Period와 동일",
        )

        st.markdown("---")

        # 6. 인터벌
        interval = st.selectbox(
            "데이터 인터벌",
            options=["minute240", "day", "week"],
            format_func=lambda x: {"minute240": "4시간", "day": "일봉", "week": "주봉"}[x],
            index=1,
        )

        st.markdown("---")

        # 7. 티커 선택
        st.subheader("📈 티커 선택")
        available, missing = validate_data_availability(DEFAULT_TICKERS, interval)

        selected_tickers = st.multiselect(
            "티커",
            options=available if available else DEFAULT_TICKERS,
            default=available[:2] if available else [],
        )

        st.markdown("---")

        # 8. 병렬 처리
        workers = st.slider(
            "병렬 워커 수",
            min_value=1,
            max_value=8,
            value=4,
            help="CPU 코어 수에 맞게 조정하세요",
        )

        st.markdown("---")

        # 실행 버튼
        run_button = st.button(
            "🚀 최적화 실행",
            type="primary",
            use_container_width=True,
            disabled=not selected_tickers,
        )

    # ===== 메인 화면 =====

    # 검증
    if not selected_tickers:
        st.warning("⚠️ 최소 1개 이상의 티커를 선택하세요.")
        _show_help()
        return

    # 파라미터 범위 파싱
    try:
        param_grid = _parse_param_grid(sma_range, trend_range, short_noise, long_noise)
    except ValueError as e:
        st.error(f"❌ 파라미터 범위 오류: {e}")
        return

    # 설정 요약
    _show_config_summary(
        strategy_type, method, metric, param_grid, selected_tickers, interval, n_iter
    )

    # 최적화 실행
    if run_button:
        _run_optimization(
            strategy_type=strategy_type,
            param_grid=param_grid,
            tickers=selected_tickers,
            interval=interval,
            metric=metric,
            method=method,
            n_iter=n_iter,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            max_slots=max_slots,
            workers=workers,
        )

    # 이전 결과 표시
    if "optimization_result" in st.session_state:
        _display_optimization_results()


def _parse_param_grid(
    sma_range: str,
    trend_range: str,
    short_noise: str,
    long_noise: str,
) -> dict[str, list[int]]:
    """파라미터 범위 파싱.

    Args:
        sma_range: SMA period 범위
        trend_range: Trend SMA period 범위
        short_noise: Short noise period 범위
        long_noise: Long noise period 범위

    Returns:
        파라미터 그리드 딕셔너리

    Raises:
        ValueError: 파싱 오류 시
    """
    def parse_range(s: str) -> list[int]:
        if not s.strip():
            return []
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    sma_values = parse_range(sma_range)
    trend_values = parse_range(trend_range)

    if not sma_values:
        raise ValueError("SMA Period 값을 입력하세요")
    if not trend_values:
        raise ValueError("Trend SMA Period 값을 입력하세요")

    param_grid = {
        "sma_period": sma_values,
        "trend_sma_period": trend_values,
        "short_noise_period": parse_range(short_noise) or sma_values,
        "long_noise_period": parse_range(long_noise) or trend_values,
    }

    return param_grid


def _show_config_summary(
    strategy_type: str,
    method: str,
    metric: str,
    param_grid: dict[str, list[int]],
    tickers: list[str],
    interval: str,
    n_iter: int,
) -> None:
    """설정 요약 표시."""
    with st.expander("📋 최적화 설정 요약", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📈 전략 & 방법**")
            st.write(f"- 전략: {strategy_type}")
            st.write(f"- 방법: {method}")
            st.write(f"- 메트릭: {metric}")

        with col2:
            st.markdown("**📐 파라미터 범위**")
            for key, values in param_grid.items():
                st.write(f"- {key}: {values}")

        with col3:
            st.markdown("**📊 데이터**")
            st.write(f"- 티커: {', '.join(tickers)}")
            st.write(f"- 인터벌: {interval}")

            # 총 조합 수 계산
            if method == "grid":
                total_combinations = 1
                for values in param_grid.values():
                    total_combinations *= len(values)
                st.metric("총 조합", f"{total_combinations:,}개")
            else:
                st.metric("탐색 횟수", f"{n_iter}회")


def _show_help() -> None:
    """도움말 표시."""
    st.info(
        """
        ### 🔧 파라미터 최적화 가이드

        **1. 전략 선택**
        - Vanilla VBO: 기본 변동성 돌파 전략
        - Legacy VBO: 노이즈 필터 포함 버전

        **2. 탐색 방법**
        - Grid Search: 모든 조합을 테스트 (정확하지만 느림)
        - Random Search: 무작위 샘플링 (빠르지만 최적해를 놓칠 수 있음)

        **3. 파라미터 범위**
        - 쉼표로 구분된 정수값 입력
        - 예: "3,4,5,6,7"

        **4. 최적화 메트릭**
        - Sharpe Ratio: 리스크 대비 수익 (권장)
        - CAGR: 연간 복리 수익률
        - Calmar Ratio: MDD 대비 수익률
        """
    )


def _run_optimization(
    strategy_type: str,
    param_grid: dict[str, list[int]],
    tickers: list[str],
    interval: str,
    metric: str,
    method: str,
    n_iter: int,
    initial_capital: float,
    fee_rate: float,
    max_slots: int,
    workers: int,
) -> None:
    """최적화 실행."""
    st.subheader("🔄 최적화 진행 중...")

    # 진행 표시
    progress_placeholder = st.empty()
    progress_placeholder.info("최적화를 시작합니다...")

    try:
        # 전략 팩토리 생성
        def create_strategy(**kwargs):
            if strategy_type == "vanilla":
                return create_vbo_strategy(
                    name="VanillaVBO",
                    use_trend_filter=False,
                    use_noise_filter=False,
                    **kwargs,
                )
            else:
                return create_vbo_strategy(
                    name="LegacyVBO",
                    use_trend_filter=True,
                    use_noise_filter=True,
                    **kwargs,
                )

        # 설정 생성
        config = BacktestConfig(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=fee_rate,
            max_slots=max_slots,
            use_cache=True,
        )

        progress_placeholder.info("백테스트 실행 중... (시간이 걸릴 수 있습니다)")

        # 최적화 실행
        result = optimize_strategy_parameters(
            strategy_factory=create_strategy,
            param_grid=param_grid,
            tickers=tickers,
            interval=interval,
            config=config,
            metric=metric,
            maximize=True,
            method=method,
            n_iter=n_iter,
            n_workers=workers,
        )

        # 결과 저장
        st.session_state.optimization_result = result
        st.session_state.optimization_metric = metric

        progress_placeholder.success("✅ 최적화 완료!")

    except Exception as e:
        logger.error(f"Optimization error: {e}", exc_info=True)
        progress_placeholder.error(f"❌ 최적화 실패: {e}")


def _display_optimization_results() -> None:
    """최적화 결과 표시."""
    result = st.session_state.optimization_result
    metric = st.session_state.optimization_metric

    st.subheader("📊 최적화 결과")

    # 최적 파라미터
    st.markdown("### 🏆 최적 파라미터")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**파라미터**")
        for key, value in result.best_params.items():
            st.write(f"- {key}: **{value}**")

    with col2:
        st.markdown("**성능**")
        st.metric(f"Best {metric}", f"{result.best_score:.4f}")

    # 전체 결과 테이블
    st.markdown("### 📋 전체 결과")

    import pandas as pd

    # 결과를 DataFrame으로 변환
    data = []
    for params, score in zip(result.all_params, result.all_scores):
        row = params.copy()
        row[metric] = score
        data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values(metric, ascending=False)

    st.dataframe(df, use_container_width=True, height=400)

    # 상위 10개 결과
    st.markdown("### 🔝 Top 10 결과")
    top_10 = df.head(10)
    st.dataframe(top_10, use_container_width=True)
