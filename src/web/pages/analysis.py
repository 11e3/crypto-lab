"""Advanced analysis page.

고급 분석 (Monte Carlo, Walk-Forward) 페이지.
"""

import streamlit as st

from src.backtester import BacktestConfig, run_backtest, run_walk_forward_analysis
from src.backtester.analysis.monte_carlo import run_monte_carlo
from src.strategies.volatility_breakout import create_vbo_strategy
from src.utils.logger import get_logger
from src.web.services.data_loader import validate_data_availability

logger = get_logger(__name__)

__all__ = ["render_analysis_page"]

# 기본 티커
DEFAULT_TICKERS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL"]

# 최적화 메트릭 옵션
METRICS = [
    ("sharpe_ratio", "Sharpe Ratio"),
    ("cagr", "CAGR"),
    ("total_return", "Total Return"),
    ("calmar_ratio", "Calmar Ratio"),
    ("win_rate", "Win Rate"),
    ("profit_factor", "Profit Factor"),
]


def render_analysis_page() -> None:
    """고급 분석 페이지 렌더링."""
    st.header("📊 고급 분석")

    # 분석 유형 선택
    analysis_type = st.radio(
        "분석 유형 선택",
        options=["monte_carlo", "walk_forward"],
        format_func=lambda x: "🎲 Monte Carlo 시뮬레이션"
        if x == "monte_carlo"
        else "📈 Walk-Forward 분석",
        horizontal=True,
    )

    st.markdown("---")

    if analysis_type == "monte_carlo":
        _render_monte_carlo()
    else:
        _render_walk_forward()


def _render_monte_carlo() -> None:
    """Monte Carlo 시뮬레이션 페이지."""
    st.subheader("🎲 Monte Carlo 시뮬레이션")

    # ===== 사이드바 설정 =====
    with st.sidebar:
        st.title("🎲 Monte Carlo 설정")
        st.markdown("---")

        # 1. 전략 선택
        st.subheader("📈 전략")
        strategy_type = st.selectbox(
            "전략 유형",
            options=["vanilla", "minimal", "legacy", "momentum", "mean-reversion"],
            format_func=lambda x: {
                "vanilla": "Vanilla VBO",
                "minimal": "Minimal VBO",
                "legacy": "Legacy VBO",
                "momentum": "Momentum",
                "mean-reversion": "Mean Reversion",
            }.get(x, x),
            key="mc_strategy",
        )

        st.markdown("---")

        # 2. 시뮬레이션 설정
        st.subheader("🎯 시뮬레이션")
        n_simulations = st.slider(
            "시뮬레이션 횟수",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            key="mc_n_sim",
        )

        method = st.radio(
            "시뮬레이션 방법",
            options=["bootstrap", "parametric"],
            format_func=lambda x: "Bootstrap (리샘플링)"
            if x == "bootstrap"
            else "Parametric (정규분포)",
            horizontal=True,
            key="mc_method",
        )

        seed = st.number_input(
            "Random Seed (선택)",
            min_value=0,
            max_value=99999,
            value=0,
            help="0이면 랜덤, 그 외에는 재현 가능한 시드",
            key="mc_seed",
        )

        st.markdown("---")

        # 3. 거래 설정
        st.subheader("💰 거래 설정")
        initial_capital = st.number_input(
            "초기 자본",
            min_value=0.1,
            max_value=100.0,
            value=1.0,
            step=0.1,
            key="mc_capital",
        )
        fee_rate = st.number_input(
            "수수료율",
            min_value=0.0,
            max_value=0.01,
            value=0.0005,
            step=0.0001,
            format="%.4f",
            key="mc_fee",
        )
        max_slots = st.slider("최대 슬롯", min_value=1, max_value=10, value=4, key="mc_slots")

        st.markdown("---")

        # 4. 데이터 설정
        st.subheader("📊 데이터")
        interval = st.selectbox(
            "인터벌",
            options=["minute240", "day", "week"],
            format_func=lambda x: {"minute240": "4시간", "day": "일봉", "week": "주봉"}[x],
            index=1,
            key="mc_interval",
        )

        available, _ = validate_data_availability(DEFAULT_TICKERS, interval)
        selected_tickers = st.multiselect(
            "티커",
            options=available if available else DEFAULT_TICKERS,
            default=available[:2] if available else [],
            key="mc_tickers",
        )

        st.markdown("---")

        # 실행 버튼
        run_button = st.button(
            "🚀 시뮬레이션 실행",
            type="primary",
            use_container_width=True,
            disabled=not selected_tickers,
            key="mc_run",
        )

    # ===== 메인 화면 =====
    if not selected_tickers:
        st.warning("⚠️ 최소 1개 이상의 티커를 선택하세요.")
        _show_monte_carlo_help()
        return

    # 설정 요약
    with st.expander("📋 시뮬레이션 설정", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**전략**")
            st.write(f"유형: {strategy_type}")
        with col2:
            st.markdown("**시뮬레이션**")
            st.write(f"횟수: {n_simulations:,}")
            st.write(f"방법: {method}")
        with col3:
            st.markdown("**데이터**")
            st.write(f"티커: {', '.join(selected_tickers)}")
            st.write(f"인터벌: {interval}")

    # 실행
    if run_button:
        _run_monte_carlo(
            strategy_type=strategy_type,
            tickers=selected_tickers,
            interval=interval,
            n_simulations=n_simulations,
            method=method,
            seed=seed if seed > 0 else None,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            max_slots=max_slots,
        )

    # 결과 표시
    if "monte_carlo_result" in st.session_state:
        _display_monte_carlo_results()


def _show_monte_carlo_help() -> None:
    """Monte Carlo 도움말."""
    st.info(
        """
        ### 🎲 Monte Carlo 시뮬레이션 가이드

        **Monte Carlo 시뮬레이션이란?**
        - 과거 거래 결과를 무작위로 재배열하여 다양한 시나리오 생성
        - 전략의 리스크와 기대 수익률의 분포를 파악

        **시뮬레이션 방법**
        - **Bootstrap**: 실제 거래 결과를 리샘플링 (권장)
        - **Parametric**: 정규분포 가정 하에 시뮬레이션

        **결과 해석**
        - 신뢰구간: 95% 확률로 수익률이 이 범위 내에 있을 것
        - VaR: 최악의 경우 예상 손실
        """
    )


def _run_monte_carlo(
    strategy_type: str,
    tickers: list[str],
    interval: str,
    n_simulations: int,
    method: str,
    seed: int | None,
    initial_capital: float,
    fee_rate: float,
    max_slots: int,
) -> None:
    """Monte Carlo 시뮬레이션 실행."""
    progress = st.empty()
    progress.info("백테스트 실행 중...")

    try:
        # 전략 생성
        strategy = _create_strategy(strategy_type)

        # 설정
        config = BacktestConfig(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=fee_rate,
            max_slots=max_slots,
            use_cache=True,
        )

        # 백테스트 실행
        result = run_backtest(
            strategy=strategy,
            tickers=tickers,
            interval=interval,
            config=config,
        )

        progress.info(f"Monte Carlo 시뮬레이션 실행 중 ({n_simulations:,}회)...")

        # Monte Carlo 실행
        mc_result = run_monte_carlo(
            result=result,
            n_simulations=n_simulations,
            method=method,
            random_seed=seed,
        )

        # 결과 저장
        st.session_state.monte_carlo_result = mc_result
        st.session_state.backtest_result_for_mc = result

        progress.success("✅ 시뮬레이션 완료!")

    except Exception as e:
        logger.error(f"Monte Carlo error: {e}", exc_info=True)
        progress.error(f"❌ 시뮬레이션 실패: {e}")


def _display_monte_carlo_results() -> None:
    """Monte Carlo 결과 표시."""
    mc_result = st.session_state.monte_carlo_result
    backtest_result = st.session_state.backtest_result_for_mc

    st.subheader("📊 Monte Carlo 결과")

    # 기본 메트릭
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "원본 수익률",
            f"{backtest_result.total_return:.2%}",
        )
    with col2:
        st.metric(
            "평균 시뮬레이션 수익률",
            f"{mc_result.mean_return:.2%}",
        )
    with col3:
        st.metric(
            "95% VaR",
            f"{mc_result.var_95:.2%}",
            help="95% 신뢰수준에서 최대 예상 손실",
        )
    with col4:
        st.metric(
            "95% CVaR",
            f"{mc_result.cvar_95:.2%}",
            help="VaR 초과 시 평균 손실",
        )

    # 신뢰구간
    st.markdown("### 📈 수익률 신뢰구간")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("5th Percentile", f"{mc_result.percentile_5:.2%}")
    with col2:
        st.metric("50th Percentile (중앙값)", f"{mc_result.percentile_50:.2%}")
    with col3:
        st.metric("95th Percentile", f"{mc_result.percentile_95:.2%}")

    # 분포 차트
    st.markdown("### 📊 수익률 분포")
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=mc_result.simulated_returns,
            nbinsx=50,
            name="시뮬레이션 수익률",
            marker_color="lightblue",
        )
    )

    # 원본 수익률 라인
    fig.add_vline(
        x=backtest_result.total_return,
        line_dash="dash",
        line_color="red",
        annotation_text="원본 수익률",
    )

    fig.update_layout(
        title="Monte Carlo 수익률 분포",
        xaxis_title="수익률",
        yaxis_title="빈도",
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_walk_forward() -> None:
    """Walk-Forward 분석 페이지."""
    st.subheader("📈 Walk-Forward 분석")

    # ===== 사이드바 설정 =====
    with st.sidebar:
        st.title("📈 Walk-Forward 설정")
        st.markdown("---")

        # 1. 전략 선택
        st.subheader("📈 전략")
        strategy_type = st.selectbox(
            "전략 유형",
            options=["vanilla", "legacy"],
            format_func=lambda x: "Vanilla VBO" if x == "vanilla" else "Legacy VBO",
            key="wf_strategy",
        )

        st.markdown("---")

        # 2. 기간 설정
        st.subheader("📅 기간 설정")
        optimization_days = st.slider(
            "최적화 기간 (일)",
            min_value=90,
            max_value=730,
            value=365,
            step=30,
            key="wf_opt_days",
        )
        test_days = st.slider(
            "테스트 기간 (일)",
            min_value=30,
            max_value=180,
            value=90,
            step=30,
            key="wf_test_days",
        )
        step_days = st.slider(
            "스텝 크기 (일)",
            min_value=30,
            max_value=180,
            value=90,
            step=30,
            key="wf_step_days",
        )

        st.markdown("---")

        # 3. 최적화 메트릭
        st.subheader("📊 최적화 메트릭")
        metric = st.selectbox(
            "최적화 대상",
            options=[m[0] for m in METRICS],
            format_func=lambda x: next(name for code, name in METRICS if code == x),
            key="wf_metric",
        )

        st.markdown("---")

        # 4. 파라미터 범위
        st.subheader("📐 파라미터 범위")
        sma_range = st.text_input(
            "SMA Period",
            value="4,5,6",
            key="wf_sma",
        )
        trend_range = st.text_input(
            "Trend SMA Period",
            value="8,10,12",
            key="wf_trend",
        )

        st.markdown("---")

        # 5. 거래 설정
        st.subheader("💰 거래 설정")
        initial_capital = st.number_input(
            "초기 자본",
            min_value=0.1,
            max_value=100.0,
            value=1.0,
            step=0.1,
            key="wf_capital",
        )
        fee_rate = st.number_input(
            "수수료율",
            min_value=0.0,
            max_value=0.01,
            value=0.0005,
            format="%.4f",
            key="wf_fee",
        )
        max_slots = st.slider("최대 슬롯", min_value=1, max_value=10, value=4, key="wf_slots")

        st.markdown("---")

        # 6. 데이터 설정
        st.subheader("📊 데이터")
        interval = st.selectbox(
            "인터벌",
            options=["minute240", "day", "week"],
            format_func=lambda x: {"minute240": "4시간", "day": "일봉", "week": "주봉"}[x],
            index=1,
            key="wf_interval",
        )

        available, _ = validate_data_availability(DEFAULT_TICKERS, interval)
        selected_tickers = st.multiselect(
            "티커",
            options=available if available else DEFAULT_TICKERS,
            default=available[:2] if available else [],
            key="wf_tickers",
        )

        st.markdown("---")

        # 7. 병렬 처리
        workers = st.slider("병렬 워커 수", min_value=1, max_value=8, value=4, key="wf_workers")

        st.markdown("---")

        # 실행 버튼
        run_button = st.button(
            "🚀 분석 실행",
            type="primary",
            use_container_width=True,
            disabled=not selected_tickers,
            key="wf_run",
        )

    # ===== 메인 화면 =====
    if not selected_tickers:
        st.warning("⚠️ 최소 1개 이상의 티커를 선택하세요.")
        _show_walk_forward_help()
        return

    # 파라미터 파싱
    try:
        param_grid = {
            "sma_period": [int(x.strip()) for x in sma_range.split(",")],
            "trend_sma_period": [int(x.strip()) for x in trend_range.split(",")],
        }
        param_grid["short_noise_period"] = param_grid["sma_period"]
        param_grid["long_noise_period"] = param_grid["trend_sma_period"]
    except ValueError as e:
        st.error(f"❌ 파라미터 오류: {e}")
        return

    # 설정 요약
    with st.expander("📋 Walk-Forward 설정", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**기간 설정**")
            st.write(f"최적화: {optimization_days}일")
            st.write(f"테스트: {test_days}일")
            st.write(f"스텝: {step_days}일")
        with col2:
            st.markdown("**파라미터**")
            st.write(f"SMA: {param_grid['sma_period']}")
            st.write(f"Trend: {param_grid['trend_sma_period']}")
        with col3:
            st.markdown("**데이터**")
            st.write(f"티커: {', '.join(selected_tickers)}")
            st.write(f"인터벌: {interval}")

    # 실행
    if run_button:
        _run_walk_forward(
            strategy_type=strategy_type,
            param_grid=param_grid,
            tickers=selected_tickers,
            interval=interval,
            optimization_days=optimization_days,
            test_days=test_days,
            step_days=step_days,
            metric=metric,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            max_slots=max_slots,
            workers=workers,
        )

    # 결과 표시
    if "walk_forward_result" in st.session_state:
        _display_walk_forward_results()


def _show_walk_forward_help() -> None:
    """Walk-Forward 도움말."""
    st.info(
        """
        ### 📈 Walk-Forward 분석 가이드

        **Walk-Forward 분석이란?**
        - 데이터를 여러 기간으로 나누어 순차적으로 테스트
        - 과적합(Overfitting) 방지를 위한 분석 방법

        **기간 설정**
        - **최적화 기간**: 파라미터를 찾기 위한 학습 기간
        - **테스트 기간**: 찾은 파라미터를 검증하는 기간
        - **스텝 크기**: 윈도우 이동 간격

        **결과 해석**
        - In-Sample vs Out-of-Sample 성능 비교
        - WFE (Walk-Forward Efficiency): 1에 가까울수록 좋음
        """
    )


def _run_walk_forward(
    strategy_type: str,
    param_grid: dict[str, list[int]],
    tickers: list[str],
    interval: str,
    optimization_days: int,
    test_days: int,
    step_days: int,
    metric: str,
    initial_capital: float,
    fee_rate: float,
    max_slots: int,
    workers: int,
) -> None:
    """Walk-Forward 분석 실행."""
    progress = st.empty()
    progress.info("Walk-Forward 분석 실행 중... (시간이 오래 걸릴 수 있습니다)")

    try:
        # 전략 팩토리
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

        # 설정
        config = BacktestConfig(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=fee_rate,
            max_slots=max_slots,
            use_cache=True,
        )

        # Walk-Forward 실행
        result = run_walk_forward_analysis(
            strategy_factory=create_strategy,
            param_grid=param_grid,
            tickers=tickers,
            interval=interval,
            config=config,
            optimization_days=optimization_days,
            test_days=test_days,
            step_days=step_days,
            metric=metric,
            n_workers=workers,
        )

        # 결과 저장
        st.session_state.walk_forward_result = result

        progress.success("✅ Walk-Forward 분석 완료!")

    except Exception as e:
        logger.error(f"Walk-Forward error: {e}", exc_info=True)
        progress.error(f"❌ 분석 실패: {e}")


def _display_walk_forward_results() -> None:
    """Walk-Forward 결과 표시."""
    result = st.session_state.walk_forward_result

    st.subheader("📊 Walk-Forward 결과")

    # 요약 메트릭
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "In-Sample 평균 수익률",
            f"{result.avg_in_sample_return:.2%}",
        )
    with col2:
        st.metric(
            "Out-of-Sample 평균 수익률",
            f"{result.avg_out_of_sample_return:.2%}",
        )
    with col3:
        wfe = result.walk_forward_efficiency
        st.metric(
            "Walk-Forward Efficiency",
            f"{wfe:.2f}",
            help="1에 가까울수록 과적합이 적음",
        )
    with col4:
        st.metric(
            "분석 기간 수",
            f"{len(result.periods)}개",
        )

    # 기간별 결과 테이블
    st.markdown("### 📋 기간별 결과")

    import pandas as pd

    data = []
    for i, period in enumerate(result.periods):
        data.append(
            {
                "기간": i + 1,
                "시작일": period.start_date.strftime("%Y-%m-%d"),
                "종료일": period.end_date.strftime("%Y-%m-%d"),
                "최적 파라미터": str(period.best_params),
                "In-Sample 수익률": f"{period.in_sample_return:.2%}",
                "Out-of-Sample 수익률": f"{period.out_of_sample_return:.2%}",
            }
        )

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # 차트
    st.markdown("### 📈 기간별 수익률 비교")

    import plotly.graph_objects as go

    periods = list(range(1, len(result.periods) + 1))
    in_sample = [p.in_sample_return for p in result.periods]
    out_sample = [p.out_of_sample_return for p in result.periods]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="In-Sample",
            x=periods,
            y=in_sample,
            marker_color="lightblue",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Out-of-Sample",
            x=periods,
            y=out_sample,
            marker_color="orange",
        )
    )

    fig.update_layout(
        title="Walk-Forward 기간별 수익률",
        xaxis_title="기간",
        yaxis_title="수익률",
        barmode="group",
    )

    st.plotly_chart(fig, use_container_width=True)


def _create_strategy(strategy_type: str):
    """전략 객체 생성."""
    from src.strategies.mean_reversion import MeanReversionStrategy
    from src.strategies.momentum import MomentumStrategy

    if strategy_type == "vanilla":
        return create_vbo_strategy(
            name="VanillaVBO",
            use_trend_filter=False,
            use_noise_filter=False,
        )
    elif strategy_type == "minimal":
        return create_vbo_strategy(
            name="MinimalVBO",
            use_trend_filter=False,
            use_noise_filter=False,
        )
    elif strategy_type == "legacy":
        return create_vbo_strategy(
            name="LegacyVBO",
            use_trend_filter=True,
            use_noise_filter=True,
        )
    elif strategy_type == "momentum":
        return MomentumStrategy(name="Momentum")
    elif strategy_type == "mean-reversion":
        return MeanReversionStrategy(name="MeanReversion")
    else:
        return create_vbo_strategy(name="DefaultVBO")
