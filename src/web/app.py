"""Streamlit Backtest UI - Main Entry Point.

백테스팅 웹 인터페이스 메인 애플리케이션.
"""

import streamlit as st

from src.utils.logger import get_logger, setup_logging

# 로깅 초기화
setup_logging()
logger = get_logger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="Crypto Quant Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/11e3/crypto-quant-system",
        "Report a bug": "https://github.com/11e3/crypto-quant-system/issues",
        "About": "# Crypto Quant Backtest UI\n이벤트 드리븐 백테스팅 엔진 기반 웹 인터페이스",
    },
)


def main() -> None:
    """메인 애플리케이션 진입점."""
    st.title("📊 Crypto Quant Backtest System")
    st.markdown("---")

    # 멀티 페이지 구조
    pages = {
        "🏠 홈": show_home,
        "📥 데이터 수집": show_data_collect,
        "📈 백테스트": show_backtest,
        "🔧 파라미터 최적화": show_optimization,
        "📊 고급 분석": show_analysis,
    }

    # 사이드바에 페이지 선택
    st.sidebar.title("📋 Navigation")
    selection = st.sidebar.radio("페이지 선택", list(pages.keys()))

    # 선택된 페이지 실행
    pages[selection]()


def show_home() -> None:
    """홈 페이지."""
    st.header("🏠 Welcome to Crypto Quant Backtest")

    st.markdown(
        """
    ## 🎯 주요 기능

    ### 📥 데이터 수집
    - **Upbit API** 연동으로 실시간 OHLCV 데이터 수집
    - **다양한 인터벌**: 1분봉부터 월봉까지 지원
    - **증분 업데이트**: 기존 데이터에 새 데이터만 추가

    ### 📈 백테스트
    - **이벤트 드리븐 엔진** 사용으로 정확한 시뮬레이션
    - **동적 파라미터 설정**: 전략 선택 시 자동으로 파라미터 UI 생성
    - **다중 자산 지원**: 여러 암호화폐 동시 백테스트
    - **실시간 메트릭**: CAGR, Sharpe, MDD 등 30+ 메트릭
    - **인터랙티브 차트**: Plotly 기반 줌/팬 가능한 차트

    ### 🔧 파라미터 최적화
    - **Grid Search**: 모든 조합 테스트
    - **Random Search**: 빠른 탐색
    - **병렬 처리**: 멀티코어 활용
    - **메트릭 선택**: Sharpe, CAGR, Calmar 등

    ### 📊 고급 분석
    - **Walk-Forward Analysis**: 과적합 방지
    - **Monte Carlo**: 리스크 시뮬레이션
    - **VaR/CVaR**: 포트폴리오 리스크 분석

    ---

    ## 🚀 시작하기

    1. **📥 데이터 수집**: 먼저 티커와 인터벌을 선택하여 데이터 수집
    2. **📈 백테스트**: 전략과 파라미터 설정 후 백테스트 실행
    3. **🔧 최적화**: 파라미터 그리드를 설정하여 최적 조합 탐색
    4. **📊 고급 분석**: Monte Carlo, Walk-Forward로 전략 검증

    ---

    ## 📚 지원 전략

    - **VBO (Volatility Breakout)**: 변동성 돌파 전략
    - **Momentum**: 모멘텀 추세 추종
    - **Mean Reversion**: 평균 회귀 전략
    - **Pair Trading**: 페어 트레이딩
    - **ORB (Opening Range Breakout)**: 시가 범위 돌파

    """
    )

    # 시스템 상태
    with st.expander("🔍 시스템 상태"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("등록된 전략", "5개")

        with col2:
            st.metric("사용 가능한 지표", "20+")

        with col3:
            st.metric("지원 자산", "100+")


def show_data_collect() -> None:
    """데이터 수집 페이지."""
    from src.web.pages.data_collect import render_data_collect_page

    render_data_collect_page()


def show_backtest() -> None:
    """백테스트 페이지."""
    from src.web.pages.backtest import render_backtest_page

    render_backtest_page()


def show_optimization() -> None:
    """파라미터 최적화 페이지."""
    from src.web.pages.optimization import render_optimization_page

    render_optimization_page()


def show_analysis() -> None:
    """고급 분석 페이지."""
    from src.web.pages.analysis import render_analysis_page

    render_analysis_page()


if __name__ == "__main__":
    main()
