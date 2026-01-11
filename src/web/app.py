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
    - **순열 검정**: 통계적 유의성 검증
    - **Monte Carlo**: 리스크 시뮬레이션
    - **VaR/CVaR**: 포트폴리오 리스크 분석
    
    ---
    
    ## 🚀 시작하기
    
    1. 왼쪽 사이드바에서 **📈 백테스트** 선택
    2. 기간, 전략, 파라미터 설정
    3. **🚀 백테스트 실행** 버튼 클릭
    4. 결과 분석 및 최적화
    
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


def show_backtest() -> None:
    """백테스트 페이지."""
    from src.web.pages.backtest import render_backtest_page

    render_backtest_page()


def show_optimization() -> None:
    """파라미터 최적화 페이지 (구현 예정)."""
    st.header("🔧 파라미터 최적화")
    st.info("🚧 최적화 페이지는 Phase 4에서 구현됩니다.")


def show_analysis() -> None:
    """고급 분석 페이지 (구현 예정)."""
    st.header("📊 고급 분석")
    st.info("🚧 고급 분석 페이지는 Phase 5에서 구현됩니다.")


if __name__ == "__main__":
    main()
