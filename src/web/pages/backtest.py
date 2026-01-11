"""Backtest page.

백테스트 실행 및 결과 표시 페이지.
"""

import numpy as np
import streamlit as st

from src.backtester.models import BacktestConfig
from src.utils.logger import get_logger
from src.web.components.charts.equity_curve import render_equity_curve
from src.web.components.charts.monthly_heatmap import render_monthly_heatmap
from src.web.components.charts.underwater import render_underwater_curve
from src.web.components.charts.yearly_bar import render_yearly_bar_chart
from src.web.components.metrics.metrics_display import (
    render_metrics_cards,
    render_statistical_significance,
)
from src.web.components.sidebar.asset_selector import render_asset_selector
from src.web.components.sidebar.date_config import render_date_config
from src.web.components.sidebar.strategy_selector import render_strategy_selector
from src.web.components.sidebar.trading_config import render_trading_config
from src.web.services.backtest_runner import run_backtest_service
from src.web.services.data_loader import get_data_files, validate_data_availability
from src.web.services.metrics_calculator import calculate_extended_metrics

logger = get_logger(__name__)

__all__ = ["render_backtest_page"]


def render_backtest_page() -> None:
    """백테스트 페이지 렌더링."""
    st.header("📈 백테스트")

    # ===== 사이드바 설정 =====
    with st.sidebar:
        st.title("📊 백테스트 설정")
        st.markdown("---")

        # 1. 날짜 설정
        start_date, end_date = render_date_config()
        st.markdown("---")

        # 2. 거래 설정
        trading_config = render_trading_config()
        st.markdown("---")

        # 3. 전략 선택
        strategy_name, strategy_params = render_strategy_selector()
        st.markdown("---")

        # 4. 자산 선택
        selected_tickers = render_asset_selector()
        st.markdown("---")

        # 실행 버튼
        run_button = st.button(
            "🚀 백테스트 실행",
            type="primary",
            use_container_width=True,
        )

    # ===== 메인 화면 =====

    # 검증
    if not strategy_name:
        st.warning("⚠️ 전략을 선택하세요.")
        return

    if not selected_tickers:
        st.warning("⚠️ 최소 1개 이상의 자산을 선택하세요.")
        return

    # 데이터 가용성 체크
    available_tickers, missing_tickers = validate_data_availability(
        selected_tickers, trading_config.interval
    )

    if missing_tickers:
        st.warning(
            f"⚠️ 다음 자산의 데이터가 없습니다: {', '.join(missing_tickers)}\n\n"
            f"사용 가능한 자산: {', '.join(available_tickers) if available_tickers else '없음'}"
        )

        if not available_tickers:
            st.error("❌ 사용 가능한 데이터가 없습니다. 데이터 수집을 먼저 진행하세요.")
            st.code("uv run python scripts/collect_data.py")
            return

    # 백테스트 실행
    if run_button:
        with st.spinner("백테스트 실행 중..."):
            # BacktestConfig 생성
            config = BacktestConfig(
                initial_capital=trading_config.initial_capital,
                fee_rate=trading_config.fee_rate,
                slippage_rate=trading_config.slippage_rate,
                max_slots=trading_config.max_slots,
                stop_loss_pct=trading_config.stop_loss_pct,
                take_profit_pct=trading_config.take_profit_pct,
                trailing_stop_pct=trading_config.trailing_stop_pct,
            )

            # 데이터 파일 경로
            data_files = get_data_files(available_tickers, trading_config.interval)

            # 백테스트 실행 (캐시됨)
            result = run_backtest_service(
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                data_files_dict={k: str(v) for k, v in data_files.items()},
                config_dict={
                    "initial_capital": config.initial_capital,
                    "fee_rate": config.fee_rate,
                    "slippage_rate": config.slippage_rate,
                    "max_slots": config.max_slots,
                    "stop_loss_pct": config.stop_loss_pct,
                    "take_profit_pct": config.take_profit_pct,
                    "trailing_stop_pct": config.trailing_stop_pct,
                },
                start_date_str=start_date.isoformat(),
                end_date_str=end_date.isoformat(),
            )

            if result:
                # 세션 스테이트에 저장
                st.session_state.backtest_result = result
                st.success("✅ 백테스트 완료!")
            else:
                st.error("❌ 백테스트 실행 실패. 로그를 확인하세요.")
                return

    # 결과 표시
    if "backtest_result" in st.session_state:
        result = st.session_state.backtest_result
        _display_results(result)
    else:
        # 안내 메시지
        st.info(
            "👈 왼쪽 사이드바에서 설정을 완료하고 **🚀 백테스트 실행** 버튼을 클릭하세요."
        )

        # 설정 요약 표시
        with st.expander("📋 현재 설정 요약", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                **📅 기간**
                - 시작: {start_date}
                - 종료: {end_date}
                - 기간: {(end_date - start_date).days}일

                **⏱️ 거래 설정**
                - 인터벌: {trading_config.interval}
                - 수수료: {trading_config.fee_rate:.2%}
                - 슬리피지: {trading_config.slippage_rate:.2%}
                """)

            with col2:
                st.markdown(f"""
                **📈 전략**
                - 이름: {strategy_name or '미선택'}
                - 파라미터: {len(strategy_params)}개

                **⚙️ 포트폴리오**
                - 초기자본: {trading_config.initial_capital:,.0f} KRW
                - 최대슬롯: {trading_config.max_slots}개
                - 자산: {len(selected_tickers)}개
                """)


def _display_results(result) -> None:
    """백테스트 결과 표시.
    
    Args:
        result: BacktestResult 객체
    """
    st.subheader("📊 백테스트 결과")

    # 거래 수익률 추출
    trade_returns = [t.pnl_pct / 100 for t in result.trades if t.pnl_pct is not None]

    # 확장 메트릭 계산
    equity = np.array(result.equity_curve)
    dates = np.array(result.dates) if hasattr(result, "dates") else np.arange(len(equity))

    extended_metrics = calculate_extended_metrics(
        equity=equity,
        trade_returns=trade_returns,
    )

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 개요",
        "📊 수익률 곡선",
        "📉 드로다운",
        "📅 월별 분석",
        "📆 연도별 분석",
        "🔬 통계 분석",
    ])

    with tab1:
        render_metrics_cards(extended_metrics)

        # 거래 내역
        if result.trades:
            with st.expander(f"📜 거래 내역 ({len(result.trades)}건)", expanded=False):
                import pandas as pd

                trades_df = pd.DataFrame(
                    [
                        {
                            "티커": t.ticker,
                            "진입일": t.entry_date,
                            "진입가": f"{t.entry_price:,.0f}",
                            "청산일": t.exit_date or "-",
                            "청산가": f"{t.exit_price:,.0f}" if t.exit_price else "-",
                            "수량": f"{t.amount:.4f}",
                            "손익": f"{t.pnl:,.0f}",
                            "수익률": f"{t.pnl_pct:.2f}%",
                        }
                        for t in result.trades[:100]  # 최대 100개만 표시
                    ]
                )

                st.dataframe(trades_df, use_container_width=True, height=400)

    with tab2:
        render_equity_curve(dates, equity)

    with tab3:
        render_underwater_curve(dates, equity)

    with tab4:
        render_monthly_heatmap(dates, equity)

    with tab5:
        render_yearly_bar_chart(dates, equity)

    with tab6:
        render_statistical_significance(extended_metrics)
