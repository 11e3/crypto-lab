"""Backtest page.

백테스트 실행 및 결과 표시 페이지.
"""

from __future__ import annotations

from datetime import date as date_type
from typing import TYPE_CHECKING

import numpy as np
import streamlit as st

from src.backtester.models import BacktestConfig, BacktestResult
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.web.components.sidebar.trading_config import TradingConfig
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
    """백테스트 페이지 렌더링 (탭 기반 UI)."""
    st.header("📈 백테스트")

    # 탭 생성: 설정 탭과 결과 탭
    if "backtest_result" in st.session_state:
        # 결과가 있으면 설정과 결과 탭 모두 표시
        tab1, tab2 = st.tabs(["⚙️ 설정", "📊 결과"])
    else:
        # 결과가 없으면 설정 탭만 표시
        tab1 = st.tabs(["⚙️ 설정"])[0]
        tab2 = None

    # ===== 설정 탭 =====
    with tab1:
        _render_settings_tab()

    # ===== 결과 탭 =====
    if tab2 is not None:
        with tab2:
            if "backtest_result" in st.session_state:
                _display_results(st.session_state.backtest_result)
            else:
                st.info("백테스트를 실행하면 결과가 여기에 표시됩니다.")


def _render_settings_tab() -> None:
    """설정 탭 렌더링."""
    st.subheader("⚙️ 백테스트 설정")

    # 3개 컬럼으로 설정 구분
    col1, col2, col3 = st.columns([1, 1, 1])

    # ===== 컬럼 1: 날짜 & 거래 설정 =====
    with col1:
        st.markdown("### 📅 기간 설정")
        start_date, end_date = render_date_config()

        st.markdown("### 💰 거래 설정")
        trading_config = render_trading_config()

    # ===== 컬럼 2: 전략 설정 =====
    with col2:
        st.markdown("### 📈 전략 설정")
        strategy_name, strategy_params = render_strategy_selector()

    # ===== 컬럼 3: 자산 선택 =====
    with col3:
        st.markdown("### 🪙 자산 선택")
        selected_tickers = render_asset_selector()

    st.markdown("---")

    # 설정 요약
    with st.expander("📋 설정 요약", expanded=False):
        _show_config_summary(strategy_name, selected_tickers, trading_config, start_date, end_date)

    # 실행 버튼
    col_left, col_center, col_right = st.columns([1, 1, 1])
    with col_center:
        run_button = st.button(
            "🚀 백테스트 실행",
            type="primary",
            use_container_width=True,
            disabled=not strategy_name or not selected_tickers,
        )

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
                use_cache=False,
            )

            # 데이터 파일 경로
            data_files = get_data_files(available_tickers, trading_config.interval)

            if not data_files:
                st.error("❌ 데이터 파일을 찾을 수 없습니다.")
                return

            # 백테스트 실행 (캐싱을 위해 직렬화 가능한 타입으로 변환)
            data_files_dict = {ticker: str(path) for ticker, path in data_files.items()}
            config_dict = {
                "initial_capital": config.initial_capital,
                "fee_rate": config.fee_rate,
                "slippage_rate": config.slippage_rate,
                "max_slots": config.max_slots,
                "use_cache": config.use_cache,
            }
            start_date_str = start_date.isoformat() if start_date else None
            end_date_str = end_date.isoformat() if end_date else None

            result = run_backtest_service(
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                data_files_dict=data_files_dict,
                config_dict=config_dict,
                start_date_str=start_date_str,
                end_date_str=end_date_str,
            )

            if result:
                st.session_state.backtest_result = result
                st.success("✅ 백테스트 완료! '📊 결과' 탭에서 확인하세요.")
                st.rerun()  # 결과 탭 표시를 위해 페이지 새로고침
            else:
                st.error("❌ 백테스트 실행 실패")


def _show_config_summary(
    strategy_name: str,
    selected_tickers: list[str],
    trading_config: TradingConfig,
    start_date: date_type | None,
    end_date: date_type | None,
) -> None:
    """설정 요약 표시."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            **📈 전략**
            - 전략: {strategy_name}
            - 인터벌: {trading_config.interval}
            """
        )

    with col2:
        st.markdown(
            f"""
            **📅 기간**
            - 시작: {start_date if start_date else "전체"}
            - 종료: {end_date if end_date else "전체"}
            """
        )

    with col3:
        st.markdown(
            f"""
            **⚙️ 포트폴리오**
            - 초기자본: {trading_config.initial_capital:,.0f} KRW
            - 최대슬롯: {trading_config.max_slots}개
            - 자산: {len(selected_tickers)}개
            """
        )


def _display_results(result: BacktestResult) -> None:
    """백테스트 결과 표시.

    Args:
        result: BacktestResult 객체
    """
    st.subheader("📊 백테스트 결과")

    # 거래 수익률 추출
    trade_returns = [t.pnl_pct / 100 for t in result.trades if t.pnl_pct is not None]

    # 확장 메트릭 계산 (세션 스테이트에 캐싱)
    equity = np.array(result.equity_curve)
    dates = np.array(result.dates) if hasattr(result, "dates") else np.arange(len(equity))

    # 캐시 키 생성 (equity의 해시로 메트릭 캐싱)
    cache_key = f"metrics_{hash(equity.tobytes())}"

    if cache_key not in st.session_state:
        # 메트릭 계산 (처음 한 번만)
        st.session_state[cache_key] = calculate_extended_metrics(
            equity=equity,
            trade_returns=trade_returns,
        )

    extended_metrics = st.session_state[cache_key]

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📈 개요",
            "📊 수익률 곡선",
            "📉 드로다운",
            "📅 월별 분석",
            "📆 연도별 분석",
            "🔬 통계",
        ]
    )

    with tab1:
        # 메트릭 카드
        render_metrics_cards(extended_metrics)

        # 거래 내역
        if result.trades:
            st.markdown("### 📋 거래 내역")

            import pandas as pd

            trades_df = pd.DataFrame(
                [
                    {
                        "티커": t.ticker,
                        "진입일": str(t.entry_date),
                        "진입가": f"{t.entry_price:,.0f}",
                        "청산일": str(t.exit_date) if t.exit_date else "-",
                        "청산가": f"{t.exit_price:,.0f}" if t.exit_price else "-",
                        "수익": f"{t.pnl:,.0f}",
                        "수익률": f"{t.pnl_pct:.2f}%",
                    }
                    for t in result.trades[-100:]  # 최근 100개만
                ]
            )

            st.dataframe(trades_df, width="stretch", height=400)

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
