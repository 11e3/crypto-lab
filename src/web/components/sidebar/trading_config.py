"""Trading configuration component.

거래 관련 설정 (인터벌, 수수료, 슬리피지 등) UI 컴포넌트.
"""

import streamlit as st

from src.data.collector_fetch import Interval
from src.web.config import get_web_settings

__all__ = ["render_trading_config", "TradingConfig"]


class TradingConfig:
    """거래 설정 데이터 클래스."""

    def __init__(
        self,
        interval: Interval,
        fee_rate: float,
        slippage_rate: float,
        initial_capital: float,
        max_slots: int,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        trailing_stop_pct: float | None = None,
    ) -> None:
        """거래 설정 초기화."""
        self.interval = interval
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.initial_capital = initial_capital
        self.max_slots = max_slots
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct


def render_trading_config() -> TradingConfig:
    """거래 설정 UI 렌더링.
    
    Returns:
        TradingConfig 객체
    """
    settings = get_web_settings()

    st.subheader("⏱️ 캔들 인터벌")
    interval_options = {
        "1분": "minute1",
        "3분": "minute3",
        "5분": "minute5",
        "10분": "minute10",
        "15분": "minute15",
        "30분": "minute30",
        "1시간": "minute60",
        "4시간": "minute240",
        "일봉": "day",
        "주봉": "week",
        "월봉": "month",
    }

    interval_label = st.selectbox(
        "인터벌 선택",
        options=list(interval_options.keys()),
        index=8,  # 기본값: 일봉
        help="백테스트에 사용할 캔들 인터벌",
    )
    interval: Interval = interval_options[interval_label]  # type: ignore

    st.markdown("---")

    # 거래 비용
    st.subheader("💰 거래 비용")

    col1, col2 = st.columns(2)

    with col1:
        fee_rate = st.number_input(
            "수수료율 (%)",
            min_value=0.0,
            max_value=1.0,
            value=settings.default_fee_rate * 100,
            step=0.01,
            format="%.3f",
            help="거래당 수수료율 (0.05% = 업비트 기본)",
        ) / 100

    with col2:
        slippage_rate = st.number_input(
            "슬리피지율 (%)",
            min_value=0.0,
            max_value=1.0,
            value=settings.default_slippage_rate * 100,
            step=0.01,
            format="%.3f",
            help="시장 충격으로 인한 가격 미끄러짐",
        ) / 100

    st.markdown("---")

    # 포트폴리오 설정
    st.subheader("⚙️ 포트폴리오 설정")

    col1, col2 = st.columns(2)

    with col1:
        initial_capital = st.number_input(
            "초기 자본 (KRW)",
            min_value=1_000_000,
            max_value=1_000_000_000,
            value=int(settings.default_initial_capital),
            step=1_000_000,
            format="%d",
            help="백테스트 시작 자본금",
        )

    with col2:
        max_slots = st.number_input(
            "최대 슬롯 수",
            min_value=1,
            max_value=20,
            value=4,
            step=1,
            help="동시에 보유 가능한 최대 자산 수",
        )

    st.markdown("---")

    # 고급 설정
    with st.expander("🔧 고급 설정 (선택사항)"):
        enable_stop_loss = st.checkbox("스탑로스 활성화", value=False)
        stop_loss_pct = None
        if enable_stop_loss:
            stop_loss_pct = (
                st.slider(
                    "스탑로스 (%)",
                    min_value=1.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.5,
                    help="손실 제한 비율",
                )
                / 100
            )

        enable_take_profit = st.checkbox("테이크프로핏 활성화", value=False)
        take_profit_pct = None
        if enable_take_profit:
            take_profit_pct = (
                st.slider(
                    "테이크프로핏 (%)",
                    min_value=1.0,
                    max_value=50.0,
                    value=10.0,
                    step=1.0,
                    help="이익 실현 비율",
                )
                / 100
            )

        enable_trailing_stop = st.checkbox("트레일링 스탑 활성화", value=False)
        trailing_stop_pct = None
        if enable_trailing_stop:
            trailing_stop_pct = (
                st.slider(
                    "트레일링 스탑 (%)",
                    min_value=1.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.5,
                    help="최고가 대비 하락률 제한",
                )
                / 100
            )

    return TradingConfig(
        interval=interval,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        initial_capital=initial_capital,
        max_slots=max_slots,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop_pct=trailing_stop_pct,
    )
