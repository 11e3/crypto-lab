"""
Phase 2: 동적 슬리피지 모델 (Slippage Model v2)

기존 문제점:
- 고정 슬리피지율 (0.05%)
- 시장 조건 무시
- 거래량 반영 안 됨

개선사항:
- 스프레드 기반 슬리피지 (호가 범위)
- 호가 변동성 적응
- 거래량 가중 슬리피지
- 시간대별 조정 (유동성 변화)
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class MarketCondition:
    """시장 상황 정보."""

    spread_ratio: float  # 현재 스프레드 / 최근 평균 스프레드
    volume_ratio: float  # 현재 거래량 / 평균 거래량
    volatility_level: int  # 0=Low, 1=Medium, 2=High
    time_of_day: int  # 0-23 시간


class DynamicSlippageModel:
    """
    동적 슬리피지 모델.

    슬리피지 = 주문 체결 시 발생하는 가격 이동

    요인:
    1. 기본 스프레드 (호가차)
    2. 호가 변동성 (spread volatility)
    3. 거래량 (volume impact)
    4. 시장 조건 (market condition)
    """

    def __init__(
        self,
        base_spread: float = 0.01,  # Upbit 기본 스프레드 ~0.01%
        maker_fee: float = 0.0,  # Upbit maker fee
        taker_fee: float = 0.05,
    ):  # Upbit taker fee (0.05%)
        """
        Args:
            base_spread: 기본 스프레드 (%)
            maker_fee: Maker 수수료 (%)
            taker_fee: Taker 수수료 (%)
        """
        self.base_spread = base_spread
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

    def calculate_spread(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        호가차 (Bid-Ask Spread) 추정.

        데이터가 체결가만 제공하므로 근사:
        spread = (high - low) / (high + low) * 100

        실제 거래소에서 bid-ask 데이터가 있으면 직접 계산 가능.

        Args:
            data: OHLC 데이터
            window: 추정 윈도우

        Returns:
            스프레드 시리즈 (%)
        """
        high = data["high"]
        low = data["low"]

        # 호가차 추정 (mid price 기반)
        spread = (high - low) / ((high + low) / 2) * 100

        return spread

    def calculate_spread_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        호가차 변동성.

        spread_vol = 호가차의 표준편차

        → 호가차가 자주 변하면 슬리피지 증가.

        Args:
            data: OHLC 데이터
            window: 슬라이딩 윈도우

        Returns:
            호가차 변동성 시리즈
        """
        spread = self.calculate_spread(data)
        spread_vol = spread.rolling(window=window).std()

        return spread_vol

    def calculate_volume_impact(
        self,
        data: pd.DataFrame,
        order_size: float = 1.0,  # BTC 개수
        window: int = 20,
    ) -> pd.Series:
        """
        거래량 영향도 (Volume Impact).

        volume_impact = order_size / avg_volume

        주문 크기가 클수록, 평균 거래량이 작을수록 슬리피지 증가.

        예:
        - 평균 거래량 100 BTC, 주문 1 BTC → impact 1%
        - 평균 거래량 100 BTC, 주문 10 BTC → impact 10%

        Args:
            data: OHLC 데이터 (volume 컬럼 필수)
            order_size: 주문 크기 (BTC)
            window: 평균 거래량 계산 윈도우

        Returns:
            거래량 영향도 시리즈 (%)
        """
        volume = data["volume"]
        avg_volume = volume.rolling(window=window).mean()

        # 거래량 영향도 (%)
        volume_impact = (order_size / (avg_volume + 1e-8)) * 100

        # 제약: 최대 20% (극한 상황)
        volume_impact = volume_impact.clip(upper=20.0)

        return volume_impact

    def calculate_dynamic_slippage(
        self, data: pd.DataFrame, condition: MarketCondition, order_size: float = 1.0
    ) -> float:
        """
        동적 슬리피지 계산.

        Total Slippage = Base + Spread + Volume Impact + Condition Adjustment

        Base: Taker 수수료 (0.05%)
        Spread: 현재 호가차 * 시장 조건
        Volume Impact: 거래량 영향도
        Condition: 시장 상황 추가 조정

        Args:
            data: OHLC 데이터
            condition: 시장 조건 정보
            order_size: 주문 크기

        Returns:
            슬리피지율 (%)
        """
        # 1. 기본: Taker 수수료
        slippage = self.taker_fee

        # 2. 호가차 + 호가차 변동성
        spread = self.calculate_spread(data).iloc[-1]
        spread_vol = self.calculate_spread_volatility(data).iloc[-1]

        # 스프레드 기여도: 기본 스프레드 + 변동성 * 시장 스트레스
        spread_component = spread + (spread_vol * (1.0 + condition.spread_ratio * 0.5))
        slippage += spread_component

        # 3. 거래량 영향도
        volume_impact = self.calculate_volume_impact(data, order_size).iloc[-1]
        slippage += volume_impact * (0.05 * condition.volume_ratio)  # 거래량 가중

        # 4. 시장 조건 추가 조정
        if condition.volatility_level == 2:  # High 변동성
            slippage *= 1.2  # +20%
        elif condition.volatility_level == 0:  # Low 변동성
            slippage *= 0.85  # -15%

        # 5. 시간대 조정 (유동성 패턴)
        # 9-18시 (거래량 높음): -10%
        # 0-6시 (거래량 낮음): +15%
        if 9 <= condition.time_of_day < 18:
            slippage *= 0.9
        elif 0 <= condition.time_of_day < 6:
            slippage *= 1.15

        return float(slippage)

    def estimate_execution_price(
        self,
        price: float,
        order_side: str,  # 'buy' or 'sell'
        slippage_pct: float,
    ) -> float:
        """
        슬리피지를 적용한 예상 체결가.

        Buy:  execution_price = price * (1 + slippage)
        Sell: execution_price = price * (1 - slippage)

        Args:
            price: 공시가
            order_side: 'buy' 또는 'sell'
            slippage_pct: 슬리피지율 (%)

        Returns:
            예상 체결가
        """
        slippage_ratio = slippage_pct / 100

        if order_side.lower() == "buy":
            return price * (1 + slippage_ratio)
        elif order_side.lower() == "sell":
            return price * (1 - slippage_ratio)
        else:
            raise ValueError(f"Invalid order_side: {order_side}")


class UpbitSlippageEstimator(DynamicSlippageModel):
    """
    Upbit 거래소 전용 슬리피지 추정기.

    Upbit 특성:
    - Maker fee: 0% (일부 조건)
    - Taker fee: 0.05%
    - 24/7 운영 (시간대 변동성 있음)
    - 암호화폐 특성상 변동성 높음
    """

    def __init__(self) -> None:
        super().__init__(
            base_spread=0.01,  # Upbit 평균 스프레드
            maker_fee=0.0,
            taker_fee=0.05,
        )

    def estimate_transaction_cost(
        self,
        entry_price: float,
        exit_price: float,
        entry_slippage: float,
        exit_slippage: float,
        position_size: float = 1.0,
    ) -> dict[str, float]:
        """
        왕복 거래 비용 계산.

        총 비용 = Entry 슬리피지 + Exit 슬리피지 + 수수료

        Args:
            entry_price: 진입가
            exit_price: 청산가
            entry_slippage: 진입 슬리피지율 (%)
            exit_slippage: 청산 슬리피지율 (%)
            position_size: 포지션 크기

        Returns:
            비용 분석 딕셔너리
        """
        # 1. Entry 비용
        actual_entry_price = entry_price * (1 + entry_slippage / 100)
        entry_cost = actual_entry_price * position_size

        # 2. Exit 비용
        actual_exit_price = exit_price * (1 - exit_slippage / 100)
        exit_value = actual_exit_price * position_size

        # 3. 수익률
        gross_pnl = exit_value - entry_cost
        gross_pnl_pct = (gross_pnl / entry_cost) * 100 if entry_cost > 0 else 0

        # 4. 총 슬리피지
        total_slippage_pct = (entry_slippage + exit_slippage) + (
            (actual_entry_price - entry_price) + (exit_price - actual_exit_price)
        ) / entry_price * 100

        return {
            "entry_price": entry_price,
            "actual_entry_price": actual_entry_price,
            "exit_price": exit_price,
            "actual_exit_price": actual_exit_price,
            "entry_cost": entry_cost,
            "exit_value": exit_value,
            "gross_pnl": gross_pnl,
            "gross_pnl_pct": gross_pnl_pct,
            "entry_slippage_pct": entry_slippage,
            "exit_slippage_pct": exit_slippage,
            "total_slippage_pct": total_slippage_pct,
        }


if __name__ == "__main__":
    print("Slippage Model v2 loaded successfully")
    print("Features: Dynamic spread, volume impact, time-based adjustment")
